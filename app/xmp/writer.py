from __future__ import annotations

import html
import os
import re
import struct
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from app.core.models import PhotoFile

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"

# Formats in which Adobe XMP can live in the image itself and for which we can
# update the XMP payload without rebuilding the image container. Unknown or
# unsupported RAW containers are deliberately left untouched; their sidecar is
# used instead.
_TIFF_LIKE_EXTENSIONS = {
    ".dng", ".tif", ".tiff", ".cr2", ".nef", ".nrw", ".arw", ".pef", ".srw"
}

_XMLNS_RE = re.compile(
    r"\bxmlns(?::(?P<prefix>[A-Za-z_][\w.\-]*))?\s*=\s*(?P<q>['\"])(?P<uri>.*?)(?P=q)",
    re.DOTALL,
)


def red_label_value(config: dict) -> str:
    x = config["xmp"]
    scheme = str(x.get("scheme", "bridge")).lower()
    if scheme == "lightroom":
        return str(x["lightroom_red"])
    if scheme == "custom":
        return str(x["custom_red"])
    return str(x["bridge_red"])


def yellow_label_value(config: dict) -> str:
    x = config["xmp"]
    scheme = str(x.get("scheme", "bridge")).lower()
    if scheme == "lightroom":
        return str(x.get("lightroom_yellow", "Yellow"))
    if scheme == "custom":
        return str(x.get("custom_yellow", "Second"))
    return str(x.get("bridge_yellow", "Second"))


class XmpWriter:
    """Change only xmp:Label while preserving all other metadata.

    Existing XMP is never rebuilt through an XML serializer.  The original XMP
    text is edited surgically so Camera Raw settings, masks, crop, keywords,
    ratings, custom namespaces, packet padding and formatting stay untouched.
    """

    def __init__(self, config: dict):
        self.config = config
        self.red = red_label_value(config)
        self.yellow = yellow_label_value(config)

    def clear_label(self, photo: PhotoFile, role: str) -> bool:
        """Remove only this profile's configured RED or YELLOW xmp:Label.

        Both embedded XMP (where safely supported) and an existing sidecar are
        checked.  No XMP file is deleted and no unrelated metadata is changed.
        """
        if role not in {"red", "yellow"}:
            raise ValueError(f"Unsupported selection role: {role}")
        label = self.red if role == "red" else self.yellow
        changed = False
        ext = photo.extension.lower()

        if ext in {".jpg", ".jpeg"} and bool(self.config["xmp"].get("jpeg_embedded", True)):
            changed = self._clear_jpeg_embedded_label(photo.path, label) or changed
        elif ext in _TIFF_LIKE_EXTENSIONS:
            changed = self._clear_fixed_embedded_label(photo.path, label, kind="tiff") or changed
        elif ext == ".psd":
            changed = self._clear_fixed_embedded_label(photo.path, label, kind="psd") or changed

        sidecar = photo.path.with_suffix(".xmp")
        update_sidecar = not (
            ext in {".jpg", ".jpeg"}
            and not bool(self.config["xmp"].get("update_existing_jpeg_sidecar", True))
        )
        if sidecar.exists() and update_sidecar:
            # A sidecar may coexist with embedded XMP.  Keep both stores free of
            # stale Photo Select AI labels while preserving every other field.
            changed = self._clear_sidecar_label(sidecar, label) or changed
        return changed

    def clear_red_label(self, photo: PhotoFile) -> bool:
        return self.clear_label(photo, "red")

    def _clear_sidecar_label(self, path: Path, expected_label: str) -> bool:
        original = path.read_bytes()
        updated, changed = _mutate_xmp_label_bytes(original, clear_value=expected_label)
        if not changed:
            return False
        _atomic_write_bytes(path, updated, suffix=".photosel-xmp.tmp")
        return True

    def _clear_jpeg_embedded_label(self, path: Path, expected_label: str) -> bool:
        # Standard JPEG XMP lives in an APP1 header segment.  Probe the header
        # first; the full JPEG is copied only when the target label is present.
        with path.open("rb") as fh:
            head = fh.read(2 * 1024 * 1024)
        if len(head) < 4 or head[:2] != b"\xff\xd8":
            return False
        existing_payload = _find_standard_xmp_payload(head)
        if existing_payload is None:
            return False
        updated_payload, changed = _mutate_xmp_label_bytes(
            existing_payload, clear_value=expected_label
        )
        if not changed:
            return False

        data = path.read_bytes()
        # Re-read from the complete JPEG in case the probe ended inside APP1.
        complete_payload = _find_standard_xmp_payload(data)
        if complete_payload is None:
            return False
        updated_payload, changed = _mutate_xmp_label_bytes(
            complete_payload, clear_value=expected_label
        )
        if not changed:
            return False
        replacement_segment = _jpeg_xmp_segment(updated_payload, path)
        rewritten = _replace_or_insert_standard_xmp(data, replacement_segment)
        _atomic_write_bytes(path, rewritten, suffix=".photosel.tmp", preserve_stat=True)
        return True

    def _clear_fixed_embedded_label(self, path: Path, expected_label: str, kind: str) -> bool:
        region = _find_tiff_xmp_region(path) if kind == "tiff" else _find_psd_xmp_region(path)
        if region is None:
            return False
        offset, size = region
        packet = _read_region(path, offset, size)
        updated, changed = _mutate_xmp_label_bytes(packet, clear_value=expected_label)
        if not changed:
            return False
        fitted = _fit_xmp_to_fixed_region(updated, size)
        if fitted is None:
            # Clearing a label always makes a packet smaller, so this is a
            # corruption/encoding edge case.  Refuse to rewrite the container.
            raise RuntimeError(f"Cannot safely clear embedded XMP label: {path}")
        _write_region(path, offset, fitted)
        return True

    def write(self, photo: PhotoFile, role: str) -> Path:
        if role not in {"red", "yellow"}:
            raise ValueError(f"Unsupported selection role: {role}")
        label = self.red if role == "red" else self.yellow
        ext = photo.extension.lower()

        if ext in {".jpg", ".jpeg"} and bool(self.config["xmp"].get("jpeg_embedded", True)):
            self._write_jpeg_embedded(photo.path, label)
            sidecar = photo.path.with_suffix(".xmp")
            if sidecar.exists() and bool(self.config["xmp"].get("update_existing_jpeg_sidecar", True)):
                self._write_sidecar(sidecar, label)
            return photo.path

        # DNG/TIFF and several TIFF-based RAW formats can contain XMP directly.
        # If an embedded packet already exists, update only its Label in place.
        # We never grow/rebuild the image container; if the packet has no room
        # for a new Label, fall back to a normal sidecar instead.
        if ext in _TIFF_LIKE_EXTENSIONS:
            if self._write_fixed_embedded_if_possible(photo.path, label, kind="tiff"):
                sidecar = photo.path.with_suffix(".xmp")
                if sidecar.exists():
                    self._write_sidecar(sidecar, label)
                return photo.path
        elif ext == ".psd":
            if self._write_fixed_embedded_if_possible(photo.path, label, kind="psd"):
                sidecar = photo.path.with_suffix(".xmp")
                if sidecar.exists():
                    self._write_sidecar(sidecar, label)
                return photo.path

        path = photo.path.with_suffix(".xmp")
        self._write_sidecar(path, label)
        return path

    def _write_sidecar(self, path: Path, label: str) -> None:
        if path.exists():
            original = path.read_bytes()
            updated, changed = _mutate_xmp_label_bytes(original, set_value=label)
            if not changed:
                return
            _atomic_write_bytes(path, updated, suffix=".photosel-xmp.tmp")
            return
        _atomic_write_bytes(path, _new_xmp_packet(label), suffix=".photosel-xmp.tmp")

    def _write_jpeg_embedded(self, path: Path, label: str) -> None:
        data = path.read_bytes()
        if len(data) < 4 or data[:2] != b"\xff\xd8":
            raise RuntimeError(f"Not a valid JPEG file: {path}")

        existing_payload = _find_standard_xmp_payload(data)
        if existing_payload is not None:
            packet, _changed = _mutate_xmp_label_bytes(existing_payload, set_value=label)
        else:
            packet = _new_xmp_packet(label)
        replacement_segment = _jpeg_xmp_segment(packet, path)
        rewritten = _replace_or_insert_standard_xmp(data, replacement_segment)
        _atomic_write_bytes(path, rewritten, suffix=".photosel.tmp", preserve_stat=True)

    def _write_fixed_embedded_if_possible(self, path: Path, label: str, kind: str) -> bool:
        region = _find_tiff_xmp_region(path) if kind == "tiff" else _find_psd_xmp_region(path)
        if region is None:
            return False
        offset, size = region
        original = _read_region(path, offset, size)
        updated, changed = _mutate_xmp_label_bytes(original, set_value=label)
        if not changed:
            return True
        fitted = _fit_xmp_to_fixed_region(updated, size)
        if fitted is None:
            return False
        _write_region(path, offset, fitted)
        return True


# ---------------------------------------------------------------------------
# Surgical XMP editing
# ---------------------------------------------------------------------------

def _decode_xml_bytes(data: bytes) -> tuple[str, str, bytes]:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8"), "utf-8", b"\xef\xbb\xbf"
    if data.startswith(b"\xff\xfe\x00\x00"):
        return data[4:].decode("utf-32-le"), "utf-32-le", b"\xff\xfe\x00\x00"
    if data.startswith(b"\x00\x00\xfe\xff"):
        return data[4:].decode("utf-32-be"), "utf-32-be", b"\x00\x00\xfe\xff"
    if data.startswith(b"\xff\xfe"):
        return data[2:].decode("utf-16-le"), "utf-16-le", b"\xff\xfe"
    if data.startswith(b"\xfe\xff"):
        return data[2:].decode("utf-16-be"), "utf-16-be", b"\xfe\xff"

    head = data[:256]
    match = re.search(br"<\?xml[^>]*\bencoding\s*=\s*['\"]([^'\"]+)['\"]", head, re.I)
    encoding = match.group(1).decode("ascii", "strict") if match else "utf-8"
    try:
        return data.decode(encoding), encoding, b""
    except (LookupError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Unsupported/invalid XMP text encoding: {encoding}") from exc


def _encode_xml_text(text: str, encoding: str, bom: bytes) -> bytes:
    try:
        return bom + text.encode(encoding)
    except (LookupError, UnicodeEncodeError) as exc:
        raise RuntimeError(f"Cannot preserve XMP text encoding: {encoding}") from exc


def _namespace_prefixes(text: str, uri: str) -> list[str]:
    prefixes: list[str] = []
    for match in _XMLNS_RE.finditer(text):
        if html.unescape(match.group("uri")) == uri and match.group("prefix"):
            prefixes.append(match.group("prefix"))
    # Most Adobe packets use these conventional prefixes.  Add them only when
    # actually declared for the requested URI; do not guess namespace meaning.
    return list(dict.fromkeys(prefixes))


def _declared_prefix_map(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in _XMLNS_RE.finditer(text):
        prefix = match.group("prefix")
        if prefix:
            result[prefix] = html.unescape(match.group("uri"))
    return result


def _xml_value(value: str) -> str:
    return html.unescape(value)


def _escape_attr(value: str, quote: str) -> str:
    escaped = xml_escape(value, {"\r": "&#13;", "\n": "&#10;", "\t": "&#9;"})
    if quote == '"':
        return escaped.replace('"', "&quot;")
    return escaped.replace("'", "&apos;")


def _label_attribute_regex(prefixes: list[str]) -> re.Pattern[str] | None:
    if not prefixes:
        return None
    names = "|".join(re.escape(p) for p in prefixes)
    return re.compile(
        rf"(?P<space>\s+)(?P<name>(?:{names}):Label)\s*=\s*(?P<q>['\"])(?P<value>.*?)(?P=q)",
        re.DOTALL,
    )


def _label_element_regex(prefixes: list[str]) -> re.Pattern[str] | None:
    if not prefixes:
        return None
    names = "|".join(re.escape(p) for p in prefixes)
    return re.compile(
        rf"<(?P<name>(?:{names}):Label)\b(?P<attrs>[^>]*)>(?P<value>.*?)</(?P=name)\s*>",
        re.DOTALL,
    )


def _clear_xmp_label_text(text: str, expected: str) -> tuple[str, bool]:
    prefixes = _namespace_prefixes(text, XMP_NS)
    changed = False

    attr_re = _label_attribute_regex(prefixes)
    if attr_re is not None:
        def attr_sub(match: re.Match[str]) -> str:
            nonlocal changed
            if _xml_value(match.group("value")) == expected:
                changed = True
                return ""
            return match.group(0)
        text = attr_re.sub(attr_sub, text)

    elem_re = _label_element_regex(prefixes)
    if elem_re is not None:
        def elem_sub(match: re.Match[str]) -> str:
            nonlocal changed
            if _xml_value(match.group("value")) == expected:
                changed = True
                return ""
            return match.group(0)
        text = elem_re.sub(elem_sub, text)

    return text, changed


def _set_xmp_label_text(text: str, value: str) -> tuple[str, bool]:
    prefixes = _namespace_prefixes(text, XMP_NS)
    changed = False
    found = False

    attr_re = _label_attribute_regex(prefixes)
    if attr_re is not None:
        def attr_sub(match: re.Match[str]) -> str:
            nonlocal changed, found
            found = True
            old = _xml_value(match.group("value"))
            if old == value:
                return match.group(0)
            changed = True
            q = match.group("q")
            return (
                f"{match.group('space')}{match.group('name')}={q}"
                f"{_escape_attr(value, q)}{q}"
            )
        text = attr_re.sub(attr_sub, text)

    elem_re = _label_element_regex(prefixes)
    if elem_re is not None:
        def elem_sub(match: re.Match[str]) -> str:
            nonlocal changed, found
            found = True
            old = _xml_value(match.group("value"))
            if old == value:
                return match.group(0)
            changed = True
            return (
                f"<{match.group('name')}{match.group('attrs')}>"
                f"{xml_escape(value)}</{match.group('name')}>"
            )
        text = elem_re.sub(elem_sub, text)

    if found:
        return text, changed

    # No existing Label.  Add one to the first rdf:Description without
    # reserializing any existing metadata.
    rdf_prefixes = _namespace_prefixes(text, RDF_NS)
    if not rdf_prefixes:
        raise RuntimeError("Existing XMP has no RDF namespace; refusing to rewrite metadata")
    rdf_names = "|".join(re.escape(p) for p in rdf_prefixes)
    desc_re = re.compile(rf"<(?P<name>(?:{rdf_names}):Description)\b(?P<body>[^>]*?)(?P<close>/?)>", re.DOTALL)
    desc = desc_re.search(text)
    if desc is None:
        raise RuntimeError("Existing XMP has no rdf:Description; refusing to rewrite metadata")

    prefix_map = _declared_prefix_map(text)
    xmp_prefix = prefixes[0] if prefixes else "xmp"
    if not prefixes and xmp_prefix in prefix_map and prefix_map[xmp_prefix] != XMP_NS:
        index = 1
        while f"psaiXmp{index}" in prefix_map:
            index += 1
        xmp_prefix = f"psaiXmp{index}"

    body = desc.group("body")
    addition = ""
    if not prefixes:
        addition += f' xmlns:{xmp_prefix}="{XMP_NS}"'
    addition += f' {xmp_prefix}:Label="{_escape_attr(value, chr(34))}"'
    replacement = f"<{desc.group('name')}{body}{addition}{desc.group('close')}>"
    return text[:desc.start()] + replacement + text[desc.end():], True


def _mutate_xmp_label_bytes(
    data: bytes,
    *,
    set_value: str | None = None,
    clear_value: str | None = None,
) -> tuple[bytes, bool]:
    if (set_value is None) == (clear_value is None):
        raise ValueError("Exactly one of set_value or clear_value must be provided")
    text, encoding, bom = _decode_xml_bytes(data)
    if set_value is not None:
        updated, changed = _set_xmp_label_text(text, set_value)
    else:
        updated, changed = _clear_xmp_label_text(text, str(clear_value))
    if not changed:
        return data, False
    return _encode_xml_text(updated, encoding, bom), True


def _new_xmp_packet(label: str) -> bytes:
    value = _escape_attr(label, '"')
    return (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + (
            f'<x:xmpmeta xmlns:x="{X_NS}"><rdf:RDF xmlns:rdf="{RDF_NS}">'
            f'<rdf:Description rdf:about="" xmlns:xmp="{XMP_NS}" xmp:Label="{value}"/>'
            f'</rdf:RDF></x:xmpmeta>'
        ).encode("utf-8")
        + b'\n<?xpacket end="w"?>'
    )


# Compatibility helpers retained for tests/tools that inspect a packet.
def _serialize_xmp_packet(tree) -> bytes:  # pragma: no cover - legacy helper
    import xml.etree.ElementTree as ET
    xml = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=False)
    return (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + xml
        + b'\n<?xpacket end="w"?>'
    )


def _parse_xmp_payload(payload: bytes):
    import xml.etree.ElementTree as ET
    cleaned = re.sub(br"<\?xpacket\b.*?\?>", b"", payload, flags=re.DOTALL).strip()
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError as exc:
        raise RuntimeError("Existing embedded XMP is not valid XML") from exc
    return ET.ElementTree(root)


# ---------------------------------------------------------------------------
# JPEG container
# ---------------------------------------------------------------------------

def _iter_jpeg_segments(data: bytes):
    """Yield (start, end, marker, payload) for header segments before SOS."""
    pos = 2
    n = len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            return
        marker_start = pos
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            return
        marker = data[pos]
        pos += 1
        if marker == 0xDA:  # Start of Scan; entropy-coded data follows.
            yield marker_start, n, marker, data[pos:]
            return
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            yield marker_start, pos, marker, b""
            continue
        if pos + 2 > n:
            return
        seg_len = int.from_bytes(data[pos:pos + 2], "big")
        if seg_len < 2 or pos + seg_len > n:
            return
        payload_start = pos + 2
        end = pos + seg_len
        yield marker_start, end, marker, data[payload_start:end]
        pos = end


def _find_standard_xmp_payload(data: bytes) -> bytes | None:
    for _start, _end, marker, payload in _iter_jpeg_segments(data):
        if marker == 0xDA:
            break
        if marker == 0xE1 and payload.startswith(XMP_JPEG_HEADER):
            return payload[len(XMP_JPEG_HEADER):]
    return None


def _jpeg_xmp_segment(packet: bytes, path: Path) -> bytes:
    payload = XMP_JPEG_HEADER + packet
    if len(payload) + 2 > 0xFFFF:
        raise RuntimeError(
            f"Embedded XMP is too large for a standard JPEG APP1 segment; refusing to overwrite metadata: {path}"
        )
    return b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload


def _replace_or_insert_standard_xmp(data: bytes, replacement_segment: bytes) -> bytes:
    segments = list(_iter_jpeg_segments(data))
    for start, end, marker, payload in segments:
        if marker == 0xDA:
            break
        if marker == 0xE1 and payload.startswith(XMP_JPEG_HEADER):
            return data[:start] + replacement_segment + data[end:]

    # No existing standard XMP. Insert after initial metadata segments while
    # leaving Exif/ICC/other JPEG data byte-for-byte unchanged.
    insert_at = 2
    for _start, end, marker, _payload in segments:
        if marker in {0xE0, 0xE1, 0xE2}:
            insert_at = end
            continue
        break
    return data[:insert_at] + replacement_segment + data[insert_at:]


# ---------------------------------------------------------------------------
# Fixed-size embedded XMP regions (TIFF/DNG/PSD)
# ---------------------------------------------------------------------------

def _fit_xmp_to_fixed_region(packet: bytes, size: int) -> bytes | None:
    if len(packet) > size:
        # Adobe packets commonly reserve whitespace around the closing xpacket
        # PI. Consume only that padding; metadata bytes themselves stay intact.
        excess = len(packet) - size
        end_pi = packet.rfind(b"<?xpacket end=")
        if end_pi >= 0:
            pi_close = packet.find(b"?>", end_pi)
            if pi_close >= 0:
                pi_close += 2
                trailing_end = len(packet)
                trailing_start = trailing_end
                while trailing_start > pi_close and packet[trailing_start - 1] in b" \t\r\n\x00":
                    trailing_start -= 1
                take = min(excess, trailing_end - trailing_start)
                if take:
                    packet = packet[:len(packet) - take]
                    excess -= take
                    end_pi = packet.rfind(b"<?xpacket end=")
            if excess > 0 and end_pi >= 0:
                pad_start = end_pi
                while pad_start > 0 and packet[pad_start - 1] in b" \t\r\n\x00":
                    pad_start -= 1
                take = min(excess, end_pi - pad_start)
                if take:
                    packet = packet[:pad_start] + packet[pad_start + take:]
                    excess -= take
        else:
            # Packets without xpacket processing instructions may still have
            # harmless trailing XML whitespace reserved by the container.
            trailing_start = len(packet)
            while trailing_start > 0 and packet[trailing_start - 1] in b" \t\r\n\x00":
                trailing_start -= 1
            take = min(excess, len(packet) - trailing_start)
            if take:
                packet = packet[:len(packet) - take]
                excess -= take
        if len(packet) > size:
            return None
    if len(packet) < size:
        # Whitespace after the XML packet is safe and keeps the TIFF/PSD field
        # length unchanged, so no offsets in the image container are touched.
        packet = packet + b" " * (size - len(packet))
    return packet


def _find_tiff_xmp_region(path: Path) -> tuple[int, int] | None:
    """Return (offset, byte_count) for TIFF tag 700 (XMP), if safely readable."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(16)
            if len(head) < 8:
                return None
            if head[:2] == b"II":
                endian = "<"
            elif head[:2] == b"MM":
                endian = ">"
            else:
                return None
            magic = struct.unpack(endian + "H", head[2:4])[0]

            if magic == 42:  # Classic TIFF
                first_ifd = struct.unpack(endian + "I", head[4:8])[0]
                return _scan_classic_tiff_ifds(fh, endian, first_ifd, file_size)
            if magic == 43 and len(head) >= 16:  # BigTIFF
                offset_size, zero = struct.unpack(endian + "HH", head[4:8])
                if offset_size != 8 or zero != 0:
                    return None
                first_ifd = struct.unpack(endian + "Q", head[8:16])[0]
                return _scan_bigtiff_ifds(fh, endian, first_ifd, file_size)
    except (OSError, struct.error, OverflowError):
        return None
    return None


def _scan_classic_tiff_ifds(fh, endian: str, offset: int, file_size: int) -> tuple[int, int] | None:
    visited: set[int] = set()
    for _ in range(8):
        if offset == 0 or offset in visited or offset + 2 > file_size:
            return None
        visited.add(offset)
        fh.seek(offset)
        raw = fh.read(2)
        if len(raw) != 2:
            return None
        count = struct.unpack(endian + "H", raw)[0]
        if count > 10000 or offset + 2 + count * 12 + 4 > file_size:
            return None
        for index in range(count):
            entry_pos = offset + 2 + index * 12
            fh.seek(entry_pos)
            entry = fh.read(12)
            if len(entry) != 12:
                return None
            tag, typ, item_count = struct.unpack(endian + "HHI", entry[:8])
            if tag == 700:
                size = _tiff_data_size(typ, item_count)
                if size is None or size <= 0:
                    return None
                if size <= 4:
                    data_offset = entry_pos + 8
                else:
                    data_offset = struct.unpack(endian + "I", entry[8:12])[0]
                if data_offset < 0 or data_offset + size > file_size:
                    return None
                return data_offset, size
        fh.seek(offset + 2 + count * 12)
        raw_next = fh.read(4)
        if len(raw_next) != 4:
            return None
        offset = struct.unpack(endian + "I", raw_next)[0]
    return None


def _scan_bigtiff_ifds(fh, endian: str, offset: int, file_size: int) -> tuple[int, int] | None:
    visited: set[int] = set()
    for _ in range(8):
        if offset == 0 or offset in visited or offset + 8 > file_size:
            return None
        visited.add(offset)
        fh.seek(offset)
        raw = fh.read(8)
        if len(raw) != 8:
            return None
        count = struct.unpack(endian + "Q", raw)[0]
        if count > 10000 or offset + 8 + count * 20 + 8 > file_size:
            return None
        for index in range(count):
            entry_pos = offset + 8 + index * 20
            fh.seek(entry_pos)
            entry = fh.read(20)
            if len(entry) != 20:
                return None
            tag, typ = struct.unpack(endian + "HH", entry[:4])
            item_count = struct.unpack(endian + "Q", entry[4:12])[0]
            if tag == 700:
                size = _tiff_data_size(typ, item_count)
                if size is None or size <= 0:
                    return None
                if size <= 8:
                    data_offset = entry_pos + 12
                else:
                    data_offset = struct.unpack(endian + "Q", entry[12:20])[0]
                if data_offset < 0 or data_offset + size > file_size:
                    return None
                return data_offset, size
        fh.seek(offset + 8 + count * 20)
        raw_next = fh.read(8)
        if len(raw_next) != 8:
            return None
        offset = struct.unpack(endian + "Q", raw_next)[0]
    return None


def _tiff_data_size(typ: int, count: int) -> int | None:
    sizes = {
        1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2,
        9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8,
    }
    unit = sizes.get(typ)
    if unit is None:
        return None
    if count < 0 or count > (1 << 31):
        return None
    return unit * count


def _find_psd_xmp_region(path: Path) -> tuple[int, int] | None:
    """Return the Photoshop image-resource XMP payload (resource id 1060)."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as fh:
            header = fh.read(26)
            if len(header) != 26 or header[:4] != b"8BPS":
                return None
            version = int.from_bytes(header[4:6], "big")
            if version not in {1, 2}:
                return None
            color_len_raw = fh.read(4)
            if len(color_len_raw) != 4:
                return None
            color_len = int.from_bytes(color_len_raw, "big")
            fh.seek(color_len, 1)
            res_len_raw = fh.read(4)
            if len(res_len_raw) != 4:
                return None
            res_len = int.from_bytes(res_len_raw, "big")
            resources_start = fh.tell()
            resources_end = resources_start + res_len
            if resources_end > file_size:
                return None
            while fh.tell() + 12 <= resources_end:
                if fh.read(4) != b"8BIM":
                    return None
                resource_id_raw = fh.read(2)
                if len(resource_id_raw) != 2:
                    return None
                resource_id = int.from_bytes(resource_id_raw, "big")
                name_len_raw = fh.read(1)
                if not name_len_raw:
                    return None
                name_len = name_len_raw[0]
                fh.seek(name_len, 1)
                # Pascal string including its one-byte length is padded to even.
                if (1 + name_len) % 2:
                    fh.seek(1, 1)
                size_raw = fh.read(4)
                if len(size_raw) != 4:
                    return None
                data_size = int.from_bytes(size_raw, "big")
                data_offset = fh.tell()
                if data_offset + data_size > resources_end:
                    return None
                if resource_id == 1060:
                    return data_offset, data_size
                fh.seek(data_size + (data_size % 2), 1)
    except OSError:
        return None
    return None


def _read_region(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read(size)
    if len(data) != size:
        raise RuntimeError(f"Cannot read embedded XMP safely: {path}")
    return data


def _write_region(path: Path, offset: int, data: bytes) -> None:
    # Same-length in-place update only.  No image offsets, pixel data or other
    # metadata blocks are rewritten.
    with path.open("r+b") as fh:
        fh.seek(offset)
        written = fh.write(data)
        if written != len(data):
            raise RuntimeError(f"Cannot write embedded XMP safely: {path}")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Safe file replacement
# ---------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, data: bytes, *, suffix: str, preserve_stat: bool = False) -> None:
    stat = path.stat() if preserve_stat and path.exists() else None
    tmp = path.with_name(path.name + suffix)
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if stat is not None:
            try:
                os.chmod(tmp, stat.st_mode)
            except OSError:
                pass
        os.replace(tmp, path)
        if stat is not None:
            # Keep capture-file timestamps stable; only metadata inside changed.
            try:
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            except OSError:
                pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
