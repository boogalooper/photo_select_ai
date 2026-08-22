from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from app.core.models import PhotoFile

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("xmp", XMP_NS)


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
    def __init__(self, config: dict):
        self.config = config
        self.red = red_label_value(config)
        self.yellow = yellow_label_value(config)

    def clear_label(self, photo: PhotoFile, role: str) -> bool:
        """Remove only this profile's configured RED or YELLOW label."""
        label = self.red if role == "red" else self.yellow
        changed = False
        if photo.extension.lower() in {".jpg", ".jpeg"} and bool(self.config["xmp"].get("jpeg_embedded", True)):
            changed = self._clear_jpeg_embedded_label(photo.path, label) or changed
            sidecar = photo.path.with_suffix(".xmp")
            if sidecar.exists() and bool(self.config["xmp"].get("update_existing_jpeg_sidecar", True)):
                changed = self._clear_sidecar_label(sidecar, label) or changed
            return changed
        sidecar = photo.path.with_suffix(".xmp")
        if sidecar.exists():
            changed = self._clear_sidecar_label(sidecar, label) or changed
        return changed

    def clear_red_label(self, photo: PhotoFile) -> bool:
        return self.clear_label(photo, "red")

    def _clear_sidecar_label(self, path: Path, expected_label: str) -> bool:
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise RuntimeError(f"Existing XMP is not valid XML: {path}") from exc
        if not self._remove_matching_label(tree, expected_label):
            return False
        tmp = path.with_name(path.name + ".tmp")
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
        os.replace(tmp, path)
        return True

    def _clear_jpeg_embedded_label(self, path: Path, expected_label: str) -> bool:
        # XMP APP1 lives in the JPEG header. Probe only the first MiB so a
        # repeated evaluation run does not read every 20–50 MB JPEG in full.
        # The complete file is read/re-written only when the target RED label
        # is actually present and must be removed.
        with path.open("rb") as fh:
            head = fh.read(1024 * 1024)
        if len(head) < 4 or head[:2] != b"\xff\xd8":
            return False
        existing_payload = _find_standard_xmp_payload(head)
        if existing_payload is None:
            return False
        tree = _parse_xmp_payload(existing_payload)
        if not self._remove_matching_label(tree, expected_label):
            return False

        data = path.read_bytes()
        packet = _serialize_xmp_packet(tree)
        payload = XMP_JPEG_HEADER + packet
        replacement_segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
        rewritten = _replace_or_insert_standard_xmp(data, replacement_segment)
        tmp = path.with_name(path.name + ".photosel.tmp")
        tmp.write_bytes(rewritten)
        os.replace(tmp, path)
        return True

    @staticmethod
    def _remove_matching_label(tree: ET.ElementTree, expected_label: str) -> bool:
        root = tree.getroot()
        changed = False
        for desc in root.findall(f".//{{{RDF_NS}}}Description"):
            key = f"{{{XMP_NS}}}Label"
            if desc.attrib.get(key) == expected_label:
                del desc.attrib[key]
                changed = True
            for child in list(desc):
                if child.tag == key and (child.text or "") == expected_label:
                    desc.remove(child)
                    changed = True
        return changed

    def write(self, photo: PhotoFile, role: str) -> Path:
        if role not in {"red", "yellow"}:
            raise ValueError(f"Unsupported selection role: {role}")
        label = self.red if role == "red" else self.yellow
        if photo.extension.lower() in {".jpg", ".jpeg"} and bool(self.config["xmp"].get("jpeg_embedded", True)):
            self._write_jpeg_embedded(photo.path, label)
            # If a JPEG already has a sidecar (for example from Camera Raw or an
            # older Photo Select AI build), keep its Label consistent too. Do
            # not create a sidecar for a normal JPEG.
            sidecar = photo.path.with_suffix(".xmp")
            if sidecar.exists() and bool(self.config["xmp"].get("update_existing_jpeg_sidecar", True)):
                self._write_sidecar(sidecar, label)
            return photo.path

        path = photo.path.with_suffix(".xmp")
        self._write_sidecar(path, label)
        return path

    def _write_sidecar(self, path: Path, label: str) -> None:
        tree = self._load_or_create(path)
        self._set_label(tree, label)
        tmp = path.with_name(path.name + ".tmp")
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
        os.replace(tmp, path)

    def _write_jpeg_embedded(self, path: Path, label: str) -> None:
        data = path.read_bytes()
        if len(data) < 4 or data[:2] != b"\xff\xd8":
            raise RuntimeError(f"Not a valid JPEG file: {path}")

        existing_payload = _find_standard_xmp_payload(data)
        if existing_payload is not None:
            tree = _parse_xmp_payload(existing_payload)
        else:
            tree = self._new_tree()
        self._set_label(tree, label)
        packet = _serialize_xmp_packet(tree)
        payload = XMP_JPEG_HEADER + packet
        if len(payload) + 2 > 0xFFFF:
            raise RuntimeError(f"Embedded XMP is too large for a standard JPEG APP1 segment: {path}")

        replacement_segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
        rewritten = _replace_or_insert_standard_xmp(data, replacement_segment)

        tmp = path.with_name(path.name + ".photosel.tmp")
        tmp.write_bytes(rewritten)
        os.replace(tmp, path)

    def _set_label(self, tree: ET.ElementTree, label: str) -> None:
        root = tree.getroot()
        desc = root.find(f".//{{{RDF_NS}}}Description")
        if desc is None:
            rdf = root.find(f".//{{{RDF_NS}}}RDF")
            if rdf is None:
                rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
            desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
            desc.set(f"{{{RDF_NS}}}about", "")
        desc.set(f"{{{XMP_NS}}}Label", label)

    def _load_or_create(self, path: Path) -> ET.ElementTree:
        if path.exists():
            try:
                return ET.parse(path)
            except ET.ParseError as exc:
                raise RuntimeError(f"Existing XMP is not valid XML: {path}") from exc
        return self._new_tree()

    @staticmethod
    def _new_tree() -> ET.ElementTree:
        root = ET.Element(f"{{{X_NS}}}xmpmeta")
        rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
        desc.set(f"{{{RDF_NS}}}about", "")
        return ET.ElementTree(root)


def _serialize_xmp_packet(tree: ET.ElementTree) -> bytes:
    xml = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=False)
    return (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + xml
        + b'\n<?xpacket end="w"?>'
    )


def _parse_xmp_payload(payload: bytes) -> ET.ElementTree:
    # Remove Adobe xpacket processing instructions around the XML root. ElementTree
    # otherwise reports "junk after document element" for a trailing xpacket PI.
    cleaned = re.sub(br"<\?xpacket\b.*?\?>", b"", payload, flags=re.DOTALL).strip()
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError as exc:
        raise RuntimeError("Existing embedded JPEG XMP is not valid XML; refusing to overwrite metadata") from exc
    return ET.ElementTree(root)


def _iter_jpeg_segments(data: bytes):
    """Yield (start, end, marker, payload) for header segments before SOS."""
    pos = 2  # after SOI
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


def _replace_or_insert_standard_xmp(data: bytes, replacement_segment: bytes) -> bytes:
    segments = list(_iter_jpeg_segments(data))
    for start, end, marker, payload in segments:
        if marker == 0xDA:
            break
        if marker == 0xE1 and payload.startswith(XMP_JPEG_HEADER):
            return data[:start] + replacement_segment + data[end:]

    # No existing standard XMP. Insert after the initial APP0/APP1 metadata
    # segments (JFIF/Exif/ICC-related headers remain ahead of it).
    insert_at = 2
    for start, end, marker, _payload in segments:
        if marker in {0xE0, 0xE1}:
            insert_at = end
            continue
        break
    return data[:insert_at] + replacement_segment + data[insert_at:]
