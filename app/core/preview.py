from __future__ import annotations

from io import BytesIO
import mmap
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf",
    ".orf", ".rw2", ".dng", ".pef", ".srw", ".3fr", ".fff", ".iiq",
    ".mos", ".mrw", ".rwl", ".x3f",
}

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF,
}
_MAX_EMBEDDED_JPEG_CANDIDATES = 8192
_JPEG_HEADER_SCAN_BYTES = 512 * 1024


def _resize_rgb(image: Image.Image, long_edge: int) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > long_edge:
        image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def load_preview(path: Path, long_edge: int, raw_fallback_half_size: bool = True) -> np.ndarray:
    if path.suffix.lower() in RAW_EXTENSIONS:
        return _load_raw(path, long_edge, raw_fallback_half_size)
    with Image.open(path) as image:
        return _resize_rgb(image, long_edge)


def _jpeg_dimensions(data: mmap.mmap, start: int, end: int) -> tuple[int, int] | None:
    """Read JPEG dimensions from marker headers without decoding image pixels."""
    pos = start + 2
    while pos + 3 < end:
        if data[pos] != 0xFF:
            pos += 1
            continue

        while pos < end and data[pos] == 0xFF:
            pos += 1
        if pos >= end:
            return None

        marker = data[pos]
        pos += 1

        # Stand-alone markers have no segment length.
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue

        # Image data begins after SOS; SOF must have appeared before it.
        if marker == 0xDA:
            return None

        if pos + 2 > end:
            return None
        segment_length = (data[pos] << 8) | data[pos + 1]
        if segment_length < 2 or pos + segment_length > end:
            return None

        if marker in _JPEG_SOF_MARKERS:
            if segment_length < 7:
                return None
            height = (data[pos + 3] << 8) | data[pos + 4]
            width = (data[pos + 5] << 8) | data[pos + 6]
            if width > 0 and height > 0:
                return width, height
            return None

        pos += segment_length

    return None


def _extract_largest_embedded_jpeg(path: Path, *, larger_than_area: int = 0) -> bytes | None:
    """Find the largest valid JPEG embedded anywhere inside a RAW container.

    This is intentionally a fallback behind LibRaw/rawpy.  It is useful for
    newer RAW containers that carry several JPEG previews but where LibRaw
    exposes only a smaller preview (or cannot expose one at all).  ``mmap``
    keeps the scan memory-efficient even for very large RAW files.
    """
    try:
        with path.open("rb") as file_obj:
            with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as data:
                candidates: list[tuple[int, int, int]] = []  # area, start, end
                search_from = 0
                checked = 0
                file_size = len(data)

                while checked < _MAX_EMBEDDED_JPEG_CANDIDATES:
                    start = data.find(_JPEG_SOI, search_from)
                    if start < 0:
                        break
                    checked += 1

                    # A random RAW byte sequence can contain FF D8 by chance.
                    # Validate JPEG marker headers before looking for its EOI so
                    # a false SOI cannot hide a real JPEG that begins later.
                    header_end = min(file_size, start + _JPEG_HEADER_SCAN_BYTES)
                    dimensions = _jpeg_dimensions(data, start, header_end)
                    if dimensions is not None:
                        end_marker = data.find(_JPEG_EOI, start + 2)
                        if end_marker >= 0:
                            width, height = dimensions
                            area = width * height
                            if area > larger_than_area:
                                candidates.append((area, start, end_marker + len(_JPEG_EOI)))

                    # Always continue from the next byte after this SOI.  This
                    # is deliberate: invalid/random SOI markers may surround a
                    # perfectly valid embedded JPEG later in the RAW stream.
                    search_from = start + len(_JPEG_SOI)

                # Prefer the largest candidate, but validate it with Pillow.
                # A random RAW region can very rarely mimic enough JPEG marker
                # structure to pass the cheap header parser.
                for _area, start, end in sorted(candidates, reverse=True):
                    jpeg = bytes(data[start:end])
                    try:
                        with Image.open(BytesIO(jpeg)) as image:
                            if image.format != "JPEG":
                                continue
                            image.verify()
                        return jpeg
                    except Exception:
                        continue
                return None
    except (OSError, ValueError):
        return None


def _load_raw(path: Path, long_edge: int, half_size: bool) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        thumb_rgb: np.ndarray | None = None
        thumb_area = 0
        thumb_long_edge = 0

        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(BytesIO(thumb.data)) as image:
                    image.load()
                    thumb_area = int(image.width) * int(image.height)
                    thumb_long_edge = max(image.size)
                    thumb_rgb = _resize_rgb(image, long_edge)
            else:
                arr = np.asarray(thumb.data, dtype=np.uint8)
                with Image.fromarray(arr, mode="RGB") as image:
                    thumb_area = int(image.width) * int(image.height)
                    thumb_long_edge = max(image.size)
                    thumb_rgb = _resize_rgb(image, long_edge)
        except Exception:
            # Some new RAW variants contain a usable embedded JPEG that LibRaw
            # cannot expose through extract_thumb().  The direct container scan
            # below gives those files another fast path before raw demosaicing.
            thumb_rgb = None

        # rawpy/LibRaw already returns the larger of its thumbnail/preview.
        # Avoid scanning a large RAW file when that preview already satisfies
        # the requested working resolution.
        if thumb_rgb is not None and thumb_long_edge >= long_edge:
            return thumb_rgb

        embedded_jpeg = _extract_largest_embedded_jpeg(path, larger_than_area=thumb_area)
        if embedded_jpeg is not None:
            try:
                with Image.open(BytesIO(embedded_jpeg)) as image:
                    image.load()
                    return _resize_rgb(image, long_edge)
            except Exception:
                pass

        # Even a smaller embedded preview is preferable to expensive RAW
        # demosaicing; preserve the old fallback only for files with no usable
        # embedded image at all.
        if thumb_rgb is not None:
            return thumb_rgb

        arr = raw.postprocess(
            half_size=bool(half_size),
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8,
        )
        with Image.fromarray(arr, mode="RGB") as image:
            return _resize_rgb(image, long_edge)
