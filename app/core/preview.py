from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf",
    ".orf", ".rw2", ".dng", ".pef", ".srw", ".3fr", ".fff", ".iiq",
    ".mos", ".mrw", ".rwl", ".x3f",
}


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


def _load_raw(path: Path, long_edge: int, half_size: bool) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(BytesIO(thumb.data)) as image:
                    return _resize_rgb(image, long_edge)
            arr = np.asarray(thumb.data, dtype=np.uint8)
            with Image.fromarray(arr, mode="RGB") as image:
                return _resize_rgb(image, long_edge)
        except Exception:
            arr = raw.postprocess(
                half_size=bool(half_size),
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
            with Image.fromarray(arr, mode="RGB") as image:
                return _resize_rgb(image, long_edge)
