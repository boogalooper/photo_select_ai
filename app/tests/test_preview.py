from __future__ import annotations

from io import BytesIO
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.core.preview import _extract_largest_embedded_jpeg, _load_raw


def _jpeg_bytes(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_extract_largest_embedded_jpeg_prefers_largest_preview(tmp_path):
    small = _jpeg_bytes((320, 240))
    large = _jpeg_bytes((1600, 1200))
    path = tmp_path / "sample.cr3"
    path.write_bytes(b"RAW-HEADER" + small + b"RAW-DATA" + large + b"RAW-TAIL")

    result = _extract_largest_embedded_jpeg(path)

    assert result is not None
    with Image.open(BytesIO(result)) as image:
        assert image.size == (1600, 1200)


def test_embedded_jpeg_scan_ignores_false_soi_before_real_preview(tmp_path):
    large = _jpeg_bytes((1200, 900))
    path = tmp_path / "sample.dng"
    # Random RAW payload can contain FF D8 / FF D9 byte pairs by chance.
    path.write_bytes(b"RAW" + b"\xff\xd8not-a-jpeg" + b"NOISE" + large + b"TAIL")

    result = _extract_largest_embedded_jpeg(path)

    assert result is not None
    with Image.open(BytesIO(result)) as image:
        assert image.size == (1200, 900)


def test_raw_loader_uses_larger_embedded_jpeg_before_postprocess(tmp_path, monkeypatch):
    small = _jpeg_bytes((320, 240))
    large = _jpeg_bytes((1600, 1200))
    path = tmp_path / "sample.nef"
    path.write_bytes(b"RAW-HEADER" + large + b"RAW-TAIL")
    postprocess_called = False

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_thumb(self):
            return SimpleNamespace(format="JPEG", data=small)

        def postprocess(self, **kwargs):
            nonlocal postprocess_called
            postprocess_called = True
            return np.zeros((600, 800, 3), dtype=np.uint8)

    fake_rawpy = SimpleNamespace(
        ThumbFormat=SimpleNamespace(JPEG="JPEG"),
        imread=lambda _path: FakeRaw(),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    rgb = _load_raw(path, long_edge=1000, half_size=True)

    assert rgb.shape[:2] == (750, 1000)
    assert postprocess_called is False


def test_raw_loader_uses_postprocess_only_when_no_embedded_preview(tmp_path, monkeypatch):
    path = tmp_path / "sample.arw"
    path.write_bytes(b"RAW-WITHOUT-JPEG")
    postprocess_called = False

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_thumb(self):
            raise RuntimeError("no thumbnail")

        def postprocess(self, **kwargs):
            nonlocal postprocess_called
            postprocess_called = True
            return np.zeros((600, 800, 3), dtype=np.uint8)

    fake_rawpy = SimpleNamespace(
        ThumbFormat=SimpleNamespace(JPEG="JPEG"),
        imread=lambda _path: FakeRaw(),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    rgb = _load_raw(path, long_edge=500, half_size=True)

    assert rgb.shape[:2] == (375, 500)
    assert postprocess_called is True
