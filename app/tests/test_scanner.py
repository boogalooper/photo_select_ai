from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

from app.core import scanner


def test_capture_time_scan_does_not_extract_raw_thumbnail(monkeypatch, tmp_path):
    photo = tmp_path / "IMG_0001.CR2"
    photo.write_bytes(b"raw container placeholder")
    received = {}

    def fake_process_file(_file, **kwargs):
        received.update(kwargs)
        return {"EXIF DateTimeOriginal": "2026:08:28 10:11:12"}

    monkeypatch.setattr(scanner.exifread, "process_file", fake_process_file)
    capture_time = scanner.read_capture_time(photo)

    assert capture_time == datetime(2026, 8, 28, 10, 11, 12)
    assert received["details"] is False
    assert received["extract_thumbnail"] is False
    assert received["stop_tag"] == "EXIF DateTimeOriginal"


def test_expected_exif_container_warning_is_suppressed(monkeypatch, tmp_path, caplog):
    photo = tmp_path / "supported.psd"
    photo.write_bytes(b"not needed by the mocked metadata reader")

    def fake_process_file(*_args, **_kwargs):
        logging.getLogger("exifread").warning("File format not recognized.")
        return {}

    monkeypatch.setattr(scanner.exifread, "process_file", fake_process_file)
    with caplog.at_level(logging.WARNING, logger="exifread"):
        capture_time = scanner.read_capture_time(photo)

    assert isinstance(capture_time, datetime)
    assert "File format not recognized." not in caplog.messages


def test_other_exif_warnings_are_not_hidden(caplog):
    with caplog.at_level(logging.WARNING, logger="exifread"):
        logging.getLogger("exifread").warning("Unexpected EXIF corruption")

    assert "Unexpected EXIF corruption" in caplog.messages


def test_parallel_scan_reports_live_progress_and_preserves_sorted_order(monkeypatch, tmp_path):
    paths = [tmp_path / f"IMG_{number:04d}.CR2" for number in (4, 1, 3, 2)]
    for path in paths:
        path.write_bytes(b"raw")

    threads: set[str] = set()

    def fake_capture_time(path):
        threads.add(threading.current_thread().name)
        time.sleep(0.01)
        return datetime(2026, 1, 1, 10, 0, int(path.stem[-1]))

    monkeypatch.setattr(scanner, "read_capture_time", fake_capture_time)
    events = []
    photos = scanner.scan_photos(
        tmp_path, [".cr2"], workers=2,
        progress=lambda done, total, path: events.append((done, total, path)),
    )

    assert [photo.path.name for photo in photos] == [
        "IMG_0001.CR2", "IMG_0002.CR2", "IMG_0003.CR2", "IMG_0004.CR2"
    ]
    assert events[0][:2] == (0, 4)
    assert events[-1][:2] == (4, 4)
    assert len({name for name in threads if name.startswith("photo-metadata")}) == 2


def test_scan_checks_cancellation_during_metadata_read(monkeypatch, tmp_path):
    for number in range(8):
        (tmp_path / f"IMG_{number:04d}.CR2").write_bytes(b"raw")

    monkeypatch.setattr(scanner, "read_capture_time", lambda _path: datetime(2026, 1, 1))
    calls = 0

    def check_cancelled():
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise RuntimeError("cancelled")

    try:
        scanner.scan_photos(tmp_path, [".cr2"], workers=2, check_cancelled=check_cancelled)
    except RuntimeError as exc:
        assert str(exc) == "cancelled"
    else:
        raise AssertionError("scan did not stop after cancellation")
