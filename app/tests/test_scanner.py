from __future__ import annotations

import logging
from datetime import datetime

from app.core import scanner


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
