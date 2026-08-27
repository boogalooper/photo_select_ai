from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import exifread

from .models import PhotoFile

_DIGITS = re.compile(r"(\d+)(?!.*\d)")
_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")


class _IgnoreUnsupportedExifContainer(logging.Filter):
    """Hide ExifRead's expected warning for otherwise supported images.

    ExifRead does not understand every container that the preview loader can
    decode (notably PSD and some newer RAW variants).  In that case capture
    time deliberately falls back to the file timestamp, so the warning is not
    an image-read failure and should not be shown to the user.  Other ExifRead
    warnings remain visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno == logging.WARNING
            and record.getMessage() == "File format not recognized."
        )


logging.getLogger("exifread").addFilter(_IgnoreUnsupportedExifContainer())


def _sequence_number(path: Path) -> int | None:
    match = _DIGITS.search(path.stem)
    return int(match.group(1)) if match else None


def _parse_date(value: str) -> datetime | None:
    value = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            pass
    return None


def read_capture_time(path: Path) -> datetime:
    try:
        with path.open("rb") as fh:
            tags = exifread.process_file(
                fh,
                details=False,
                stop_tag="EXIF DateTimeOriginal",
            )
        for key in ("EXIF DateTimeOriginal", "Image DateTime"):
            if key in tags:
                dt = _parse_date(str(tags[key]))
                if dt:
                    return dt
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def count_supported_photos(folder: Path, extensions: Iterable[str], recursive: bool = True) -> int:
    """Count supported files quickly for GUI validation; does not parse EXIF."""
    if not folder.is_dir():
        return 0
    extset = {e.lower() for e in extensions}
    count = 0
    if recursive:
        try:
            for _root, _dirs, files in os.walk(folder):
                count += sum(1 for name in files if Path(name).suffix.lower() in extset)
        except OSError:
            return 0
        return count
    try:
        return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extset)
    except OSError:
        return 0


def has_supported_photos(folder: Path, extensions: Iterable[str], recursive: bool = True) -> bool:
    """Fast existence check for the GUI; does not parse EXIF."""
    if not folder.is_dir():
        return False
    extset = {e.lower() for e in extensions}
    if recursive:
        try:
            for root, _dirs, files in os.walk(folder):
                if any(Path(name).suffix.lower() in extset for name in files):
                    return True
        except OSError:
            return False
        return False
    try:
        return any(p.is_file() and p.suffix.lower() in extset for p in folder.iterdir())
    except OSError:
        return False


def scan_photos(folder: Path, extensions: Iterable[str], recursive: bool = True) -> list[PhotoFile]:
    extset = {e.lower() for e in extensions}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = [p for p in iterator if p.is_file() and p.suffix.lower() in extset]

    photos = [
        PhotoFile(
            path=p,
            capture_time=read_capture_time(p),
            sequence_number=_sequence_number(p),
            extension=p.suffix.lower(),
        )
        for p in paths
    ]
    photos.sort(key=lambda x: (x.capture_time, x.sequence_number if x.sequence_number is not None else 10**15, x.path.name.lower()))
    return photos
