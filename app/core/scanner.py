from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import exifread

from app.paths import ROOT
from .models import PhotoFile
from .preview import RAW_EXTENSIONS

_DIGITS = re.compile(r"(\d+)(?!.*\d)")
_TRAILING_DIGITS = re.compile(r"(\d+)$")
_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")
_CACHE_VERSION = 1
_DEFAULT_CACHE_PATH = ROOT / "runtime" / "cache" / "capture_times.json"
_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
_SENTINEL_SEQUENCE = 10**15
_LOG = logging.getLogger("photo_select_ai")


@dataclass(slots=True)
class _QuickFile:
    path: Path
    sequence_number: int | None
    source_key: tuple[str, str]
    size: int
    mtime_ns: int
    mtime: float


@dataclass(slots=True)
class ScanReport:
    files_discovered: int = 0
    files_returned: int = 0
    files_skipped: int = 0
    exif_reads: int = 0
    exif_failures: int = 0
    cache_hits: int = 0
    time_from_exif: int = 0
    time_from_file: int = 0
    order_from_name: int = 0
    paired_jpeg_skipped: int = 0
    metadata_only_photos: list[PhotoFile] = field(default_factory=list)
    pair_preview_fallbacks: dict[Path, Path] = field(default_factory=dict)


class _IgnoreUnsupportedExifContainer(logging.Filter):
    """Hide ExifRead's expected warning for otherwise supported images."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno == logging.WARNING
            and record.getMessage() == "File format not recognized."
        )


logging.getLogger("exifread").addFilter(_IgnoreUnsupportedExifContainer())


def _sequence_number(path: Path) -> int | None:
    match = _DIGITS.search(path.stem)
    return int(match.group(1)) if match else None


def _sequence_source(path: Path) -> tuple[str, str]:
    # Sequence counters are comparable only when the stem actually ends in
    # digits.  Names such as IMG_0001_edit must not share the IMG_ source of
    # camera-native IMG_0002 files merely because they contain a number.
    match = _TRAILING_DIGITS.search(path.stem)
    prefix = path.stem[: match.start()] if match else path.stem
    try:
        parent = str(path.parent.resolve(strict=False)).casefold()
    except OSError:
        parent = str(path.parent.absolute()).casefold()
    return parent, prefix.casefold()


def _resource_key(path: Path) -> tuple[str, str]:
    try:
        parent = str(path.parent.resolve(strict=False)).casefold()
    except OSError:
        parent = str(path.parent.absolute()).casefold()
    return parent, path.stem.casefold()


def _cache_key(path: Path) -> str:
    try:
        return str(path.resolve(strict=False)).casefold()
    except OSError:
        return str(path.absolute()).casefold()


def _parse_date(value: str) -> datetime | None:
    value = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            pass
    return None


def _read_exif_capture_time(path: Path) -> datetime | None:
    """Read only a trustworthy EXIF capture time; let real I/O errors escape."""
    with path.open("rb") as fh:
        tags = exifread.process_file(
            fh,
            details=False,
            stop_tag="EXIF DateTimeOriginal",
            extract_thumbnail=False,
        )
    for key in ("EXIF DateTimeOriginal", "Image DateTime"):
        if key in tags:
            dt = _parse_date(str(tags[key]))
            if dt:
                return dt
    return None


def read_capture_time(path: Path) -> datetime:
    """Compatibility helper: EXIF first, then filesystem modification time."""
    try:
        dt = _read_exif_capture_time(path)
        if dt is not None:
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
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if entry.is_file() and Path(entry.name).suffix.lower() in extset:
                        count += 1
                except OSError:
                    continue
        return count
    except OSError:
        return 0


def has_supported_photos(folder: Path, extensions: Iterable[str], recursive: bool = True) -> bool:
    """Fast existence check for the GUI; does not parse EXIF."""
    if not folder.is_dir():
        return False
    extset = {e.lower() for e in extensions}
    if recursive:
        try:
            for _root, _dirs, files in os.walk(folder):
                if any(Path(name).suffix.lower() in extset for name in files):
                    return True
        except OSError:
            return False
        return False
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if entry.is_file() and Path(entry.name).suffix.lower() in extset:
                        return True
                except OSError:
                    continue
        return False
    except OSError:
        return False


def _discover_paths(folder: Path, extset: set[str], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            root_path = Path(root)
            for name in files:
                if Path(name).suffix.lower() in extset:
                    paths.append(root_path / name)
    else:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if entry.is_file() and Path(entry.name).suffix.lower() in extset:
                        paths.append(Path(entry.path))
                except OSError:
                    continue
    paths.sort(key=lambda p: str(p).casefold())
    return paths


def _build_quick_index(
    folder: Path,
    extset: set[str],
    recursive: bool,
    report: ScanReport,
    progress: Callable[[int, int, Path | None], None] | None,
    check_cancelled: Callable[[], None] | None,
) -> list[_QuickFile]:
    try:
        paths = _discover_paths(folder, extset, recursive)
    except OSError as exc:
        _LOG.warning("Fast filesystem index failed for %s: %s", folder, exc)
        return []

    report.files_discovered = len(paths)
    total = len(paths)
    if progress:
        progress(0, total, None)

    quick: list[_QuickFile] = []
    for completed, path in enumerate(paths, start=1):
        if check_cancelled:
            check_cancelled()
        try:
            stat = path.stat()
        except OSError as exc:
            report.files_skipped += 1
            _LOG.warning("Cannot stat %s; file skipped: %s", path, exc)
        else:
            quick.append(
                _QuickFile(
                    path=path,
                    sequence_number=_sequence_number(path),
                    source_key=_sequence_source(path),
                    size=int(stat.st_size),
                    mtime_ns=int(stat.st_mtime_ns),
                    mtime=float(stat.st_mtime),
                )
            )
        if progress:
            progress(completed, total, path)
    return quick


def _metadata_only_photo(item: _QuickFile) -> PhotoFile:
    return PhotoFile(
        path=item.path,
        capture_time=datetime.fromtimestamp(item.mtime),
        sequence_number=item.sequence_number,
        extension=item.path.suffix.lower(),
        capture_time_source="file",
        order_source="name",
        metadata_cached=False,
        sequence_source=item.source_key,
    )


def _collapse_raw_jpeg_pairs(
    quick: list[_QuickFile], report: ScanReport
) -> tuple[list[_QuickFile], dict[Path, _QuickFile]]:
    by_resource: dict[tuple[str, str], list[_QuickFile]] = {}
    for item in quick:
        by_resource.setdefault(_resource_key(item.path), []).append(item)

    excluded: set[Path] = set()
    paired_jpeg_items: dict[Path, _QuickFile] = {}
    for items in by_resource.values():
        raws = sorted(
            (item for item in items if item.path.suffix.lower() in RAW_EXTENSIONS),
            key=lambda item: str(item.path).casefold(),
        )
        jpegs = sorted(
            (item for item in items if item.path.suffix.lower() in _JPEG_EXTENSIONS),
            key=lambda item: str(item.path).casefold(),
        )
        if not raws or not jpegs:
            continue
        fallback = jpegs[0]
        for raw in raws:
            report.pair_preview_fallbacks[raw.path] = fallback.path
            paired_jpeg_items[raw.path] = fallback
        for jpeg in jpegs:
            excluded.add(jpeg.path)
            report.metadata_only_photos.append(_metadata_only_photo(jpeg))
            report.paired_jpeg_skipped += 1

    return [item for item in quick if item.path not in excluded], paired_jpeg_items


def _load_cache(path: Path) -> dict[str, dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != _CACHE_VERSION:
            return {}
        entries = raw.get("entries", {})
        return entries if isinstance(entries, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _valid_cached_time(item: _QuickFile, entry: object) -> datetime | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("source") != "exif":
        return None
    if entry.get("size") != item.size or entry.get("mtime_ns") != item.mtime_ns:
        return None
    value = entry.get("capture_time")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _save_cache(path: Path, entries: dict[str, dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = {"version": _CACHE_VERSION, "entries": entries}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        _LOG.debug("Cannot save EXIF cache %s: %s", path, exc)


def _group_sort_key(item: _QuickFile) -> tuple:
    return (
        item.source_key[0],
        item.source_key[1],
        item.sequence_number if item.sequence_number is not None else _SENTINEL_SEQUENCE,
        item.path.name.casefold(),
    )


def _group_exif_probe_indices(
    items: list[_QuickFile],
    max_filename_gap: int,
    max_gap_seconds: float,
) -> set[int]:
    """Return only indices around ambiguous filename/filesystem boundaries."""
    result: set[int] = set()
    duplicate_map: dict[tuple[tuple[str, str], int], list[int]] = {}
    for idx, item in enumerate(items):
        if item.sequence_number is None:
            result.add(idx)
        else:
            duplicate_map.setdefault((item.source_key, item.sequence_number), []).append(idx)
    for indices in duplicate_map.values():
        if len(indices) > 1:
            result.update(indices)

    for idx in range(1, len(items)):
        prev = items[idx - 1]
        cur = items[idx]
        suspicious = False
        if prev.source_key != cur.source_key:
            suspicious = True
        elif prev.sequence_number is None or cur.sequence_number is None:
            suspicious = True
        else:
            delta = cur.sequence_number - prev.sequence_number
            if delta <= 0 or delta > max_filename_gap:
                suspicious = True
        if abs(cur.mtime - prev.mtime) > max_gap_seconds:
            suspicious = True
        if suspicious:
            result.update((idx - 1, idx))
    return result


def _read_exif_indices(
    items: list[_QuickFile],
    indices: set[int],
    workers: int,
    report: ScanReport,
    check_cancelled: Callable[[], None] | None,
) -> dict[int, tuple[datetime | None, Exception | None]]:
    if not indices:
        return {}

    ordered = sorted(indices)

    def read_one(index: int) -> tuple[int, datetime | None, Exception | None]:
        try:
            return index, _read_exif_capture_time(items[index].path), None
        except Exception as exc:
            return index, None, exc

    report.exif_reads += len(ordered)
    worker_count = max(1, min(8, int(workers), len(ordered)))
    results: dict[int, tuple[datetime | None, Exception | None]] = {}
    if worker_count == 1:
        for index in ordered:
            if check_cancelled:
                check_cancelled()
            _idx, capture_time, error = read_one(index)
            results[index] = (capture_time, error)
        return results

    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="photo-exif")
    futures = {executor.submit(read_one, index): index for index in ordered}
    try:
        for future in as_completed(futures):
            if check_cancelled:
                check_cancelled()
            index, capture_time, error = future.result()
            results[index] = (capture_time, error)
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    return results


def scan_photos(
    folder: Path,
    extensions: Iterable[str],
    recursive: bool = True,
    *,
    workers: int = 1,
    mode: str = "portrait",
    max_filename_gap: int = 5,
    max_gap_seconds: float = 12.0,
    cache_path: Path | None = None,
    report: ScanReport | None = None,
    progress: Callable[[int, int, Path | None], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> list[PhotoFile]:
    report = report if report is not None else ScanReport()
    extset = {e.lower() for e in extensions}
    quick = _build_quick_index(folder, extset, recursive, report, progress, check_cancelled)
    if not quick:
        report.files_returned = 0
        return []

    quick, paired_jpeg_items = _collapse_raw_jpeg_pairs(quick, report)
    if not quick:
        report.files_returned = 0
        return []

    cache_file = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE_PATH
    cache_entries = _load_cache(cache_file)
    cached_times: dict[int, datetime] = {}
    for idx, item in enumerate(quick):
        cached = _valid_cached_time(item, cache_entries.get(_cache_key(item.path)))
        if cached is not None:
            cached_times[idx] = cached
            report.cache_hits += 1

    mode = str(mode).lower()
    if mode == "group":
        quick.sort(key=_group_sort_key)
        # Cache indices changed with the sort, so rebuild the lookup cheaply.
        cached_times = {}
        for idx, item in enumerate(quick):
            cached = _valid_cached_time(item, cache_entries.get(_cache_key(item.path)))
            if cached is not None:
                cached_times[idx] = cached
        probe = _group_exif_probe_indices(quick, int(max_filename_gap), float(max_gap_seconds))
        exif_indices = {idx for idx in probe if idx not in cached_times}
    else:
        exif_indices = {idx for idx in range(len(quick)) if idx not in cached_times}

    exif_results = _read_exif_indices(quick, exif_indices, workers, report, check_cancelled)
    failed_indices: set[int] = set()
    fresh_times: dict[int, datetime] = {}
    promoted_jpegs: list[tuple[_QuickFile, datetime, str, bool, _QuickFile]] = []
    promoted_jpeg_paths: set[Path] = set()
    cache_changed = False
    for idx, (capture_time, error) in exif_results.items():
        item = quick[idx]
        if error is not None:
            report.exif_failures += 1
            report.files_skipped += 1
            failed_indices.add(idx)

            paired_jpeg = paired_jpeg_items.get(item.path)
            if paired_jpeg is None or paired_jpeg.path in promoted_jpeg_paths:
                _LOG.warning("ExifRead failed for %s; file skipped: %s", item.path, error)
                continue

            # The RAW itself is unusable for the scan, but an exact same-stem
            # JPEG represents the same exposure. Promote that JPEG back into
            # analysis instead of losing the whole logical frame.  The RAW is
            # retained below as metadata-only so the final logical label can
            # still be mirrored to both physical files.
            jpeg_cached_time = _valid_cached_time(
                paired_jpeg, cache_entries.get(_cache_key(paired_jpeg.path))
            )
            jpeg_cached = jpeg_cached_time is not None
            jpeg_capture_time = jpeg_cached_time
            jpeg_source = "exif" if jpeg_cached else "file"
            if jpeg_cached:
                report.cache_hits += 1
            else:
                report.exif_reads += 1
                try:
                    jpeg_capture_time = _read_exif_capture_time(paired_jpeg.path)
                except Exception as jpeg_error:
                    report.exif_failures += 1
                    report.files_skipped += 1
                    _LOG.warning(
                        "ExifRead failed for RAW %s and exact paired JPEG %s; exposure skipped: %s / %s",
                        item.path, paired_jpeg.path, error, jpeg_error,
                    )
                    continue
                if jpeg_capture_time is not None:
                    jpeg_source = "exif"
                    cache_entries[_cache_key(paired_jpeg.path)] = {
                        "size": paired_jpeg.size,
                        "mtime_ns": paired_jpeg.mtime_ns,
                        "capture_time": jpeg_capture_time.isoformat(),
                        "source": "exif",
                    }
                    cache_changed = True

            if jpeg_capture_time is None:
                jpeg_capture_time = datetime.fromtimestamp(paired_jpeg.mtime)
                jpeg_source = "file"

            promoted_jpegs.append((paired_jpeg, jpeg_capture_time, jpeg_source, jpeg_cached, item))
            promoted_jpeg_paths.add(paired_jpeg.path)
            report.pair_preview_fallbacks.pop(item.path, None)
            _LOG.warning(
                "ExifRead failed for RAW %s; using exact paired JPEG %s as the analysed frame",
                item.path, paired_jpeg.path,
            )
            continue
        if capture_time is not None:
            fresh_times[idx] = capture_time
            cache_entries[_cache_key(item.path)] = {
                "size": item.size,
                "mtime_ns": item.mtime_ns,
                "capture_time": capture_time.isoformat(),
                "source": "exif",
            }
            cache_changed = True

    photos: list[PhotoFile] = []
    for idx, item in enumerate(quick):
        if idx in failed_indices:
            continue
        if idx in cached_times:
            capture_time = cached_times[idx]
            capture_source = "exif"
            cached = True
        elif idx in fresh_times:
            capture_time = fresh_times[idx]
            capture_source = "exif"
            cached = False
        else:
            capture_time = datetime.fromtimestamp(item.mtime)
            capture_source = "file"
            cached = False

        order_source = "name" if mode == "group" else "time"
        photo = PhotoFile(
            path=item.path,
            capture_time=capture_time,
            sequence_number=item.sequence_number,
            extension=item.path.suffix.lower(),
            capture_time_source=capture_source,
            order_source=order_source,
            metadata_cached=cached,
            sequence_source=item.source_key,
        )
        photos.append(photo)
        if capture_source == "exif":
            report.time_from_exif += 1
        else:
            report.time_from_file += 1
        if order_source == "name":
            report.order_from_name += 1

    for jpeg, capture_time, capture_source, cached, failed_raw in promoted_jpegs:
        order_source = "name" if mode == "group" else "time"
        photos.append(
            PhotoFile(
                path=jpeg.path,
                capture_time=capture_time,
                sequence_number=jpeg.sequence_number,
                extension=jpeg.path.suffix.lower(),
                capture_time_source=capture_source,
                order_source=order_source,
                metadata_cached=cached,
                sequence_source=jpeg.source_key,
            )
        )
        if capture_source == "exif":
            report.time_from_exif += 1
        else:
            report.time_from_file += 1
        if order_source == "name":
            report.order_from_name += 1

        # JPEG is no longer metadata-only because it is now analysed. Keep the
        # failed RAW as the metadata mirror for the same logical resource.
        report.metadata_only_photos = [
            photo for photo in report.metadata_only_photos if photo.path != jpeg.path
        ]
        report.metadata_only_photos.append(_metadata_only_photo(failed_raw))

    if cache_changed:
        _save_cache(cache_file, cache_entries)

    if mode == "group":
        photos.sort(
            key=lambda photo: (
                photo.sequence_source[0],
                photo.sequence_source[1],
                photo.sequence_number if photo.sequence_number is not None else _SENTINEL_SEQUENCE,
                photo.capture_time,
                photo.path.name.casefold(),
            )
        )
    else:
        photos.sort(
            key=lambda photo: (
                photo.capture_time,
                photo.sequence_number if photo.sequence_number is not None else _SENTINEL_SEQUENCE,
                photo.path.name.casefold(),
            )
        )

    active_paths = {photo.path for photo in photos}
    report.pair_preview_fallbacks = {
        raw: jpeg
        for raw, jpeg in report.pair_preview_fallbacks.items()
        if raw in active_paths
    }
    # Metadata mirrors are tied to the logical same-stem resource, not to a
    # particular RAW->JPEG direction. This also preserves the failed RAW when
    # its exact JPEG had to be promoted into analysis after an EXIF failure.
    active_resources = {_resource_key(photo.path) for photo in photos}
    report.metadata_only_photos = [
        photo
        for photo in report.metadata_only_photos
        if _resource_key(photo.path) in active_resources
    ]

    report.files_returned = len(photos)
    return photos
