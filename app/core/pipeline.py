from __future__ import annotations

import gc
import logging
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Callable

from app.analysis.factory import create_face_analyzer
from app.analysis.grouping import (
    has_group_face_count,
    looks_like_group_series,
    merge_adjacent_group_blocks,
    select_group_series,
    split_group_candidate_by_identity,
    stabilize_group_frame_order,
)
from app.analysis.scoring import select_portrait
from app.core.models import FrameAssessment, RunStats, Selection
from app.core.preview import load_preview
from app.core.scanner import ScanReport, scan_photos
from app.core.series import (
    build_candidate_series,
    refine_assessments_detailed,
    merge_adjacent_portrait_series,
    link_repeated_portrait_series,
    portrait_pose_signature,
    portrait_pose_difference,
)
from app.utils.cleanup import cleanup_temp
from app.xmp.writer import XmpWriter

ProgressCallback = Callable[[float, str], None]
MessageCallback = Callable[[str], None]


class AnalysisPipeline:
    def __init__(
        self,
        config: dict,
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
        message: MessageCallback | None = None,
    ):
        self.config = config
        self.cancel_event = cancel_event or threading.Event()
        self._progress_callback = progress or (lambda _p, _m: None)
        self._progress_lock = threading.Lock()
        self._last_progress = 0.0
        self.progress = self._emit_progress
        self.message = message or (lambda _m: None)
        self.log = logging.getLogger("photo_select_ai")
        self.mode = str(config.get("runtime", {}).get("mode", "portrait")).lower()
        self._pair_preview_fallbacks: dict[Path, Path] = {}
        self._preview_source_lock = threading.Lock()
        self._active_stats: RunStats | None = None


    def _emit_progress(self, value: float, message: str) -> None:
        """Publish a monotonic progress value.

        Several analysis stages have their own local progress scales.  Keeping
        this guard in the pipeline prevents a late/stale callback from making
        the GUI jump backwards, which otherwise looks like a restarted or
        frozen analysis.
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self._last_progress
        value = max(0.0, min(100.0, value))
        with self._progress_lock:
            value = max(self._last_progress, value)
            self._last_progress = value
        self._progress_callback(value, str(message))

    def _preview_workers(self) -> int:
        """Number of CPU workers used for metadata I/O and preview prefetch.

        InsightFace inference itself remains serialized on one analyzer/GPU.
        The worker threads overlap file I/O, RAW/JPEG decoding, EXIF rotation
        and resize of the next image with CUDA inference of the current image.
        """
        try:
            value = int(self.config.get("runtime", {}).get("cpu_workers", 2))
        except (TypeError, ValueError):
            value = 2
        return max(1, min(8, value))

    @staticmethod
    def _metadata_resource_key(photo_or_path) -> str:
        path = photo_or_path.path if hasattr(photo_or_path, "path") else Path(photo_or_path)
        return str(path.with_suffix("")).casefold()

    def _build_metadata_plan(self, selections: list[Selection]) -> dict[str, str]:
        """Build one RED/YELLOW role per logical path-without-extension resource."""
        plan: dict[str, str] = {}
        for selection in selections:
            role = str(selection.label_role).lower()
            if role not in {"red", "yellow"}:
                continue
            key = self._metadata_resource_key(selection.photo)
            previous = plan.get(key)
            if previous is None or previous == role:
                plan[key] = role
                continue
            # A resource must never end the run with two logical roles. RED is
            # the deterministic winner because it represents the main choice.
            winner = "red" if "red" in {previous, role} else role
            plan[key] = winner
            self.log.warning(
                "Metadata role conflict for %s: %s vs %s; using %s",
                selection.photo.path.with_suffix(""), previous, role, winner,
            )
        return plan

    def _apply_metadata_changes(
        self,
        photos,
        writer: XmpWriter,
        selections: list[Selection],
        clear_red: bool,
        clear_yellow: bool,
        stats: RunStats,
        metadata_plan: dict[str, str] | None = None,
    ) -> None:
        """Commit the complete metadata plan after all analysis has succeeded.

        This is intentionally a deferred commit rather than a rollback-capable
        database transaction. Cancellation is checked once immediately before
        the first write; after that the small metadata stage is completed so a
        cancel cannot leave half the shoot on the old plan and half on the new.
        """
        plan = metadata_plan if metadata_plan is not None else self._build_metadata_plan(selections)
        self._check_cancelled()  # final cancellation point before any XMP change

        resources: dict[str, list] = {}
        seen_paths: set[str] = set()
        for photo in photos:
            path_key = str(photo.path).casefold()
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            resources.setdefault(self._metadata_resource_key(photo), []).append(photo)
        for selection in selections:
            photo = selection.photo
            path_key = str(photo.path).casefold()
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                resources.setdefault(self._metadata_resource_key(photo), []).append(photo)

        total = max(1, len(resources))
        self.message(
            f"Финальная запись: ресурсов={len(resources)}, выбранных={len(plan)}; "
            "после начала commit отмена применяется только к следующему запуску."
        )
        for completed, key in enumerate(sorted(resources), start=1):
            physical = sorted(
                resources[key],
                key=lambda photo: (
                    photo.extension.lower() in {".jpg", ".jpeg"},
                    str(photo.path).casefold(),
                ),
            )
            role = plan.get(key)
            resource_error = False
            resource_cleared = False
            if role is not None:
                for photo in physical:
                    try:
                        destination = writer.write(photo, role)
                        self._count_metadata_write(stats, photo, destination)
                    except Exception as exc:
                        resource_error = True
                        self.log.error(
                            "Final metadata write failed for %s (%s): %s",
                            photo.path, role, exc, exc_info=(type(exc), exc, exc.__traceback__),
                        )
            elif clear_red or clear_yellow:
                for photo in physical:
                    try:
                        if clear_red:
                            resource_cleared = writer.clear_label(photo, "red") or resource_cleared
                        if clear_yellow:
                            resource_cleared = writer.clear_label(photo, "yellow") or resource_cleared
                    except Exception as exc:
                        resource_error = True
                        self.log.error(
                            "Final metadata clear failed for %s: %s",
                            photo.path, exc, exc_info=(type(exc), exc, exc.__traceback__),
                        )
            if resource_cleared:
                stats.labels_cleared_before_run += 1
            if resource_error:
                stats.metadata_errors += 1
            self.progress(99.0 + 0.9 * completed / total, f"Финальная запись {completed}/{len(resources)}")

    def _record_preview_source(self, source: str) -> None:
        stats = self._active_stats
        if stats is None:
            return
        with self._preview_source_lock:
            if source == "rawpy_preview":
                stats.raw_preview_rawpy += 1
            elif source == "embedded_jpeg":
                stats.raw_preview_embedded_jpeg += 1
            elif source == "demosaic":
                stats.raw_preview_demosaic += 1

    def _load_preview_with_pair_fallback(self, path: Path, long_edge: int, fallback_half: bool):
        try:
            return load_preview(
                path, long_edge, fallback_half, source_callback=self._record_preview_source
            )
        except Exception:
            fallback = self._pair_preview_fallbacks.get(path)
            if fallback is None:
                raise
            self.log.warning(
                "RAW preview failed for %s; using exact paired JPEG %s",
                path.name, fallback.name,
            )
            return load_preview(fallback, long_edge, fallback_half)

    def _iter_loaded_previews(self, jobs, long_edge: int, fallback_half: bool):
        """Yield ``(payload, rgb, error)`` in input order with bounded prefetch.

        At most ``cpu_workers`` decoded previews can be resident ahead of the
        analyzer, which prevents the 3200/4800 px group modes from consuming
        unbounded RAM.  A decode failure is returned as an item-level error so
        the rest of the shoot continues exactly like the old serial loop.
        """
        jobs = list(jobs)
        workers = min(self._preview_workers(), max(1, len(jobs)))
        if workers <= 1 or len(jobs) <= 1:
            for payload, path in jobs:
                self._check_cancelled()
                try:
                    yield payload, self._load_preview_with_pair_fallback(path, long_edge, fallback_half), None
                except Exception as exc:
                    yield payload, None, exc
            return

        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="photo-preview")
        pending = deque()
        iterator = iter(jobs)

        def submit_one(job) -> None:
            payload, path = job
            pending.append((payload, path, executor.submit(self._load_preview_with_pair_fallback, path, long_edge, fallback_half)))

        try:
            for _ in range(workers):
                try:
                    submit_one(next(iterator))
                except StopIteration:
                    break

            while pending:
                self._check_cancelled()
                payload, _path, future = pending.popleft()
                try:
                    rgb = future.result()
                    error = None
                except Exception as exc:
                    rgb = None
                    error = exc

                # Keep the pipeline full before handing the current preview to
                # the GPU. While inference runs, this future decodes the next.
                try:
                    submit_one(next(iterator))
                except StopIteration:
                    pass
                yield payload, rgb, error
        finally:
            for _payload, _path, future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)


    def _parallel_face_workers(self) -> int:
        """Number of independent InsightFace analyzers requested by the user."""
        runtime = self.config.get("runtime", {})
        if not bool(runtime.get("parallel_face_analysis", False)):
            return 1
        try:
            value = int(runtime.get("parallel_face_workers", 2))
        except (TypeError, ValueError):
            value = 2
        return max(2, min(4, value))

    def _gpu_memory_safe_mode(self) -> bool:
        return bool(self.config.get("runtime", {}).get("gpu_memory_safe_mode", True))

    def _secondary_face_workers(self) -> int:
        """Worker cap for Group high-res/gaze stages.

        The main pass over hundreds of files can benefit from all requested GPU
        sessions.  A group gaze shortlist is normally only three frames, so
        keeping four complete buffalo_l/ORT applications resident there wastes
        VRAM without useful parallelism.  Safe mode therefore applies a separate
        cap to the secondary Group stages.
        """
        requested = self._parallel_face_workers()
        if requested <= 1 or not self._gpu_memory_safe_mode():
            return requested
        runtime = self.config.get("runtime", {})
        try:
            cap = int(runtime.get("group_secondary_face_workers", 2))
        except (TypeError, ValueError):
            cap = 2
        return max(1, min(requested, 4, cap))

    def _group_gpu_recycle_every(self) -> int:
        if not self._gpu_memory_safe_mode():
            return 0
        try:
            value = int(self.config.get("runtime", {}).get("group_gpu_recycle_every", 8))
        except (TypeError, ValueError):
            value = 8
        return max(0, min(50, value))

    def _create_analyzer_pool(
        self, config: dict, label: str, *, max_workers: int | None = None
    ) -> tuple[list, str]:
        """Create one or more independent analyzers, degrading gracefully."""
        requested_by_user = self._parallel_face_workers()
        requested = requested_by_user
        if max_workers is not None:
            requested = max(1, min(requested, int(max_workers)))
        if requested < requested_by_user:
            self.log.info(
                "GPU MEMORY SAFE CAP | stage=%s | requested=%d | active=%d",
                label, requested_by_user, requested,
            )
            self.message(
                f"{label}: защита VRAM ограничила GPU-сессии {requested_by_user} → {requested}."
            )
        analyzers: list = []
        backend_name = ""
        for worker_no in range(1, requested + 1):
            self._check_cancelled()
            try:
                analyzer, backend = create_face_analyzer(
                    config,
                    message=self.message if worker_no == 1 else None,
                )
                analyzers.append(analyzer)
                if not backend_name:
                    backend_name = backend
            except Exception as exc:
                if not analyzers:
                    raise
                self.log.exception("Could not create parallel InsightFace worker %d for %s", worker_no, label)
                self.message(
                    f"{label}: не удалось создать InsightFace-сессию {worker_no}/{requested} ({exc}). "
                    f"Продолжаю с {len(analyzers)} сессиями."
                )
                break
        if len(analyzers) > 1:
            self.log.info("PARALLEL FACE ANALYSIS | stage=%s | workers=%d | backend=%s", label, len(analyzers), backend_name)
            self.message(f"{label}: параллельный анализ, независимых InsightFace-сессий: {len(analyzers)}")
        return analyzers, backend_name

    @staticmethod
    def _close_analyzer_pool(analyzers) -> None:
        for analyzer in analyzers or []:
            try:
                analyzer.close()
            except Exception:
                pass
        # Make destruction of ORT sessions deterministic at stage boundaries.
        gc.collect()

    def _iter_analyzed_frames(self, jobs, analyzers, long_edge: int, fallback_half: bool):
        """Yield ``(payload, assessment, error)`` as frames finish.

        With one analyzer this keeps the bounded CPU preview prefetch path.
        With two or more analyzers, each task decodes one image and borrows one
        dedicated analyzer from a queue.  No analyzer instance is ever used
        concurrently by multiple tasks.
        """
        jobs = list(jobs)
        if not jobs:
            return
        if len(analyzers) <= 1:
            analyzer = analyzers[0]
            for packed, rgb, load_error in self._iter_loaded_previews(
                [((payload, photo), photo.path) for payload, photo in jobs], long_edge, fallback_half
            ):
                self._check_cancelled()
                payload, photo = packed
                if load_error is not None:
                    yield payload, None, load_error
                    continue
                try:
                    yield payload, analyzer.analyze(photo, rgb), None
                except Exception as exc:
                    yield payload, None, exc
            return

        analyzer_queue: queue.Queue = queue.Queue()
        for analyzer in analyzers:
            analyzer_queue.put(analyzer)

        def work(payload, photo):
            self._check_cancelled()
            analyzer = analyzer_queue.get()
            try:
                rgb = self._load_preview_with_pair_fallback(photo.path, long_edge, fallback_half)
                self._check_cancelled()
                return payload, analyzer.analyze(photo, rgb), None
            except CancelledError:
                raise
            except Exception as exc:
                return payload, None, exc
            finally:
                analyzer_queue.put(analyzer)

        executor = ThreadPoolExecutor(max_workers=len(analyzers), thread_name_prefix="face-analysis")
        futures = [executor.submit(work, payload, photo) for payload, photo in jobs]
        try:
            for future in as_completed(futures):
                self._check_cancelled()
                yield future.result()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def run(self, folder: Path) -> tuple[RunStats, list[Selection]]:
        stats = RunStats(run_mode=self.mode)
        selections: list[Selection] = []
        self._active_stats = stats
        if self.config["runtime"].get("cleanup_temp_on_start", True):
            cleanup_temp()

        try:
            self.progress(1.0, "Сканирование файлов...")
            self.message("Сканирование файлов...")

            def scan_progress(completed: int, total: int, path: Path | None) -> None:
                if total <= 0:
                    self.progress(1.0, "Сканирование файлов: поддерживаемые файлы не найдены")
                    return
                pct = 1.0 + 0.9 * completed / total
                detail = f"Сканирование файлов: индекс {completed}/{total}"
                if path is not None:
                    detail += f" | {path.name}"
                self.progress(pct, detail)

            scan_workers = self._preview_workers()
            scan_report = ScanReport()
            if self.mode == "group":
                scan_cfg = self.config.get("group", {})
                scan_max_filename_gap = int(scan_cfg.get("max_filename_gap", self.config["series"]["max_filename_gap"]))
                scan_max_gap_seconds = float(scan_cfg.get("max_gap_seconds", self.config["series"]["max_gap_seconds"]))
            else:
                scan_max_filename_gap = int(self.config["series"]["max_filename_gap"])
                scan_max_gap_seconds = float(self.config["series"]["max_gap_seconds"])
            self.message(f"Быстрый индекс: файловые данные + selective EXIF, потоков EXIF до {scan_workers}.")
            photos = scan_photos(
                folder,
                self.config["scan"]["extensions"],
                bool(self.config["scan"].get("recursive", True)),
                workers=scan_workers,
                mode=self.mode,
                max_filename_gap=scan_max_filename_gap,
                max_gap_seconds=scan_max_gap_seconds,
                report=scan_report,
                progress=scan_progress,
                check_cancelled=self._check_cancelled,
            )
            stats.files_found = len(photos)
            stats.scan_exif_reads = scan_report.exif_reads
            stats.scan_exif_cache_hits = scan_report.cache_hits
            stats.scan_time_from_exif = scan_report.time_from_exif
            stats.scan_time_from_file = scan_report.time_from_file
            stats.scan_order_from_name = scan_report.order_from_name
            stats.scan_exif_failures = scan_report.exif_failures
            stats.scan_files_skipped = scan_report.files_skipped
            stats.raw_jpeg_pairs_collapsed = scan_report.paired_jpeg_skipped
            self._pair_preview_fallbacks = dict(scan_report.pair_preview_fallbacks)
            metadata_photos = list(photos) + list(scan_report.metadata_only_photos)
            self.message(
                f"Индекс готов: для анализа={len(photos)}, EXIF-чтений={scan_report.exif_reads}, "
                f"кэш EXIF={scan_report.cache_hits}, порядок по имени={scan_report.order_from_name}, "
                f"RAW+JPEG пар свернуто={scan_report.paired_jpeg_skipped}, "
                f"пропущено из-за ошибок={scan_report.files_skipped}"
            )
            if not photos:
                self.message("Поддерживаемые фотографии не найдены.")
                return stats, []

            self.progress(2.0, f"Сканирование завершено: {len(photos)} кадров для анализа")

            candidates = build_candidate_series(photos, self.config)
            stats.candidate_series = len(candidates)
            self.message(f"Найдено файлов: {len(photos)}, предварительных временных блоков: {len(candidates)}")
            writer = XmpWriter(self.config)
            portrait_repeat_mode = str(
                self.config.get("portrait", {}).get("repeat_pose_mode", "off")
            ).lower() in {"red_yellow", "first_red_rest_yellow", "best_red_pose_yellow"}
            if self.mode == "group":
                self.message(f"XMP: главный кадр RED='{writer.red}', дополнительные YELLOW='{writer.yellow}'")
            elif portrait_repeat_mode:
                self.message(
                    f"XMP: лучший кадр ребёнка RED='{writer.red}', "
                    f"явно отличающиеся позы YELLOW='{writer.yellow}'"
                )
            else:
                self.message(f"XMP: лучший портрет RED='{writer.red}'")

            clear_red = bool(self.config.get("xmp", {}).get("clear_red_before_run", True))
            clear_yellow = (
                bool(self.config.get("xmp", {}).get("clear_yellow_before_run", True))
                if self.mode == "group" or (self.mode == "portrait" and portrait_repeat_mode)
                else False
            )
            analyzers, backend_name = self._create_analyzer_pool(self.config, "Основной анализ")
            self.message(f"Распознавание лиц: {backend_name}")
            processed_files = 0
            long_edge = int(self.config["preview"]["group_long_edge" if self.mode == "group" else "portrait_long_edge"])
            all_candidate_assessments: list[list[FrameAssessment]]

            try:
                all_candidate_assessments = [
                    [None] * len(candidate.photos) for candidate in candidates
                ]
                analysis_jobs = [
                    ((candidate_no, photo_idx), photo)
                    for candidate_no, candidate in enumerate(candidates, start=1)
                    for photo_idx, photo in enumerate(candidate.photos)
                ]
                fallback_half = bool(self.config["preview"].get("raw_fallback_half_size", True))
                if len(analyzers) == 1:
                    workers = self._preview_workers()
                    self.log.info("PREVIEW PREFETCH | workers=%d | long_edge=%d", workers, long_edge)
                    self.message(
                        f"RAW preview: предзагрузка изображений, {workers} "
                        + ("поток" if workers == 1 else "потока" if workers in (2, 3, 4) else "потоков")
                        + "; InsightFace-анализ: 1 сессия"
                    )
                for (candidate_no, photo_idx), assessment, error in self._iter_analyzed_frames(
                    analysis_jobs, analyzers, long_edge, fallback_half
                ):
                    self._check_cancelled()
                    photo = candidates[candidate_no - 1].photos[photo_idx]
                    if error is not None or assessment is None:
                        exc = error or RuntimeError("Unknown analysis error")
                        stats.analysis_errors += 1
                        self.log.error("Failed to analyze %s: %s", photo.path, exc)
                        assessment = FrameAssessment(photo=photo, faces=[], technical=0.0, error=str(exc))
                    elif not assessment.faces:
                        stats.frames_without_faces += 1
                    all_candidate_assessments[candidate_no - 1][photo_idx] = assessment
                    processed_files += 1
                    stats.files_analyzed = processed_files
                    analysis_end = 55.0 if self.mode == "group" else 78.0
                    pct = 8.0 + (analysis_end - 8.0) * processed_files / len(photos)
                    self.progress(pct, f"Анализ лиц: {photo.path.name} | блок {candidate_no}/{len(candidates)}")
                all_candidate_assessments = [
                    [frame for frame in candidate_frames if frame is not None]
                    for candidate_frames in all_candidate_assessments
                ]
            finally:
                # Release every primary analyzer before creating the more
                # demanding high-resolution stage. Parallel mode therefore
                # multiplies VRAM only within one stage, not across stages.
                self._close_analyzer_pool(analyzers)

            if stats.analysis_errors:
                self.log.error(
                    "Primary analysis finished with %d error(s); metadata commit is blocked and XMP remains unchanged",
                    stats.analysis_errors,
                )
                self.message(
                    f"Анализ завершён с ошибками ({stats.analysis_errors}). "
                    "Финальная запись меток не начата; существующие XMP не изменены."
                )
                raise AnalysisIncompleteError(
                    f"Не удалось полностью проанализировать {stats.analysis_errors} файл(а/ов). "
                    "Для безопасности RED/YELLOW не изменялись. Подробности записаны в журнал."
                )

            if self.mode == "group":
                self._run_group_mode(all_candidate_assessments, writer, stats, selections)
            else:
                self._run_portrait_mode(all_candidate_assessments, writer, stats, selections)

            self.progress(99.0, "План меток: формирование итогового RED/YELLOW плана")
            metadata_plan = self._build_metadata_plan(selections)
            self.message(f"План меток: логических ресурсов выбрано={len(metadata_plan)}")
            self._apply_metadata_changes(
                metadata_photos, writer, selections, clear_red, clear_yellow, stats, metadata_plan
            )
        finally:
            self._active_stats = None
            if self.config["runtime"].get("cleanup_temp_on_exit", True):
                cleanup_temp()
        self.progress(100.0, "Готово")
        return stats, selections

    def _run_portrait_mode(self, candidates: list[list[FrameAssessment]], writer: XmpWriter, stats: RunStats, selections: list[Selection]) -> None:
        min_frames = int(self.config["series"]["min_frames"])
        all_refined: list[list[FrameAssessment]] = []
        total_candidates = max(1, len(candidates))
        for candidate_idx, assessments in enumerate(candidates, start=1):
            self.progress(78.0 + 7.0 * candidate_idx / total_candidates, f"Кластеризация портретов {candidate_idx}/{len(candidates)}")
            detail = refine_assessments_detailed(assessments, self.config)
            stats.weak_series_rejected += detail.rejected_weak_series
            stats.local_fragments_merged += detail.merged_fragments
            stats.portrait_boundary_guard_proposals += detail.boundary_guard_proposals
            stats.portrait_boundary_guard_splits += detail.boundary_guard_splits
            if detail.boundary_guard_splits:
                self.log.info(
                    "PORTRAIT BOUNDARY GUARD | block=%d | proposals=%d | recovered_splits=%d",
                    candidate_idx, detail.boundary_guard_proposals, detail.boundary_guard_splits
                )
            all_refined.extend(detail.groups)

        all_refined, cross_merges = merge_adjacent_portrait_series(all_refined, self.config)
        stats.cross_block_series_merged += cross_merges
        stats.refined_series = len(all_refined)

        repeat_mode = str(self.config.get("portrait", {}).get("repeat_pose_mode", "off")).lower()
        repeat_enabled = repeat_mode in {
            "red_yellow", "first_red_rest_yellow", "best_red_pose_yellow"
        }
        repeat_links = link_repeated_portrait_series(all_refined, self.config)
        child_ids = (
            repeat_links.child_ids
            if len(repeat_links.child_ids) == len(all_refined)
            else list(range(len(all_refined)))
        )
        stats.portrait_repeat_links = repeat_links.links
        stats.portrait_repeat_ambiguous = repeat_links.ambiguous_rejected
        stats.portrait_repeat_children = len(set(child_ids)) if repeat_enabled else len(all_refined)

        repeat_text = ""
        if repeat_enabled:
            repeat_text = (
                f", найдено детей={stats.portrait_repeat_children}, "
                f"связано повторных серий={stats.portrait_repeat_links}, "
                f"неоднозначных совпадений={stats.portrait_repeat_ambiguous}"
            )
        self.message(
            f"После кластеризации: серий={len(all_refined)}, "
            f"объединено фрагментов={stats.local_fragments_merged + stats.cross_block_series_merged}, "
            f"восстановлено пропущенных границ={stats.portrait_boundary_guard_splits}, "
            f"отброшено слабых/случайных серий={stats.weak_series_rejected}"
            f"{repeat_text}"
        )

        if not repeat_enabled:
            total_refined = max(1, len(all_refined))
            for refined_idx, frames in enumerate(all_refined, start=1):
                self.progress(
                    85.0 + 14.0 * refined_idx / total_refined,
                    f"Выбор портрета {refined_idx}/{len(all_refined)}",
                )
                usable = [f for f in frames if not f.error]
                if len(usable) < min_frames:
                    stats.skipped_short_series += 1
                    continue
                stats.series_processed += 1
                stats.portrait_series += 1
                sel = select_portrait(usable, self.config)
                if not sel:
                    stats.series_without_selection += 1
                    continue
                sel.label_role = "red"
                stats.portrait_selected += 1
                selections.append(sel)
            self.progress(99.0, "Завершение портретного анализа...")
            return

        # Multi-pose mode has two deliberately separate concepts:
        #   series = chronological shooting run; child = linked identity.
        # We keep every series intact, choose one global RED from all usable
        # frames of the child, then add YELLOW only for *clearly different*
        # pose clusters. Small head/composition changes are intentionally folded
        # into the RED pose instead of generating extra labels.
        usable_by_series: dict[int, list[FrameAssessment]] = {}
        children: dict[int, list[int]] = {}

        for series_idx, frames in enumerate(all_refined):
            usable = [f for f in frames if not f.error]
            if len(usable) < min_frames:
                stats.skipped_short_series += 1
                continue
            stats.series_processed += 1
            stats.portrait_series += 1
            sel = select_portrait(usable, self.config)
            if not sel:
                stats.series_without_selection += 1
                continue
            usable_by_series[series_idx] = usable
            child_id = child_ids[series_idx] if series_idx < len(child_ids) else series_idx
            children.setdefault(child_id, []).append(series_idx)

        stats.portrait_repeat_children = len(children)
        max_yellows = max(0, int(self.config.get("portrait", {}).get("repeat_pose_max_yellows", 3)))
        child_items = sorted(children.items(), key=lambda item: min(item[1]))
        total_children = max(1, len(child_items))

        for child_no, (child_id, series_indices) in enumerate(child_items, start=1):
            self.progress(
                85.0 + 14.0 * child_no / total_children,
                f"Выбор портрета и поз: ребёнок {child_no}/{len(child_items)}",
            )
            all_child_frames = [
                frame
                for series_idx in series_indices
                for frame in usable_by_series.get(series_idx, [])
            ]
            red = select_portrait(all_child_frames, self.config)
            if red is None:
                stats.series_without_selection += 1
                continue

            red_series_idx = next(
                (
                    series_idx
                    for series_idx in series_indices
                    if any(frame.photo.path == red.photo.path for frame in usable_by_series.get(series_idx, []))
                ),
                series_indices[0],
            )
            red.label_role = "red"
            red.reason = f"child={child_id + 1}; best_across_series={len(series_indices)}; " + red.reason
            selections.append(red)
            stats.portrait_selected += 1
            self.log.info(
                "PORTRAIT CHILD RED (%s) | child=%d | series_count=%d | source_series=%d | %s | score=%.3f",
                writer.red, child_id + 1, len(series_indices), red_series_idx + 1,
                red.photo.path.name, red.score,
            )

            # Detect poses across *all frames* of the linked child, not only
            # across refined series. In normal shooting several poses of the
            # same child often stay inside one identity series, so series-level
            # pose detection would miss exactly the intended use case.
            red_frame = next(
                (frame for frame in all_child_frames if frame.photo.path == red.photo.path),
                all_child_frames[0],
            )
            ordered_frames = sorted(
                all_child_frames,
                key=lambda frame: (
                    frame.photo.capture_time,
                    frame.photo.sequence_number if frame.photo.sequence_number is not None else -1,
                    frame.photo.path.name.casefold(),
                ),
            )
            pose_clusters: list[list[FrameAssessment]] = [[red_frame]]
            for frame in ordered_frames:
                if frame is red_frame:
                    continue
                sig = portrait_pose_signature([frame])
                if sig is None:
                    pose_clusters[0].append(frame)
                    continue

                comparisons: list[tuple[bool, dict[str, float], int]] = []
                for cluster_no, cluster_frames in enumerate(pose_clusters):
                    distinct, metrics = portrait_pose_difference(
                        portrait_pose_signature(cluster_frames), sig, self.config
                    )
                    comparisons.append((distinct, metrics, cluster_no))

                compatible = [item for item in comparisons if not item[0]]
                if compatible:
                    _distinct, _metrics, cluster_no = min(
                        compatible, key=lambda item: item[1]["score"]
                    )
                    pose_clusters[cluster_no].append(frame)
                else:
                    pose_clusters.append([frame])

            min_pose_frames = max(1, int(
                self.config.get("portrait", {}).get("repeat_pose_min_pose_frames", 2)
            ))
            red_pose_sig = portrait_pose_signature(pose_clusters[0])
            accepted_pose_signatures = [red_pose_sig]
            yellow_candidates: list[Selection] = []
            for pose_no, cluster in enumerate(pose_clusters[1:], start=2):
                if len(cluster) < min_pose_frames:
                    self.log.info(
                        "PORTRAIT POSE REJECT | child=%d | pose=%d | frames=%d < min=%d",
                        child_id + 1, pose_no, len(cluster), min_pose_frames,
                    )
                    continue
                pose_sig = portrait_pose_signature(cluster)
                distinct_from_red, metrics = portrait_pose_difference(
                    red_pose_sig, pose_sig, self.config
                )
                if not distinct_from_red:
                    self.log.info(
                        "PORTRAIT POSE SAME | child=%d | pose=%d | frames=%d | score=%.2f | yaw=%.1f pitch=%.1f center=%.3f scale=%.2f",
                        child_id + 1, pose_no, len(cluster), metrics["score"], metrics["yaw"],
                        metrics["pitch"], metrics["center"], metrics["scale"],
                    )
                    pose_clusters[0].extend(cluster)
                    continue

                # Final conservative de-duplication: the candidate must also be
                # clearly different from every already accepted pose.
                if any(
                    not portrait_pose_difference(existing, pose_sig, self.config)[0]
                    for existing in accepted_pose_signatures
                    if existing is not None
                ):
                    continue
                accepted_pose_signatures.append(pose_sig)

                yellow = select_portrait(cluster, self.config)
                if yellow is None:
                    continue
                yellow.label_role = "yellow"
                yellow.reason = (
                    f"child={child_id + 1}; distinct_pose={len(accepted_pose_signatures)}; "
                    f"pose_frames={len(cluster)}; pose_delta={metrics['score']:.2f}; "
                    + yellow.reason
                )
                yellow_candidates.append(yellow)
                self.log.info(
                    "PORTRAIT POSE DISTINCT | child=%d | pose=%d | frames=%d | score=%.2f | yaw=%.1f pitch=%.1f center=%.3f scale=%.2f",
                    child_id + 1, pose_no, len(cluster), metrics["score"], metrics["yaw"],
                    metrics["pitch"], metrics["center"], metrics["scale"],
                )

            # When there are more genuine poses than the configured cap, retain
            # the strongest portrait from each pose, then keep the best ones.
            yellow_candidates.sort(key=lambda sel: sel.score, reverse=True)
            for yellow in yellow_candidates[:max_yellows]:
                selections.append(yellow)
                stats.portrait_repeat_yellow_selected += 1
                self.log.info(
                    "PORTRAIT POSE YELLOW (%s) | child=%d | %s | score=%.3f",
                    writer.yellow, child_id + 1, yellow.photo.path.name, yellow.score,
                )

        self.progress(99.0, "Завершение портретного анализа...")

    def _group_rescue_config(self) -> dict:
        """Second-pass InsightFace profile for missed small/weak group faces."""
        result = deepcopy(self.config)
        analysis = result.setdefault("analysis", {})
        group = result.setdefault("group", {})
        source_group = self.config.get("group", {})
        analysis["insightface_det_size_group"] = int(source_group.get("highres_rescue_det_size", 1536))
        analysis["insightface_det_thresh_group"] = float(source_group.get("highres_rescue_det_thresh", 0.14))
        rescue_fraction = float(source_group.get("highres_rescue_min_face_fraction", 0.00018))
        analysis["face_min_fraction"] = min(float(analysis.get("face_min_fraction", 0.0005)), rescue_fraction)
        group["track_det_thresh"] = float(source_group.get("highres_rescue_det_thresh", 0.14))
        group["min_track_face_fraction"] = rescue_fraction
        group["promote_singleton_tracks"] = False
        return result


    def _group_attention_config(self) -> dict:
        """Final-stage high-resolution profile for camera-attention analysis."""
        result = deepcopy(self.config)
        analysis = result.setdefault("analysis", {})
        group = result.setdefault("group", {})
        source_group = self.config.get("group", {})
        analysis["insightface_det_size_group"] = int(source_group.get("camera_attention_det_size", 1280))
        analysis["insightface_det_thresh_group"] = float(source_group.get("camera_attention_det_thresh", 0.20))
        # Keep the normal stable-roster face floor. Camera attention is not a
        # face-rescue stage and should not introduce distant background faces.
        analysis["face_min_fraction"] = float(source_group.get("min_track_face_fraction", analysis.get("face_min_fraction", 0.0005)))
        group["track_det_thresh"] = float(source_group.get("camera_attention_det_thresh", 0.20))
        group["camera_attention_enabled"] = True
        group["promote_singleton_tracks"] = False
        return result

    def _analyze_group_attention_shortlist(
        self,
        frames: list[FrameAssessment],
        shortlist: list[int],
        analyzers,
        group_idx: int,
        group_count: int,
        progress_start: float,
        progress_end: float,
    ) -> dict[int, FrameAssessment]:
        """Re-read only the strongest ordinary candidates for gaze/head pose."""
        cfg = self.config.get("group", {})
        preview_cfg = self.config.get("preview", {})
        long_edge = int(cfg.get("camera_attention_preview_long_edge", 4800))
        fallback_half = bool(preview_cfg.get("raw_fallback_half_size", True))
        out: dict[int, FrameAssessment] = {}
        valid = [i for i in shortlist if 0 <= i < len(frames)]
        total = max(1, len(valid))
        jobs = [(frame_idx, frames[frame_idx].photo) for frame_idx in valid]
        for no, (frame_idx, assessment, error) in enumerate(
            self._iter_analyzed_frames(jobs, analyzers, long_edge, fallback_half), start=1
        ):
            self._check_cancelled()
            frame = frames[frame_idx]
            if error is not None or assessment is None:
                exc = error or RuntimeError("Unknown camera-attention analysis error")
                self.log.error("Camera-attention analysis failed for %s: %s", frame.photo.path, exc)
                assessment = FrameAssessment(photo=frame.photo, faces=[], technical=frame.technical, error=str(exc))
            out[frame_idx] = assessment
            pct = progress_start + (progress_end - progress_start) * no / total
            self.progress(
                pct,
                f"Группа {group_idx}/{group_count}: взгляд в камеру {no}/{len(valid)}",
            )
        return out

    def _analyze_group_highres(
        self,
        frames: list[FrameAssessment],
        analyzers,
        rescue_config: dict,
        group_idx: int,
        group_count: int,
        progress_start: float,
        progress_end: float,
    ) -> list[FrameAssessment]:
        """Re-run one confirmed physical group with the sensitive detector."""
        preview_cfg = self.config.get("preview", {})
        group_cfg = self.config.get("group", {})
        long_edge = int(group_cfg.get("highres_rescue_preview_long_edge", preview_cfg.get("group_long_edge", 3200)))
        fallback_half = bool(preview_cfg.get("raw_fallback_half_size", True))
        out: list[FrameAssessment | None] = [None] * len(frames)
        total = max(1, len(frames))
        jobs = [(idx, frame.photo) for idx, frame in enumerate(frames)]
        for no, (frame_idx, assessment, error) in enumerate(
            self._iter_analyzed_frames(jobs, analyzers, long_edge, fallback_half), start=1
        ):
            self._check_cancelled()
            frame = frames[frame_idx]
            if error is not None or assessment is None:
                exc = error or RuntimeError("Unknown high-res analysis error")
                self.log.error("High-res group rescue failed for %s: %s", frame.photo.path, exc)
                assessment = FrameAssessment(photo=frame.photo, faces=[], technical=frame.technical, error=str(exc))
            out[frame_idx] = assessment
            pct = progress_start + (progress_end - progress_start) * no / total
            self.progress(
                pct,
                f"Группа {group_idx}/{group_count}: high-res поиск пропущенных лиц {no}/{len(frames)}",
            )
        return [frame for frame in out if frame is not None]

    def _run_group_mode(self, candidates: list[list[FrameAssessment]], writer: XmpWriter, stats: RunStats, selections: list[Selection]) -> None:
        min_frames = int(self.config.get("group", {}).get("min_frames", self.config.get("series", {}).get("group_min_frames", 2)))
        group_like: list[list[FrameAssessment]] = []
        rejected = 0
        total_candidates = max(1, len(candidates))
        split_weight_total = max(1, sum(max(1, len(a)) for a in candidates))
        split_weight_done = 0
        self.progress(55.0, f"Разделение физических групп: 0/{len(candidates)} временных блоков")
        for candidate_idx, assessments in enumerate(candidates, start=1):
            self._check_cancelled()
            candidate_weight = max(1, len(assessments))
            stage_start = 55.0 + 4.0 * split_weight_done / split_weight_total
            stage_end = 55.0 + 4.0 * (split_weight_done + candidate_weight) / split_weight_total
            usable = [f for f in assessments if not f.error]
            if not usable:
                split_weight_done += candidate_weight
                self.progress(stage_end, f"Разделение физических групп: блок {candidate_idx}/{len(candidates)} — нет пригодных кадров")
                continue
            # Group mode must not trust EXIF ordering blindly: exported JPEGs can
            # have coarse or copied timestamps.  A safe filename sequence order
            # makes both identity splitting and later tracking deterministic.
            usable = stabilize_group_frame_order(usable)

            def split_progress(local: float, detail: str, *, _start=stage_start, _end=stage_end, _idx=candidate_idx):
                local = max(0.0, min(1.0, float(local)))
                self.progress(
                    _start + (_end - _start) * local,
                    f"Разделение физических групп: блок {_idx}/{len(candidates)} — {detail}",
                )

            identity_groups, split_count = split_group_candidate_by_identity(
                usable, self.config, progress=split_progress
            )
            stats.group_identity_splits += split_count
            for subgroup in identity_groups:
                # Keep even a one-frame group-like fragment for now. A pause or
                # imperfect EXIF timestamp may have cut one physical burst into
                # several hard temporal blocks; those fragments must get a chance
                # to merge before min_frames is enforced.
                if has_group_face_count(subgroup, self.config):
                    group_like.append(subgroup)
                else:
                    rejected += 1
            split_weight_done += candidate_weight
            self.progress(
                stage_end,
                f"Разделение физических групп: блок {candidate_idx}/{len(candidates)} готов; найдено границ={split_count}",
            )

        self.progress(59.0, f"Склейка частей одной группы: найдено фрагментов={len(group_like)}")

        def merge_progress(local: float, detail: str) -> None:
            local = max(0.0, min(1.0, float(local)))
            self.progress(59.0 + 2.0 * local, f"Склейка частей одной группы — {detail}")

        group_like, merged_blocks = merge_adjacent_group_blocks(
            group_like, self.config, progress=merge_progress
        )
        stats.group_blocks_merged += merged_blocks

        merged_group_like: list[list[FrameAssessment]] = []
        validate_total = max(1, len(group_like))
        self.progress(61.0, f"Проверка найденных групп: 0/{len(group_like)}")
        for block_idx, block in enumerate(group_like, start=1):
            self._check_cancelled()
            block = stabilize_group_frame_order(block)
            if len(block) < min_frames:
                stats.skipped_short_series += 1
            elif looks_like_group_series(block, self.config):
                merged_group_like.append(block)
            else:
                rejected += 1
            self.progress(
                61.0 + 1.0 * block_idx / validate_total,
                f"Проверка найденных групп: {block_idx}/{len(group_like)}",
            )
        group_like = merged_group_like

        stats.refined_series = len(group_like)
        stats.weak_series_rejected += rejected
        self.message(
            f"Групповых серий подтверждено: {len(group_like)}; "
            f"границ найдено по смене состава: {stats.group_identity_splits}; "
            f"объединено частей той же группы: {merged_blocks}; "
            f"отброшено слабых/негрупповых блоков: {rejected}"
        )

        if not group_like:
            self.progress(99.0, "Групповые серии не найдены")
            return

        self.progress(62.0, f"Группы определены: {len(group_like)}. Подготовка выбора лучших дублей...")
        secondary_workers = self._secondary_face_workers()
        recycle_every = self._group_gpu_recycle_every()
        if self._gpu_memory_safe_mode():
            try:
                mem_limit_gb = float(self.config.get("runtime", {}).get("gpu_session_mem_limit_gb", 6.0))
            except (TypeError, ValueError):
                mem_limit_gb = 6.0
            self.log.info(
                "GPU MEMORY SAFE MODE | secondary_workers=%d | worker_arena_budget_gb=%.1f | recycle_every=%d",
                secondary_workers, mem_limit_gb, recycle_every,
            )
            self.message(
                f"Защита VRAM: secondary Group-этапы до {secondary_workers} GPU-сессий; "
                f"CUDA-arena бюджет на InsightFace-worker {mem_limit_gb:.1f} ГБ"
                + (f"; перезапуск gaze-сессий каждые {recycle_every} групп" if recycle_every else "")
            )

        rescue_enabled = bool(self.config.get("group", {}).get("highres_rescue_enabled", False))
        rescue_analyzers: list = []
        rescue_config: dict | None = None
        if rescue_enabled:
            self.progress(62.0, "Загрузка high-res модели для поиска пропущенных лиц...")
            rescue_config = self._group_rescue_config()
            det_size = int(rescue_config.get("analysis", {}).get("insightface_det_size_group", 1536))
            det_thresh = float(rescue_config.get("analysis", {}).get("insightface_det_thresh_group", 0.14))
            min_fraction = float(rescue_config.get("group", {}).get("min_track_face_fraction", 0.00018))
            self.log.info(
                "GROUP HIGHRES PASS START | det_size=%d | det_thresh=%.3f | min_face_fraction=%.6f",
                det_size, det_thresh, min_fraction,
            )
            self.message(
                f"High-res поиск пропущенных лиц: detector={det_size}px, threshold={det_thresh:.2f}; "
                "новое лицо требуется подтвердить на нескольких дублях"
            )
            try:
                rescue_analyzers, rescue_backend = self._create_analyzer_pool(
                    rescue_config, "High-res поиск лиц", max_workers=secondary_workers
                )
                self.message(f"High-res распознавание: {rescue_backend}")
            except Exception as exc:
                self.log.exception("Could not start high-res group rescue; using stable primary roster only")
                self.message(f"High-res поиск лиц недоступен ({exc}). Продолжаю по устойчивому основному составу.")
                rescue_analyzers = []

        attention_enabled = bool(self.config.get("group", {}).get("camera_attention_enabled", False))
        attention_analyzers: list = []
        attention_config: dict | None = None
        if attention_enabled:
            attention_config = self._group_attention_config()
            preview_edge = int(self.config.get("group", {}).get("camera_attention_preview_long_edge", 4800))
            shortlist = int(self.config.get("group", {}).get("camera_attention_shortlist", 3))
            self.log.info(
                "GROUP CAMERA ATTENTION START | shortlist=%d | preview=%d | det_size=%d | det_thresh=%.3f",
                shortlist,
                preview_edge,
                int(attention_config.get("analysis", {}).get("insightface_det_size_group", 1280)),
                float(attention_config.get("analysis", {}).get("insightface_det_thresh_group", 0.20)),
            )
            self.message(
                f"Взгляд в камеру: финальная high-res проверка {shortlist} лучших дублей каждой группы"
            )
            # When face rescue is enabled, reuse its already-loaded pool for the
            # shortlist. Otherwise create a dedicated pool with the gaze profile.
            if not rescue_analyzers:
                self.progress(62.0, "Загрузка модели финальной проверки взгляда...")
                try:
                    attention_analyzers, attention_backend = self._create_analyzer_pool(
                        attention_config, "Проверка взгляда",
                        max_workers=min(secondary_workers, max(1, shortlist)),
                    )
                    self.message(f"Проверка взгляда: {attention_backend}")
                except Exception as exc:
                    self.log.exception("Could not start camera-attention analyzer; continuing without gaze")
                    self.message(f"Проверка взгляда недоступна ({exc}). Продолжаю без критерия взгляда.")
                    attention_enabled = False
                    attention_analyzers = []

        def recycle_dedicated_attention_pool(group_idx: int) -> None:
            nonlocal attention_analyzers, attention_enabled
            # Only a dedicated gaze pool is recycled here. If gaze reuses the
            # high-res rescue pool, the per-session memory cap + worker cap still
            # protect VRAM without repeatedly rebuilding the rescue models.
            if (
                recycle_every <= 0
                or not attention_analyzers
                or attention_config is None
                or group_idx >= len(group_like)
                or group_idx % recycle_every != 0
            ):
                return
            self.message(
                f"Защита VRAM: освобождение CUDA-кэша после группы {group_idx}; "
                "перезапуск модели взгляда..."
            )
            self.log.info("GPU MEMORY RECYCLE | stage=gaze | after_group=%d", group_idx)
            self._close_analyzer_pool(attention_analyzers)
            attention_analyzers = []
            try:
                shortlist = int(self.config.get("group", {}).get("camera_attention_shortlist", 3))
                attention_analyzers, attention_backend = self._create_analyzer_pool(
                    attention_config, "Проверка взгляда",
                    max_workers=min(secondary_workers, max(1, shortlist)),
                )
                self.log.info("GPU MEMORY RECYCLE DONE | backend=%s", attention_backend)
            except Exception as exc:
                self.log.exception("Could not recreate camera-attention analyzer after VRAM recycle")
                self.message(
                    f"Не удалось перезапустить модель взгляда после очистки VRAM ({exc}). "
                    "Оставшиеся группы будут обработаны без критерия взгляда."
                )
                attention_enabled = False

        total_weight = max(1, sum(max(1, len(frames)) for frames in group_like))
        completed_weight = 0
        try:
            for group_idx, frames in enumerate(group_like, start=1):
                self._check_cancelled()
                weight = max(1, len(frames))
                group_start = 62.0 + 36.0 * completed_weight / total_weight
                group_end = 62.0 + 36.0 * (completed_weight + weight) / total_weight
                span = group_end - group_start
                if rescue_analyzers and rescue_config is not None:
                    rescue_end = group_start + span * (0.34 if attention_enabled else 0.42)
                    rescue_frames = self._analyze_group_highres(
                        frames, rescue_analyzers, rescue_config, group_idx, len(group_like),
                        group_start, rescue_end,
                    )
                    preliminary_start = rescue_end
                else:
                    rescue_frames = None
                    preliminary_start = group_start

                if attention_enabled:
                    preliminary_end = group_start + span * 0.56
                    gaze_start = preliminary_end
                    gaze_end = group_start + span * 0.78
                    selection_start = gaze_end
                    selection_end = group_start + span * 0.92
                else:
                    preliminary_end = group_start + span * 0.88
                    selection_start = preliminary_start
                    selection_end = preliminary_end

                def preliminary_progress(local: float, detail: str, *, _start=preliminary_start, _end=preliminary_end, _idx=group_idx):
                    local = max(0.0, min(1.0, float(local)))
                    pct = _start + (_end - _start) * local
                    self.progress(pct, f"Группа {_idx}/{len(group_like)}: {detail}")

                stats.series_processed += 1
                stats.group_series += 1
                result, diag = select_group_series(
                    frames, self.config, progress=preliminary_progress, rescue_frames=rescue_frames,
                    diagnostic_log=not attention_enabled,
                )

                if attention_enabled and result.main is not None:
                    gaze_analyzers = rescue_analyzers if rescue_analyzers else attention_analyzers
                    attention_frames = None
                    if gaze_analyzers:
                        attention_frames = self._analyze_group_attention_shortlist(
                            frames, diag.camera_attention_shortlist_indices, gaze_analyzers,
                            group_idx, len(group_like), gaze_start, gaze_end,
                        )
                    if attention_frames:
                        def final_progress(local: float, detail: str, *, _start=selection_start, _end=selection_end, _idx=group_idx):
                            local = max(0.0, min(1.0, float(local)))
                            pct = _start + (_end - _start) * local
                            self.progress(pct, f"Группа {_idx}/{len(group_like)}: {detail}")
                        result, diag = select_group_series(
                            frames, self.config, progress=final_progress, rescue_frames=rescue_frames,
                            attention_frames=attention_frames, diagnostic_log=True,
                        )
                    else:
                        # Preliminary pass suppressed frame logs because gaze was
                        # expected. If gaze failed, emit one normal diagnostic pass.
                        result, diag = select_group_series(
                            frames, self.config, progress=None, rescue_frames=rescue_frames,
                            diagnostic_log=True,
                        )
                stats.group_tracks_confirmed += diag.confirmed_tracks
                stats.group_eye_problems += diag.eye_problems
                stats.group_missing_problems += diag.missing_problems
                stats.group_sharpness_problems += diag.sharpness_problems
                stats.group_quality_problems += diag.quality_problems
                stats.group_camera_attention_known += diag.camera_attention_known
                stats.group_camera_attention_away += diag.camera_attention_away
                if diag.camera_attention_known > 0:
                    stats.group_camera_attention_series += 1
                stats.group_problems_covered += diag.covered_problems
                stats.group_problems_unresolved += diag.unresolved_problems
                stats.group_backup_candidates += diag.backup_extras
                if result.main is not None:
                    roster_text = f"состав={diag.confirmed_tracks} (stable={diag.stable_tracks}"
                    if diag.highres_rescue_tracks:
                        roster_text += f", high-res +{diag.highres_rescue_tracks}"
                    roster_text += ")"
                    self.message(
                        f"Группа {group_idx}: {roster_text}; на RED проблем: глаза={diag.eye_problems}, "
                        f"нет лица={diag.missing_problems}, резкость={diag.sharpness_problems}, "
                        f"качество={diag.quality_problems}; "
                        f"взгляд-в-сторону={diag.camera_attention_away}/{diag.camera_attention_known}; "
                        f"закрыто YELLOW={diag.covered_problems}, без подходящего дубля={diag.unresolved_problems}"
                    )
                if result.main is None:
                    stats.series_without_selection += 1
                    completed_weight += weight
                    self.progress(group_end, f"Группа {group_idx}/{len(group_like)}: пропущена")
                    recycle_dedicated_attention_pool(group_idx)
                    continue

                self.progress(selection_end, f"Группа {group_idx}/{len(group_like)}: выбор RED/YELLOW")
                result.main.label_role = "red"
                self.log.info("GROUP PLAN RED (%s): %s", writer.red, result.main.photo.path.name)
                stats.group_main_selected += 1
                selections.append(result.main)
                for extra_no, extra in enumerate(result.extras, start=1):
                    self._check_cancelled()
                    extra.label_role = "yellow"
                    self.log.info("GROUP PLAN YELLOW (%s): %s", writer.yellow, extra.photo.path.name)
                    stats.group_extra_selected += 1
                    selections.append(extra)
                    meta_pct = selection_end + (group_end - selection_end) * extra_no / max(1, len(result.extras) + 1)
                    self.progress(meta_pct, f"Группа {group_idx}/{len(group_like)}: YELLOW {extra_no}/{len(result.extras)}")

                completed_weight += weight
                self.progress(group_end, f"Группа {group_idx}/{len(group_like)} готова")
                recycle_dedicated_attention_pool(group_idx)
        finally:
            self._close_analyzer_pool(rescue_analyzers)
            self._close_analyzer_pool(attention_analyzers)

        self.progress(99.0, "Завершение группового анализа...")

    @staticmethod
    def _count_metadata_write(stats: RunStats, item, destination: Path) -> None:
        """Count the actual metadata destination returned by XmpWriter.

        The old implementation inferred storage from the source extension and
        therefore reported every PSD/TIFF/DNG write as a sidecar even when XMP
        was embedded successfully.
        """
        photo = item.photo if isinstance(item, Selection) else item
        stats.xmp_written += 1
        destination = Path(destination)
        if destination.suffix.lower() == ".xmp":
            stats.sidecar_xmp_written += 1
            return
        stats.embedded_xmp_written += 1
        if photo.extension.lower() in {".jpg", ".jpeg"}:
            stats.jpeg_embedded_written += 1

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("Analysis cancelled by user")


class AnalysisIncompleteError(RuntimeError):
    """Primary frame analysis was incomplete, so deferred metadata commit is forbidden."""


class CancelledError(RuntimeError):
    pass
