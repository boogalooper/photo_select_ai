from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import load_config, merged_config
from app.core.logging_setup import setup_logging
from app.core.pipeline import AnalysisPipeline, AnalysisIncompleteError, CancelledError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Photo Select AI — portrait and group research/evaluation build")
    parser.add_argument("folder", nargs="?", help="Shoot folder")
    parser.add_argument("--mode", choices=("portrait", "group"), default=None, help="Analysis mode")
    parser.add_argument("--cli", action="store_true", help="Run immediately without Tkinter")
    parser.add_argument("--config", type=Path, default=None, help="Alternative JSON config")
    parser.add_argument("--label-scheme", choices=("bridge", "lightroom", "custom"), default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.label_scheme:
        config = merged_config(config, {"xmp": {"scheme": args.label_scheme}})
    if args.mode:
        config = merged_config(config, {"runtime": {"mode": args.mode}})

    if args.cli:
        if not args.folder:
            print("CLI mode requires a shoot folder.", file=sys.stderr)
            return 2
        folder = Path(args.folder).expanduser().resolve()
        if not folder.is_dir():
            print(f"Folder does not exist: {folder}", file=sys.stderr)
            return 2
        try:
            pipeline = AnalysisPipeline(
                config,
                cancel_event=threading.Event(),
                progress=lambda p, m: print(f"\r{p:6.2f}% {m[:90]:90}", end="", flush=True),
                message=lambda m: print(f"\n{m}"),
            )
            stats, _ = pipeline.run(folder)
            print("\n")
            print(f"Файлов найдено: {stats.files_found}")
            print(f"Файлов проанализировано: {stats.files_analyzed}")
            print(f"Предварительных временных блоков: {stats.candidate_series}")
            print(f"Серий после уточнения: {stats.refined_series}")
            print(f"Отброшено слабых/случайных серий: {stats.weak_series_rejected}")
            if stats.run_mode == "group":
                print(f"Границ по смене состава детей: {stats.group_identity_splits}")
                print(f"Объединено частей той же группы: {stats.group_blocks_merged}")
                print(f"Групповых серий: {stats.group_series}")
                print(f"Выбрано RED групп: {stats.group_main_selected}")
                print(f"Выбрано YELLOW кандидатов: {stats.group_extra_selected}")
                print(f"Людей учтено в составах групп: {stats.group_tracks_confirmed}")
                print(f"Проблем на RED — глаза: {stats.group_eye_problems}")
                print(f"Проблем на RED — лицо не найдено: {stats.group_missing_problems}")
                print(f"Проблем на RED — резкость глаз/лица: {stats.group_sharpness_problems}")
                print(f"Проблем на RED — техническое качество: {stats.group_quality_problems}")
                print(f"Проблем на RED — поворот головы: {stats.group_pose_problems}")
                print(f"Проверка взгляда — серий с надёжными данными: {stats.group_camera_attention_series}")
                print(f"Проверка взгляда — лиц оценено на RED: {stats.group_camera_attention_known}")
                print(f"Проверка взгляда — уверенно смотрят в сторону на RED: {stats.group_camera_attention_away}")
                print(f"Проблем закрыто целевыми YELLOW: {stats.group_problems_covered}")
                print(f"Резервных YELLOW добавлено: {stats.group_backup_candidates}")
                print(f"Проблем без подходящего дубля: {stats.group_problems_unresolved}")
            else:
                print(f"Объединено соседних фрагментов: {stats.local_fragments_merged}")
                print(f"Объединено серий через паузу: {stats.cross_block_series_merged}")
                print(f"Портретных серий: {stats.portrait_series}")
                print(f"Выбрано лучших кадров: {stats.portrait_selected}")
            print(f"Серий без выбора: {stats.series_without_selection}")
            print(
                "Сканирование: "
                f"EXIF={stats.scan_exif_reads}, cache={stats.scan_exif_cache_hits}, "
                f"время EXIF={stats.scan_time_from_exif}, mtime={stats.scan_time_from_file}, "
                f"по имени={stats.scan_order_from_name}, ошибки EXIF={stats.scan_exif_failures}, "
                f"пропущено={stats.scan_files_skipped}"
            )
            print(f"RAW+JPEG пар обработано как один кадр: {stats.raw_jpeg_pairs_collapsed}")
            print(
                "RAW preview: "
                f"rawpy={stats.raw_preview_rawpy}, JPEG из контейнера={stats.raw_preview_embedded_jpeg}, "
                f"demosaic={stats.raw_preview_demosaic}"
            )
            print(f"Ресурсов со старыми RED/YELLOW очищено: {stats.labels_cleared_before_run}")
            print(f"Меток записано: {stats.xmp_written}")
            print(f"Встроено в исходные файлы: {stats.embedded_xmp_written}")
            print(f"  из них JPG/JPEG: {stats.jpeg_embedded_written}")
            print(f"Sidecar XMP: {stats.sidecar_xmp_written}")
            print(f"Ошибок финальной записи меток: {stats.metadata_errors}")
            print(f"Кадров без лица: {stats.frames_without_faces}")
            print(f"Ошибок анализа: {stats.analysis_errors}")
            return 0
        except CancelledError:
            return 130
        except AnalysisIncompleteError as exc:
            print(f"\nОшибка анализа: {exc}", file=sys.stderr)
            return 1

    from app.gui.main_window import MainWindow
    MainWindow(config, initial_folder=args.folder).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
