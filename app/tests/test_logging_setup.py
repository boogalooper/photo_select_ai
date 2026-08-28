from __future__ import annotations

import logging

from app.core.logging_setup import ConsoleFormatter


def _record(message: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord("photo_select_ai", level, "", 0, message, (), None)


def test_console_formatter_colours_summary_and_marks_stage():
    formatter = ConsoleFormatter(use_color=True)
    summary = formatter.format(_record("===== ИТОГ АНАЛИЗА =====\nФайлов: 500"))
    stage = formatter.format(_record("Сканирование файлов: метаданные 25/500"))

    assert "\x1b[1;32m" in summary and "[ИТОГ]" in summary
    assert "\x1b[1;36m" in stage and "[ЭТАП]" in stage
    assert summary.endswith("\x1b[0m")


def test_ordinary_frame_diagnostic_remains_neutral():
    formatter = ConsoleFormatter(use_color=True)
    rendered = formatter.format(_record("GROUP FRAME IMG_0001.CR2 | score=0.8"))

    assert "\x1b[" not in rendered
    assert "[ЭТАП]" not in rendered


def test_file_style_formatter_is_not_changed():
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    rendered = formatter.format(_record("===== ИТОГ АНАЛИЗА ====="))

    assert "\x1b[" not in rendered
    assert "[ИТОГ]" not in rendered
