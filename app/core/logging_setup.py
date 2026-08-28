from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from app.paths import ROOT
LOG_PATH = ROOT / "logs" / "photo_select_ai.log"

_RESET = "\x1b[0m"
_BOLD_CYAN = "\x1b[1;36m"
_BOLD_GREEN = "\x1b[1;32m"
_BOLD_YELLOW = "\x1b[1;33m"
_BOLD_RED = "\x1b[1;31m"


def _enable_windows_ansi(stream) -> bool:
    """Enable virtual-terminal colours in the attached Windows console."""
    if not getattr(stream, "isatty", lambda: False)():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        kernel32 = ctypes.windll.kernel32
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError, ValueError):
        return False


class ConsoleFormatter(logging.Formatter):
    """Add sparse colour badges to important console records only."""

    _STAGE_PREFIXES = (
        "Сканирование файлов",
        "Быстрый индекс",
        "План меток",
        "Финальная запись",
        "RAW preview",
        "Распознавание лиц",
        "Предзагрузка изображений",
        "Групповых серий подтверждено",
        "Группы определены",
        "High-res",
        "Проверка взгляда",
        "Загрузка",
    )
    _WARNING_MARKERS = (
        "CUDA → CPU",
        "CUDA -> CPU",
        "fallback",
        "недоступ",
        "не удалось",
    )

    def __init__(self, *, use_color: bool):
        super().__init__("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
        self.use_color = bool(use_color)

    def _badge(self, record: logging.LogRecord) -> tuple[str, str]:
        message = record.getMessage().lstrip()
        lowered = message.casefold()
        if record.levelno >= logging.ERROR:
            return "ОШИБКА", _BOLD_RED
        if record.levelno >= logging.WARNING or any(marker.casefold() in lowered for marker in self._WARNING_MARKERS):
            return "ВНИМАНИЕ", _BOLD_YELLOW
        if "===== ИТОГ АНАЛИЗА =====" in message:
            return "ИТОГ", _BOLD_GREEN
        if message.startswith("=== Новый"):
            return "ЗАПУСК", _BOLD_CYAN
        if message.startswith(("Готово", "Анализ завершён")):
            return "ГОТОВО", _BOLD_GREEN
        if message.startswith(self._STAGE_PREFIXES):
            return "ЭТАП", _BOLD_CYAN
        return "", ""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        badge, colour = self._badge(record)
        if not badge:
            return rendered
        tagged = rendered.replace(" | " + record.getMessage(), f" | [{badge}] " + record.getMessage(), 1)
        if self.use_color:
            return f"{colour}{tagged}{_RESET}"
        return tagged


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("photo_select_ai")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(ConsoleFormatter(use_color=_enable_windows_ansi(stream.stream)))
    logger.addHandler(stream)
    return logger
