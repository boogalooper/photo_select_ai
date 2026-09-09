from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.paths import ROOT
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.json"
UI_STATE_PATH = ROOT / "config" / "ui_state.json"
UI_STATE_VERSION = 20

# Values that are safe to preserve when an unknown/very old UI-state schema is
# encountered. Algorithm thresholds intentionally reset to current defaults.
_STABLE_UI_STATE_KEYS = {
    "folder", "mode", "scheme", "custom_red", "custom_yellow",
    "advanced_visible", "insightface_provider", "cuda_conv_algo",
    "cuda_fallback", "cpu_workers",
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path or DEFAULT_CONFIG_PATH
    with source.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_ui_state() -> dict[str, Any]:
    if not UI_STATE_PATH.exists():
        return {}
    try:
        with UI_STATE_PATH.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
        if not isinstance(value, dict):
            return {}

        try:
            version = int(value.get("state_version", 0))
        except (TypeError, ValueError):
            version = 0

        if version == UI_STATE_VERSION:
            return value
        if version in {18, 19}:
            # The immediately preceding UI schema uses compatible controls. Mark
            # the schema migrated so later versions can make explicit choices.
            migrated = dict(value)
            migrated["state_version"] = UI_STATE_VERSION
            return migrated

        # Very old or future state: keep only durable UI/environment choices.
        # Selection thresholds return to the current release defaults instead
        # of silently overriding newly recommended values.
        migrated = {k: value[k] for k in _STABLE_UI_STATE_KEYS if k in value}
        migrated["state_version"] = UI_STATE_VERSION
        return migrated
    except Exception:
        return {}


def save_ui_state(state: dict[str, Any]) -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = UI_STATE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    tmp.replace(UI_STATE_PATH)


def merged_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    _deep_update(result, overrides)
    return result


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
