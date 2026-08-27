from __future__ import annotations

import shutil

from app.paths import ROOT
TEMP_PREVIEWS = ROOT / "temp" / "previews"


def cleanup_temp() -> None:
    TEMP_PREVIEWS.mkdir(parents=True, exist_ok=True)
    for child in TEMP_PREVIEWS.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass
