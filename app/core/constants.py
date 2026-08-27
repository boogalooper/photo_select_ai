from __future__ import annotations

# Group mode can keep several alternate takes for manual head replacement.
# Keep the user-facing limit and the normal default in one place so the GUI
# and selection engine cannot silently drift apart.
GROUP_YELLOW_MAX_LIMIT = 5
GROUP_YELLOW_DEFAULT_MAX = 2
