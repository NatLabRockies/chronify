"""CSV utilities for auto-detection features."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=32)
def _get_file_size(path: str | Path) -> int:
    try:
        return os.path.getsize(path)
    except (OSError, FileNotFoundError):
        return 0


def _is_large_file(path: str | Path, threshold_mb: int = 50) -> bool:
    size_mb = _get_file_size(path) / (1024 * 1024)
    return size_mb > threshold_mb


def _should_use_auto_detect(auto_detect: Optional[bool] = None) -> bool:
    """Check if auto-detection should be used: parameter > environment > False."""
    if auto_detect is not None:
        return auto_detect
    return os.environ.get("CHRONIFY_AUTO_DETECT_CSV", "").lower() in ("true", "1", "yes")
