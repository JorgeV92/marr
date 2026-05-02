from __future__ import annotations

from typing import Any


def _log_api_usage_once(obj: Any) -> None:
    """Compatibility hook for torchvision-style dataset classes."""
    del obj
