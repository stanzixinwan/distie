"""Load the C++ distie_core extension.

The .so lives in core/build after `make core`. Unit tests do not import
this module; they inject a fake allocator instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_block_pool(num_blocks: int):
    _ensure_core_on_path()
    try:
        import distie_core
    except ImportError as exc:
        raise ImportError("distie_core not found. Build it with: make core") from exc
    return distie_core.BlockPool(num_blocks)


def _ensure_core_on_path() -> None:
    override = os.getenv("DISTIE_CORE_DIR")
    repo = Path(__file__).resolve().parents[3]
    directory = Path(override) if override else repo / "core" / "build"
    path = str(directory)
    if directory.is_dir() and path not in sys.path:
        sys.path.insert(0, path)
