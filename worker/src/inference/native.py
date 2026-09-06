"""Load the C++ distie_core extension.

The .pyd lives in core/build after scripts/build_core.ps1. Unit tests do
not import this module; they inject a fake allocator instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_block_pool(num_blocks: int, block_size_bytes: int):
    _ensure_core_on_path()
    try:
        import distie_core
    except ImportError as exc:
        raise ImportError(
            "distie_core not found. Build it with: powershell -File scripts/build_core.ps1"
        ) from exc
    return distie_core.BlockPool(num_blocks, block_size_bytes)


def _ensure_core_on_path() -> None:
    for directory in _candidate_dirs():
        if not directory.is_dir():
            continue
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
        return


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    override = os.getenv("DISTIE_CORE_DIR")
    if override:
        dirs.append(Path(override))
    repo = Path(__file__).resolve().parents[3]
    dirs.append(repo / "core" / "build")
    dirs.append(repo / "core" / "build" / "Release")
    return dirs
