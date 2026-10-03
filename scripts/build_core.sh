#!/usr/bin/env bash
# Build the C++ core (BlockPool + distie_core Python module) and run its tests.
# Requires: cmake >= 3.18, a C++17 compiler, pybind11 in the active Python.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_DIR="$REPO_ROOT/core"
BUILD_DIR="$CORE_DIR/build"
PYTHON="${PYTHON:-python3}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

command -v cmake >/dev/null || { echo "cmake not found" >&2; exit 1; }
"$PYTHON" -c "import pybind11" 2>/dev/null || {
  echo "pybind11 missing. Run: $PYTHON -m pip install pybind11" >&2
  exit 1
}

cmake -S "$CORE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DPython_EXECUTABLE="$(command -v "$PYTHON")"
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "C++ core build OK. Python binding tests:"
"$PYTHON" "$CORE_DIR/tests/test_block_pool.py"
