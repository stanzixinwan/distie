#!/usr/bin/env bash
# Create .venv (Python 3.12) with worker deps and C++ build tools, then
# install the protoc Go plugins pinned to the versions in proto/gen/go.
# Requires: uv (curl -LsSf https://astral.sh/uv/install.sh | sh). Go optional.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY_VERSION="${PY_VERSION:-3.12}"
PROTOC_GEN_GO=v1.36.12
PROTOC_GEN_GO_GRPC=v1.6.2

command -v uv >/dev/null || {
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

uv venv --python "$PY_VERSION" --allow-existing .venv
uv pip install --python .venv/bin/python -r worker/requirements.txt cmake ninja

if command -v go >/dev/null; then
  go install "google.golang.org/protobuf/cmd/protoc-gen-go@${PROTOC_GEN_GO}"
  go install "google.golang.org/grpc/cmd/protoc-gen-go-grpc@${PROTOC_GEN_GO_GRPC}"
else
  echo "warning: go not found; skipping protoc Go plugins (needed by make proto)" >&2
fi

echo "Dev environment ready: .venv ($(.venv/bin/python --version))"
