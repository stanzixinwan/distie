#!/bin/bash
# 从 proto 文件生成 Go 和 Python 的 gRPC 代码

set -e

PROTO_DIR="proto"
GO_OUT="proto/gen/go"
PY_OUT="worker/src/proto_gen"

mkdir -p "$GO_OUT" "$PY_OUT"

# Go 代码生成
protoc \
  --proto_path="$PROTO_DIR" \
  --go_out="$GO_OUT" \
  --go-grpc_out="$GO_OUT" \
  "$PROTO_DIR/inference.proto"

# Python 代码生成
python -m grpc_tools.protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$PY_OUT" \
  --grpc_python_out="$PY_OUT" \
  "$PROTO_DIR/inference.proto"

echo "Proto generation complete."
