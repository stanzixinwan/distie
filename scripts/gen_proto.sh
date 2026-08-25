#!/bin/bash
# 从 proto 文件生成 Go 和 Python 的 gRPC 代码

set -e

PROTO_DIR="proto"
PY_OUT="worker/src/proto_gen"
MODULE="github.com/omniserve/llm_inference_server"

mkdir -p proto/gen/go "$PY_OUT"

python -c "
import grpc_tools.protoc, sys
sys.exit(grpc_tools.protoc.main([
    '',
    '--proto_path=${PROTO_DIR}',
    '--go_out=.',
    '--go_opt=module=${MODULE}',
    '--go-grpc_out=.',
    '--go-grpc_opt=module=${MODULE}',
    '${PROTO_DIR}/inference.proto',
]))
"

python -m grpc_tools.protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$PY_OUT" \
  --grpc_python_out="$PY_OUT" \
  "$PROTO_DIR/inference.proto"

echo "Proto generation complete."
