#!/bin/bash
# 从 proto 文件生成 Go 和 Python 的 gRPC 代码

set -e

PROTO_DIR="proto"
PY_OUT="worker/src/proto_gen"
MODULE="github.com/stanzixinwan/llm_inference_server"

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

# protoc emits `import inference_pb2`; make it a package-relative import.
python - "$PY_OUT/inference_pb2_grpc.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
path.write_text(
    text.replace(
        "import inference_pb2 as inference__pb2",
        "from . import inference_pb2 as inference__pb2",
        1,
    )
)
PY

echo "Proto generation complete."
