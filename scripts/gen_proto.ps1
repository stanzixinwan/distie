# 从 proto 文件生成 Go 和 Python 的 gRPC 代码 (Windows)

$ErrorActionPreference = "Stop"

$PROTO_DIR = "proto"
$GO_OUT = "proto/gen/go"
$PY_OUT = "worker/src/proto_gen"

New-Item -ItemType Directory -Force -Path $GO_OUT | Out-Null
New-Item -ItemType Directory -Force -Path $PY_OUT | Out-Null

# Go 代码生成
Write-Host "Generating Go code..."
protoc `
  --proto_path="$PROTO_DIR" `
  --go_out="$GO_OUT" `
  --go-grpc_out="$GO_OUT" `
  "$PROTO_DIR/inference.proto"

# Python 代码生成
Write-Host "Generating Python code..."
python -m grpc_tools.protoc `
  --proto_path="$PROTO_DIR" `
  --python_out="$PY_OUT" `
  --grpc_python_out="$PY_OUT" `
  "$PROTO_DIR/inference.proto"

Write-Host "Proto generation complete."
