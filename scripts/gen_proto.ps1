# 从 proto 文件生成 Go 和 Python 的 gRPC 代码 (Windows)

$ErrorActionPreference = "Stop"

$PROTO_DIR = "proto"
$PY_OUT = "worker/src/proto_gen"
$MODULE = "github.com/stanzixinwan/llm_inference_server"

New-Item -ItemType Directory -Force -Path "proto/gen/go" | Out-Null
New-Item -ItemType Directory -Force -Path $PY_OUT | Out-Null

$env:Path += ";$env:USERPROFILE\go\bin"

Write-Host "Generating Go code..."
python -c @"
import grpc_tools.protoc, sys
sys.exit(grpc_tools.protoc.main([
    '',
    '--proto_path=$PROTO_DIR',
    '--go_out=.',
    '--go_opt=module=$MODULE',
    '--go-grpc_out=.',
    '--go-grpc_opt=module=$MODULE',
    '$PROTO_DIR/inference.proto',
]))
"@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Generating Python code..."
python -m grpc_tools.protoc `
  --proto_path="$PROTO_DIR" `
  --python_out="$PY_OUT" `
  --grpc_python_out="$PY_OUT" `
  "$PROTO_DIR/inference.proto"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# protoc emits `import inference_pb2`; make it a package-relative import.
$grpcFile = Join-Path $PY_OUT "inference_pb2_grpc.py"
$grpcSrc = Get-Content -Raw $grpcFile
$grpcSrc = $grpcSrc -replace '(?m)^import inference_pb2 as inference__pb2', 'from . import inference_pb2 as inference__pb2'
Set-Content -Path $grpcFile -Value $grpcSrc -NoNewline

Write-Host "Proto generation complete."
