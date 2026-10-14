#!/usr/bin/env bash
# GPU check for batched-decode CUDA graphs: the CUDA-only unit test, then the
# same ShareGPT load with graphs off and on. Everything lands in OUT_DIR.
#
#   bash scripts/ab_decode_graphs.sh            # uses .venv/bin/python
#   PYTHON=/path/to/python bash scripts/ab_decode_graphs.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PYTHON:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-benchmarks/results/local-ab}"
CONCURRENCY="${MAX_CONCURRENCY:-64}"
NUM_REQUESTS="${NUM_REQUESTS:-200}"
DATASET=benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json
mkdir -p "$OUT_DIR"
exec > >(tee "$OUT_DIR/run.log") 2>&1

if [ ! -x "$PY" ]; then
  echo "python not found at $PY; set PYTHON=/path/to/venv/bin/python" >&2
  exit 2
fi

echo "== environment"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'flash_attn', flash_attn.__version__)"
git log --oneline -1

echo "== unit tests (flash)"
DISTIE_ATTN=flash "$PY" -m unittest discover -s worker/tests -t worker/tests -p 'test_batch_decode.py' -v
UNIT=$?
echo "== unit tests (full worker suite)"
"$PY" -m unittest discover -s worker/tests -t worker/tests 2>&1 | tail -3

wait_port() {
  for _ in $(seq 1 300); do
    (exec 3<>/dev/tcp/127.0.0.1/50052) 2>/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

for G in 0 1; do
  echo "== load test, WORKER_DECODE_GRAPHS=$G"
  PYTHONPATH=worker/src DISTIE_ATTN=flash WORKER_NUM_BLOCKS=512 WORKER_DECODE_GRAPHS=$G \
    "$PY" -m worker.main > "$OUT_DIR/worker-graphs$G.log" 2>&1 &
  WPID=$!
  if ! wait_port $WPID; then
    echo "worker failed to start; tail of its log:"; tail -30 "$OUT_DIR/worker-graphs$G.log"
    kill $WPID 2>/dev/null; continue
  fi
  "$PY" benchmarks/load/bench_serving.py --backend grpc --target localhost:50052 \
    --workload sharegpt --dataset "$DATASET" --num-requests "$NUM_REQUESTS" --rate inf \
    --max-concurrency "$CONCURRENCY" --out "$OUT_DIR/graphs$G-c$CONCURRENCY.json"
  kill -INT $WPID; wait $WPID 2>/dev/null
  grep -E "captured batch decode|batch decode graph|batch decode buckets" "$OUT_DIR/worker-graphs$G.log" | head -12
done

echo "== summary"
"$PY" - "$OUT_DIR" "$CONCURRENCY" <<'PY'
import json, sys
out, c = sys.argv[1], sys.argv[2]
runs = {}
for g in ("0", "1"):
    try:
        runs[g] = json.load(open(f"{out}/graphs{g}-c{c}.json"))
    except OSError:
        print(f"graphs={g}: no report")
for g, r in runs.items():
    s = r["summary"]
    print(f"graphs={g}  ok {s['completed']}/{s['requests']}  {s['output_throughput_tok_s']:.1f} tok/s  "
          f"TTFT p50 {s['ttft_ms']['p50']:.1f} ms  TPOT p50 {s['tpot_ms']['p50']:.2f} ms")
if len(runs) == 2:
    a = {x["request_id"]: x["output_tokens"] for x in runs["0"]["records"]}
    b = {x["request_id"]: x["output_tokens"] for x in runs["1"]["records"]}
    same = sum(a[k] == b.get(k) for k in a)
    print(f"requests with identical output length: {same}/{len(a)}  "
          f"(total tokens {sum(a.values())} vs {sum(b.values())})")
PY
echo "unit test exit code: $UNIT"
