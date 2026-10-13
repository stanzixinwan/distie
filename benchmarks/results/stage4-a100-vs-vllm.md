# 阶段 4 云上对比：DistIE vs vLLM（2026-10-13）

同一台机器、同一份数据、同一个压测客户端，依次跑 vLLM 和 DistIE。原始结果在 `cloud/`。

## 环境

| 项 | 值 |
|----|----|
| GPU | NVIDIA A100-SXM4-80GB，驱动 570.172.08（最高 CUDA 12.8），见 `cloud/gpu.txt` |
| 模型 | Qwen2.5-1.5B-Instruct，fp16 |
| DistIE | `.venv`：torch 2.8.0+cu128，flash-attn 2.8.3，`DISTIE_ATTN=flash` |
| vLLM | 0.11.0（独立 venv `.venv-vllm`，torch 2.8.0+cu128）。新版 vLLM 0.31 的 torch 要求更新的驱动，在这台机器上起不来 |
| 负载 | ShareGPT 200 条，`--rate inf`，`max_output_len=256`，贪心，遇 EOS 停止，客户端 warmup 2 条 |

对齐采样：vLLM 默认会读模型自带的 `generation_config.json`（`repetition_penalty=1.05`），所以启动时加 `--generation-config vllm`，和 DistIE 一样不加惩罚。六次运行的输出 token 总数都在 34,917–34,970 之间（相差 < 0.2%），两边生成长度一致，吞吐可以直接比。

## 配置

| 名称 | 座位数 | 每步 token 预算 | KV 容量 | 其他 |
|------|--------|-----------------|---------|------|
| DistIE | `WORKER_MAX_SEQS=32` | 2048 | 512 页 × 256 = 131,072 token | 只给 batch=1 decode 录 CUDA graph |
| DistIE seqs64 | `WORKER_MAX_SEQS=64` | 2048 | 1024 页 = 262,144 token | 同上 |
| vLLM 对齐 | `--max-num-seqs 32` | `--max-num-batched-tokens 2048` | `--num-gpu-blocks-override 8192` × 16 = 131,072 token | `--no-enable-prefix-caching`；chunked prefill 默认开启 |
| vLLM 默认 | 256 | 默认 | 显存的 90%（约 255 万 token） | prefix caching、chunked prefill 都开 |

vLLM 两种配置都开了 torch.compile，并按批大小 1–64 录制 CUDA graph（decode 用 FULL，混合批用 PIECEWISE）。

## 结果

| 运行 | 并发 | output tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 | 墙钟 |
|------|------|-------------:|---------:|---------:|---------:|--------:|-----:|
| vLLM 对齐 `vllm-aligned-c1` | 1 | 207.7 | 18.7 ms | 32.4 ms | 4.72 ms | 0.99 s | 168.3 s |
| DistIE `distie-c1` | 1 | 173.2 | 19.0 ms | 30.6 ms | 5.68 ms | 1.18 s | 201.7 s |
| vLLM 对齐 `vllm-aligned-c64` | 64 | 4739.2 | 1.04 s | 2.04 s | 5.74 ms | 2.09 s | 7.4 s |
| DistIE `distie-c64` | 64 | 1475.4 | 3.40 s | 5.17 s | 19.75 ms | 6.71 s | 23.7 s |
| DistIE seqs64 `distie-c64-seqs64` | 64 | 2382.8 | 44.8 ms | 258 ms | 21.38 ms | 4.42 s | 14.7 s |
| vLLM 默认 `vllm-default-c64` | 64 | 6751.2 | 110 ms | 631 ms | 6.79 ms | 1.50 s | 5.2 s |

DistIE / vLLM（同配置）：并发 1 为 **83%**，并发 64 为 **31%**。

DistIE 并发 64 解码阶段的 GPU 利用率（`nvidia-smi dmon -s u`，1 s 采样一次）：开头 prefill 时为 96% / 78%，进入 decode 后稳定在 **45–47%**。

## 分析

**1. 单条 decode 已经接近 vLLM。** 并发 1 时 TTFT 基本相同，TPOT 5.7 ms 对 4.7 ms。这条路径已经录了 CUDA graph（阶段 3）；剩下约 1 ms/token 来自每步的 Python 调度循环、gRPC 流式返回和逐 token detokenize。

**2. 主要差距在批 decode，瓶颈是 CPU。** 有三点证据：

- **TPOT 随批大小的变化。** vLLM 从并发 1 到并发 64，TPOT 只从 4.7 涨到 5.7 ms；DistIE 从 5.7 涨到 19.8 ms。
- **换更快的 GPU 几乎没变快。** DistIE 并发 64 的 TPOT 在 3080 上是 29.7 ms，换成 A100 只降到 19.8 ms；并发 1 的 TPOT 则从 13.2 降到了 5.7 ms，跟着算力走。批路径的步长没有随 GPU 算力缩短，说明时间不花在 GPU 上。
- **GPU 一半时间在等。** decode 阶段 SM 利用率只有 45–47%（这个指标本身还偏乐观），GPU 一半以上时间没有 kernel 在跑。

**3. 每步开销基本固定，和批大小无关。** 座位数从 32 加到 64 后，每步要处理的序列翻倍，TPOT 只从 19.8 涨到 21.4 ms（+8%），吞吐 ×1.6（尾部请求逐渐结束、批变小，所以到不了 ×2）。对照 vLLM 默认配置的 TPOT 6.8 ms，DistIE 每步大约多 14–15 ms。这部分是：批 decode 没有 CUDA graph，Python 逐层发射 kernel（28 层，每层十几个算子），再加上每步在 Python 里组装 `cu_seqlens` / `slot_mapping` / block table。

**4. 并发 64 的 TTFT 是排队，不是 prefill 慢。** 两边对齐配置都只有 32 个座位，200 条请求同时到，后来的请求要等座位释放。DistIE 每步慢 3 倍多，座位释放得也慢，所以 TTFT p50 是 3.4 s 对 1.0 s。座位加到 64 后，DistIE 的 TTFT p50 降到 45 ms。

**5. vLLM 默认配置比对齐配置快 42%，这部分来自配置，不是实现。** 256 个座位加上约 20 倍的 KV 空间，让 64 条请求全部同时进批。DistIE 调大 `WORKER_MAX_SEQS` 也能拿到这部分提升（见 seqs64 那行），只是每步的固定开销还在。

## 下一步（按收益排序）

1. **批 decode 录 CUDA graph。** 按批大小分桶（1、2、4、8、16、24、32…）预先录制，实际批补齐到最近的桶；`slot_mapping`、block table 等用固定地址的缓冲区原地更新。预计可以补上大部分每步 14 ms 的差距。
2. **把每步的 metadata 组装移出 Python 热路径。** 例如预分配张量、增量更新，或者挪到 C++ core 里。
3. **默认座位数改成按 KV 容量推算，不再固定为 32。**
4. chunked prefill、prefix caching（原计划阶段 5）。

## 复现

```bash
# vLLM（独立 venv；驱动只支持 CUDA 12.8 时用 0.11.0）
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python "vllm==0.11.0" "transformers<5"

# 对齐配置
.venv-vllm/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --served-model-name distie \
  --dtype float16 --port 8000 --generation-config vllm \
  --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --num-gpu-blocks-override 8192 --no-enable-prefix-caching
# 默认配置：去掉最后两行参数

# DistIE（先停 vLLM）
PYTHONPATH=worker/src DISTIE_ATTN=flash WORKER_NUM_BLOCKS=512 .venv/bin/python -m worker.main
# seqs64：WORKER_NUM_BLOCKS=1024 WORKER_MAX_SEQS=64

# 压测（另开终端）。直接调用脚本、用 --out 指定路径，避免覆盖 3080 的结果
.venv/bin/python benchmarks/load/bench_serving.py --backend openai --target http://localhost:8000 \
  --workload sharegpt --dataset benchmarks/data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct --num-requests 200 --rate inf \
  --max-concurrency 64 --out benchmarks/results/cloud/vllm-aligned-c64.json
# DistIE：--backend grpc --target localhost:50052
```
