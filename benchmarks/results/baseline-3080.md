# 3080 Laptop 基线（2026-10-09）

当前引擎仍是阶段 1 的路径：HuggingFace 前向，每步把分页 KV gather 成稠密 cache 再 scatter 回去，一次只跑一条请求。下面的数字是这条路径的本地基线，正式对比仍以云上同环境为准。

环境：WSL2，RTX 3080 Laptop 16 GB，Qwen2.5-1.5B-Instruct，torch 2.14.1+cu130，CUDA 13.0。Worker 默认 fp16。压测打在 Worker `:50052`，不经过 Gateway。

## 正确性

| 报告 | 精度 | 结果 |
|------|------|------|
| `correctness-fp32-cuda.json` | fp32 | 8/8 与 HF 贪心逐 token 一致，logits 最大绝对误差 4.2e-5 |
| `correctness-fp16.json` | fp16 | 8/8 逐 token 一致（阈值只要求 top-1 ≥ 0.98），logits 最大绝对误差 1.6e-2 |

更早的 Mac CPU fp32 结果仍在 `correctness-fp32.json`，没有被这次覆盖。

## 吞吐

`serving-sharegpt-rateinf.json`：ShareGPT 200 条，全部成功，墙钟 858 s。

| 指标 | p50 | p99 |
|------|-----|-----|
| TTFT | 29.6 ms | 188 ms |
| TPOT | 23.7 ms | 33.5 ms |
| E2E | 5.08 s | 7.81 s |

吞吐 0.23 req/s，40.8 output tok/s（共 35040 个输出 token）。

读法：并发被限制为 1，因为 `TorchEngine` 的前向丢进默认线程池，KV slab 又是所有请求共用的；多条同时跑会互相踩缓存。`make bench` 因此默认 `MAX_CONCURRENCY=1`。TTFT 低，是因为没有排队，prefill 只是单条 1.5B 的一次前向。吞吐被串行 decode 卡住，这是阶段 4 Continuous Batching 要超过的数。
