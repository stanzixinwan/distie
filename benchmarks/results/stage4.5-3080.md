# 阶段 4.5 本地验收：批 decode CUDA graph（2026-10-13）

环境：WSL2，RTX 3080 Laptop 16 GB，Qwen2.5-1.5B-Instruct fp16，flash-attn。Worker：`DISTIE_ATTN=flash WORKER_NUM_BLOCKS=512`，调度默认 32 座位、每步 2048 token。分支 `batch-decode-graph`（92bd545）。

## 正确性

`worker/tests/test_batch_decode.py` 在 GPU 上 5/5 通过，`test_graph_replay_matches_varlen` 实际执行，没有被跳过：分桶 graph replay 的 logits 和写入的 K/V 与 varlen eager 路径一致，覆盖批大小变化、序列加入和退出、跨页。

## 吞吐

ShareGPT 200 条，`--rate inf`，并发 64，两次都是 200/200 成功。同一个 worker 命令，只切换 `WORKER_DECODE_GRAPHS`。

| `WORKER_DECODE_GRAPHS` | output tok/s | TTFT p50 | TPOT p50 | 墙钟 |
|---|---:|---:|---:|---:|
| 0（eager） | 956.5 | 5274.7 ms | 29.75 ms | 36.5 s |
| 1（graph） | **1271.2** | 4031.9 ms | **22.68 ms** | 27.5 s |

开启后日志显示录制了 1、2、4、8、16、24、32 七个桶；关闭时日志为 `batch decode graphs disabled`。关闭时的数字与阶段 4（978.9 tok/s，TPOT 29.7 ms）一致，说明对比基线可信。

## 读法

- 吞吐 +33%，TPOT −24%。CUDA graph 去掉了批 decode 中 Python 逐层发射 kernel 的开销。
- 22.7 ms 仍是下限的 3 倍左右：3080 Laptop 显存带宽约 450 GB/s，每步读一遍约 3 GB 的 fp16 权重就要约 7 ms。
- 剩余时间有两个候选来源，尚未测量：
  1. 新请求补位时的 prefill 步仍走 eager，可能上千个 token，同批的 decode 都要等它；
  2. forward 之外的 CPU 工作：调度、采样同步、逐 token detokenize 和 gRPC 推送，后者在 asyncio 线程里与 GPU 线程争 GIL。
- 下一步用 `WORKER_STEP_STATS=200` 重跑同一负载，按 decode / prefill 分开看 sched、gpu、wait、emit 各占多少，再决定是做 chunked prefill，还是把采样和 detokenize 移出热路径。

原始报告在 WSL 的 `/tmp/graphs-off.json`、`/tmp/graphs-on.json`，尚未放进仓库。
