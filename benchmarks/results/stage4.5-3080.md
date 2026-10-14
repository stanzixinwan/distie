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

## 每步耗时拆分（`WORKER_STEP_STATS=200`）

同一负载，graph 开。稳态窗口（32 座位满载）每 200 步的典型值：

| 步类型 | 占步数 | 每步 rows / 新 token | sched | gpu | wait | emit | 占墙钟 |
|---|---:|---|---:|---:|---:|---:|---:|
| decode（graph） | ~85% | 32 / 32 | 0.2 ms | **17.8–18.4 ms** | 3.0–3.5 ms | 0.2 ms | 67–76% |
| prefill（eager，混合批） | ~15% | 32 / 137–227 | 0.2 ms | 38–45 ms | 3.2–3.8 ms | 0.2 ms | 24–33% |

全窗口合计：gpu 84–87%，wait 11–14%，sched + emit 约 2%。

读法：

- **调度器不是瓶颈。** 纯 Python 的 `Scheduler` 每步 0.2 ms，约占 1%。这也意味着阶段 5 把调度挪到 C++，对吞吐帮助有限，价值在前缀缓存等功能本身。
- **wait 占 11–14%。** GPU 线程算完后，事件循环平均要 3 ms 才回来处理，这段时间被 32 路 detokenize 和 gRPC 推送占着（同时争 GIL）。
- **大头在 GPU 线程，而且与批大小无关。** decode 步 rows=32 时 17.8 ms，rows=21 时 17.8 ms，rows=6 时 20.0 ms，rows=1 时 15.1 ms。每步约 15–18 ms 的固定成本，是 3080 读一遍权重下限（约 8–9 ms）的 2 倍左右。
- **batch=1 比阶段 3 慢约 3 ms。** 阶段 3 单条 graph 的 TPOT 是 12.2 ms，这里 rows=1 的 gpu 是 15.1 ms。一个嫌疑：新 graph 的 block table 固定 513 列，flash-attn 按「列数 × 页大小」估计 KV 长度来决定 split-KV 份数，小批时可能切得过细。待验证。
- prefill 步每步 38–45 ms，占约 15% 的步数和四分之一以上的墙钟。它不影响总工作量，但会让同批 decode 等它，拉高 TPOT。

下一步：`gpu` 已拆成 `kernel`（CUDA event 量出的 forward 本身）和 `host`（GPU 线程里 forward 之外的时间：组装输入、采样、同步）。重跑一次即可判断这 18 ms 是 kernel 本身慢（需要算子融合 / 调 split 参数），还是 host 侧开销。

## GPU 线程再拆：kernel 与 host

`gpu` 拆成 `kernel`（forward 前后各一个 CUDA event，graph 步即纯 GPU 执行时间）和 `host`（GPU 线程里 forward 之外的时间）。稳态（rows=32）：

| 步类型 | gpu | kernel | host |
|---|---:|---:|---:|
| decode（graph） | 17.7–18.2 ms | **15.9–16.4 ms** | 1.8–2.2 ms |
| prefill（eager） | 34–45 ms | 31–42 ms | 3.0–3.5 ms |

rows=1 时 decode kernel 13.9 ms，rows≈6 时 17.8 ms，rows=32 时 15.9 ms：kernel 时间与批大小基本无关。

结论：**慢在 kernel 本身，host 侧只占约 2 ms。** 按 3080 Laptop 的带宽估算，每步读一遍约 3.1 GB 权重需要约 8 ms，attention 读 KV 约 1 ms，剩下约 6–7 ms 是大量小 kernel 的固定开销。

在 CPU 上数了一遍 `forward_decode` 的算子：每层约 56 个（flash 路径约 46 个），其中 RMSNorm 拆成 to / pow / mean / rsqrt / mul 等 7 个左右，RoPE 有 cat / neg / mul / add 十几个，Q/K/V 和 gate/up 分别是独立 GEMM。28 层合计约 1300 个 kernel。graph 去掉了 CPU 发射开销，但每个 kernel 在 GPU 上仍有几微秒的启动和尾部开销，1300 个累计约 6 ms，与上面的缺口吻合。eager 的 prefill 步每个 kernel 还要加上 Python 发射时间，所以 150–230 个新 token 就要 31–42 ms。

下一步是减少 kernel 数量（算子融合），对 graph decode 和 eager prefill 都有效：

1. RoPE 交给 `flash_attn_with_kvcache` 的 `rotary_cos` / `rotary_sin` 参数在 kernel 内完成；
2. RMSNorm 和残差相加融合成一个 Triton kernel；
3. Q/K/V 合并为一个 GEMM，gate/up 合并为一个 GEMM；
4. 对照实验：`torch.compile` 自动融合，看手工融合与编译器各能拿到多少。

原始报告：[`stage4.5-3080/graphs-off.json`](stage4.5-3080/graphs-off.json)、[`stage4.5-3080/graphs-on.json`](stage4.5-3080/graphs-on.json)。A/B 复现脚本：`scripts/ab_decode_graphs.sh`。
