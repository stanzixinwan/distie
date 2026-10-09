# DistIE 开发日志

| 日期 | 说明 |
|------|------|
| 2026-08-19 | 初始化仓库：mini vLLM / 高性能 LLM 推理 serving |
| 2026-08-22 | 生成 Python & Go 代码，配置工具链 |
| 2026-08-22 | 更新提案 |
| 2026-08-24 | 网关源码 |
| 2026-08-24 | Worker 骨架：独立 gRPC（默认 :50052），不接 Gateway，不加载真模型 |
| 2026-08-27 | Gateway 路径对齐 GitHub 仓库 |
| 2026-08-27 | Protobuf package 改名 |
| 2026-08-27 | 项目名改为 DistIE |
| 2026-08-27 | Gateway：config + upstream.Client + handler 转发 |
| 2026-08-27 | 漏桶限流；超限返回 `ResourceExhausted: rate limit exceeded`（默认 20 rps / burst 40）。|
| 2026-09-06 | C++ Block 内存池 + Pybind11 |
| 2026-09-06 | FakeEngine 接入 BlockPool：生成只拿整数 ID，结束或取消时归还。|
| 2026-09-06 | TorchEngine：默认 Qwen2.5-1.5B-Instruct，CUDA fp16 逐 token 流式生成；KV 仍由 PyTorch 管。`WORKER_ENGINE=fake` 可回退。|
| 2026-09-06 | Windows 用 cu128 轮子装 GPU 版 torch（PyPI 默认常为 `+cpu`）。|
| 2026-09-09 | .cursorrules：每次改动最短记入 `devlog.md`。|
| 2026-09-09 | PagedKvCache：C++ Block ID 当页表，torch 槽位存 KV；TorchEngine scatter/gather，HF cache 不再是唯一存储。|
| 2026-10-03 | 仓库清理：删 `bin/gateway.exe` 与 `.ps1`，新增 `build_core.sh` + 顶层 `Makefile`；C++ `BlockPool` 去掉 arena/`block_view`，只管块 ID；proposal 统一称 Gateway。|
| 2026-10-03 | 正确性测试：`benchmarks/correctness/hf_parity.py` 对比分页 KV 路径与 HF `generate`（贪心 token 一致 + teacher-forced top-1/logits 误差），`make correctness`；`TorchEngine` 加 `trace()` 与 `dtype`；修复 transformers 5 下 `DynamicCache` 构造崩溃，依赖升至 `>=5.0`。|
| 2026-10-03 | 压测：`benchmarks/load/` 开环 Poisson 到达的 gRPC 客户端，ShareGPT / synthetic 负载，输出 TTFT/TPOT/E2E p50/p90/p99 与吞吐；`make sharegpt`、`make bench`。|
| 2026-10-03 | 开发环境：`make setup`（uv 建 `.venv` + cmake/ninja + protoc Go 插件）；Makefile 自动使用 `.venv`（绝对路径，兼容 macOS Make 3.81）。|
| 2026-10-03 | Mac 实跑：修 transformers 5 下 `apply_chat_template` 返回 dict；HF 参考强制 `repetition_penalty=1.0` 并自检纯贪心。Qwen2.5-1.5B fp32 CPU 8/8 逐 token 一致（max err 4.8e-5）。|
| 2026-10-09 | 阶段 2 基线（RTX 3080 Laptop，fp16 serving）：fp32/fp16 正确性均 8/8 贪心一致；ShareGPT 200 请求、并发 1，40.8 output tok/s，TTFT p50 29.6 ms，TPOT p50 23.7 ms。`make bench` 默认并发 1，避免当前引擎并发踩共用 KV。|

