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

