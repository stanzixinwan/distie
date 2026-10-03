# DistIE：面向 KV-Cache 复用的分布式 LLM 推理引擎

**Project Proposal: DistIE — Distributed Inference Engine with Cache-Aware Scheduling and Routing**

> 状态：进行中（阶段 0、1 已完成原型，见 §6）｜最后更新：2026-10

---

## 1. 项目目标

构建一个从零实现、可度量的 LLM 推理服务系统，主题是 **KV-Cache 的跨请求复用与跨节点复用**：

- **单节点**：实现 Paged KV-Cache、自有模型前向与 Continuous Batching，使吞吐与延迟显著优于 HuggingFace 原生 `generate`，并能解释与 vLLM 之间的差距来源。
- **跨请求**：通过前缀缓存（Prefix Caching）让共享 system prompt / 多轮对话的请求复用已计算的 KV 块，降低首字延迟（TTFT）。
- **跨节点**：Go 路由层根据各 Worker 上报的缓存摘要做前缀感知路由，把请求送到"已经算过这段前缀"的 Worker 上。

本项目在本地单卡上开发，正式基准测试与多 Worker 扩展实验在云上多卡环境完成（见 §5.3），所有结论以基准测试数据为准。

## 2. 非目标

为保证范围可控，以下内容明确不做（或仅在主线完成后考虑）：

- 张量并行 / 流水线并行等多卡模型切分（与 KV 复用主题正交；列为主线完成后的可选扩展，见阶段 8）
- 量化、投机解码（Speculative Decoding）
- 鉴权、多租户、会话存储等业务层功能
- 生产级部署（Kubernetes 编排等）



## 3. 系统架构

```mermaid
flowchart LR
    C[Client] -->|gRPC stream| R[Router - Go]
    R -->|前缀感知路由| W1
    R --> W2[Worker 2]
    W1 -.->|健康 + 缓存摘要| R
    W2 -.-> R
    subgraph W1[Worker 1 - Python]
        S[gRPC Servicer<br/>asyncio] --> Q[Scheduler Loop]
        Q --> M[Model Runner<br/>自有 Qwen2 前向]
        M --> K[Paged Attention Kernel]
        Q <--> B[KV Block Manager<br/>C++ / Pybind11]
    end
```





### 3.1 路由层（Router — Go）

- **职责**：接入与流式转发、限流（漏桶）、多 Worker 负载均衡、**前缀感知路由**。
- **路由策略**：Worker 通过 `WorkerControl.ReportHealth` 流定期上报负载与缓存摘要（前缀块哈希集合）。Router 对请求 prompt 计算同样的块哈希，按「前缀命中长度 − 负载惩罚」打分选择 Worker；无命中时退化为最少负载。
- **为什么用 Go**：路由层是 I/O 密集、高并发、需要长连接流式转发的组件，与 SGLang Router、TGI Router 等工业实现的定位一致。



### 3.2 Worker（Python）

- **gRPC Servicer（asyncio）**：只负责 I/O——接收请求、放入等待队列、把生成的 token 流式推回。
- **Scheduler Loop**：独占 GPU 的单一调度循环。每一步：
  1. 准入：等待队列中的请求若能拿到足够的 KV 块则加入运行批次（优先查前缀缓存）；
  2. 抢占：块耗尽时按策略抢占（先实现 recompute，后续可加 swap 到 CPU）；
  3. 构造批次输入：展平的 token、position、每条序列的 block table 与长度；
  4. 前向 + 采样，追加 token，释放已完成序列的块。
- **Model Runner**：自己实现 Qwen2 模型结构（权重从 HuggingFace 加载），attention 层直接读写 Paged KV slab，区分 prefill 与 decode 两条路径。



### 3.3 KV 块管理器（C++）

- **职责**：KV 块的**元数据**管理——分配/释放、引用计数、前缀哈希索引、LRU 淘汰、GPU/CPU 两级存储的换入换出计划。
- **数据面与控制面分离**：K/V 张量本身由 torch 在 GPU（以及 CPU pinned memory）上预分配；C++ 只管理块 ID 及其状态，不持有张量数据，避免每步跨设备拷贝。
- **为什么用 C++**：引用计数、哈希索引、淘汰、写时复制（Copy-on-Write）组合在一起是一个状态复杂、需要强不变式保证的数据结构，适合用 C++ 实现并用 GTest 做细粒度测试。其性能收益将通过与同接口 Python 实现的对比来验证（见 §5），而不是预设。



## 4. 关键设计决策


| 决策        | 选择                                                                                                 | 理由                                                                               |
| --------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 注意力计算     | 自有模型前向 + Paged Attention kernel                                                                    | HF 的 attention 只接受稠密 KV，batching 时需每步 gather + padding，抵消分页收益                    |
| Kernel 路线 | 先用 PyTorch 索引实现正确版本，再接入 `flash_attn_with_kvcache`（`block_table`）或 FlashInfer，最后视情况自写 Triton kernel | 先正确再快；每一步都有基准对照                                                                  |
| 调度模型      | 单调度循环独占 GPU，asyncio 仅做 I/O                                                                         | 避免多请求争抢 GPU；分配器无需加锁                                                              |
| 抢占策略      | 先 recompute，后 swap                                                                                 | recompute 实现简单；swap 依赖 CPU 存储层                                                   |
| 前缀缓存粒度    | 以完整块为单位、链式哈希（前缀哈希 + 本块 token）                                                                      | 与 vLLM 方案一致，便于跨 Worker 共享同一哈希空间                                                  |
| 开发环境      | macOS 写无 GPU 组件，WSL2 跑 GPU 组件，统一用类 Unix 构建脚本                                                       | flash-attn、FlashInfer、vLLM 均以 Linux 为主；Go / C++ / 调度逻辑借助 FakeEngine 可在 Mac 上开发测试 |
| 数值精度      | 本地 fp16 / bf16，不做 FP8                                                                              | RTX 3080（Ampere，sm_86）支持 bf16 与 flash-attn 2 / FlashInfer，但不支持 FP8               |




## 5. 评测方案



### 5.1 正确性

- fp32 下贪心解码输出与 HF `generate` **逐 token 一致**；
- fp16 下比较前 N 个 token 的 top-1 一致率与 logits 最大绝对误差；
- 前缀缓存命中与未命中两种路径的输出一致；
- C++ 块管理器：GTest 覆盖引用计数、淘汰、重复释放、耗尽等边界情况。



### 5.2 性能

**负载**

- A：ShareGPT 采样（通用对话，长度分布真实）
- B：合成共享前缀负载（K 个 system prompt × M 个用户问题）
- C：多轮对话（每轮携带完整历史）

**对比对象**

- 下限：HF `generate`（逐请求）
- 本项目各阶段版本（展示每一步的增量）
- 上限：vLLM（同模型、同 GPU、同负载）

**指标**：吞吐（output tok/s）、TTFT p50/p99、TPOT p50/p99、KV 块利用率、前缀命中率、多 Worker 负载均衡度。

所有结果（包括不理想的结果）连同分析记录在 `benchmarks/` 目录。

### 5.3 开发与实验环境


| 环境             | 硬件                            | 用途                                                                   |
| -------------- | ----------------------------- | -------------------------------------------------------------------- |
| macOS          | 无 CUDA                        | Go Router、C++ 块管理器、调度逻辑（FakeEngine）的开发与单元测试                          |
| Windows + WSL2 | RTX 3080 Laptop，16 GB 显存      | 模型前向、kernel、正确性测试、日常性能迭代；本地可跑 Qwen2.5-0.5B / 1.5B / 3B，多 Worker 功能联调 |
| 云 GPU（按需租用）    | A100 / H100 / L40S 等，单机 1–8 卡 | 正式基准测试（7B/8B 级模型）、与 vLLM 同环境对比、多 Worker 路由实验、可选的张量并行实验               |


本地数据用于迭代和定位问题；对外报告的结论以云上同环境测得的数据为准，并注明 GPU 型号、驱动、CUDA 与依赖版本。

## 6. 里程碑



### 阶段 0：网关与 RPC 基础 ✅

- Go Gateway：配置、上游 gRPC 客户端、流式转发、漏桶限流（超限返回 `ResourceExhausted`）。
- Protobuf 定义推理服务与 Worker 控制面。



### 阶段 1：块分配与分页 KV 原型 ✅

- C++ `BlockPool` + Pybind11 绑定；`TorchEngine` 以块 ID 为页表，KV 存于预分配 torch slab。
- **已知局限**：仍依赖 HF 前向，每步需 gather/scatter；单请求串行执行。这些将在阶段 3、4 解决。



### 阶段 2：度量基线

- 清理仓库（移除构建产物与缓存文件）；项目迁入 WSL 文件系统，`.ps1` 脚本替换为 `.sh` / Makefile。
- 正确性测试集与压测脚本（§5），测出当前版本与 HF `generate` 的基线数据。
- **验收**：一条命令产出 TTFT / TPOT / 吞吐报告。



### 阶段 3：自有模型前向 + Paged Attention

- 实现 Qwen2 前向，attention 直接读写 paged slab，移除 gather/scatter。
- 先 PyTorch 索引版，再接入 paged attention kernel。
- **验收**：通过正确性测试；batch=1 decode 延迟不劣于 HF `generate`。



### 阶段 4：Continuous Batching

- Scheduler Loop：准入、recompute 抢占、prefill/decode 混合批次。
- Servicer 改为纯 I/O，请求通过队列与调度循环交互；支持客户端取消。
- **验收**：负载 A 下吞吐显著高于 HF 基线；在云上与 vLLM 同环境对比，给出差距分析。



### 阶段 5：C++ KV 块管理器

- 在 `BlockPool` 基础上增加引用计数、前缀哈希索引、LRU 淘汰、Copy-on-Write。
- CPU 存储层：pinned memory slab + swap 抢占，C++ 负责生成换入换出计划，Python 执行异步拷贝。
- 同接口 Python 实现作为对照。
- **验收**：负载 B、C 下 TTFT 明显下降；给出 C++ 与 Python 实现的调度开销对比。



### 阶段 6：前缀感知多 Worker 路由

- Worker 经 `ReportHealth` 上报负载与前缀哈希摘要；Router 实现打分路由与退化策略。
- 本地在单卡上起多个 Worker 做功能联调；正式实验在云上每卡一个 Worker（4–8 卡）。
- **验收**：多 Worker、负载 B 下，前缀感知路由相比轮询 / 最少负载的命中率与 TTFT 对比。



### 阶段 7：可观测性

- Prometheus 指标：吞吐、TTFT、TPOT、队列长度、KV 利用率、前缀命中率、路由分布。
- Grafana 面板。



### 阶段 8（可选）：张量并行

- 在自有模型前向中按 Megatron 方式切分 attention / MLP，NCCL all-reduce 通信；KV 块管理器按 rank 管理各自的 KV 分片。
- **验收**：云上 2/4 卡运行单卡放不下或吃紧的模型，给出扩展效率与通信开销分析。



## 7. 风险与应对


| 风险                           | 应对                                                |
| ---------------------------- | ------------------------------------------------- |
| Paged attention kernel 开发难度高 | 优先使用 flash-attn / FlashInfer 现成接口，自写 kernel 作为加分项 |
| 笔记本 GPU 性能受功耗与散热影响，数据波动大     | 本地数据只用于迭代；正式结论用云上数据，多次运行取统计值                      |
| 本地与云上环境不一致导致结果不可复现           | 固定依赖版本，提供环境脚本 / Docker 镜像，报告中记录完整环境信息             |
| C++ 块管理器性能收益可能不明显            | 如实报告；其价值同时体现在不变式保证与可测试性上                          |




## 8. 设计修订记录

- **2026-10**：原第二阶段以「Python GC 导致 KV 分配卡顿」为动机引入 C++显存池。复盘后认为块分配频率低（每序列每 16 token 一次），GC 并非瓶颈，且显存实际由 torch 管理。C++ 层职责调整为 KV 块元数据管理（引用计数、前缀缓存、淘汰、两级存储），见 §3.3。
- **2026-10**：发现基于 HF 前向的 gather/scatter 方案无法高效支持 Continuous Batching，将「自有模型前向 + Paged Attention」调整到 batching 之前。
- **2026-10**：移除多线程 C++ Tokenizer（HF tokenizers 已是 Rust 实现）、Redis / JWT（超出项目主题）、LibTorch（模型前向保留在 Python）。项目主题收敛为 KV-Cache 复用。
- **2026-10**：明确实验环境：本地 RTX 3080 Laptop（16 GB）开发，云上多卡做正式基准与扩展实验；张量并行由非目标调整为可选阶段 8。

