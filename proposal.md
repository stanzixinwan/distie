# 项目提案基于 C++/Go 的分布式高性能推理引擎
**Project Proposal: DistIE: Distributed Inference Engine with C++ Performance Core**

### 1. 项目愿景 (Objective)
构建一个支持多机多卡、低延迟、高吞吐的 LLM 推理系统。通过 **Go 处理高并发网络请求**，**C++ 负责底层内存与算子优化**，**Python 进行模型逻辑编排**，实现一个完整的生产级高性能系统。

### 2. 三层架构设计 (The 3-Tier Architecture)

*   **接入层 (Gateway - Go)**
    *   **职责**：处理数万级并发连接，负载均衡，请求去重。
    *   **关键技术**：Goroutines, gRPC Server, Redis (Session state), JWT Auth.
*   **调度层 (Orchestrator - Python/C++)**
    *   **职责**：请求的批处理（Batching）策略，控制 Continuous Batching 逻辑。
    *   **关键技术**：Pybind11 (调用 C++ 核心), Asyncio.
*   **性能内核 (Engine Core - C++) —— *核心竞争力***
    *   **职责**：**KV-Cache 管理器**、**显存池管理**、**高性能请求队列**。
    *   **关键技术**：C++17, Smart Pointers, Multi-threading, CUDA Kernels (可选).

### 3. 技术栈 (Technical Stack)
*   **语言**：C++ (内核), Go (网关), Python (编排)
*   **通信**：gRPC, Protobuf, Shared Memory (本地进程间通信优化)
*   **库/工具**：Pybind11, LibTorch (C++ 版 PyTorch), NVIDIA Triton, GTest (C++ 测试框架)
*   **部署**：Docker, Kubernetes, Prometheus, Grafana

### 4. 核心研发里程碑 (Key Milestones)

#### 第一阶段：Go-Python 分布式基础 (SWE 信号)
*   实现 Go Gateway，通过 **gRPC 流式传输（Streaming）** 将 Prompt 发送到推理节点。
*   在 Go 中实现 **Leaky Bucket 算法** 进行流量整形，确保系统不会在瞬间高并发下崩溃。

#### 第二阶段：C++ 显存管理器 (MLSys/Infra 信号)
*   **问题**：Python 的垃圾回收（GC）在高频分配 KV-Cache 时会导致系统卡顿。
*   **解决**：用 **C++ 实现一个 Block-based Memory Pool**。
    *   预先分配大块显存，手动管理 Block 的分配与释放（类似 vLLM 的 PagedAttention 思想）。
    *   通过 **Pybind11** 将此管理器封装给 Python 层的推理循环使用。

#### 第三阶段：C++ 性能加速 (Low-level Engineering 信号)
*   **多线程 Tokenizer**：使用 C++ 实现并行化的文本编码/解码，消除 Python GIL 带来的延迟。
*   **Zero-copy 数据传输**：研究并实现如何减少 Tensor 在 CPU 和 GPU 之间的拷贝次数。

#### 第四阶段：系统监控与压测
*   使用 **Prometheus** 记录每秒处理的 Token 数（TPS）和首字延迟（TTFT）。
*   使用 **Grafana** 展示 C++ 显存池的利用率曲线。

