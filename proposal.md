
# 项目提案：OmniServe 分布式高性能推理引擎
**Project Proposal: OmniServe - High-Performance Distributed LLM Inference System**

### 1. 项目背景 (Problem Statement)
当前大语言模型（LLM）在生产环境面临两大瓶颈：
*   **计算资源浪费**：传统的静态 Batching 导致 GPU 在等待长文本生成时出现空闲。
*   **内存碎片化**：KV Cache 占用大量显存且动态变化，导致 OOM 或低吞吐量。
*   **系统扩展性差**：单机推理难以支撑高并发用户需求。

**OmniServe** 旨在构建一个分布式的推理后端，通过系统级优化（Systems for ML）实现高吞吐、低延迟的模型服务。

### 2. 核心架构 (Architecture)
项目采用**解耦架构**，分为三层：
*   **接入层 (Control Plane - Go)**：基于 Go 语言实现的 API Gateway，负责高并发请求接收、鉴权、速率限制（Rate Limiting）。
*   **调度层 (Orchestrator - Go/gRPC)**：核心调度引擎。实现请求队列管理，根据 Worker 节点的显存水位进行负载均衡。
*   **推理层 (Data Plane - Python/C++/CUDA)**：基于 `vLLM` 或 `Triton` 思想的推理节点。实现 **Continuous Batching** 和 **PagedAttention**。

### 3. 技术栈 (Technical Stack)
*   **后端/系统**：Go, gRPC, Protobuf, Redis (状态存储)
*   **算法/推理**：Python, PyTorch, NVIDIA Triton, CUDA (选学)
*   **基础设施**：Docker, Kubernetes, Prometheus, Grafana
*   **模型**：Llama-3 (8B) 或 Mistral-7B

### 4. 关键技术点 (Key Features / Milestones)
*   **Phase 1: 分布式通信框架**
    *   使用 **gRPC** 定义双向流式通信协议，实现 Go 网关与 Python 推理节点的低延迟互联。
*   **Phase 2: 高效调度策略 (SWE 核心)**
    *   实现基于 **优先级队列** 的请求调度，支持请求在网关层的缓冲与聚合，防止后端过载。
*   **Phase 3: 推理性能优化 (MLE 核心)**
    *   实现 **Continuous Batching**：允许新请求在已有请求生成过程中动态加入 Batch。
    *   实现 **KV Cache 管理**：引入类似虚拟内存的分页机制，减少显存碎片，提升 2-3 倍吞吐量。
*   **Phase 4: 可观测性与部署**
    *   集成 **Prometheus** 监控指标（如：Tokens/sec, Time-to-First-Token, GPU Util）。
    *   编写 **Helm Charts**，支持在 K8s 上的快速水平扩容。

### 5. 预期成果 (Expected Impact)
*   **性能指标**：在高并发场景下，吞吐量相较于常规 FastAPI + Transformers 提升 **200%+**。
*   **系统稳定性**：支持节点故障自动剔除与请求重试，保证 99.9% 的服务可用性。
*   **工程价值**：产出一个具有工业级水准的开源代码库，并在简历上填补分布式系统和 ML Infra 的空白。