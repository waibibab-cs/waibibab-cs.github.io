本编程指南的第 1‑3 章已经对 CUDA 与 GPU 编程进行了介绍（对应“CUDA官方文档学习1-12”），通过理论概念与简单代码示例讲解了各类基础主题。本指南第 4 部分中介绍各项 CUDA 具体特性的章节（对应“CUDA官方文档学习13+”），要求读者已经掌握之前章节所涵盖的相关概念。

CUDA 拥有大量适用于不同问题场景的特性，并非所有特性都适用于每一种业务场景。本章将对这些特性逐一进行介绍，说明其设计用途以及能够解决的问题。特性按照其所要解决问题的类型做了粗略分类。部分特性（例如 CUDA 图）可归属于多个类别。

更多完整细节可参见：https://docs.nvidia.com/cuda/cuda-programming-guide/part4.html#cuda-features
# 一、提升核函数性能
本节所列的全部特性，旨在帮助核函数开发者最大化核程序的执行性能。
## 1.1 异步屏障
异步屏障在《高级核编程》章节中首次引入，它支持对线程间同步进行更精细的控制。异步屏障将**抵达（arrive）** 与**等待（wait）** 两个屏障操作相互拆分开。这使得应用程序在等待其他线程抵达屏障的同时，可以执行那些不依赖该屏障结果的计算任务。

异步屏障可配置为不同的[线程作用域（thread scopes）](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-thread-scopes)。 异步屏障的完整详细说明参见：https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#asynchronous-barriers
## 1.2 异步数据拷贝与张量内存加速器（TMA）
在内核代码语境下，[异步数据拷贝](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies)指的是：在执行计算的同时，完成共享内存与 GPU 显存之间的数据搬移。注意不要把它和 CPU‑GPU 之间的异步内存拷贝相混淆。该特性需要依托异步屏障来实现。

异步数据拷贝的完整使用说明参见文档： https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies
## 1.3 流水线
[流水线](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines)是一种用于任务分段调度、协调多缓冲区生产者‑消费者模式的机制，常被用来实现计算与[异步拷贝操作](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies)的执行重叠。

关于在 CUDA 中使用流水线的完整说明与代码示例，请参见文档： https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines
## 1.4 基于集群发射控制的工作窃取
**工作窃取**是一种用于在负载不均衡场景下维持硬件利用率的技术：已经完成自身任务的工作单元，可以从其他工作单元那里 “窃取” 任务来执行。

集群发射控制是算力 10.0（Blackwell 架构）引入的特性，允许内核直接控制正在运行的线程块调度，以此实时适配不均衡的负载。 一个线程块可以取消另一个尚未开始执行的线程块或者线程块集群的发射，接管被取消块的索引，并立刻执行窃取过来的任务。

这套工作窃取执行流程能够让流式多处理器（SM）保持忙碌，在数据不规则或者运行时耗时波动的场景下减少空闲时间；实现更细粒度的负载均衡，而不完全依赖硬件调度器。

该特性的完整使用说明参见文档：https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#cluster-launch-control
# 二、降低延迟
本节介绍的各类特性拥有共同目标：削减某一类延迟，但不同特性所针对的延迟类型各有区别。总体而言，这些特性主要聚焦于内核发射层级及之上的延迟，内核内部的 GPU 显存访问延迟不在本节的讨论范畴之内。
## 2.1 绿色上下文（Green Contexts）
绿色上下文，也称作执行上下文（execution contexts），是一项 CUDA 特性。该特性允许程序创建一类 CUDA 上下文，限定其内提交的任务仅能在 GPU 的一部分流式多处理器（SM）上执行。

默认情况下，内核启动产生的线程块会被调度到 GPU 内任意满足该内核资源需求的 SM 上。会影响线程块可在哪些 SM 上执行的因素非常多，包括（但不限于）：共享内存占用、寄存器占用、线程块集群的使用，以及线程块内的总线程数。

借助执行上下文，可以将内核提交到专门创建的上下文当中，进一步限制可供该内核使用的 SM 数量。关键特性：当程序创建一个绿色上下文并分配了一组 SM 之后，GPU 上的其他上下文（包括 CUDA 运行时默认使用的主上下文）不会再把线程块调度到分配给该绿色上下文的这些 SM 上。由此，这些 SM 可以预留给高优先级、对延迟敏感的工作负载使用。
## 2.2 局部域（Locality Domains）
[局部域](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/locality-domains.html#locality-domains)是 GPU 显存与流式多处理器（SM）的一个子集。对于具备多个局部域的 GPU，将内存分配在某个指定局部域中，再通过绿色上下文执行内核，且该绿色上下文仅使用同一局部域内的 SM，这种做法可以带来性能收益。

局部域属于性能优化特性：即便不对内存、计算资源做局部绑定，CUDA 代码依旧可以正确执行。

[《CUDA 编程指南》编程模型章节](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-locality-domains)介绍了局部域的基础概念；下面的文档链接给出完整说明，包含如何查询 GPU 的局部域数量、如何把内存分配到指定局部域、如何创建仅包含该局部域内 SM 的绿色上下文：https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-locality-domains
## 2.3 流序内存分配（Stream-Ordered Memory Allocation）
[流序内存分配器](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator)允许程序将 GPU 内存的分配与释放操作编排到 [CUDA 流](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)中执行。 `cudaMalloc`与`cudaFree`会立即执行；与之不同，`cudaMallocAsync`和`cudaFreeAsync`是将内存分配或释放操作插入到 CUDA 流队列。
该页面包含这些 API 的全部详细说明：https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator
## 2.4 CUDA图
[CUDA图](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs)允许应用程序指定一系列 CUDA 操作（例如内核启动、内存拷贝）以及这些操作之间的依赖关系，从而让这些操作可以在 GPU 上高效执行。使用[CUDA流](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)也可以实现类似的行为；实际上，创建 CUDA 图的其中一种机制叫做[流捕获（Stream Capture）](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture)，它可以将流上执行的操作记录到一张 CUDA 图当中。CUDA 图也可以通过[图 API](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-graph-apis)进行创建。

图一旦创建完成，就可以完成实例化，并且被反复执行。这对于描述需要重复执行的工作负载十分有用。CUDA 图能够带来一定性能收益：一方面它可以降低调用 CUDA 操作带来的 CPU 提交开销；另一方面，当完整工作负载提前全部指定后，还可以启用一些仅在此条件下可用的优化手段。

CUDA 编程指南的[ CUDA Graph 章节](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs)对 CUDA 图的用法进行了完整说明并附带示例代码。
## 2.5 可编程依赖启动
[可编程依赖启动](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)是一项 CUDA 特性：允许存在依赖关系的内核（即需要使用前一个内核输出结果的内核），在其所依赖的主内核尚未执行完毕时就开始运行。该依赖内核可以执行初始化代码以及与主内核无关的计算任务，直到它需要读取主内核的数据时再发生阻塞。主内核可以发出信号，通知依赖内核所需的数据已经准备就绪，随后依赖内核便可恢复继续执行。

该机制能够让两个内核的执行过程产生部分重叠，有助于维持较高的 GPU 利用率，同时缩短关键数据通路的延迟。CUDA 编程指南的[可编程依赖启动章节](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)对该特性做了详细阐述。
## 2.6 延迟加载
[延迟加载](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#lazy-loading)是一项用于控制应用程序启动阶段 JIT 编译器工作方式的特性。 如果应用程序存在大量内核需要从 PTX 即时编译为 cubin，若在程序启动阶段就对全部内核执行 JIT 编译，则会出现启动时间过长的问题。 默认行为是：模块直到真正被需要时才会完成编译。 该行为可以通过 [CUDA 环境变量](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-environment-variables)进行修改，具体说明参见[延迟加载](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#lazy-loading)章节文档。
# 三、功能特性
本节所描述的各项特性拥有一个共同点：用于启用更多附加能力与功能。
## 3.1 扩展GPU内存（EGM）
[扩展 GPU 内存（EGM）](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#extended-gpu-memory)是仅在 NVLink‑C2C 互连系统上可用的一项特性，它允许 GPU 高效访问系统内的全部内存。扩展 GPU 内存相关内容可参见官方文档：https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html#extended-gpu-memory
## 3.2 动态并行
CUDA 应用程序最常见的方式是从 CPU 侧代码发起内核启动。也可以在 GPU 上正在运行的内核内部发起新的内核调用，该特性称为[CUDA 动态并行](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism)。[CUDA 动态并行文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism)详细介绍了如何在 GPU 设备代码中发起新的 GPU 内核启动。
# 四、CUDA互操作性
## 4.1 CUDA与其他API的互操作性
除 CUDA 之外，还有其他可在 GPU 上执行代码的机制。GPU 最初设计用于加速计算机图形任务，图形领域拥有 Direct3D、Vulkan 等一套专属 API。应用程序可能希望使用图形 API 完成 3D 渲染，同时借助 CUDA 执行计算任务。CUDA 提供了相应机制，可以在 CUDA 上下文与各类 3D 图形 API 的 GPU 上下文之间交换 GPU 上存储的数据。例如，应用可以用 CUDA 完成仿真计算，再使用 3D API 对计算结果做可视化；实现方式是让部分缓冲区同时对 CUDA 和图形 API 具备读、写访问权限。

这套用于和图形 API 共享缓冲区的机制，同样也可以用于通信组件之间的缓冲区共享，支持多节点环境下高速的 GPU‑to‑GPU 直接通信。

[CUDA 互操作文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#cuda-interoperability)阐述了 CUDA 如何与其他 GPU API 协同工作，以及怎样在 CUDA 与其他 API 之间共享数据，并给出了多种不同 API 的具体示例。
## 4.2 进程间通信
在大规模计算场景中，经常会联合使用多块 GPU，以此获得更大的内存容量与更多计算资源来共同处理同一计算任务。在集群计算术语里，单台服务器也称作一个**节点（node）**，可以在单个主机进程内使用多块 GPU，相关内容可参考[多 GPU 系统介绍文档](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-gpu-introduction)。

同时也经常会使用分布在单台或多台服务器上的多个独立主机进程。当多个进程协同工作时，进程之间的数据交互就称为**进程间通信（IPC）**。CUDA 进程间通信（CUDA IPC）提供了在不同进程之间共享 GPU 缓冲区的机制。[CUDA IPC 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication)介绍并演示了如何借助 CUDA IPC 在不同主机进程之间完成协同工作与数据通信。
# 五、细粒度控制
## 5.1 虚拟内存管理
正如[统一虚拟地址空间章节](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-virtual-address-space)所述，系统内的全部 GPU 以及 CPU 内存共享同一个统一虚拟地址空间。绝大多数应用程序可以直接使用 CUDA 提供的默认内存管理机制，无需修改其行为。但对于有特殊需求的开发者，[**CUDA 驱动 API**](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-virtual-address-space)提供了对该虚拟地址空间布局的高级精细化控制能力。该特性主要适用于需要管控缓冲区在单节点内部以及跨多节点场景下的 GPU 之间共享行为。

[虚拟内存管理章节](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management)详细介绍了 CUDA 驱动 API 所提供的各项控制能力、工作原理，以及开发者在哪些场景下可以从中获益。
## 5.2 计算架构传输
[计算架构传输文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html#compute-fabric-transport)介绍了一套**以资源为中心**的模型，用于在 NVLink 互联架构上实现由 GPU 发起的数据搬运。该模型中，远端目标通过**逻辑端点 ID（logical endpoint id）+ 偏移量**进行寻址，而非依靠已映射的虚拟地址。该特性面向通信原码库的开发人员；对于普通应用，如果业务通信模式能够匹配，应优先选用[ NCCL 设备端 API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) 或者 [NVSHMEM 库。](https://docs.nvidia.com/nvshmem/api/latest/index.html)
## 5.3 驱动入口点访问
[驱动入口点访问](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#driver-entry-point-access)是 CUDA 11.3 起提供的能力：可以获取 CUDA 驱动 API 与 CUDA 运行时 API 的函数指针。开发者还可以获取驱动函数特定变体版本的函数指针，并且能够调用 CUDA Toolkit 版本中并未提供、但由更高版本驱动所实现的驱动函数。[驱动入口点访问章节](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html#driver-entry-point-access)详细讲解该功能。
## 5.4 错误日志管理
[错误日志管理](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management)提供用于处理、记录 CUDA API 报错的工具。只需要设置环境变量`CUDA_LOG_FILE`，就可以将 CUDA 错误直接输出到标准错误 (stderr)、标准输出 (stdout) 或者指定文件。错误日志管理还支持应用注册回调函数，当 CUDA 遇到错误时会触发该回调。[错误日志管理章节](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html#error-log-management)给出该功能的更多细节。