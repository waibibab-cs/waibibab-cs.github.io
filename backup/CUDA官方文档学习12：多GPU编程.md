多 GPU 编程可充分利用多 GPU 系统更高的总算力、显存容量与显存带宽，让应用能够处理单 GPU 无法承载的问题规模，并实现单 GPU 达不到的性能水平。

CUDA 依托主机 API、驱动底层组件以及配套 GPU 硬件技术，支持多 GPU 编程，包含以下能力：
- 主机线程的 CUDA 上下文管理
>解释：1.宿主线程就是CPU线程，一个CPU线程同一时刻只能绑定一个 “当前CUDA上下文”，上下文就是这块GPU给你的独立运行环境；2.可以在CPU线程上压入/弹出多个上下文（栈结构），切换当前绑定的GPU环境，但切换有开销，不是免费操作；3.多 GPU 编程时，CPU 线程可以轮流把不同 GPU 的上下文设为 current，从而向多张 GPU 提交任务，也可以开多个 CPU 线程，每个线程绑定一张 GPU 的上下文
- 系统内所有处理器的统一内存寻址
>解释：1.统一内存寻址：CPU 和多张 GPU 使用同一套虚拟地址空间，同一个指针，CPU、GPU 都能识别；2.不用手动调用`cudaMemcpy`显式搬运数据，驱动会在背后自动把内存页迁移到当前正在访问它的处理器（CPU/GPU）；3.不是所有内存都物理共享，只是地址视图统一，底层还是 PCIe/NVLink 做数据搬移，访问不在本地的内存会有延迟；4.多 GPU 场景下，任意 GPU 可以直接用这个统一指针访问数据，简化跨卡编程，但要注意远端访问带来的性能开销。
- GPU 间点对点批量内存传输
>解释：1.GPU 点对点（P2P）批量传输：两块 GPU 之间直接搬运显存数据，不走 CPU 内存中转；2.批量意味着一次性传输一大块显存，适合大规模数据迁移，比逐字节访问效率高很多；3.硬件支持（NVLink/PCIe）前提下，源 GPU 显存 → 目标 GPU 显存，CPU 只需要下发启动命令，不参与数据拷贝；4.优点：省掉 CPU 内存缓冲区、减少 CPU 开销；局限：受链路带宽限制，不是所有 GPU / 主板都能开启 P2P。
- 细粒度的 GPU 点对点加载 / 存储内存访问
>解释：1.细粒度 P2P load/store：GPU 核函数里可以直接读写另一张 GPU 显存上的单个变量 / 少量数据，不是一次性整块批量拷贝；2.批量 P2P 传输是提前发起一次大块 DMA 拷贝，搬完再使用；细粒度是 kernel 运行时按需远程读 / 写零散数据；3.不需要 CPU 参与发起拷贝指令，GPU SM 直接发请求访问远端 GPU 显存地址；单次访问数据量很小；4.优势：编程灵活，不用预先规划数据分片；缺点：远程访问延迟很高，大量零散访问性能会暴跌。
- 高层抽象与配套系统软件，例如 CUDA 进程间通信、使用 NCCL 实现的并行归约，以及依托 NVLink 和 / 或 GPU-Direct RDMA、通过 NVSHMEM 与 MPI 等 API 完成通信
>解释：1.高层抽象：高层抽象：把底层 P2P、NVLink 这些硬件通信能力封装起来，不用手动写底层显存拷贝、上下文切换，开发者调用现成库 API 即可；2.CUDA IPC：同一台机器上不同进程的 GPU 显存互相访问，让多进程共享 GPU 显存句柄；3.NCCL：专门做多 GPU 集合通信（归约、广播、allreduce），自动选 NVLink/PCIe 最优路径，并行归约就是把多卡数据汇总求和 / 求最大值这类操作；4.GPU-Direct RDMA + NVSHMEM/MPI：GPU 可以绕过主机内存，直接和远端 GPU 甚至远端机器网卡通信；NVSHMEM 提供类似共享内存风格的分布式编程接口，MPI 是经典分布式并行 API。

从最基础层面来说，多 GPU 编程要求应用并发管理多个活跃 CUDA 上下文，将数据分发至各个 GPU，在 GPU 上启动核函数完成计算，并通信、汇总结果，供应用后续处理。具体实现方式，取决于如何将应用算法、可用并行度以及现有代码结构，最高效地映射到合适的多 GPU 编程方案。几种最常用的多 GPU 编程方案包括：
* 单个主机线程驱动多个 GPU
>解释：1.单个 CPU 宿主线程，依靠CUDA上下文栈，轮流切换当前上下文，依次给不同 GPU 提交任务；2.同一时刻线程只能绑定一个GPU上下文，所以任务是分时下发，GPU之间可以并行跑计算；3.线程只负责下发kernel、拷贝指令；各个GPU硬件独立执行，互不阻塞；4.短板：频繁切换上下文会带来开销；下发任务的并发上限受单 CPU 线程能力限制。
* 多个主机线程，每个线程管理专属 GPU
>解释：1.多个 CPU 宿主线程，每个线程固定绑定一个 GPU 上下文，专门负责给这一张 GPU 提交任务；2.线程之间互不抢占上下文，不需要频繁做上下文切换；各线程可以并行向各自 GPU 下发指令；3.每个线程只关心自己那张卡的显存、kernel；GPU 之间硬件独立并行执行；4.代价：线程资源开销更大，多线程之间同步、数据协调逻辑会变复杂。
* 多个单线程主机进程，每个进程管理专属 GPU
>解释：1.多个独立 CPU 进程，每个进程独占管理一张 GPU，进程之间互相隔离，各自维护自己的 CUDA 上下文、显存与模块；2.进程天然内存隔离，不能直接用指针互访显存；跨进程 GPU 通信需要 CUDA IPC、NCCL 等机制；3.优势：一个进程崩溃不会连累其他进程 / 其他 GPU；资源隔离好，适合多用户任务；4.缺点：进程本身开销大；进程间数据交换成本高于线程；同卡不能被多个进程无限制共用（看 MIG / 虚拟化配置）。
* 多线程的多个主机进程，每个线程管理专属 GPU
>解释：1.存在多个独立 CPU 进程，每个进程内部又开多条宿主线程，每条线程固定负责一张 GPU，线程绑定对应 GPU 上下文；2.进程之间内存地址空间隔离；同一进程内的线程之间地址空间共享；3.同进程内线程之间通信成本低；跨进程 GPU 交互依然需要 CUDA IPC/NCCL；4.优点：可充分利用多核 CPU 提交任务，故障隔离粒度是进程级；缺点：模型复杂，需要同时处理进程间、线程间两套同步逻辑，资源管理繁琐。
* 多节点 NVLink 互联集群：集群跨多个操作系统实例，由其上运行的线程与进程驱动 GPU
>解释：1.多节点 NVLink 集群：多台物理服务器组成集群，NVLink 只在单机内部直连 GPU，跨节点靠 GPU-Direct RDMA（网卡）通信，每个节点是独立 OS 实例，内存空间完全隔离；2.每个操作系统内可以跑多个进程 / 线程，由这些宿主线程 / 进程管理本地 GPU 上下文，向 GPU 下发计算、通信任务；3.节点内 GPU 之间走 NVLink 高速 P2P；跨节点 GPU 不能直接 NVLink 访问，依赖 RDMA 把 GPU 显存数据直接传到远端 GPU，绕开主机内存；4.编程上依赖 NCCL、MPI、NVSHMEM 这类高层库，屏蔽单机 NVLink 和跨机 RDMA 差异；故障隔离粒度为节点 / OS 级别。

GPU 之间可通过设备显存间的数据传输与点对点访问完成通信，上述所有多设备任务分发方式均支持该通信模式。要实现高性能、低延迟的 GPU 通信，需要查询并开启 GPU 点对点显存访问能力，并利用 NVLink 在设备间实现高带宽传输以及更细粒度的加载 / 存储操作。

CUDA 统一虚拟寻址支持同一主机进程内多 GPU 之间通信，只需极少额外步骤，即可查询并启用基于 NVLink 等的高性能点对点显存访问与数据传输。

由不同主机进程管理的多 GPU 之间的通信，则依靠进程间通信（IPC）与虚拟内存管理（VMM）API 实现。高层 IPC 概念简介以及节点内 CUDA IPC 相关 API，可参考文档章节：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication)。高级虚拟内存管理（VMM）API 同时支持节点内与跨节点 IPC，可在 Linux 与 Windows 系统上使用；并且如文档所述，支持以单次显存分配为粒度，控制内存缓冲区的 IPC 共享：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management)。

计算结构传输（Compute fabric transport）针对 NVLink 结构上的 GPU 采用一套不同的机制：不同于上述方案通过映射虚拟地址访问对等端显存，该机制使用逻辑端点 ID对一组资源命名，通过偏移定位组内特定资源，并使用显式、异步的结构操作访问资源。详见下文：[https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-gpu-compute-fabric-transport](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-gpu-compute-fabric-transport)。

CUDA 本身提供在一组 GPU（可包含主机）内实现集合通信操作所需的底层 API，但 CUDA 自身并不提供高层多 GPU 集合通信接口。高层多 GPU 集合通信由更高抽象层级的 CUDA 通信库实现，例如 NCCL 与 NVSHMEM。
# 一、多设备上下文与执行管理
应用程序使用多 GPU 的首要步骤：枚举可用 GPU 设备，依据硬件属性、CPU 亲和性以及设备间的互联拓扑，在可用设备中按需筛选；并为应用将要使用的每一块设备创建 CUDA 上下文。
## 1.1 设备枚举
下面的代码示例演示如何查询支持 CUDA 的设备总数、遍历枚举所有设备，并查询设备各项属性。
```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```
## 1.2 设备选择
宿主线程可随时调用`cudaSetDevice()`，设定其当前要操作的设备。

设备显存分配、核函数启动，均在**当前设备**上执行；流（stream）与事件（event）会和当前选定的设备绑定创建。在宿主线程调用`cudaSetDevice()`之前，当前设备默认为 0 号设备。

下面的代码示例展示：设置当前设备，会如何影响后续的显存分配以及核函数执行操作。
```cpp
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // Set device 0 as current
float* p0;
cudaMalloc(&p0, size);       // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

cudaSetDevice(1);            // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);       // Allocate memory on device 1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
```
## 1.3 多设备下的流、事件与内存拷贝行为
如果向一个不属于当前设备的流提交核函数启动操作，该调用会失败，如下代码示例所示：
```cpp
cudaSetDevice(0);                // 将0号设备设为当前设备
cudaStream_t s0;
cudaStreamCreate(&s0);            // 在0号设备上创建流s0
MyKernel1<<<100, 64, 0, s0>>>(); // 在0号设备的流s0中启动核函数

cudaSetDevice(1);                // 将1号设备设为当前设备
cudaStream_t s1;
cudaStreamCreate(&s1);            // 在1号设备上创建流s1
MyKernel1<<<100, 64, 0, s1>>>(); // 在1号设备的流s1中启动核函数

// 本次核启动会失败，因为流s0不属于1号设备：
MyKernel1<<<100, 64, 0, s0>>>(); // 试图在1号设备上，使用流s0启动核函数
```
内存拷贝操作：即使提交到不属于当前设备的流，**仍然可以成功执行**。
`cudaEventRecord()`：如果传入的事件和流隶属于不同设备，调用会失败。
`cudaEventElapsedTime()`：如果两个输入事件隶属于不同设备，调用会失败。
`cudaEventSynchronize()` 和 `cudaEventQuery()`：即便输入事件所属设备与当前设备不同，调用**仍可成功执行**。
`cudaStreamWaitEvent()`：即便输入流和输入事件隶属于不同设备，调用**仍可成功执行**。因此，可以使用`cudaStreamWaitEvent()`完成多个设备之间的同步。
每个设备都拥有独立的默认流。因此，向某设备默认流提交的指令，和提交到其他任意设备默认流的指令之间，执行顺序无序，可并发运行。
# 二、多设备点对点传输与内存访问
## 2.1 点对点内存传输
CUDA 支持设备间内存传输。当点对点内存访问可用时，CUDA 会利用专用拷贝引擎与 NVLink 硬件，最大化传输性能。
`cudaMemcpy` 可搭配拷贝类型 `cudaMemcpyDeviceToDevice` 或者 `cudaMemcpyDefault` 使用。
若不满足上述条件，则必须使用 `cudaMemcpyPeer()`、`cudaMemcpyPeerAsync()`、`cudaMemcpy3DPeer()` 或 `cudaMemcpy3DPeerAsync()` 执行拷贝操作，代码示例见下文。
>此时`cudaMemcpyPeer`仍然是宿主机发起的拷贝，底层会自动申请主机侧临时缓存，性能远低于 P2P 直传，只是 API 帮你省去手动写「GPU→主机→GPU」两次拷贝的代码。
```cpp
cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0

cudaSetDevice(1);                   // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);              // Allocate memory on device 1

cudaSetDevice(0);                   // Set device 0 as current
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0

cudaSetDevice(1);                   // Set device 1 as current
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1
```
在隐式 NULL 默认流上执行两块不同设备显存之间的拷贝操作：
1. 必须等待下发到任意一个设备的所有先前指令全部完成后，拷贝才会启动；
2. 并且，必须等本次拷贝执行完毕，后续下发到这两个设备中任意一个的所有指令才能够开始执行。
和流的常规行为保持一致：两块设备显存之间的异步拷贝，可以与其他流中的拷贝任务或核函数并行重叠执行。
如果两台设备之间已开启点对点（P2P）访问（参考文档：……），那么二者之间的点对点内存拷贝不再需要经过主机内存中转缓存，因此速度更快。
## 2.2 点对点内存访问
根据系统特性（尤其是 PCIe 和 / 或 NVLink 拓扑），设备之间可以互相寻址对方的显存（即在一个设备上执行的核函数，可以解引用指向另一块设备显存的指针）。 若`cudaDeviceCanAccessPeer()`针对指定的两块设备返回 true，则代表这两个设备之间支持点对点内存访问。

必须调用`cudaDeviceEnablePeerAccess()`，才能在两块设备之间启用点对点内存访问，代码示例见下文。 **在未开启 NVSwitch 的系统上，每块设备全局最多支持 8 条对等连接。**

两块设备共用统一虚拟地址空间（参见统一虚拟地址空间章节），因此如下面代码示例所示，可以用同一个指针访问两块设备上的内存。
```cpp
cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0

cudaSetDevice(1);                   // Set device 1 as current
cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                    // with device 0

// Launch kernel on device 1
// This kernel launch can access memory on device 0 at address p0
MyKernel<<<1000, 128>>>(p0);
```
**注意**
调用`cudaDeviceEnablePeerAccess()`启用对等显存访问，作用是全局生效：对等设备上此前以及后续所有GPU 显存分配都会受到该设置影响。
通过`cudaDeviceEnablePeerAccess()`开启对某设备的对等访问后，该对等设备上的显存分配操作会引入运行时开销。这是因为分配出来的内存需要立即可被当前设备、以及其他所有拥有访问权限的对等设备访问；该开销是倍增式的，会随着对等设备数量增加而放大。
如果不想让全部显存分配都开启对等访问，可采用更具备可扩展性的替代方案：使用 CUDA 虚拟内存管理 API，仅在需要的时候、在内存分配阶段显式创建支持对等访问的内存区域。 在内存分配时显式指定对等访问权限，则不需要对等访问的内存分配不会产生额外运行时开销；同时支持对等访问的数据结构拥有明确的作用域，便于软件调试，提升程序可靠性（参考：虚拟内存管理章节）。
## 2.3 点对点内存一致性
对于分布在多个设备上的网格内并发执行的线程，必须借助同步操作，保证内存访问的访存顺序与正确性。跨设备进行同步的线程，其同步作用域为 `thread_scope_system`，参考链接[https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#thread-scopes)。同理，这类内存操作归属 `thread_scope_system` 作用域，参考链接：[https://docs.nvidia.com/cuda/cuda-programming-guide/#memory-synchronization-domains](https://docs.nvidia.com/cuda/cuda-programming-guide/#memory-synchronization-domains)。

CUDA 原子函数（参考链接：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions)）可以对对等设备显存中的对象执行读 - 改 - 写操作，但前提是**仅有单个 GPU 访问该对象**。对等访问场景下原子操作的要求与限制，可参阅 CUDA 内存模型中原子性相关讨论，参考链接：[https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity) 
## 2.4 多设备的托管内存（Managed Memory）
托管内存可在具备点对点（P2P）支持的多 GPU 系统上使用。多设备并发访问托管内存的详细要求，以及用于 GPU 独占访问托管内存的相关 API，请参见：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-legacy-multi-gpu](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-legacy-multi-gpu)
## 2.5 主机 IOMMU 硬件、PCI 访问控制服务与虚拟机
>**背景补充1：PCIe DMA分类**
   PCIe 设备发起的 DMA，本质就是设备在总线上发起 Memory 读写 TLP，分为两类：
   1.DMA 目标 = 系统 DRAM 地址（内存 DMA） GPU 读写主机内存，`cudaMemcpy Host <-> Device` 这种。 目标地址属于系统内存地址域。IOMMU专门处理这种DMA访问
   2.DMA 目标 = 其他 PCI 设备的 BAR 空间（PCI P2P DMA）GPU A 发起 DMA 读写 GPU B 的显存窗口 BAR，也就是 CUDA P2P。 目标地址属于PCI 配置的 BAR 地址域，不是 DRAM。IOMMU无法处理这种地址翻译
>**背景补充2：IOMMU（输入输出内存管理单元）**
>CPU侧硬件，作用是DMA地址翻译+设备隔离。
>外设使用的 IO 虚拟地址 → 物理内存地址转换；阻止一个 PCI 设备直接随意读写系统物理内存，提升安全性，是虚拟化隔离的核心硬件。
>裸机（物理机）：IOMMU 开启后，所有 PCIe 设备的 DMA 都要经过 IOMMU 地址转换。
>虚拟机 VFIO 直通：IOMMU**必须打开**，才能把一个 PCI 设备（GPU）放到独立 IOMMU 分组，安全透传给 VM，防止虚拟机直接篡改宿主机内存。
>**背景补充3：PCIe Root Complex（根复合体）**
>CPU 里 PCIe 的根端口，所有 PCIe 链路最终连到 Root Complex。GPU 之间 PCIe P2P 有两种路径：
>1.直连P2P（理想）：GPU A ↔ GPU B，流量不经过 CPU / 根复合体，直接 PCIe 对等传输，就是 CUDA P2P。
>2.绕 Root Complex：GPU A → Root Complex → GPU B，流量必须经过 CPU 侧，带宽大幅下降。
>**背景补充4：PCI ACS（Access Control Services，PCI访问控制服务）**
>PCIe 的安全扩展。ACS 一旦开启，强制所有 PCI 点对点流量，全部转发经过 Root Complex。
>1.目的：安全隔离，防止同 PCIe 域内恶意 PCI 设备互相访问。
>2.代价：GPU 之间再也走不了原生 PCIe 直连 P2P，全部流量绕 CPU，二分带宽暴跌。很多服务器主板 BIOS 默认开启 ACS，这是多 GPU P2P 踩坑高频点。
>**背景补充5：虚拟机直通（VFIO PCI Pass-through）**
>VFIO（Virtual Function I/O）是 Linux 内核提供的安全设备直通框架。 作用：把一台物理机上的整块物理 PCIe 设备（比如整张 GPU 网卡、SSD 控制器）完整剥离出来，直接分配给一台虚拟机。
>典型场景：1.云服务器；2.隔离需求等
>**背景补充6：几种地址**
>![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261012120814.png)

仅在 Linux 平台下：CUDA 与显示驱动**不支持开启 IOMMU 的裸机环境下的 PCIe 点对点内存传输**。但是，CUDA 与显示驱动支持虚拟机直通场景下的 IOMMU。在裸机 Linux 系统运行时，**必须关闭 IOMMU，避免静默的设备显存损坏**。与之相反，虚拟机环境中应当开启 IOMMU，并使用 VFIO 驱动完成 PCIe 直通。 
>**Linux裸机+IOMMU开启**
>1.场景：GPU 是宿主机系统的原生 PCIe 设备；当 GPU A 发起 PCIe P2P，想要直接访问 GPU B 的显存：
>2.现象：理想 P2P：GPUA → PCIe 总线 → GPU B，不经过 CPU、不经过 IOMMU；但一旦开启裸机 IOMMU：PCIe 硬件规则强制所有 PCIe DMA 流量都经过 IOMMU 做地址转换。这里的 P2P DMA，是GPU 直接访问另一块 GPU 的 BAR 空间（显存窗口），不是访问系统内存。 老版 NVIDIA Linux 驱动没有实现对 IOMMU 翻译后的 GPU-to-GPU 对等 DMA 的支持。 IOMMU 会篡改 PCIe 事务地址，造成 DMA 读写出错。
>3.注意：机 IOMMU 会拦截GPU→GPU 的 PCIe 对等 DMA，而 NVIDIA 驱动无法处理这种带 IOMMU 地址翻译的跨 GPU 访问。 但这条限制只针对 PCIe P2P，NVLink 不受影响，NVLink 不走 PCIe 总线。
>**VFIO虚拟机直通+IOMMU开启**
>在 VFIO 直通场景：GPU 虽然物理上插在宿主机主板，但在 IOMMU 层面，GPU 被隔离给虚拟机，宿主机的 IOMMU 做的是「宿主机物理地址 ↔ 虚拟机 IO 虚拟地址」的转换。
>此时：1.宿主机 IOMMU 只负责把虚拟机发出的 IO 地址翻译成宿主机的物理地址，做 VM 内存隔离；2.GPU 之间的 P2P 事务（GPUA ↔ GPU B）发生在虚拟机内部，属于 VM 地址空间内部的 PCIe 对等访问；3.虚拟机内部的 NVIDIA 驱动，看到的 GPU 是原生 PCI 设备，虚拟机内部的 PCIe 对等 DMA 不再被宿主机 IOMMU 介入翻译。

Windows 平台不存在上述 IOMMU 相关限制。

另参见：[https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html](https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html)

此外，支持 IOMMU 的系统可以启用 PCI 访问控制服务（ACS）。PCI ACS 特性会将所有 PCI 点对点流量重定向，全部经过 CPU 根复合体转发；这会降低全局二分带宽，造成显著的性能损失。
