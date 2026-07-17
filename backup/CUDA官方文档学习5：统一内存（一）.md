原文链接：
[2.6. Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
# 前言
异构系统拥有多处可存储数据的物理内存：主机 CPU 外接有 DRAM 内存，系统内每一块 GPU 也都配备独立的专属 DRAM 显存。数据存放于访问端处理器的本地内存中时，程序性能最优。CUDA 提供可手动管控数据内存布局的应用程序接口（例如：cudaMalloc、cudaMallocHost等），但这类接口代码冗余度高，还会增加软件开发难度。为此 CUDA 推出多项功能特性，简化不同物理内存间的数据分配、布局调度与数据迁移工作。

本章旨在讲解上述相关功能特性，阐明其在功能实现与性能优化层面对应用开发者的实际作用。统一内存存在多种实现形式，具体形态由操作系统、驱动版本以及所用 GPU 硬件共同决定。本章将说明如何判定设备所适配的统一内存运行模式，以及各类模式下统一内存的运行机制。在”[CUDA官方文档学习6：统一内存（二）](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-6%EF%BC%9A-tong-yi-nei-cun-%EF%BC%88-er-%EF%BC%89.html)“中会对统一内存展开更为详尽的讲解。

本章将定义并阐释以下核心概念：
**统一虚拟地址空间（Unified Virtual Address Space，UVA）**：CPU内存与各GPU内存，在同一个虚拟地址空间内划分出相互独立的地址区间

**统一内存（Unified Memory，UM）** ：CUDA专属功能，支持托管内存在CPU与GPU之间完成自动数据迁移
>**受限统一内存（Limited Unified Memory）**：存在功能约束的统一内存运行模式
  **完整统一内存（Full Unified Memory）**：全面兼容统一内存各项功能
  **具有硬件一致性的完整统一内存（Full Unified Memory）**：依托硬件能力实现全功能统一内存
  **统一内存提示（Unified Memory Hints）**：CUDA 提供的一组 API，用于指导 Unified Memory 的行为

**页锁定主机内存（Page-locked Host Memory，Pinned Memory）**：不可被操作系统换页的系统内存，是部分 CUDA 运算操作的必备条件
>**映射内存(Mapped Memory)** ：可在内核函数中直接访问主机内存的机制，与统一内存分属不同技术方案

同时，本章介绍研究统一内存与系统内存时常用专业术语：
**异构托管内存（Heterogeneous Managed Memory，HMM）** ：Linux 内核自带功能，通过软件方式实现 Full Unified Memory 所需的一致性管理
**地址转换服务（Address Translation Services，ATS）**：硬件级特性，当 GPU 通过 NVLink Chip-to-Chip（C2C）与 CPU 相连时，ATS 可以提供 Hardware Coherency（硬件一致性），从而支持 Full Unified Memory。
# 1.统一虚拟地址空间
在同一个操作系统进程（OS process）中，所有主机内存以及系统中所有 GPU 的全局内存共同使用一个统一的虚拟地址空间（Unified Virtual Address Space，UVA）。无论是主机内存还是 GPU 内存，所有的内存分配都会位于这一统一虚拟地址空间中。这一点对于以下两类内存分配方式都成立：
* 使用CUDA API分配的内存，例如：`cudaMalloc()`、`cudaMallocHost()`
* 使用操作系统或C/C++提供的内存分配接口，例如：`new`、`malloc()`、`mmap()`
在该统一虚拟地址空间中：CPU 内存以及每一块 GPU 的显存都分别占据一个互不重叠的、唯一的地址区间（unique range）。

这意味着：
1. **可以仅根据指针的数值判断该内存属于哪里**。通过调用`cudaPointerGetAttributes()`，即可根据一个指针的值判断：该指针指向的是 CPU 主机内存，还是某一块 GPU 的显存（以及具体是哪一块 GPU）。也就是说，指针本身已经携带了所属内存区域的信息，不需要程序员额外记录该指针来自哪里。
2. **cudaMemcpy()可以自动判断拷贝方向**。调用`cudaMemcpy()`时，参数`cudaMemcpyKind`可以设置为`cudaMemcpyDefault`，此时，CUDA Runtime 会根据**源指针（source pointer）** 和**目标指针（destination pointer）** 所在的地址区间，自动判断本次数据拷贝属于哪一种类型，而无需程序员手动指定。
# 2.统一内存
统一内存是 CUDA 提供的一项内存管理特性，它允许一种称为 **Managed Memory（托管内存）** 的内存分配，被运行在CPU或GPU上的代码共同访问。在[CUDA官方文档学习2：CUDA C++入门](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-2%EF%BC%9ACUDA%20C%2B%2B-ru-men.html#2.1-%E7%BB%9F%E4%B8%80%E5%86%85%E5%AD%98)中的2.1小节已经初步介绍过，统一内存适用于所有CUDA支持的系统。

在某些系统上，托管内存必须显式分配。 在CUDA中，可以通过以下几种方式显式分配 Managed Memory：
* 使用`cudaMallocManaged()`
* 使用`cudaMallocFromPoolAsync()`，其中所使用的Memory Pool必须设置：`allocType=cudaMemAllocationTypeManaged`
* 使用带有`__managed__`修饰符的全局变量（可参考：[5.4. C/C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-space-specifiers)）
而在支持 HMM（异构托管内存）或 ATS（地址转换服务）的系统上，所有系统内存（System Memory）都会被隐式视为 Managed Memory，而不论它们是通过何种方式分配的。因此无需进行任何特殊的Managed Memory分配。
## 2.1 统一内存范式
Unified Memory（统一内存）的功能和行为会随着以下因素的不同而有所变化：
- 操作系统（Operating System）
- Linux 的内核版本（Kernel Version）
- GPU 硬件（GPU Hardware）
- CPU 与 GPU 之间的互连方式（GPU-CPU Interconnect）
因此，不同平台所支持的 Unified Memory 能力并不完全相同。

可以通过调用`cudaDeviceGetAttribute()`查询以下几个设备属性，从而判断当前系统支持的是哪一种 Unified Memory 形式：
1. `cudaDevAttrConcurrentManagedAccess`：用于判断是否支持**Full Unified Memory**，返回值是1，表明支持Full Unified Memory，返回值为0，表明仅支持Limited Unified Memory
2. `cudaDevAttrPageableMemoryAccess`：用于判断普通系统内存（Pageable System Memory）是否也属于 Fully-supported Unified Memory。如果返回值为1，表示所有系统内存都被视为 Fully-supported Unified Memory。无论它们是通过`new`、`malloc()`、`mmap()`还是其他普通方式分配，GPU都可以直接按照Unified Memory的方式访问。如果返回值为0，表示只有显式分配的 Managed Memory 才属于 Fully-supported Unified Memory。也就是说必须使用：`cudaMallocManaged()`等方式分配，GPU 才能按照 Unified Memory 的方式访问，普通`malloc()`得到的内存，并不具备完整 Unified Memory 语义。
3. `cudaDevAttrPageableMemoryAccessUsesHostPageTables`：用于判断CPU 与 GPU 如何保持一致性（Coherence）。如果返回值为1，表明使用Hardware Coherency（硬件一致性），如果返回值为0，表明使用软件一致性。

图 21（Figure 21）给出了一个可视化流程，用于判断当前系统属于哪一种 Unified Memory 模式
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260716160459.png)
总之，Unified Memory 共有四种工作模式（paradigms）：
- 显式分配的托管内存（Managed Memory）提供完整支持（Full support for explicit managed memory allocations）
- 对所有内存分配提供完整支持，并采用软件一致性（Full support for all allocations with software coherence）（见2.2.2）
- 对所有内存分配提供完整支持，并采用硬件一致性（Full support for all allocations with hardware coherence）（见2.2.1）
- 有限统一内存支持（Limited unified memory support）（见2.3）
>图21解释：目前所有 GPU 都使用统一虚拟地址空间（Unified Virtual Address Space），并支持 Unified Memory。当 `cudaDevAttrConcurrentManagedAccess` 的值为 1 时，表示系统支持完整 Unified Memory（Full Unified Memory Support）；否则，仅支持有限 Unified Memory（Limited Unified Memory Support）。当支持完整 Unified Memory 时，如果 `cudaDevAttrPageableMemoryAccess`的值也为1，则表示所有系统内存（System Memory）都是 Unified Memory。否则，只有通过 CUDA API（例如 `cudaMallocManaged`）分配的内存才属于 Unified Memory。当所有系统内存都属于 Unified Memory 时，`cudaDevAttrPageableMemoryAccessUsesHostPageTables`用于指示一致性的实现方式：当其值为 1时，一致性由硬件（Hardware）提供；当其值为0时，一致性由软件（Software）提供。

下表展示了与图 21相同的信息，并提供了指向本章相关章节的链接，以及本指南后续章节中更完整的文档说明。

|         统一内存工作模式          |                                                                     设备属性                                                                      |                                                                                                                                                                                                       完整文档/参考资料                                                                                                                                                                                                        |
| :-----------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   有限 Unified Memory 支持    |                                                   `cudaDevAttrConcurrentManagedAccess`的值为 0                                                   | [1.Unified Memory on Windows, WSL, and Tegra](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-legacy-devices)[2.CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)[3.Unified memory on Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#effective-usage-of-unified-memory-on-tegra) |
| 显式分配 Managed Memory 的完整支持 |                              `cudaDevAttrPageableMemoryAccess` 的值为0，且 `cudaDevAttrConcurrentManagedAccess` 的值为 1                              |                                                                                                                  [Unified Memory on Devices with only CUDA Managed Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-no-pageable-systems)                                                                                                                   |
|   对所有内存分配提供完整支持（软件一致性）    | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 的值为0，且 `cudaDevAttrPageableMemoryAccess` 的值为1，且 `cudaDevAttrConcurrentManagedAccess`的值为 1 |                                                                                                                    [Unified Memory on Devices with Full CUDA Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-pageable-systems)                                                                                                                    |
|   对所有内存分配提供完整支持（硬件一致性）    | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 的值为1，且 `cudaDevAttrPageableMemoryAccess` 的值为1，且 `cudaDevAttrConcurrentManagedAccess` 的值为1 |                                                                                                                    [Unified Memory on Devices with Full CUDA Unified Memory Support](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-pageable-systems)                                                                                                                    |
## 2.2 完整统一内存支持
大多数 Linux 系统都支持完整 Unified Memory（Full Unified Memory）。如果设备属性 `cudaDevAttrPageableMemoryAccess`的值为1，则所有系统内存（System Memory），无论是通过 CUDA API 还是系统 API 分配——都以 Unified Memory 的方式运行，并支持完整的功能。也包括使用 `mmap` 创建的文件映射内存（file-backed memory allocations）。

如果 `cudaDevAttrPageableMemoryAccess`的值为0，则只有通过CUDA分配为Managed Memory 的内存才会以Unified Memory的方式工作。通过系统API分配的内存不是Managed Memory，并且不一定能够被GPU Kernel访问。

通常情况下，对于支持完整功能的Unified Memory分配：
- Managed Memory通常会分配在首次访问它的处理器的内存空间中。
- 当Managed Memory 被当前驻留处理器之外的另一个处理器使用时，通常会发生迁移（migrated）。
- Managed Memory的迁移或访问，以**内存页**为粒度（软件一致性，software coherence），或以**缓存行**为粒度（硬件一致性，hardware coherence）。
- 允许**超额分配（Oversubscription）**：应用程序可以分配比 GPU 实际物理内存容量更多的 Managed Memory。

内存分配和迁移行为可能与上述情况有所不同。程序员可以通过提示（hints）和预取（prefetches）来影响这些行为。（见2.4）
关于完整 Unified Memory 支持的详细内容，可参见[CUDA官方文档学习6：统一内存（二）](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-6%EF%BC%9A-tong-yi-nei-cun-%EF%BC%88-er-%EF%BC%89.html)中的**《Unified Memory on Devices with Full CUDA Unified Memory Support》**小节。
### 2.2.1 具有硬件一致性的完整统一内存
在Grace Hopper和Grace Blackwell等硬件平台上，当使用NVIDIA CPU，且CPU与GPU之间通过 NVLink Chip-to-Chip（C2C）互连时，可提供Address Translation Services（ATS）。当 ATS 可用时，设备属性 `cudaDevAttrPageableMemoryAccessUsesHostPageTables`的值为1。

在 ATS 可用的情况下，除支持所有主机内存分配（host allocations）的完整 Unified Memory之外，还具有以下特性：
- 驻留在GPU上的Managed Memory分配（例如通过 `cudaMallocManaged`分配）可以被 CPU 访问，而无需迁移（此时 `cudaDevAttrDirectManagedMemAccessFromHost`的值为1）。
- CPU与GPU之间的互连支持原生原子操作（native atomics）（此时 `cudaDevAttrHostNativeAtomicSupported`的值为1）
- 相比软件一致性（software coherence），硬件一致性（hardware coherence）能够提升性能。

ATS提供了HMM的全部能力。当ATS可用时，HMM会被自动禁用。关于硬件一致性与软件一致性的进一步讨论，请参见[CUDA官方文档学习6：统一内存（二）](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-6%EF%BC%9A-tong-yi-nei-cun-%EF%BC%88-er-%EF%BC%89.html)中的 **《CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency》** 小节。

注意：硬件一致性并不能使主机访问仅分配给 GPU 的内存，例如使用 `cudaMalloc`分配的内存
### 2.2.2 HMM——具有软件一致性的完整统一内存
Heterogeneous Memory Management（HMM，异构内存管理）是 Linux 操作系统（需要合适版本的内核）提供的一项功能，它能够实现采用软件一致性（software-coherent）的完整 Unified Memory 支持（Full Unified Memory Support）。

异构内存管理（HMM）为通过PCIe连接的 GPU 提供了 ATS 所具备的部分能力和便利性。
在 Linux 系统上，当内核版本至少为 **Linux Kernel 6.1.24、6.2.11 或 6.3 及以上**时，可能支持异构内存管理（HMM）。
可以使用以下命令查看当前系统是否采用HMM地址模式：
```
$ nvidia-smi -q | grep Addressing
Addressing Mode : HMM
```
当HMM可用时：
- 支持完整 Unified Memory（Full Unified Memory）；
- 所有系统内存分配（System Allocations）都会隐式地作为 Unified Memory。
## 2.3 有限统一内存支持
在Windows（包括 Windows Subsystem for Linux，WSL）以及部分Tegra系统上，仅提供 Unified Memory功能的有限子集（limited subset）。
在这些系统上，虽然支持Managed Memory，但Managed Memory 在 CPU 与 GPU 之间的迁移方式（migration）与完整 Unified Memory 不同。

其行为如下：
- Managed Memory 最初分配在 CPU 的物理内存（physical memory）中。
- Managed Memory 的迁移粒度大于虚拟内存页（virtual memory pages）。
- 当 GPU 开始执行时，Managed Memory 会迁移到 GPU。
- 当 GPU 处于活动状态（active）时，CPU 不得访问Managed Memory。
- 当 GPU 完成同步（synchronized）后，Managed Memory 会迁移回 CPU。
- **不允许** GPU 内存超额分配（Oversubscription）。
- **只有**通过 CUDA 显式分配为 **Managed Memory** 的内存才属于 Unified Memory。

关于这一工作模式的完整说明，请参见[CUDA官方文档学习6：统一内存（二）](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-6%EF%BC%9A-tong-yi-nei-cun-%EF%BC%88-er-%EF%BC%89.html)中的 **《Unified Memory on Windows, WSL, and Tegra》**。
## 2.4 内存建议与预取
程序员可以向负责管理**Unified Memory**的**NVIDIA Driver**提供**提示（hints）**，帮助其尽可能提高应用程序的性能。

CUDA API `cudaMemAdvise`允许程序员为内存分配指定一些属性（properties），这些属性会影响：
- 内存应当放置（placed）在哪里；
- 当该内存被其他设备访问时，是否需要迁移（migrated）。
CUDA API `cudaMemPrefetchAsync`允许程序员建议（suggest）系统异步地（asynchronously）将某一块指定的内存分配迁移（migrate）到另一个位置，并开始这一迁移过程。

一种常见的用法是在 **Kernel 启动之前**，提前开始传输该 Kernel 将要使用的数据。这样，数据拷贝可以在**其他 GPU Kernel 正在执行**时同时进行。

[CUDA官方文档学习6：统一内存（二）](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-6%EF%BC%9A-tong-yi-nei-cun-%EF%BC%88-er-%EF%BC%89.html)中的**《Performance Hints》** 一节介绍了可以传递给 `cudaMemAdvise`的各种提示（hints），并给出了使用`cudaMemPrefetchAsync`的示例。
# 3.页锁定主机内存
在之前博客的入门代码示例中，我们使用 `cudaMallocHost`在CPU上分配内存。该函数在主机（Host）上分配页锁定内存（Page-locked Memory），也称为**固定内存（Pinned Memory）**。

通过传统内存分配机制（如 `malloc`、`new` 或 `mmap`）分配的主机内存不是页锁定内存，这意味着它们可能会被操作系统换出到磁盘（swapped to disk），或者重新映射到其他物理地址（physically relocated）。

而页锁定主机内存是 **CPU 与 GPU 之间进行异步数据拷贝（asynchronous copies）** 所必需的。页锁定主机内存还能够提高**同步数据拷贝（synchronous copies）** 的性能。此外，页锁定内存还可以被**映射（mapped）** 到 GPU，使 GPU Kernel 能够直接访问。

CUDA Runtime 提供了用于分配页锁定主机内存，或将已有内存页锁定的 API：
- `cudaMallocHost`：分配页锁定主机内存。
- `cudaHostAlloc`：默认行为与 `cudaMallocHost`相同，但还可以通过传入标志（flags）指定其他内存参数。
- `cudaFreeHost`：释放由 `cudaMallocHost`或`cudaHostAlloc`分配的内存。
- `cudaHostRegister`：将 CUDA API 之外分配的一段已有内存（例如通过 `malloc`或 `mmap`分配）设置为页锁定内存。

>**注意（Note）**：页锁定主机内存可用于系统中所有 GPU的异步数据拷贝和映射内存。在不支持 I/O 一致性（non I/O coherent）的 Tegra 设备上，页锁定主机内存不会被缓存（cached）。此外，`cudaHostRegister()`在不支持 I/O 一致性的 Tegra 设备上不受支持。

## 3.1 映射内存
在支持HMM或ATS的系统上，所有主机内存（Host Memory）都可以通过主机指针（host pointers）被 GPU 直接访问。
当ATS或HMM不可用时，可以通过**将主机内存映射（mapping）到 GPU 的内存空间**，使主机内存能够被 GPU 访问。**Mapped Memory 始终是页锁定内存（page-locked memory）**。

下面的代码示例将演示以下数组复制 Kernel，它直接在映射后的主机内存（mapped host memory）上执行。
```cpp
__global__ void copyKernel(float* a, float* b)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    a[idx] = b[idx];
}
```
虽然在某些场景下，Mapped Memory 很有用，例如某些不会复制到 GPU、但又需要在 Kernel 中访问的数据，但在 Kernel 中访问 Mapped Memory，需要通过 **CPU–GPU 互连（CPU-GPU interconnect）** 发起访问事务（transactions），例如
- PCIe
- NVLink C2C
与访问**设备内存（device memory）** 相比，这种访问具有**更高的延迟（higher latency）** 和**更低的带宽（lower bandwidth）**。
对于大多数 Kernel 的内存需求而言，**Mapped Memory 不应被视为 Unified Memory 或显式内存管理（explicit memory management）的高性能替代方案。**

## 3.2 cudaMallocHost与cudaHostAlloc
使用 `cudaMallocHost` 或 `cudaHostAlloc` 分配的主机内存（Host Memory）会自动完成映射（mapped）。这些 API 返回的指针可以直接在 Kernel 代码中使用，以访问位于主机上的这块内存。对这块主机内存的访问，是通过 CPU–GPU 互连（CPU-GPU interconnect）完成的。

一个使用cudaMallocHost的例子：
```cpp
void usingMallocHost() {
  float* a = nullptr;
  float* b = nullptr;
  
  CUDA_CHECK(cudaMallocHost(&a, vLen*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&b, vLen*sizeof(float)));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaMallocHost: ");
  checkAnswer(a,b);
}
```
一个使用cudaHostAlloc的例子：
```cpp
void usingCudaHostAlloc() {
  float* a = nullptr;
  float* b = nullptr;

  CUDA_CHECK(cudaHostAlloc(&a, vLen*sizeof(float), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&b, vLen*sizeof(float), cudaHostAllocMapped));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaHostAlloc: ");
  checkAnswer(a, b);
}
```
## 3.3 cudaHostRegister
当ATS和HMM不可用时，通过系统内存分配器（system allocators）分配的内存，仍然可以使用 `cudaHostRegister`进行映射（mapped），从而允许 GPU Kernel 直接访问。
但是，与通过 CUDA API 创建的内存不同，这类内存不能在 Kernel 中直接使用主机指针（host pointer）进行访问。必须通过 `cudaHostGetDevicePointer()`获取一个位于设备内存地址空间（device's memory region）中的指针，并且 Kernel 中必须使用该设备指针（device pointer）来访问这块内存。
```cpp
void usingRegister() {
  float* a = nullptr;
  float* b = nullptr;
  float* devA = nullptr;
  float* devB = nullptr;

  a = (float*)malloc(vLen*sizeof(float));
  b = (float*)malloc(vLen*sizeof(float));
  CUDA_CHECK(cudaHostRegister(a, vLen*sizeof(float), 0 ));
  CUDA_CHECK(cudaHostRegister(b, vLen*sizeof(float), 0  ));

  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devA, (void*)a, 0));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devB, (void*)b, 0));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(devA, devB);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaHostRegister: ");
  checkAnswer(a, b);
}
```
## 3.4 比较统一内存和映射内存
Mapped Memory使CPU 内存能够被GPU访问，但并不保证所有类型的访问（例如原子操作（atomics））在所有系统上都受到支持。而 **Unified Memory** 保证支持所有类型的访问。

Mapped Memory 始终驻留在CPU 内存中，这意味着 GPU 对它的所有访问都必须经过CPU与 GPU之间的互连，即：
- PCIe
- NVLink
通过这些互连进行访问的延迟（latency）明显高于访问 GPU 内存，同时可用的总带宽（bandwidth）也更低。

因此，将 Mapped Memory 用于 Kernel 的全部内存访问，通常**无法充分利用 GPU 的计算资源。**

Unified Memory 通常会迁移（migrated）到当前访问它的处理器（processor）的物理内存中。在第一次迁移之后，如果 Kernel 重复访问同一个内存页（memory page）或缓存行（cache line），则可以利用 GPU 的全部内存带宽。

> **注意（Note）**
> Mapped Memory 在以前的文档中也称为 **Zero-copy Memory（零拷贝内存）**。
> 在所有 CUDA 应用程序都采用统一虚拟地址空间之前，需要额外的 API 来启用内存映射，例如：`cudaSetDeviceFlags`、`cudaDeviceMapHost`，现在已经不再需要这些 API。
> 作用于映射主机内存（mapped host memory）上的原子函数（Atomic Functions）（参见 [5.4. C/C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-space-specifiers)），从主机（Host）或其他 GPU 的角度来看，并不是原子的。
> CUDA Runtime 要求：由设备（Device）发起、对主机内存进行的以下自然对齐（naturally aligned）读写操作：1 字节、2 字节、4 字节、8 字节、16 字节，从主机和其他设备的角度来看，都必须保持为一次单独的访问（single accesses）。
> 在某些平台上，硬件可能会将对内存执行的原子操作拆分为独立的加载（load）和存储（store）操作。这些拆分后的加载和存储操作，同样需要满足上述关于自然对齐访问保持完整性的要求。
   CUDA Runtime 不支持这样一种PCI Express总线拓扑：即 PCI Express Bridge（PCI Express 桥）会将一个自然对齐的 8 字节访问拆分成多个访问。此外，NVIDIA 目前尚不了解任何会将自然对齐的 16 字节访问拆分开的 PCI Express 拓扑。

# 4.总结
在支持 Heterogeneous Memory Management（HMM） 或 Address Translation Services（ATS） 的 Linux 平台上，所有通过系统分配的内存（system-allocated memory）都是 Managed Memory。

在不支持 HMM 或 ATS 的 Linux 平台、Tegra 处理器以及所有 Windows 平台上，Managed Memory 必须通过 CUDA 显式分配，可采用以下方式：cudaMallocManaged()；或cudaMallocFromPoolAsync()，其中使用的内存池（memory pool）需设置：allocType = cudaMemAllocationTypeManaged

在 Windows 和 Tegra 处理器上，Unified Memory 存在一定的限制（limitations）。
在采用 NVLink C2C 互连并支持 ATS 的系统上，通过 cudaMallocManaged 分配的设备内存（device memory）可以被 CPU 或其他 GPU 直接访问。