原文链接：
[4.1. Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-legacy-devices)
# 前言
本节将详细介绍目前可用的各种统一内存工作模式（paradigms）的行为及其使用方式。之前的文章[CUDA官方文档学习5：统一内存（一）](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-5%EF%BC%9A-tong-yi-nei-cun-%EF%BC%88-yi-%EF%BC%89.html)一节已经介绍了如何判断当前系统所采用的 Unified Memory 工作模式，并对每一种工作模式进行了简要说明。

如前所述，Unified Memory 编程共有四种工作模式：
- 对显式分配的 Managed Memory 提供完整支持（Full support for explicit managed memory allocations）（见1）
- 对所有内存分配提供完整支持，并采用软件一致性（Full support for all allocations with software coherence）（见2）
- 对所有内存分配提供完整支持，并采用硬件一致性（Full support for all allocations with hardware coherence）（见2）
- 有限 Unified Memory 支持（Limited unified memory support）（见3）
前三种工作模式均属于**完整 Unified Memory 支持（Full Unified Memory Support）**，它们具有非常相似的行为和编程模型，将在后续 **《具备完整CUDA统一内存支持的设备上的统一内存》** 一节中介绍，并指出它们之间存在的差异。

最后一种工作模式，即**Unified Memory 支持受限**的情况，将在后续 **《Windows、WSL与Tegra上的统一内存》** 一节中进行详细介绍。
# 1.具备完整CUDA统一内存支持的统一内存
这些系统包括**硬件一致性内存系统（hardware-coherent memory systems）**，例如NVIDIA Grace Hopper，以及启用了**异构内存管理（Heterogeneous Memory Management，HMM）** 的现代 Linux 系统提供了软件一致性。（本小节“具备完整CUDA统一内存支持”默认指的是支持ATS/HMM的平台，排除了“仅对显式分配的 Managed Memory 提供完整支持”的平台）

HMM 是一种基于软件的内存管理系统，其提供的编程模型与硬件一致性内存系统相同。Linux HMM要求：
- Linux 内核版本 6.1.24+、6.2.11+ 或 6.3+；
- 计算能力（Compute Capability）7.5 或更高的设备；
- 安装 535+版本的 CUDA 驱动，并使用 [Open Kernel Modules](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/kernel-modules.html#open-gpu-kernel-modules-installation)。

>注意（Note）：我们将 CPU 和 GPU 共用同一页表（combined page table）的系统称为**硬件一致性系统（hardware-coherent systems）**。CPU 和 GPU 分别使用独立页表（separate page tables）的系统称为**软件一致性系统（software-coherent systems）**。

像NVIDIA Grace Hopper这样的硬件一致性系统，为 CPU 和 GPU 提供了逻辑上的统一页表，将在后续 **《CPU 与 GPU 页表：硬件一致性 vs 软件一致性》** 中介绍。

后续的 **《访问计数器迁移》** 这一小节仅适用于硬件一致性系统。
## 1.1 统一内存：深度实例解析
具有完整 CUDA Unified Memory 支持的系统允许设备（Device）访问与该设备交互的主机进程（host process）所拥有的任意内存。

本节将通过几个高级用例进行说明，这些示例均使用一个简单的 Kernel，该 Kernel 仅将输入字符数组的前8个字符输出到标准输出流（standard output stream）。
```cpp
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```
接下来展示了使用系统分配内存调用该内核的多种方式：
**Malloc**
```cpp
void test_malloc() {
  const char test_string[] = "Hello World";
  char* heap_data = (char*)malloc(sizeof(test_string));
  strncpy(heap_data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("malloc", heap_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  free(heap_data);
}
```
这段代码的作用是验证GPU能否直接访问由`malloc`分配的普通系统内存，一般来说如果是ATS/HMM平台上可以，否则不能。

**Managed**
```cpp
void test_managed() {
  const char test_string[] = "Hello World";
  char* data;
  cudaMallocManaged(&data, sizeof(test_string));
  strncpy(data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("managed", data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  cudaFree(data);
}
```
这个示例的意义就是验证 `cudaMallocManaged()` 分配的 Managed Memory 在所有支持 Full CUDA Unified Memory 的平台（包括 Explicit Managed、HMM 和 ATS）上，都能够被 GPU 直接访问。

**Stack variable**
```cpp
void test_stack() {
  const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("stack", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```
该案例验证GPU能否直接访问CPU栈上的局部变量

**File-scope static variable**
```cpp
void test_static() {
  static const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("static", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```
该案例验证GPU能否直接访问CPU静态存储区中的数据

**Global-scope variable**
```cpp
const char global_string[] = "Hello World";

void test_global() {
  kernel<<<1, 1>>>("global", global_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```
该案例验证GPU能否直接访问CPU全局变量中的数据

**Global-scope extern variable**
```cpp
// declared in separate file, see below
extern char* ext_data;

void test_extern() {
  kernel<<<1, 1>>>("extern", ext_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}

/** This may be a non-CUDA file(the other file) */
char* ext_data;
static const char global_string[] = "Hello World";

void __attribute__ ((constructor)) setup(void) {
  ext_data = (char*)malloc(sizeof(global_string));
  strncpy(ext_data, global_string, sizeof(global_string));
}

void __attribute__ ((destructor)) tear_down(void) {
  free(ext_data);
}
```
该案例验证GPU能否直接访问其他源文件中定义的全局指针所指向的系统内存。其中`ext_data`这个指针变量位于CPU全局存储区，`ext_data`指向的数据位于CPU堆，`global_string`位于CPU静态存储区
`__attribute__`介绍：这是GCC/Clang的扩展语法，不是C/C++标准语法，其作用是：**给编译器添加一些特殊属性，控制函数或变量的行为**，语法形式：
```cpp
__attribute__((属性名))
or
__attribute__((属性1,属性2,...))
```
上述代码中使用的属性是：`constructor`，表示：**程序启动时，在`main()`执行之前自动调用`setup()`这个函数**，同时`destructor`表示程序结束时，自动调用`tear_down()`

注意，对于 `extern` 变量而言，它可能是在**第三方库**中声明的，其所指向的内存也是由该第三方库负责分配和管理，即使这块内存完全由一个**从未接触 CUDA 的第三方库**负责分配和管理，GPU 仍然可以直接访问它（前提是 HMM 或 ATS 平台）。
还需要注意的是，栈变量（stack variables）、文件作用域变量（file-scope variables）以及全局作用域变量（global-scope variables），GPU 只能通过指针（pointer）来访问它们。在本例中，这一点比较方便，因为字符数组本身就是以指针类型 `const char*` 的形式使用的。
然而，请考虑下面这个使用全局作用域整数变量（global-scope integer）的示例：
```cpp
// this variable is declared at global scope
int global_variable;

__global__ void kernel_uncompilable() {
  // this causes a compilation error: global (__host__) variables must not
  // be accessed from __device__ / __global__ code
  printf("%d\n", global_variable);
}

// On systems with pageableMemoryAccess set to 1, we can access the address
// of a global variable. The below kernel takes that address as an argument
__global__ void kernel(int* global_variable_addr) {
  printf("%d\n", *global_variable_addr);
}
int main() {
  kernel<<<1, 1>>>(&global_variable);
  ...
  return 0;
}
```
在上面的示例中，我们需要确保将指向全局变量的指针传递给内核，而非在内核中直接访问该全局变量。这是因为未添加 `__managed__` 说明符的全局变量默认会被声明为仅 `__host__` 可用，因此目前大多数编译器都不允许在设备代码中直接使用这类变量（即使是在ATS/HMM平台上），而如果添加了__managed__说明符，那么在ATS/HMM以及非ATS与HMM的全统一内存支持的平台上都可以直接访问这个变量
### 1.1.1 文件后端的统一内存
由于支持完整 CUDA Unified Memory（Full CUDA Unified Memory Support）的系统允许设备访问属于主机进程的任意内存，因此GPU也可以直接访问文件后端内存（file-backed memory）。

>文件后端内存（file-backed memory）简介
>Linux内存分类两大种：File-backed memory与Anonymous memory
>File-backed memory的内存页关联磁盘上真实文件，来源包括读取程序二进制、库文件、配置、日志、数据文件、mmap映射文件，特点为：内存不足时直接写回原磁盘文件，不用单独开辟交换分区，且回收极快，系统优先回收这类内存。
>Anonymous memory无对应磁盘文件，进程堆、栈、临时数据、malloc分配的内存都属于它，内存不够只能丢去swap交换分区

下面我们将前一节中的初始示例稍作修改，改为使用文件后端内存，使 GPU 能够直接从输入文件中读取字符串并输出。在下面的示例中，这块内存以后端为物理文件（physical file）的方式进行管理；不过，该示例同样适用于内存后端文件（memory-backed file）。
```cpp
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```
```cpp
void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}
```
mmap函数解释：
```cpp
void *mmap(
	void *addr, //映射的起始地址（由内核选择时填0）
	size_t length, //映射的长度（字节数）（一般是文件的完整大小）
	int prot, //内存保护权限（PROT_READ表示映射区仅允许读操作，写操作会触发段错误SIGSEGV）
	int flags, //映射类型与行为标志（MAP_PRIVATE表示私有映射，对映射区修改不会写回原文件
	int fd, //要映射的文件描述符
	off_t offset //文件内的偏移量（从哪里开始映射）
)
```
请注意，在不具备 hostNativeAtomicSupported 属性（见后续 **《主机原生原子操作》** 小节）的系统上，包括启用了 Linux HMM 的系统，均不支持对文件映射内存执行原子访问。
### 1.1.2 统一内存的进程间通信（IPC）
>注意：将 IPC 与统一内存结合使用可能会带来显著的性能影响。

许多应用倾向于让每个进程管理一个 GPU，但仍然需要使用统一内存，例如用于超额分配内存，以及从多个 GPU 访问该内存。

CUDA IPC（参见“[进程间通信](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html#interprocess-communication)”）不支持托管内存（managed memory）：此类内存的句柄不能通过本节讨论的任何机制进行共享。在支持完整 CUDA 统一内存的系统上，系统分配的内存具备 IPC 能力。一旦系统分配的内存已与其他进程共享，就适用相同的编程模型，这与文件后端统一内存类似。

有关在 Linux 下创建具备 IPC 能力的系统分配内存的各种方法，请参阅以下资料：
- [使用 `MAP_SHARED` 的 `mmap`](https://man7.org/linux/man-pages/man2/mmap.2.html)
- [POSIX IPC API](https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html)
- [Linux `memfd_create`](https://man7.org/linux/man-pages/man2/memfd_create.2.html)
请注意，使用这种技术无法在不同主机及其设备之间共享内存。
## 1.2 性能调优
为了通过统一内存获得良好性能，以下几点非常重要：
- 了解系统中的分页机制如何工作，以及如何避免不必要的缺页；
- 了解各种能够使数据保持在访问该数据的处理器本地的机制；
- 考虑根据系统的内存传输粒度对应用程序进行调优。
作为一般性建议，性能提示（参见后续章节 **《性能提示》** ）可能会提升性能，但如果使用不当，与默认行为相比，性能反而可能下降。还需要注意，任何提示都会在主机端产生相应的性能开销，因此，有效的提示至少必须带来足够的性能提升，以抵消这一开销。
### 1.2.1 内存分页与页面大小
为了更好地理解统一内存对性能的影响，有必要先理解**虚拟寻址（virtual addressing）**、**内存页（memory pages）** 以及**页大小（page sizes）** 等概念。本小节旨在定义这些必要术语，并解释为什么**分页（paging）** 会对系统性能产生重要影响。

目前所有支持统一内存的系统都采用**虚拟地址空间（virtual address space）**。这意味着，应用程序所使用的内存地址表示的是一个**虚拟地址（virtual location）**，它会被映射（mapped）到内存实际存放数据的**物理地址（physical location）**。

此外，目前所有受支持的处理器（包括 CPU 和 GPU）都采用了**内存分页机制（memory paging）**。由于系统采用虚拟地址空间，因此内存页可以分为两类：
- **虚拟页（Virtual pages）**：表示操作系统为每个进程维护的一块**固定大小、连续的虚拟内存区域**，该区域可以映射到物理内存中。需要注意的是，**虚拟页本身对应的是一种映射关系**。例如，同一段虚拟地址空间，可以采用不同大小的页映射到物理内存中。
- **物理页（Physical pages）**：表示处理器主**内存管理单元（Memory Management Unit，MMU）** 所支持的一块**固定大小、连续的物理内存区域**，虚拟页最终会被映射到这些物理页上。

目前，各类处理器支持的物理页大小如下：
- 所有x86_64 CPU默认采用4 KiB的物理页大小。
- Arm CPU根据具体型号支持多种物理页大小，包括4 KiB、16 KiB、32 KiB 和 64 KiB。
- NVIDIA GPU同样支持多种物理页大小，但通常更倾向于使用2 MiB 或更大的物理页。
需要注意的是，这些页大小可能会随着未来硬件的发展而发生变化。

通常情况下，虚拟页的默认大小会与物理页大小保持一致。不过，只要操作系统和硬件均支持，应用程序也可以选择不同的虚拟页大小。
一般来说，支持的虚拟页大小需要满足以下两个条件：
1. 必须是 **2 的幂（power of 2）**；
2. 必须是**物理页大小的整数倍**。

负责维护虚拟页到物理页映射关系的逻辑结构称为**页表（Page Table）**。
其中，每一条将某个虚拟页（具有特定页大小）映射到一个或多个物理页的记录，称为**页表项（Page Table Entry，PTE）**。

为了加速虚拟地址到物理地址的转换，目前所有受支持的处理器都提供了专门缓存页表内容的硬件缓存，这种缓存称为：**Translation Lookaside Buffer（TLB，快表/地址转换后备缓冲区）** ，TLB 可以缓存最近使用的页表项，从而避免每次访问内存时都查询页表，大幅降低地址转换开销。

对于应用程序的性能调优而言，有两个方面尤为重要：
1. **虚拟页大小（virtual page size）的选择。**
2. **系统采用的是 CPU 与 GPU 共用同一套页表（combined page table），还是 CPU 与 GPU 分别维护各自独立的页表（separate page tables）。**
这两个因素都会显著影响统一内存的数据访问效率、地址转换开销以及页面迁移（page migration）的性能。
#### 1.2.1.1 选择正确的页大小
一般来说，较小的页大小能够减少（虚拟）内存碎片（memory fragmentation），但会导致**TLB 未命中（TLB misses）** 增多；而较大的页大小则会增加内存碎片，但能够减少 TLB 未命中的发生。

此外，由于内存迁移（memory migration）通常是**以整个内存页为单位进行迁移**，因此较大页大小的迁移开销通常高于较小页大小。这意味着，在使用大页的应用程序中，一次页面迁移可能需要移动更多的数据，从而导致更大的访问延迟峰值（latency spikes）。关于页面错误（page faults）的更多细节，可参考后续内容。

对于性能调优而言，一个十分重要的特点是：**GPU 上发生一次 TLB 未命中的代价通常远高于 CPU。** 这意味着，如果一个 GPU 线程频繁随机访问采用较小页大小映射的统一内存，其访问速度可能会比访问采用较大页大小映射的统一内存慢得多。

类似的情况在 CPU 上也可能发生。例如，一个 CPU 线程随机访问一大片采用小页映射的内存时，也会因为 TLB 未命中增加而导致性能下降。但与 GPU 相比，这种性能下降通常没有那么明显。
因此，对于 CPU 而言，应用程序往往可以在以下两者之间进行权衡：
- 使用较小页，获得更低的内存碎片；
- 接受一定程度的 TLB 未命中带来的性能下降。

最后需要注意的是：通常情况下，应用程序不应该针对某种处理器的物理页大小（physical page size）进行性能优化。这是因为**物理页大小可能会随着硬件平台的不同而发生变化**。因此，上述关于性能调优的建议仅适用于虚拟页大小（virtual page size），而不适用于物理页大小。
#### 1.2.1.2 CPU与GPU页表：硬件一致性vs软件一致性
像NVIDIA Grace Hopper这样的硬件一致性（hardware-coherent）系统，为 CPU 和 GPU 提供了一套**逻辑上共享的页表（logically combined page table）**。这一点十分重要，因为GPU 在访问由操作系统分配（system-allocated）的内存时，会直接使用 CPU 为该内存创建的页表项（Page Table Entry，PTE）。如果该页表项采用的是 CPU 默认的页大小（例如4 KiB或 64 KiB），那么当 GPU 访问一大片连续的虚拟内存区域时，就会产生大量 TLB 未命中，从而导致显著的性能下降。

相反，在软件一致性（software-coherent）系统中，CPU 和 GPU **各自维护独立的逻辑页表（logical page table）**。在这类系统中，性能调优需要考虑另外一些因素。为了保证 CPU 与 GPU 之间的数据一致性（coherency），这类系统通常会在某个处理器访问了一块当前映射到另一个处理器物理内存中的地址时触发页面错误（page fault）。发生页面错误后，系统需要完成以下几个步骤：
1. 保证当前拥有该物理页的处理器（即当前该物理页所在的处理器）不能继续访问该页面。
    为此，系统需要：
    - 删除对应的页表项（Page Table Entry），或者
    - 更新该页表项，使其失效。
2. 保证请求访问该页面的处理器能够访问该页面。
    为此，系统需要：
    - 创建新的页表项，或者
    - 更新已有的页表项，使其重新变为有效（valid/active）。
3. 将支撑该虚拟页的物理页迁移（move/migrate）到请求访问的处理器。
    这一操作通常代价较高，并且迁移所需的工作量与页大小成正比——页越大，需要搬迁的数据越多，因此迁移开销也越高。

总体而言，在CPU 和 GPU 线程频繁并发访问同一块内存页的场景下，**硬件一致性系统相比软件一致性系统具有明显的性能优势**，主要体现在以下两个方面：
1. 更少的页面错误：硬件一致性系统**无需依赖页面错误（page fault）来实现数据一致性或页面迁移**。也就是说：
	* 不需要因为 CPU/GPU 交替访问同一块内存而频繁触发 page fault
	* 不需要不断修改页表
	* 不需要频繁迁移整页数据
	因此能够显著降低访问延迟
2. 更少的竞争：硬件一致性系统是在**缓存行（cache line）粒度**而不是**页面（page）粒度**上维护一致性的。也就是说：
	* 当多个处理器竞争同一个缓存行时，系统只需要交换这一条缓存行（通常64B或128B），而缓存行远小于最小的内存页（通常4KiB）。
	* 如果 CPU 和 GPU 分别访问的是同一页面中的不同缓存行，  那么它们之间不会发生竞争。

相比之下，软件一致性系统通常以整页（4KB、64KB甚至更大）作为一致性维护单位，因此即使 CPU 和 GPU 修改的是同一页中的不同数据，也可能导致整个页面迁移，造成大量额外开销。上述差异会直接影响以下典型场景的性能：
* CPU 和 GPU 同时对同一个内存地址执行原子更新（atomic updates）。例如`atomicAdd(counter, 1);`，CPU与GPU可以同时更新同一个计数器，而无需迁移整个页面
* CPU线程与GPU线程之间进行信号通知（Signaling）。例如：CPU 写入一个 flag，通知 GPU 开始执行；GPU 修改一个 flag，通知 CPU 某项任务已经完成。在硬件一致性系统中，这种同步通常只涉及缓存行状态更新，因此延迟较低；而在软件一致性系统中，则可能触发 page fault 和页面迁移，导致同步成本显著增加。
#### 1.2.1.3 混合硬件与软件一致性
一些支持硬件一致性（hardware coherence）的系统（例如 NVIDIA DGX Station）还支持安装独立的、非一致性（non-coherent）的 GPU。
在这种情况下：
- 硬件一致性 GPU仍然采用前文中介绍的硬件一致性机制进行内存访问。
- 而**独立 GPU（discrete GPU）** 则仍采用软件一致性机制进行访问。

这两类GPU允许共享同一个统一虚拟地址空间。但是，由于两类 GPU 采用不同的一致性机制，因此它们访问统一内存时，会表现出不同的性能特征和内存迁移行为。
具体来说：
- 软件一致性 GPU访问统一内存时，会产生更多的页面错误以及内存迁移。
- 硬件一致性 GPU访问统一内存时，则发生页面错误的次数更少；并且在可能的情况下，它会直接远程映射已有的数据，而不是将数据迁移到本地。
>远程映射（Remote Mapping）：数据仍然保留在原来的物理内存位置（例如 CPU 内存），GPU 通过一致性互连（如 NVLink-C2C）直接访问，而不是把整页数据复制到自己的本地内存。

为了获得最佳性能，应尽量减少两类 GPU 之间共享同一份数据，或者通过显式的数据拷贝（explicit copies）来完成数据交换。
另一种优化方式是调用：`cudaMemAdviseSetPreferredLocation()`指定某块数据的首选驻留位置（preferred location），例如，可以让频繁共享的数据始终驻留在：
- CPU 内存，或者
- 硬件一致性 GPU 可直接访问的内存。
这样做的原因是：默认情况下，当软件一致性 GPU访问一块共享数据时，通常需要先触发页面错误，随后再执行页面迁移，才能完成访问。而将数据放置在更合适位置，可以减少这种额外开销。

在这种**混合一致性系统（mixed coherency system）** 中，`cudaHostRegister`以及其他**主机内存访问 API（host memory access APIs）** 对于软件一致性 GPU 的行为也会发生变化。
传统情况下：
- `cudaHostRegister()` 会将一块 CPU 内存注册为**Pinned Memory（页锁定内存）**，
- GPU 可以直接通过固定映射（pinned mapping）访问这块内存，而无需发生页面迁移。
但是，在混合一致性系统中，对于软件一致性 GPU而言，情况有所不同：
它不再使用 Pinned Mapping，而是采用：CPU 页表的软件镜像（software mirroring of CPU page tables）也就是说，GPU 会维护一份 CPU 页表的软件副本，而不是建立固定映射。这意味着：软件一致性 GPU 的某些内存访问可能会发生页面错误（page fault），而这些访问在普通（非混合一致性）系统中本来是不需要发生页面错误的。不过，这种额外的页面错误通常比较少见，一般只有在系统内存压力较大时才会发生。
### 1.2.2 主机端直接统一内存访问
某些设备在硬件层面支持主机（Host，即 CPU）直接对驻留在 GPU 上的统一内存（GPU-resident unified memory）执行一致性的读取（read）、写入（store）以及原子操作（atomic access）。这类设备具有属性：`cudaDevAttrDirectManagedMemAccessFromHost=1`

需要注意的是：所有采用硬件一致性架构的系统，对于通过 NVLink 连接的设备，都会将该属性设置为 1。在这些系统上，CPU可以直接访问驻留在 GPU 内存中的数据，而无需发生页面错误或数据迁移。换句话说：GPU 上的数据无需迁移回 CPU 内存，CPU 就能够直接读取、写入甚至执行原子操作。不过需要注意的是，对于CUDA Managed Memory，若希望CPU能够实现这种无页面错误的直接访问，还必须使用：`cudaMemAdviseSetAccessedBy`，并将位置类型设置为`cudaMemLocationTypeHost`作为一个**Hint（访问提示）**，这样 CUDA 运行时会提前建立 CPU 对该统一内存的访问权限，使 CPU 后续访问该内存时无需再触发页面错误。

**System Allocator**
```cpp
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  // for shared page table systems, the following hint is not necesary
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory()
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  free(ret);
}
```

**Managed**
```cpp
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_managed() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);  // set direct access hint

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  cudaFree(ret); 
```
在 `write` Kernel 执行完成之后，`ret` 将会在 GPU 内存中创建并初始化。随后，CPU 将访问 `ret`，接着 `append` Kernel 会再次使用同一块 `ret` 内存。
根据系统架构以及是否支持硬件一致性，这段代码将表现出不同的行为：
- 对于 `directManagedMemAccessFromHost = 1` 的设备：
    CPU 对这块 Managed Memory 缓冲区的访问不会触发任何内存迁移（migration）；数据将继续驻留在 GPU 内存中，之后任何再次访问该数据的 GPU Kernel 都可以直接访问它，而不会产生页面错误或内存迁移。
- 对于 `directManagedMemAccessFromHost = 0` 的设备：
    CPU 对这块 Managed Memory 缓冲区的访问将会触发页面错误（page fault），并启动数据迁移（data migration）；随后，任何第一次再次访问这块数据的 GPU Kernel 都会再次触发页面错误，并将这些页面迁移回GPU内存。
### 1.2.3 主机原生原子操作
部分设备（包括硬件一致性系统中通过 NVLink 连接的设备）支持对驻留在 CPU 上的内存进行硬件加速原子访问。这意味着对主机内存的原子访问无需通过页故障来模拟。对于这类设备，cudaDevAttrHostNativeAtomicSupported 属性会被设置为 1。
### 1.2.4 原子访问与同步原语
CUDA统一内存支持主机线程和设备线程所支持的所有原子操作（atomic operations），从而使所有线程能够通过并发访问同一共享内存位置进行协同工作。

libcu++库提供了许多针对主机线程与设备线程之间并发使用而优化的**异构同步原语（heterogeneous synchronization primitives）**，其中包括 `cuda::atomic`、`cuda::atomic_ref`、`cuda::barrier`、`cuda::semaphore`等多种同步机制。

在软件一致性系统中，设备（GPU）对文件后端（file-backed）的主机内存（host memory）执行原子访问（atomic accesses）是不受支持的。下面给出的示例代码在硬件一致性系统上是合法的，但在其他系统上则表现为未定义行为。
```cpp
#include <cuda/atomic> //引入CUDA C++原子操作库

#include <cstdio>
#include <fcntl.h> //引入文件控制相关接口
#include <sys/mman.h> //引入内存映射相关接口

#define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

__global__ void kernel(int* ptr) {
  cuda::atomic_ref{*ptr}.store(2);//把2这个值原子写入ptr指向的内存
}

int main() {
  // this will be closed/deleted by default on exit
  FILE* tmp_file = tmpfile64();
  // need to allocate space in the file, we do this with posix_fallocate here
  int status = posix_fallocate(fileno(tmp_file), 0, 4096);
  //fileno把FILE*转换成文件描述符的int
  if (status != 0) ERR("Failed to allocate space in temp file\n");
  int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
  if (ptr == MAP_FAILED) ERR("Failed to map temp file\n");

  // initialize the value in our file-backed memory
  *ptr = 1;
  printf("Atom value: %d\n", *ptr);

  // device and host thread access ptr concurrently, using cuda::atomic_ref
  kernel<<<1, 1>>>(ptr);
  while (cuda::atomic_ref{*ptr}.load() != 2);
  // this will always be 2
  printf("Atom value: %d\n", *ptr);

  return EXIT_SUCCESS;
}
```
在软件一致性系统上，对统一内存执行原子访问可能会触发页面错误，从而导致显著的访问延迟。需要注意的是，这并不意味着这类系统中所有由 GPU 对 CPU 内存执行的原子操作都会触发页面错误。通过以下命令列出的操作：
```
nvidia-smi -q | grep "Atomic Caps Outbound"
```
可能能够避免页面错误。
在硬件一致性系统（hardware-coherent systems）上，主机与设备之间的原子操作不需要通过页面错误来实现；但由于其他原因，访问仍然可能发生页面错误，而这些原因同样可能导致任意内存访问触发页面错误。
### 1.2.5 统一内存中的Memcpy()/Memset()行为
`cudaMemcpy*()` 和 `cudaMemset*()` 均可接受统一内存指针作为参数。

对于 `cudaMemcpy*()`而言，参数 `cudaMemcpyKind`所指定的数据传输方向（direction）是一种性能提示（performance hint）。当 `cudaMemcpy*()` 的任一参数是统一内存指针时，这一提示可能会对性能产生较大的影响。

因此，建议遵循以下性能优化建议：
- 当已知统一内存的物理驻留位置（physical location）时，应准确指定 `cudaMemcpyKind`。
- 如果无法准确确定统一内存的物理驻留位置，则优先使用 `cudaMemcpyDefault`，而不要使用错误的 `cudaMemcpyKind` 提示。
- 始终使用已经完成物理页建立并初始化（populated / initialized）的缓冲区（buffers）；避免使用这些 API 来初始化内存。
- 如果 `cudaMemcpy*()` 的源指针和目标指针都指向由系统分配（system-allocated）的内存，则应避免使用 `cudaMemcpy*()`。此时，更推荐启动一个 CUDA Kernel 完成数据拷贝，或者直接使用 CPU 内存拷贝算法（如 `std::memcpy`）进行复制。
### 1.2.6 统一内存分配器概述
对于具备完整 CUDA 统一内存支持的系统，可使用多种不同的分配器来分配统一内存。下表列出了部分分配器及其对应功能的概览。请注意，本节中的所有信息可能会在未来的 CUDA 版本中发生变更。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260717160603.png)
【1】对于 `mmap`，文件后端内存默认放置在 CPU 上，除非通过 `cudaMemAdviseSetPreferredLocation`（或 `mbind`，见后续内容）另行指定。
【2】该特性可以通过 `cudaMemAdvise` 进行覆盖。即使禁用了基于访问的迁移（access-based migrations），如果底层内存空间已满，内存仍然可能发生迁移。
【3】文件后端内存不会根据访问情况发生迁移。
【4】在大多数系统中，默认的系统页大小为4 KiB或64 KiB，除非显式指定了大页大小（huge page size）（例如，通过 `mmap` 的 `MAP_HUGETLB` / `MAP_HUGE_SHIFT`）。在这种情况下，系统所配置的任意大页大小均受支持。
【5】GPU 驻留内存的页大小在未来的 CUDA 版本中可能会发生变化。
【6】目前，当内存迁移到 GPU，或由于 GPU 首次访问（first-touch）而放置到 GPU 上时，大页大小（huge page size）可能无法得到保留。

上表展示了几种可用于分配“可同时被多个处理器访问”的数据的内存分配器在语义上的差异。有关 `cudaMemPoolCreate` 的更多细节，参见[4.3. Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-pools)中“Memory Pools”小节。有关 `cuMemCreate` 的更多细节，参加[4.16.Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-memory-management)的“Virtual Memory Management”小节

在硬件一致性系统中，如果设备内存作为一个NUMA域（NUMA domain）暴露给系统，则可以使用诸如 `numa_alloc_on_node` 之类的特殊分配器，将内存固定（pin）到指定的 NUMA 节点（无论是主机还是设备）。这类内存可同时被主机和设备访问，并且不会发生迁移。类似地，也可以使用 `mbind` 将内存固定到指定的一个或多个 NUMA 节点，并且可以使文件后端内存在首次访问之前就放置到指定的一个或多个 NUMA 节点上。
### 1.2.7 访问计数器迁移
在硬件一致性系统中，**访问计数器（access counters）** 功能会跟踪 GPU 对位于其他处理器上的内存的访问频率。这样做的目的是确保内存页被迁移到访问该页面最频繁的处理器的物理内存中。访问计数器可以指导 CPU 与 GPU 之间，以及对等 GPU（peer GPUs）之间的内存迁移，这一过程称为访问计数器迁移（access counter migration）。

从CUDA 12.4开始，访问计数器支持系统分配内存（system-allocated memory）。

需要注意的是，文件后端内存不会根据访问情况发生迁移。
对于系统分配内存，可以通过使用 `cudaMemAdviseSetAccessedBy` 提示，并指定相应设备的设备 ID，来开启访问计数器迁移。如果启用了访问计数器，则可以使用 `cudaMemAdviseSetPreferredLocation`，并将其设置为 host，以阻止内存迁移。

默认情况下，`cudaMallocManaged` 采用页面错误并迁移（fault-and-migrate）机制进行迁移。驱动程序还可能在**更高效地缓解内存抖动（thrashing mitigation）** 或**内存超额订阅（memory oversubscription）** 场景中使用访问计数器。
### 1.2.8 避免从CPU频繁写入驻留于GPU的内存
如果主机（Host，CPU）访问统一内存（Unified Memory），缓存未命中（cache misses）可能会在主机与设备之间引入比预期更多的数据传输。

许多 CPU 架构要求所有内存操作（包括写操作）都必须经过缓存层次结构（cache hierarchy）。如果系统内存当前驻留（resident）在 GPU 上，这意味着 CPU 对这块内存进行频繁写入时，可能会导致缓存未命中（cache misses）。因此，在真正将新的值写入目标内存区域之前，必须先将相应的数据从 GPU 传输到 CPU。

在软件一致性系统（software-coherent systems）中，这可能会引入额外的页面错误（page faults）；而在硬件一致性系统（hardware-coherent systems）中，则可能导致CPU 操作之间更高的访问延迟（higher latencies）。

因此，为了与设备（Device，GPU）共享由主机产生的数据，建议将数据写入驻留在 CPU 上的内存（CPU-resident memory），并由设备直接读取这些数据。下面的代码展示了如何利用统一内存实现这一过程。

**System Allocator**
```cpp
  size_t data_size = sizeof(int);
  int* data = (int*)malloc(data_size);
  // 设置数据优先驻留在CPU内存，避免迁移到GPU
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
  
  // 告诉CUDA：Host会访问这块内存，减少CPU访问时的Page Fault
  cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

  // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
  // and GPU directly accesses that data, we can avoid the CPU caches re-loading
  // data if it was evicted in between writes
  for (int i = 0; i < 10; ++i) {
    *data = 42 + i;
    kernel<<<1, 1>>>(data);
    cudaDeviceSynchronize();
    // CPU cache potentially evicted data here
  }
  free(data);
```
**Managed**
```cpp
  int* data;
  size_t data_size = sizeof(int);
  cudaMallocManaged(&data, data_size);
  // ensure that data stays local to the host and avoid faults
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

  // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
  // and GPU directly accesses that data, we can avoid the CPU caches re-loading
  // data if it was evicted in between writes
  for (int i = 0; i < 10; ++i) {
    *data = 42 + i;
    kernel<<<1, 1>>>(data);
    cudaDeviceSynchronize();
    // CPU cache potentially evicted data here
  }
  cudaFree(data);
```
### 1.2.9 利用对系统内存的异步访问
如果应用程序需要将设备（Device，GPU）上的计算结果共享给主机（Host，CPU），可以采用以下几种方式：
1. 设备将计算结果写入驻留在 GPU 上的内存（GPU-resident memory），随后使用 `cudaMemcpy*` 将结果传输到主机，最后由主机读取传输后的数据。
2. 设备直接将计算结果写入驻留在 CPU 上的内存（CPU-resident memory），随后由主机直接读取该数据。
3. 设备将计算结果写入驻留在 GPU 上的内存（GPU-resident memory），随后由主机直接访问该数据。

如果在**主机传输或访问计算结果的同时，设备还可以继续执行其他独立的工作**，则**方案 1 或方案 3 更优**。
如果**设备必须等待主机访问完计算结果之后才能继续执行（即设备处于空闲等待状态）**，则**方案 2 可能更合适**。
这是因为，**设备（GPU）写入数据的带宽通常高于主机（CPU）读取数据的带宽**，除非使用大量主机线程并行读取这些数据。

**Explicit Copy**
```cpp
void exchange_explicit_copy(cudaStream_t stream) {
  int* data, *host_data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  host_data = (int*)malloc(n_bytes);
  // allocate, since we touch on the device first, will be GPU-resident
  cudaMallocManaged(&data, n_bytes);
  kernel<<<1, 16, 0, stream>>>(data);
  // launch independent work on the device
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // transfer to host
  cudaMemcpyAsync(host_data, data, n_bytes, cudaMemcpyDeviceToHost, stream);
  // sync stream to ensure data has been transferred
  cudaStreamSynchronize(stream);
  // read transferred data
  printf("Got values %d - %d from GPU\n", host_data[0], host_data[15]);
  cudaFree(data);
  free(host_data);
}
```

**Device Direct Write**
```cpp
void exchange_device_direct_write(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  cudaMallocManaged(&data, n_bytes);
  // ensure that data is mapped and resident on the host
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, location);
  kernel<<<1, 16, 0, stream>>>(data);
  // sync stream to ensure data has been transferred
  cudaStreamSynchronize(stream);
  // read transferred data
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
}
```

**Host Direct Read**
```cpp
void exchange_host_direct_read(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  cudaMallocManaged(&data, n_bytes);
  // ensure that data is mapped and resident on the device
  cudaMemLocation device_loc = {};
  cudaGetDevice(&device_loc.id);
  device_loc.type = cudaMemLocationTypeDevice;
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, device_loc);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, device_loc);
  kernel<<<1, 16, 0, stream>>>(data);
  // launch independent work on the GPU
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // sync stream to ensure data may be accessed (has been written by device)
  cudaStreamSynchronize(stream);
  // read data directly from host
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
```
最后，在前面的显式拷贝示例中，除了使用`cudaMemcpy*`进行数据传输之外，还可以使用主机（Host）或设备（Device）上的 Kernel来显式执行数据拷贝。

对于连续数据而言，优先使用 CUDA 的 Copy Engine（拷贝引擎），因为由 Copy Engine 执行的数据传输可以与主机和设备两侧的计算工作同时重叠执行（overlap）。

Copy Engine 可能会被 `cudaMemcpy*` 和 `cudaMemPrefetchAsync`API 使用，但并不能保证 `cudaMemcpy*` API 调用一定会使用 Copy Engine。

出于同样的原因，对于足够大的数据，显式拷贝优于主机直接读取（direct host read）。这是因为，如果主机和设备都在执行不会占满各自内存系统带宽的计算任务，那么数据传输可以由Copy Engine在后台完成，并与主机和设备的计算工作并发执行。

通常情况下，Copy Engine不仅用于主机与设备之间的数据传输，还用于NVLink 互连系统中对等设备（peer devices）之间的数据传输。

由于系统中 Copy Engine 的总数量通常是有限的，因此某些系统上 `cudaMemcpy*` 的带宽可能低于由设备（Device）显式执行数据拷贝所能达到的带宽。在这种情况下，如果数据传输位于应用程序的关键路径上，那么采用基于设备（Device）的显式数据拷贝可能是更优的选择。
# 2.仅支持CUDA托管内存的设备上的统一内存
对于计算能力（Compute Capability）6.x 及以上、但不支持可分页内存访问（pageable memory access）的设备，CUDA Managed Memory仍然得到完整支持，并且保持一致性（coherent）但 GPU 无法访问由系统分配器（system allocator）分配的内存。

统一内存（Unified Memory）的编程模型和性能调优方法，与“具备完整CUDA统一内存支持的统一内存”一节中介绍的内容基本相同。唯一需要特别注意的区别是：不能使用系统分配器（system allocator）来分配统一内存。

因此，下面这些小节不适用于这类设备：
- **Unified Memory: In-Depth Examples**（本文1.1）
- **CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency**（本文1.2.1.2）
- **Atomic Accesses and Synchronization Primitives**（本文1.2.4）
- **Access Counter Migration**（本文1.2.7）
- **Avoid Frequent Writes to GPU-Resident Memory from the CPU**（本文1.2.8）
- **Exploiting Asynchronous Access to System Memory**（本文1.2.9）
# 3.Windows、WSL 与 Tegra 上的统一内存
暂略
# 4.性能提示
**性能提示（Performance Hints）** 允许程序员向 CUDA 提供更多关于统一内存使用方式的信息。CUDA 会利用这些性能提示，以更高效地管理统一内存，从而提高应用程序的性能。性能提示不会影响应用程序的正确性，它们只会影响性能。

> **注意（Note）**
> 应用程序只有在统一内存性能提示能够提升性能的情况下，才应该使用这些性能提示。

性能提示可以应用于任何统一内存分配，包括 CUDA Managed Memory。在支持完整 CUDA Unified Memory（Full CUDA Unified Memory Support）的系统上，性能提示还可以应用于所有由系统分配器（system allocator）分配的内存。
## 4.1 数据预取
`cudaMemPrefetchAsync` API 是一种异步流排序 API，可将数据迁移至更靠近指定处理器的位置。数据在预取过程中仍可被访问。该迁移操作需等待流中所有先前操作完成后才会启动，并会在流中任何后续操作开始前完成。
```cpp
cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                 size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags,
                                 cudaStream_t stream=0);
```
当预取任务在指定的Stream中执行时，包含 `[devPtr, devPtr + count)`范围内的内存区域，可以迁移到由 `location.id`指定的目标设备（当 `location.type`为 `cudaMemLocationTypeDevice`时），或者迁移到CPU（当 `location.type`为 `cudaMemLocationTypeHost`时）。关于`flags`参数的详细说明，请参考当前版本的[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)文档。

下面给出一个简单的代码示例：
**System Allocator**
```cpp
void test_prefetch_sam(const cudaStream_t& s) {
  // initialize data on CPU
  char *data = (char*)malloc(dataSizeBytes);
  init_data(data, dataSizeBytes);                                     
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // encourage data to move to GPU before use
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);      

  // use data on GPU
  const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);  

  // encourage data to move back to CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);      
  
  cudaStreamSynchronize(s);

  // use data on CPU
  use_data(data, dataSizeBytes);                                      
  free(data);
}
```
**Managed**
```cpp
void test_prefetch_managed(const cudaStream_t& s) {
  // initialize data on CPU
  char *data;
  cudaMallocManaged(&data, dataSizeBytes);
  init_data(data, dataSizeBytes);                                     
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // encourage data to move to GPU before use
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  // use data on GPU
  const uinsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes); 

  // encourage data to move back to CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s); 

  cudaStreamSynchronize(s);

  // use data on CPU
  use_data(data, dataSizeBytes);
  cudaFree(data);
}
```
## 4.2 数据使用提示
当多个处理器同时访问同一块数据时，可以使用 `cudaMemAdvise`向 CUDA 提供提示（hint），说明位于 `[devPtr, devPtr + count)`范围内的数据将会如何被访问。
```cpp
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```
advice可能采用以下的值：
`cudaMemAdviseSetReadMostly`：该提示表示这块数据主要用于读取，只有偶尔才会被写入。通常情况下，它允许在该内存区域上以写带宽为代价换取更高的读带宽。
`cudaMemAdviseSetPreferredLocation`：该提示将数据的首选位置设置为指定设备物理内存所在的位置。该提示会促使系统尽量将数据保留在首选位置，但并不保证数据一定驻留在那里。
当 `location.type`的值为 `cudaMemLocationTypeHost`时，表示将数据的首选位置设置为 CPU 内存。其他提示（例如 **`cudaMemPrefetchAsync`**）可能会覆盖这一提示，使内存迁移到首选位置之外。
`cudaMemAdviseSetAccessedBy`在某些系统上，在某个处理器访问数据之前，预先建立该处理器到这块内存的映射（mapping）可能有助于提高性能。
该提示告诉系统：当 `location.type`为 `cudaMemLocationTypeDevice`时，位于 `location.id`的设备将会频繁访问这块数据，因此系统可以认为提前创建这些映射是值得的。该提示并不表示数据应该驻留在哪里，但可以与 `cudaMemAdviseSetPreferredLocation` 结合使用，以同时指定数据的首选驻留位置。

在硬件一致性系统中，该提示会启用访问计数器迁移机制
上述每一种建议（advice）都可以通过对应的 **Unset** 接口进行取消，分别为：
- `cudaMemAdviseUnsetReadMostly`
- `cudaMemAdviseUnsetPreferredLocation`
- `cudaMemAdviseUnsetAccessedBy`
**System Allocator**
```cpp
void test_advise_sam(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
  
  // Allocate memory using malloc or cudaMallocManaged
  dataPtr = (char*)malloc(dataSize);

  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written by the CPU each outer loop iteration
    init_data(dataPtr, dataSize);

    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }

  free(dataPtr);
}
```

**Managed**
```cpp
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

  // Allocate memory using cudaMallocManaged
  // (malloc may be used on systems with full CUDA Unified memory support)
  cudaMallocManaged(&dataPtr, dataSize);

  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written by the CPU each outer loop iteration
    init_data(dataPtr, dataSize);

    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  
  cudaFree(dataPtr);
}
```
## 4.3 内存丢弃
`cudaMemDiscardBatchAsync`API 允许应用程序通知 CUDA Runtime：指定内存区域中的内容已经不再有用（no longer useful）。

为了支持设备内存超额订阅（device memory oversubscription），Unified Memory 驱动程序会由于基于缺页的页面迁移或页面驱逐而自动执行内存传输。这些自动内存传输有时可能是冗余的，从而严重降低性能。

将某个地址范围标记为 discard 后，Unified Memory 驱动程序会知道：应用程序已经消费了该范围内的数据内容，因此在后续执行预取或页面驱逐以腾出空间时，无需再迁移这些数据。如果在未进行后续写入或预取的情况下读取已被丢弃的页面，将得到不确定值。

而在执行 discard 操作之后，任何新的写入都保证能够被后续读取正确看到。

如果在执行 discard 操作的同时，对对应地址范围发生并发访问或预取，则会导致未定义行为。
```cpp
cudaError_t cudaMemDiscardBatchAsync(
    void **dptrs,
    size_t *sizes,
    size_t count,
    unsigned long long flags,
    cudaStream_t stream);
```
该函数对 `dptrs` 和 `sizes`数组中指定的地址范围执行一批内存 discard 操作。这两个数组的长度必须与 `count`指定的数量一致。每一个内存区域都必须引用通过 `cudaMallocManaged`分配的 Managed Memory，或者引用通过 `__managed__`声明的变量。

`cudaMemDiscardAndPrefetchBatchAsync`API 将 discard和 prefetch两种操作组合在一起。调用 `cudaMemDiscardAndPrefetchBatchAsync`在语义上等价于：
1. 调用 `cudaMemDiscardBatchAsync`
2. 然后调用 `cudaMemPrefetchBatchAsync`
但其实现更加高效（more optimal）。
当应用程序希望内存迁移到目标位置，但并不需要保留原有内容时，该接口非常有用。
```cpp
cudaError_t cudaMemDiscardAndPrefetchBatchAsync(
    void **dptrs,
    size_t *sizes,
    size_t count,
    struct cudaMemLocation *prefetchLocs,
    size_t *prefetchLocIdxs,
    size_t numPrefetchLocs,
    unsigned long long flags,
    cudaStream_t stream);
```
其中：
- **`prefetchLocs`** 数组用于指定各次预取（prefetch）的目标位置（destination）。
- **`prefetchLocIdxs`** 用于指出每个目标位置对应哪些 discard/prefetch 操作。
例如：如果一个 batch 中共有 **10 个操作**，其中：
- 前 **6 个**需要预取到一个位置；
- 后 **4 个**需要预取到另一个位置；
则：
- **`numPrefetchLocs`** 的值应为 **2**；
- **`prefetchLocIdxs`** 应为 **`{0, 6}`**；
- **`prefetchLocs`** 中保存两个对应的目标位置。
 需要注意的重要事项（Important considerations）
- 如果在 discard 之后，没有执行新的写入（write）或预取（prefetch）便读取该内存区域，将返回不确定值（indeterminate value）。
- 可以通过向该区域重新写入数据，或调用 `cudaMemPrefetchAsync` 对其进行预取，来撤销（undo）discard 操作。
- 如果在 discard 操作执行期间，同时发生读（read）、写（write）或预取（prefetch），则会导致未定义行为（undefined behavior）。
- 所有设备都必须满足属性 `cudaDevAttrConcurrentManagedAccess` 的值非零（non-zero）。
## 4.4 查询托管内存上的数据使用属性
程序可以使用下面的 API，查询通过 `cudaMemAdvise`或 `cudaMemPrefetchAsync` 为 CUDA Managed Memory设置的内存区域属性。
```cpp
cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);
```

该函数用于查询从 `devPtr`开始、大小为`count`字节的内存区域的某一项属性。该内存区域必须引用通过 `cudaMallocManaged`分配的 Managed Memory，或者引用通过 `__managed__`声明的变量，可以查询以下属性：

`cudaMemRangeAttributeReadMostly`
如果整个内存区域都设置了 `cudaMemAdviseSetReadMostly`属性，则返回1；否则返回 0。

`cudaMemRangeAttributePreferredLocation`
如果整个内存区域都将同一个处理器设置为了首选位置（preferred location），则返回对应 GPU 的设备 ID或 `cudaCpuDeviceId`。否则返回 `cudaInvalidDeviceId`。
应用程序可以根据 Managed Pointer 的 Preferred Location属性，决定是否通过 CPU或 GPU来进行数据中转（staging）。需要注意的是，查询时内存区域的实际驻留位置可能与首选位置不同。

`cudaMemRangeAttributeAccessedBy`
返回对该内存区域设置了 `cudaMemAdviseSetAccessedBy` 建议的所有设备列表。

`cudaMemRangeAttributeLastPrefetchLocation`
返回该内存区域最近一次通过 `cudaMemPrefetchAsync`显式预取的目标位置。需要注意的是，该属性仅返回应用程序最后一次请求预取到的位置。它并不能说明：
- 该预取操作是否已经完成；
- 甚至不能说明该预取操作是否已经开始执行。

`cudaMemRangeAttributePreferredLocationType`返回首选位置的类型，其返回值可能为：
- `cudaMemLocationTypeDevice`
    如果内存区域中的所有页面（pages）都将同一块 GPU作为首选位置。
- `cudaMemLocationTypeHost`
    如果内存区域中的所有页面都将 CPU作为首选位置。
- `cudaMemLocationTypeHostNuma`
    如果内存区域中的所有页面都将同一个 Host NUMA 节点作为首选位置。
- `cudaMemLocationTypeInvalid`
    如果：
    - 所有页面没有相同的首选位置；或者
    - 某些页面根本没有设置首选位置。

`cudaMemRangeAttributePreferredLocationId`
如果同一地址范围查询 `cudaMemRangeAttributePreferredLocationType` 返回的是 `cudaMemLocationTypeDevice`，则返回对应 GPU 的设备编号。如果首选位置类型为Host NUMA，则返回对应的Host NUMA 节点 ID。否则，应忽略该返回值。

`cudaMemRangeAttributeLastPrefetchLocationType`
返回整个内存区域最近一次通过 `cudaMemPrefetchAsync`显式预取到的位置类型。其返回值可能为：
- `cudaMemLocationTypeDevice`
    如果内存区域中的所有页面都被预取到同一块 GPU。
- `cudaMemLocationTypeHost`
    如果内存区域中的所有页面都被预取到CPU。
- `cudaMemLocationTypeHostNuma`
    如果内存区域中的所有页面都被预取到同一个 Host NUMA 节点。
- `cudaMemLocationTypeInvalid`
    如果：
    - 所有页面并没有预取到同一个位置；或者
    - 某些页面从未进行过预取。

`cudaMemRangeAttributeLastPrefetchLocationId`
如果同一地址范围查询 `cudaMemRangeAttributeLastPrefetchLocationType`返回的是 `cudaMemLocationTypeDevice`，则该值为一个有效的 GPU 设备编号。如果返回的是 `cudaMemLocationTypeHostNuma`，则该值为一个有效的 Host NUMA 节点 ID。否则，应忽略该返回值。此外，还可以使用对应的 `cudaMemRangeGetAttributes` 函数，一次查询多个属性。
# 4.5 GPU超额分配
统一内存使应用程序能够超额使用任意单个处理器的内存。换句话说，应用程序可以分配并共享容量超过系统中任意单个处理器内存容量的数组。
这一特性使得应用程序能够在无需显著增加编程模型复杂性的情况下，实现包括对无法完全装入单块 GPU 内存的数据集进行外存（out-of-core）处理等功能。