本节将介绍更高级的 CUDA API 与特性。本节所讨论的技术或特性，通常不需要修改 CUDA 内核代码，但依然可以在主机侧（host-side）、应用层面影响 GPU 任务执行、性能表现，同时也会影响 CPU 侧性能。
# 一、cudaLaunchKernelEx
在 CUDA 早期版本引入三尖括号语法（`<<<>>>`）时，内核的启动配置（Kernel Configuration）仅有 4 个可编程参数：
- 线程块维度（thread block dimensions）
- 网格维度（grid dimensions）
- 动态共享内存（可选；未指定时为 0）
- 流（stream；未指定时使用默认流）
部分 CUDA 特性可以受益于在内核启动时额外传入的属性与提示信息。 `cudaLaunchKernelEx` 允许程序通过 `cudaLaunchConfig_t` 结构体设置上述执行配置参数。除此之外，`cudaLaunchConfig_t` 结构体还支持传入一个或多个 `cudaLaunchAttribute`，用于对内核启动的其他行为进行控制或给出性能提示。

例如：
- `cudaLaunchAttributePreferredSharedMemoryCarveout`（后续小节会讲解），需要通过 `cudaLaunchKernelEx` 来指定；
- `cudaLaunchAttributeClusterDimension` 属性（本章后续讲解），用于指定本次内核启动期望的集群大小。
# 二、启动簇（cluster）
前文介绍的**线程块簇（Thread block clusters）**，是计算能力 9.0 及以上硬件才支持的可选线程块组织层级。它可以保证：同一个簇内的所有线程块，会在**同一个 GPC 上同时执行**。这就允许规模超过单个 SM 容纳上限的线程组，互相交换数据并执行同步操作。

[cuda c++](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-2%EF%BC%9ACUDA%20C%2B%2B-ru-men.html#6.%E7%BA%BF%E7%A8%8B%E5%9D%97%E9%9B%86%E7%BE%A4)第六小节展示了如何使用三尖括号语法来声明并启动使用簇特性的内核。在该小节中，使用`__cluster_dims__`注解指定启动内核时必须采用的簇维度。如果使用三尖括号语法启动，簇的大小是**隐式确定**的。
## 2.1 使用 `cudaLaunchKernelEx` 启动带簇特性的内核
和使用三尖括号语法启动簇内核不同：采用本 API 时，**每次启动都可以单独配置线程块簇的大小**。下方代码示例演示如何使用`cudaLaunchKernelEx`启动簇内核。
```cpp
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```
有两类和线程块簇（thread block clusters）相关的 `cudaLaunchAttribute`： `cudaLaunchAttributeClusterDimension` 与 `cudaLaunchAttributePreferredClusterDimension`。

属性 ID `cudaLaunchAttributeClusterDimension` 用于指定**必需的簇维度**，内核将按照该维度执行簇。该属性的值 `clusterDim` 是一个三维值。网格（grid）对应的 x、y、z 维度，**必须能被所指定簇维度的对应分量整除**。设置该属性的效果，类似于在内核定义时加上编译期属性 `__cluster_dims__`（参见《使用三尖括号语法启动簇》小节），但它的优势是：**同一个内核，在多次启动时，可以在运行时修改簇维度**。

对于计算能力 10.0 及以上的 GPU，另一属性 ID `cudaLaunchAttributePreferredClusterDimension` 允许应用额外指定**首选簇维度**。首选簇维度必须是内核上 `__cluster_dims__` 属性，或是传给 `cudaLaunchKernelEx` 的 `cudaLaunchAttributeClusterDimension` 所规定的**最小簇维度的整数倍**。也就是说：除首选簇维度之外，必须额外指定一个最小簇维度。网格（grid）对应的 x、y、z 维度，也必须能被该首选簇维度的对应分量整除。

所有线程块都会以至少等于最小簇维度的簇来执行。在条件允许时，会采用首选簇维度，但不保证所有簇都一定按照首选簇维度执行。所有线程块的簇大小只会是最小簇维度或者首选簇维度二者之一。使用首选簇维度的内核，必须保证：无论是在最小簇维度还是首选簇维度下运行，内核逻辑都能正确工作。
## 2.2 以簇为单位定义线程块
当内核使用 `__cluster_dims__` 注解定义时，网格内簇的数量是隐式确定的，可以用网格总大小除以指定的簇大小计算得到。
```cpp
__cluster_dims__((2, 2, 2)) __global__ void foo();
// 8x8x8 clusters each with 2x2x2 thread blocks.
foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
```
在上例中，内核启动时网格包含 16×16×16 个线程块，这等价于使用了 8×8×8 个簇。

内核还可以使用另一种注解 `__block_size__`，在内核定义阶段同时指定所需的线程块大小与簇大小。使用该注解后，三尖括号启动语法里传入的网格维度，**不再代表线程块数量，而是代表簇的数量**，如下所示。

```cpp
// Implementation detail of how many threads per block and blocks per cluster
// is handled as an attribute of the kernel.
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();
// 8x8x8 clusters.
foo<<<dim3(8, 8, 8)>>>();
```
`__block_size__` 需要传入两个三元组参数。第一个三元组表示线程块维度，第二个三元组代表簇大小；如果省略第二个三元组，则默认取值为`(1,1,1)`。在内核启动时如果要指定动态共享内存大小、或者指定流，那么三尖括号`<<<>>>`中的第二个参数必须填写占位值`1`；填写其他数值会导致未定义行为。

注意：不能同时在 `__block_size__` 和 `__cluster_dims__` 中设置第二个三元组；同时，`__block_size__` 搭配空的 `__cluster_dims__` 也是非法写法。当指定了 `__block_size__` 的第二个三元组时，代表启用 “以簇为单位定义线程块（Blocks as Clusters）” 特性，编译器会把`<<<>>>`内的第一个参数解读为**簇的数量**，而不是线程块数量。
# 三、stream与event的更多细节
CUDA 流章节介绍了 CUDA 流的基础概念。默认情况下，提交到同一个 CUDA 流上的操作是串行执行的：后一个操作必须等待前一个操作完成后才能开始执行。唯一例外是新引入的**可编程依赖启动与同步（Programmatic Dependent Launch and Synchronization）** 特性。使用多个 CUDA 流是实现并发执行的一种方式；另一种方式是使用 CUDA 图。这两种方案也可以组合使用。

提交至不同 CUDA 流的任务，在满足特定条件时可以并发执行，例如：不存在事件依赖、不存在隐式同步、GPU 资源充足等。

若在两个来自不同 CUDA 流的独立操作之间，提交了任何 NULL 流上的 CUDA 操作，那么这两个独立操作就无法并发执行；除非这些流是非阻塞 CUDA 流。这类流通过运行时 API `cudaStreamCreateWithFlags()`，搭配`cudaStreamNonBlocking`标志创建。为提升 GPU 任务并发执行的潜力，推荐用户创建非阻塞 CUDA 流。

同时建议用户选择**满足需求的最低粒度同步原语**。举例来说，如果仅需要 CPU 阻塞等待某一个指定 CUDA 流上的全部任务完成，应当优先使用针对该流的`cudaStreamSynchronize()`，而非`cudaDeviceSynchronize()`；后者会不必要地等待该设备上所有 CUDA 流中的 GPU 任务全部结束。如果需要 CPU 等待但不阻塞，则可以在轮询循环中调用`cudaStreamQuery()`并检查返回值。

借助 CUDA 事件（CUDA Events）也可以实现类似的同步效果。例如：在该流上记录事件，调用`cudaEventSynchronize()`以阻塞方式等待该事件捕获的任务完成。同样，该方式相比`cudaDeviceSynchronize()`更优、作用范围更精准。调用`cudaEventQuery()`并在轮询循环中检查返回值，则是对应的非阻塞方案。

如果同步操作处于应用程序的**关键路径**上，同步方式的选择就尤为重要。表 4 从宏观层面总结了主机端各类同步方案。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260927135328.png)
若要在多个 CUDA 流之间实现同步（即表达任务依赖关系），推荐使用**非计时型 CUDA 事件**，详见 CUDA 事件章节。开发者可以调用`cudaStreamWaitEvent()`，让某个流后续提交的所有操作，等待一个预先记录完成的事件（例如，该事件记录在另一个流上）。需要注意：对于任何等待或查询事件状态的 CUDA API，开发者必须保证已经提前调用过`cudaEventRecord`；未经记录的事件，查询时会始终返回成功。

CUDA 事件默认会携带计时信息，供`cudaEventElapsedTime()`接口调用。但如果 CUDA 事件仅用于表达流之间的任务依赖，则不需要计时信息。这类场景下，推荐创建**禁用计时信息**的事件以提升性能。可以通过`cudaEventCreateWithFlags()`接口搭配`cudaEventDisableTiming`标志实现。
## 3.1 流的优先级
流的相对优先级可在创建流时通过`cudaStreamCreateWithPriority()`指定。可用优先级范围按【最高优先级，最低优先级】顺序，可调用`cudaDeviceGetStreamPriorityRange()`获取。运行时，GPU 调度器会参考流优先级来决定任务的执行顺序，但这些优先级仅为**提示（hint）**，而非硬性保证。

在挑选待启动任务时，高优先级流中的待执行任务会优先于低优先级流任务。高优先级任务不会抢占已经正在运行的低优先级任务。GPU 不会在任务执行过程中重新评估任务队列；提升某个流的优先级，也无法中断正在执行中的任务。流优先级仅对任务执行产生倾向性影响，并不强制严格的执行顺序。开发者可以利用流优先级来调控任务执行，但不能依赖它获得严格的顺序保证。

下面的代码示例将获取当前设备支持的优先级范围，并创建两个非阻塞 CUDA 流，分别赋予设备可用的最高优先级与最低优先级。
```cpp
// get the range of stream priorities for this device
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, greatestPriority));
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, leastPriority);
```
## 3.2 显式同步
如前文所述，流之间存在多种同步方式。下文介绍不同粒度下的常用方法：
- **cudaDeviceSynchronize()**：阻塞等待，直到所有主机线程的所有流中，此前提交的全部命令都执行完毕。
- **cudaStreamSynchronize()**：接收一个流作为参数，阻塞等待该流内所有先前提交的命令执行完成。可用于让主机与指定流完成同步，设备上其他流可继续执行。
- **cudaStreamWaitEvent()**：接收一个流与一个事件作为参数（事件相关说明参见 CUDA 事件章节）。调用该函数后，添加到指定流内的后续所有命令，都会延迟执行，直到给定事件完成。
- **cudaStreamQuery()**：为应用程序提供查询接口，判断某个流中先前提交的全部命令是否已经执行完毕。
## 3.3 隐式同步
如果主机线程在来自不同流的两条命令之间，执行了下述任意一种操作，则这两条命令无法并发执行：
- 分配页锁定主机内存
- 分配设备内存
- 设备内存置值操作
- 同一设备内存地址空间内的内存拷贝
- 提交至 NULL 流（默认流）的任意 CUDA 命令
- 在 L1 缓存 / 共享内存的不同配置之间切换
需要做依赖检查的操作包含：待检查核函数所在流内的其他所有命令，以及对该流执行的`cudaStreamQuery()`调用。因此应用程序应当遵循以下准则，最大化内核并发执行的可能性：
- 所有相互独立的操作，应当在存在依赖关系的操作之前提交；
- 任何形式的同步操作，都应当尽可能延后执行。
# 四、可编程依赖内核启动（PDL）
正如前文所述，CUDA 流的语义保证内核按提交顺序执行。 如此一来，若存在两个连续内核，且第二个内核依赖第一个内核的计算结果，开发者可以放心：当第二个内核开始执行时，其所依赖的数据已经就绪。

但存在这样一种场景：第一个内核已经把后续内核所需的数据写入全局内存，而它自身还有剩余工作尚未完成；同理，被依赖的第二个内核，在需要读取第一个内核的数据之前，也可以执行一部分与该数据无关的独立计算。 在这种场景下，**两个内核的执行可以部分重叠**（前提是硬件资源充足）。这种重叠也可以覆盖第二个内核的启动开销。

除硬件资源余量之外，可实现的重叠程度取决于内核的具体结构，例如：
- 第一个内核在执行的哪个阶段，完成第二个内核所依赖的计算；
- 第二个内核在执行的哪个阶段，才开始使用第一个内核输出的数据。

由于该特性高度依赖具体内核逻辑，很难做到完全自动化。因此 CUDA 提供了一种机制，允许应用开发者手动指定两个内核之间的同步点，该技术就叫做**可编程依赖内核启动（Programmatic Dependent Kernel Launch）**，相关时序场景见下图。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260928153115.png)
i. 第一个内核（称为**主内核 primary kernel**）需要调用一个专用函数，标记它已经完成后续依赖内核（也叫**次内核 secondary kernel**）所需的全部数据准备工作。该操作用函数 `cudaTriggerProgrammaticLaunchCompletion()` 实现。

ii. 对应的依赖次内核，需要标记：它已经执行完自身所有与主内核无关的独立计算，现在开始等待主内核完成其所依赖的计算任务。该操作用函数 `cudaGridDependencySynchronize()` 实现。

iii. 启动第二个内核时，需要设置特殊启动属性：`cudaLaunchAttributeProgrammaticStreamSerialization`，并将其字段 `programmaticStreamSerializationAllowed` 设置为 `1`。
```cpp
__global__ void primary_kernel() {
    // Initial work that should finish before starting secondary kernel

    // Trigger the secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
    // Initialization, Independent work, etc.

    // Will block until all primary kernels the secondary kernel is dependent on have
    // completed and flushed results to global memory
    cudaGridDependencySynchronize();

    // Dependent work
}

// Launch the secondary kernel with the special attribute

// Set Up the attribute
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;

// Set the attribute in a kernel launch configuration
 cudaLaunchConfig_t config = {0};

// Base launch configuration
config.gridDim = grid_dim;
config.blockDim = block_dim;
config.dynamicSmemBytes= 0;
config.stream = stream;

// Add special attribute for PDL
config.attrs = attribute;
config.numAttrs = 1;

// Launch primary kernel
primary_kernel<<<grid_dim, block_dim, 0, stream>>>();

// Launch secondary (dependent) kernel using the configuration with
// the attribute
cudaLaunchKernelEx(&config, secondary_kernel);
```
# 五、批量内存传输
在 CUDA 开发中，批量处理是一种常见技术模式。宽泛来讲，批量就是将若干个（通常是小规模的）任务打包合并为单个（通常规模更大的）操作。批次内的子任务不必完全相同，尽管多数场景下它们是一致的。cuBLAS 库提供的批量矩阵乘法就是该思想的典型例子。

一般来说，和 CUDA Graph、PDL 类似，批量处理的目的是降低**逐个提交子任务带来的调度开销**。在内存传输场景下，每次发起内存拷贝都会产生一定的 CPU 与驱动开销。此外，常规的`cudaMemcpyAsync()`函数，就其现有接口形式而言，无法向驱动提供充足信息用于传输优化，例如源地址、目标地址相关的提示信息。 在 Tegra 平台上，开发者可以选择使用流式多处理器（SM）或者拷贝引擎（CE）来执行数据传输。该选择目前由驱动中的启发式策略决定。这一点十分关键：使用 SM 执行传输可能获得更快的拷贝速度，但会占用一部分可用计算资源；反过来，使用 CE 拷贝虽然传输速度可能更慢，但整体应用性能反而更高，因为 SM 可以保留下来执行其他计算任务。

上述考量推动了`cudaMemcpyBatchAsync()`（以及配套的`cudaMemcpyBatch3DAsync()`）接口的设计。这组 API 支持对批量内存传输做优化。除源地址、目标地址指针数组之外，该 API 还可以通过内存拷贝属性，指定：

- 对拷贝执行顺序的预期；
- 源端、目的端存储位置的提示；
- 是否希望让数据传输与计算任务重叠执行（该特性目前仅在带 CE 拷贝引擎的 Tegra 平台上支持）。

我们先来看最简单的用例：将锁页主机内存的数据批量传输至锁页设备内存。
```cpp
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<size_t> sizes(batch_size);

// Allocate the source and destination buffers
// initialize with the stream number
for (size_t i = 0; i < batch_size; i++) {
    cudaMallocHost(&srcs[i], sizes[i]);
    cudaMalloc(&dsts[i], sizes[i]);
    cudaMemsetAsync(srcs[i], sizes[i], stream);
}

// Setup attributes for this batch of copies
cudaMemcpyAttributes attrs = {};
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

// All copies in the batch have same copy attributes.
size_t attrsIdxs = 0;  // Index of the attributes

// Launch the batched memory transfer
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);
```
`cudaMemcpyBatchAsync()` 的前几个参数很容易理解：它们是存放源指针、目标指针以及传输长度的数组，每个数组都必须包含 `batch_size` 个元素。新增的信息来自拷贝属性。该函数需要一个指向属性数组的指针，以及对应的属性索引数组。原则上还可以传入一个 `size_t` 类型数组，用于记录发生失败的传输任务编号；但此处传入 `nullptr` 也是安全的，这种情况下不会记录失败任务的索引。

再来看属性部分。在本示例中，所有传输任务是同构的，因此我们仅使用一套属性，并将其应用于全部传输任务。这一点由属性索引（attrIndex）参数控制。该参数本质上可以是一个数组：数组第`i`个元素代表属性数组第`i`项所作用的第一个传输任务的下标。本例中，attrIndex 被视作只有单个元素的数组，取值为`0`，含义是：`attribute[0]` 将作用于下标从 0 开始的所有传输任务，也就是全部传输。

最后说明：我们将 `srcAccessOrder` 属性设置为 `cudaMemcpySrcAccessOrderStream`。这代表源数据将按照流的提交顺序进行访问。换言之，本次内存拷贝会阻塞，直到流中所有操作这些源、目标指针数据的前置内核全部执行完成。程序必须保证源内存在拷贝完成前始终有效且不被修改。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260928162725.png)

下一个示例，我们将介绍更复杂的异构批量传输场景。
```cpp
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<size_t> sizes(batch_size);

// Allocate the src and dst buffers
for (size_t i = 0; i < batch_size - 10; i++) {
    cudaMallocHost(&srcs[i], sizes[i]);
    cudaMalloc(&dsts[i], sizes[i]);
}

int buffer[10];

for (size_t i = batch_size - 10; i < batch_size; i++) {
    srcs[i] = &buffer[10 - (batch_size - i];
    cudaMalloc(&dsts[i], sizes[i]);
}

// Setup attributes for this batch of copies
cudaMemcpyAttributes attrs[2] = {};
attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
attrs[1].srcAccessOrder = cudaMemcpySrcAccessOrderDuringApiCall;

size_t attrsIdxs[2];
attrsIdxs[0] = 0;
attrsIdxs[1] = batch_size - 10;

// Launch the batched memory transfer
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 2 /*numAttrs*/, nullptr /*failIdx*/, stream);
```
这里包含两类传输任务：`batch_size - 10` 个从锁页主机内存向锁页设备内存的传输，以及 10 个从主机数组向锁页设备内存的传输。此外，该缓冲区数组仅存在于主机端，并且生命周期只在当前作用域内 —— 这类地址被称为**临时指针（ephemeral pointer）**。由于 API 是异步执行的，当 API 调用返回后，该指针可能就已经失效。想要使用这类临时指针执行拷贝，必须将属性中的`srcAccessOrder`设置为`cudaMemcpySrcAccessOrderDuringApiCall`。

本例中我们使用两套属性：第一套属性作用于下标从 0 开始、小于`batch_size-10`的所有传输任务；第二套属性作用于下标从`batch_size-10`开始、小于`batch_size`的所有传输任务。

如果缓冲区数组不是在栈上分配，而是通过`malloc`在堆上分配，那么这块数据就不再是临时数据。内存会保持有效，直到指针被显式释放。在这种场景下，拷贝的最优调度方式取决于系统硬件：
- 如果系统支持硬件托管内存，或者 GPU 可以通过地址转换实现对主机内存的一致性访问，那么推荐使用流顺序模式；
- 如果不支持上述特性，则应立即执行传输，此时属性的`srcAccessOrder`应设置为`cudaMemcpyAccessOrderAny`。

`cudaMemcpyBatchAsync`还允许开发者提供源端与目标端存储位置的提示信息。通过设置`cudaMemcpyAttributes`结构体的`srcLocation`与`dstLocation`字段实现。`srcLocation`和`dstLocation`均为`cudaMemLocation`类型，该结构体包含存储位置类型以及位置 ID。 这个`cudaMemLocation`结构体，和`cudaMemPrefetchAsync()`中用来向运行时提供预取提示的结构体是同一个。下方代码示例将演示，如何配置从设备内存传输到主机指定 NUMA 节点的提示信息。
```cpp
// Allocate the source and destination buffers
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<size_t> sizes(batch_size);

// cudaMemLocation structures we will use tp provide location hints
// Device device_id
cudaMemLocation srcLoc = {cudaMemLocationTypeDevice, dev_id};

// Host with numa Node numa_id
cudaMemLocation dstLoc = {cudaMemLocationTypeHostNuma, numa_id};

// Allocate the src and dst buffers
for (size_t i = 0; i < batch_size; i++) {
    cudaMallocManaged(&srcs[i], sizes[i]);
    cudaMallocManaged(&dsts[i], sizes[i]);

    cudaMemPrefetchAsync(srcs[i], sizes[i], srcLoc, 0, stream);
    cudaMemPrefetchAsync(dsts[i], sizes[i], dstLoc, 0, stream);
    cudaMemsetAsync(srcs[i], sizes[i], stream);
}

// Setup attributes for this batch of copies
cudaMemcpyAttributes attrs = {};

// These are managed memory pointers so Stream Order is appropriate
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

// Now we can specify the location hints here.
attrs.srcLocHint = srcLoc;
attrs.dstlocHint = dstLoc;

// All copies in the batch have same copy attributes.
size_t attrsIdxs = 0;

// Launch the batched memory transfer
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);
```
最后要介绍的是一个标志位，用于提示系统在执行传输时，优先选用 SM（流式多处理器）还是 CE（拷贝引擎）。该配置项位于 `cudaMemcpyAttributes::flags`，可选取值如下：
- `cudaMemcpyFlagDefault` —— 默认行为
- `cudaMemcpyFlagPreferOverlapWithCompute` —— 提示系统优先使用 CE 执行传输，让数据传输与计算任务实现重叠执行。
综上，关于 `cudaMemcpyBatchAsync` 的核心要点总结如下：
- `cudaMemcpyBatchAsync` 函数（及其 3D 版本）支持开发者一次性指定一批内存传输任务，摊薄每一次传输的启动开销。
- 除源指针、目标指针和传输长度之外，该函数可接收一组或多组内存拷贝属性。属性可以提供：待传输内存类型、源指针对应的流顺序行为、源端与目标端存储位置提示，以及是否尽可能让传输与计算重叠、或是使用 SM 执行传输的提示。
- 基于上述全部信息，CUDA 运行时能够尽可能对内存传输做深度优化。
# 六、环境变量
CUDA 提供了多种环境变量（参见 5.2 节），它们能够影响程序执行行为与性能。如果不显式设置，CUDA 会为这些环境变量使用合理的默认值；但部分场景下需要针对性调整，例如用于调试，或是为了提升性能。

举例而言，增大环境变量 `CUDA_DEVICE_MAX_CONNECTIONS` 的取值，有助于降低来自不同 CUDA 流的独立任务因**虚假依赖**被串行化执行的概率。当多个任务复用底层相同资源时，就可能产生这类虚假依赖。建议先使用默认值；仅当出现性能问题时（例如：不同 CUDA 流上相互独立的任务发生意料之外的串行执行，且排除了 SM 资源不足等其他原因），再探究该环境变量带来的影响。值得注意：在开启 MPS（多进程服务）时，该环境变量的默认值会更小。

与之类似，对于延迟敏感型应用，可以将环境变量 `CUDA_MODULE_LOADING` 设置为 `EAGER`。此举可以把模块加载带来的全部开销转移到应用初始化阶段，避开业务关键路径。当前默认模式为延迟模块加载（lazy）。在默认模式下，也可以在应用初始化阶段主动调用各个 Kernel 做 “预热”，强制提前加载模块，达到和立即加载相近的效果。