原文链接：https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
# 1.异步并发执行
CUDA的异步并发机制允许主机计算、设备计算以及各类数据转移（包含主机与设备间H2D/D2H、设备内部D2D及设备间P2P的内存传输）等多种任务实现时间上的重叠执行；其核心技术依赖于非阻塞的异步接口设计——即分发函数调用或核函数启动后会立即返回，使得应用程序无需等待底层操作开始或结束即可并发处理其他工作，而仅在需要提取该异步操作的最终结果时，才必须引入同步机制以确保执行的完整性。这种将内存传输过程与实际计算任务高度重叠的并发模式，是CUDA编程中用以大幅减少甚至彻底消除底层数据移动开销、实现极致性能的最关键架构准则。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260404165132.png)
在 CUDA 编程中，异步分发的任务（如核函数启动）通常会立即返回，程序需借由同步机制来获取最终计算结果。CUDA 提供了三种主要的同步范式：第一是**阻塞式（Blocking）**，应用程序挂起并等待设备任务彻底完成，典型代表为 `cudaDeviceSynchronize()` 函数，它会强制同步全局所有已下发的任务；第二是**非阻塞/轮询式（Non-blocking/Polling）**，应用程序通过非阻塞调用立即获取操作的当前状态，从而允许 CPU 继续执行其他逻辑；第三是**回调式（Callback）**，通过在主机端注册回调函数，在设备操作完成后自动触发对应逻辑。需要指出的是，程序实际能达到的并发执行能力，受限于运行时的 CUDA 版本以及硬件的计算能力。

支撑上述复杂并发逻辑与同步行为的核心 API 抽象组件是 **CUDA 流（CUDA Streams）** 和 **CUDA 事件（CUDA Events）**。这二者构成了开发者表达和控制指令排队、依赖等待以及多流重叠（Overlapping）的基础设施，贯穿了大部分并发优化的生命周期。

针对高频、固定模式的异步操作集合，CUDA 提供了 **CUDA 图（CUDA Graphs）** 作为进阶优化手段。它允许开发者通过**流捕获（Stream Capture）** 等机制，将复杂的异步任务及其依赖拓扑预先定义并实例化为一整张图。在后续调用中，只需启动该图即可重复执行，从而最大程度降低或消除 CPU 侧频繁发号施令带来的 API 调用开销（Overhead）。
# 2.CUDA流
CUDA 流（CUDA Stream）是实现指令排队与异步执行的最基础抽象组件，其运行机制类似于一个先进先出（FIFO）的工作队列。开发者可将内存拷贝或核函数（Kernel）启动等任务推入流中。**在单一流的内部，所有操作将严格遵循其入队（Enqueue）的先后顺序依次串行执行**，确保了单流内严格的数据依赖安全。更重要的是，被推入流中的 API 调用和任务启动相对于主机线程（CPU）均是完全异步的，从而实现了 CPU 与 GPU 的彻底解耦。

为突破单兵作战的性能瓶颈，应用程序允许同时创建并激活多个流。在多流架构下，CUDA 运行时系统（Runtime）会根据 GPU 底层硬件资源（如计算单元或拷贝引擎）的实际空闲状态，动态调度并重叠执行各个流中的就绪任务。虽然开发者在创建流时可以赋予其**优先级（Priority）**，但这仅仅是向底层调度器提供的一种**“提示（Hint）”** ：它既不能保证绝对的执行先后顺序，也不具备抢占（Preempt）其他已在执行任务的能力。

系统中存在一个无处不在的 **“默认流（Default Stream）”**（隐式的 0 号流）。当开发者在发起任何内存操作或核函数启动而未显式指定流参数时，这些指令都会被自动排入该默认流中。由于该流往往携带特殊的同步语义，随意使用极易破坏程序的整体并发度。最后，为了应对流的非阻塞异步特性，当 CPU 需要消费流的最终计算结果时，必须主动发起强制同步——可以通过等待特定流的任务彻底清空（如 `cudaStreamSynchronize`），或者在全局设备层级进行拦截（如 `cudaDeviceSynchronize`），以确保所需数据已安全落地。
## 2.1 创建/销毁流
可以使用cudaStreamCreate()函数来创建CUDA流
```cpp
cudaStream_t stream;        // Stream handle
cudaStreamCreate(&stream);  // Create a new stream

// stream based operations ...

cudaStreamDestroy(stream);  // Destroy the stream
```
如果应用程序在流（stream）中仍有任务执行时调用 `cudaStreamDestroy()`，该流将在销毁前完成其中的所有工作。
## 2.2 流中启动核函数
通常用于启动核函数的三重尖括号语法也可用于将核函数启动到指定的流中。流作为额外的参数在核函数启动时指定。在以下示例中，名为kernel的核函数被启动到流中，其句柄为stream，类型为cudaStream_t，并且假设之前已经创建：
```cpp
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```
内核启动是异步的，函数调用会立即返回。假设内核启动成功，内核将在指定的流中执行，同时应用程序可以自由地在CPU上执行其他任务，或在GPU的其他流上执行操作，而无需等待内核执行完成。
## 2.3 流中启动内存传输
要将内存传输启动到流中，我们可以使用函数cudaMemcpyAsync()。此函数与cudaMemcpy()函数类似，但它需要一个额外的参数来指定用于内存传输的流。下面代码块中的函数调用从src指向的主机内存复制size字节到dst指向的设备内存，并在stream指定的流中执行。
```cpp
// Copy `size` bytes from `src` to `dst` in stream `stream`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```
与其他异步函数调用类似，此函数调用会立即返回，而cudaMemcpy()函数则会阻塞，直到内存传输完成。为了安全地访问传输结果，应用程序必须通过某种形式的同步机制来确定操作已经完成。

在 CUDA 架构中，若要利用 `cudaMemcpyAsync()` 实现主机与设备间真正的异步数据传输，其对应的主机端缓冲区必须被分配为**锁页内存（Pinned and Page-locked Memory）**。若开发者错误地传入了通过常规方式分配的普通可分页内存，虽然函数仍能正确执行且不报错，但底层驱动会为保证内存安全，将传输过程隐式退化为由 CPU 介入的同步阻塞模式。这种隐式退化将彻底破坏数据传输与核函数计算相重叠的并发流水线，导致异步带来的性能收益完全丧失。因此，CUDA 高性能编程的强制规范是：所有需要与 GPU 进行直接异步数据交互的主机内存，必须使用 **`cudaMallocHost()`** 函数进行分配，以确保底层 DMA 引擎能够独立、安全地接管异步搬运任务。
## 2.4 流同步
与流同步的最简单方法是等待流中任务清空。这可以通过两种方式实现：使用cudaStreamSynchronize()函数或cudaStreamQuery()函数。cudaStreamSynchronize()函数会阻塞，直到流中的所有工作完成。
```cpp
// Wait for the stream to be empty of tasks
cudaStreamSynchronize(stream);

// At this point the stream is done
// and we can access the results of stream operations safely
```
如果我们不想阻塞，而只是需要快速检查流是否为空，可以使用cudaStreamQuery()函数。
```cpp
// Have a peek at the stream
// returns cudaSuccess if the stream is empty
// returns cudaErrorNotReady if the stream is not empty
cudaError_t status = cudaStreamQuery(stream);

switch (status) {
    case cudaSuccess:
        // The stream is empty
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // The stream is not empty
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // An error occurred - we should handle this
        break;
};
```
# 3.CUDA事件
CUDA事件（CUDA Events）是异步并发编程中用于在CUDA流（Stream）内插入执行标记（Markers）的核心底层机制。在传统的流式控制中，应用程序只能粗粒度地感知“整个流是否已清空”；而CUDA事件的引入，使得开发者能够以极高的精度监控流内特定任务（如核函数或内存拷贝）的执行节点，这是构建高效并发流水线的重要前提。

事件机制最具价值的应用在于实现局部任务的安全阻塞与依赖解锁。例如，当两个核函数（Kernel）先后被推入同一流时，若外部操作仅依赖第一个核函数的输出，开发者可在这两个核函数之间精准插入一个CUDA事件。外部程序只需等待该特定事件到达流的前端（即代表第一个核函数已完成），即可安全启动后续操作，而完全无需等待第二个核函数执行完毕。这种通过事件建立的操作与流之间的精密依赖网络，在逻辑上直接映射为有向无环图，构成了进阶优化特性“CUDA图（CUDA Graphs）”的底层理论与运作基石。

除了作为同步与依赖控制的屏障，CUDA事件还在运行时系统中隐式记录了精确的时间戳信息。这一特性使其成为性能分析（Profiling）的标准工具，被广泛应用于精准测量指定核函数的执行耗时以及底层内存传输的物理延迟。
## 3.1 创建/销毁事件
```cpp
cudaEvent_t event;

// Create the event
cudaEventCreate(&event);

// do some work involving the event

// Once the work is done and the event is no longer needed
// we can destroy the event
cudaEventDestroy(event);
```
## 3.2 向CUDA流中插入事件
```cpp
cudaEvent_t event;
cudaStream_t stream;

// Create the event
cudaEventCreate(&event);

// Insert the event into the stream
cudaEventRecord(event, stream);
```
## 3.3 CUDA流中的计时操作
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start;
cudaEvent_t stop;

// create the events
cudaEventCreate(&start);
cudaEventCreate(&stop);

 // record the start event
cudaEventRecord(start, stream);

// launch the kernel
kernel<<<grid, block, 0, stream>>>(...);

// record the stop event
cudaEventRecord(stop, stream);

// wait for the stream to complete
// both events will have been triggered
cudaStreamSynchronize(stream);

// get the timing
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

// clean up
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```
## 3.4 检查CUDA事件的状态
与检查流状态的情况类似，我们可以通过阻塞或非阻塞的方式检查事件状态。

`cudaEventSynchronize()`函数会一直阻塞，直到事件完成。在下面的代码片段中，我们先将一个内核启动到流中，接着记录一个事件，然后启动第二个内核。我们可以使用 `cudaEventSynchronize()`函数等待第一个内核后的事件完成，从而在原则上可以立即启动一个依赖任务——甚至可能在 `kernel2`完成之前就启动。
```cpp
cudaEvent_t event;
cudaStream_t stream;

// create the stream
cudaStreamCreate(&stream);

// create the event
cudaEventCreate(&event);

// launch a kernel into the stream
kernel<<<grid, block, 0, stream>>>(...);

// Record the event
cudaEventRecord(event, stream);

// launch a kernel into the stream
kernel2<<<grid, block, 0, stream>>>(...);

// Wait for the event to complete
// Kernel 1 will be  guaranteed to have completed
// and we can launch the dependent task.
cudaEventSynchronize(event);
dependentCPUtask();

// Wait for the stream to be empty
// Kernel 2 is guaranteed to have completed
cudaStreamSynchronize(stream);

// destroy the event
cudaEventDestroy(event);

// destroy the stream
cudaStreamDestroy(stream);
```
CUDA事件可以通过`cudaEventQuery()`函数以非阻塞的方式检查是否完成。在下面的示例中，我们向一个流（stream）中启动了两个内核函数。第一个内核函数`kernel1`生成了一些数据，我们希望将这些数据复制到主机（host），但同时也有一些CPU端的工作需要处理。在下面的代码中，我们首先将`kernel1`、一个事件（event）以及`kernel2`依次放入流`stream1`中。随后，我们进入一个CPU工作循环，但会偶尔检查事件是否已完成，这表示`kernel1`已执行完毕。如果是，我们就启动一个从主机到设备（device）的复制操作，并将其放入流`stream2`中。这种方法可以实现CPU工作、GPU内核执行以及设备到主机数据传输的重叠（即并发执行）。
```cpp
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;

size_t size = LARGE_NUMBER;
float *d_data;

// Create some data
cudaMalloc(&d_data, size);
float *h_data = (float *)malloc(size);

// create the streams
cudaStreamCreate(&stream1);   // Processing stream
cudaStreamCreate(&stream2);   // Copying stream
bool copyStarted = false;

//  create the event
cudaEventCreate(&event);

// launch kernel1 into the stream
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
// enqueue an event following kernel1
cudaEventRecord(event, stream1);

// launch kernel2 into the stream
kernel2<<<grid, block, 0, stream1>>>();

// while the kernels are running do some work on the CPU
// but check if kernel1 has completed because then we will start
// a device to host copy in stream2
while ( not allCPUWorkDone() || not copyStarted ) {
    doNextChunkOfCPUWork();

    // peek to see if kernel 1 has completed
    // if so enqueue a non-blocking copy into stream2
    if ( not copyStarted ) {
        if( cudaEventQuery(event) == cudaSuccess ) {
            cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
            copyStarted = true;
        }
    }
}

// wait for both streams to be done
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// destroy the event
cudaEventDestroy(event);

// destroy the streams and free the data
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```
# 4.流的回调函数
CUDA 提供了允许开发者从设备端流（Stream）的内部调度并触发主机端（CPU）函数的机制。目前用于实现此功能的核心API是 **`cudaLaunchHostFunc()`**，函数的签名要求传入三个关键参数：目标流的句柄（`stream`）、满足特定签名的主机端回调函数指针（`func`，其签名必须为 `void hostFunction(void *data)`），以及传递给该回调函数的自定义数据结构指针（`data`）。
```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```
在利用上述机制编写主机端回调函数时，存在一个最为关键的系统级约束：**在回调函数的执行体内部，严禁调用任何 CUDA API**。违反此约束可能导致未定义行为或系统级死锁，这是由于回调函数本身是由 CUDA 运行时（Runtime）的基础线程进行调度的。
```cpp
void hostFunction(void *data);
```
# 5.异步错误处理
在CUDA流中，错误可能源自流中的任何操作，包括内核启动和内存传输。这些错误在运行时可能不会立即传播回用户，直到流被同步为止，例如通过等待事件或调用cudaStreamSynchronize()。有两种方法可以检测流中可能发生的错误：

• 使用函数cudaGetLastError()——此函数返回并清除当前上下文中任意流遇到的最后一个错误。如果两次调用之间没有发生其他错误，紧接着第二次调用cudaGetLastError()将返回cudaSuccess。

• 使用函数cudaPeekAtLastError()——此函数返回当前上下文中的最后一个错误，但不会清除它。

这两个函数均以cudaError_t类型的值返回错误。可使用函数cudaGetErrorName()和cudaGetErrorString()生成错误的可打印名称。
```cpp
// Some work occurs in streams.
cudaStreamSynchronize(stream);

// Look at the last error but do not clear it
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
}

// Look at the last error and clear it
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err2));
    printf("Error description: %s\n", cudaGetErrorString(err2));
}

if (err2 == err) {
    printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}

// Check again
cudaError_t err3 = cudaGetLastError();
if (err3 == cudaSuccess) {
    printf("As expected, cudaGetLastError() cleared the error\n");
}
```
## 6.阻塞流、非阻塞流、默认流
在CUDA中有两种流：阻塞流和非阻塞流。这个名称可能有点误导，因为阻塞和非阻塞的语义仅指**这些流与默认流的同步方式** ，通俗说：两个阻塞流之间可以互相并发，但阻塞流和默认流之间不能，具体见6.1
默认情况下，使用cudaStreamCreate()创建的流是阻塞流。要创建非阻塞流，必须使用cudaStreamCreateWithFlags()函数并指定cudaStreamNonBlocking标志，并且非阻塞流可以通过调用cudaStreamDestroy()函数以常规方式进行销毁。
```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```
## 6.1 默认流
阻塞流与非阻塞流之间的关键区别在于它们如何与默认流同步。CUDA 提供了一个传统的默认流（也称为 NULL 流或流 ID 为 0 的流），当在内核启动或阻塞式 cudaMemcpy() 调用中未指定流时会使用此默认流。这个在所有主机线程之间共享的默认流是一种阻塞流。当一个操作被启动到该默认流中时，它将与所有其他阻塞流同步，换句话说，它会等待所有其他阻塞流完成后才能执行。
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```
默认流行为意味着，在上述代码片段中，即便三个内核在理论上可以并发执行，kernel2仍会等待kernel1完成，而kernel3会等待kernel2完成。通过创建非阻塞流，我们可以避免这种同步行为。在下面的代码片段中，我们创建了两个非阻塞流。默认流将不再与这些流同步，理论上三个内核可以并发执行。因此，我们无法假设内核的执行顺序，需要进行显式同步（例如通过代价较高的cudaDeviceSynchronize()调用）以确保内核已完成执行。
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```
## 6.2 多个默认流
自CUDA 7版本起，CUDA允许每个主机线程拥有各自独立的默认流，而非共享传统的默认流。要启用此功能，必须使用nvcc编译器的`--default-stream per-thread`选项，或定义预处理器宏`CUDA_API_PER_THREAD_DEFAULT_STREAM`。启用该功能后，每个主机线程将拥有独立的默认流，这些流不会像传统默认流那样与其他流进行同步。在这种情况下，传统默认流的示例将表现出与非阻塞流示例相同的同步行为。
# 7.高级异步机制简介
## 7.1 流的优先级
如前所述，开发者可以为 CUDA 流分配优先级。需要优先处理的流必须通过函数 `cudaStreamCreateWithPriority()`来创建。此函数包含两个参数：流句柄和优先级。通常，数值越小表示优先级越高。可通过函数 `cudaDeviceGetStreamPriorityRange()`查询指定设备与上下文支持的优先级范围。流的默认优先级为 0。
```cpp
int minPriority, maxPriority;

// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the legacy default stream
// One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);  // Lowest priority
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);  // Highest priority
```
流的优先级只是对运行时的一个提示，主要适用于内核启动操作，而内存传输可能不会遵循该优先级。流优先级不会抢占已在执行的工作，也不保证任何特定的执行顺序。其核心价值在于当 GPU 资源（如 SM 单元）处于满载竞争状态时，为硬件调度器提供决策权重偏移，确保高优先级流的线程块能优先获得新释放的硬件槽位，从而有效压低关键任务的排队延迟（Queuing Latency）并降低执行时间的波动。
## 7.2 CUDA图
在 CUDA 编程中，流（Streams）用于规定一系列操作（如核函数执行或内存拷贝）的严格顺序。当开发者结合多流机制与 **`cudaStreamWaitEvent`** 函数建立跨流的执行依赖时，应用程序便能构建出一个定义明确的**有向无环图（DAG）**。然而，当特定的操作序列或完整的 DAG 需要在程序生命周期中被成百上千次地重复执行时，每次均由主机端（CPU）线程重新下发 API 链条会引发严重的指令延迟与 CPU 开销，从而限制了系统的整体吞吐量。

为彻底消除上述重复调用的开销，CUDA 引入了 **CUDA 图** 特性。该机制的核心思想是“一次定义，多次高效执行”。通过采用**流捕获（Stream capture）** 等技术手段，应用程序能够将原本分散的 API 调用固化为一张逻辑图。这使得 CPU 不再需要在每次循环时逐一下发指令，而是直接把整张图作为一个整体提交给 GPU，从而大幅缩减了主机的性能损耗。

CUDA 图的实际运作严格遵循以下三个阶段：
1. **捕获/创建（Capture）**：这是图逻辑的录制阶段。通常在图首次被执行时，应用程序会启动捕获机制记录下所有的操作路径（除流捕获外，也可使用 CUDA Graph API 进行手动组装）。此步骤仅需执行一次。
2. **实例化（Instantiate）**：这是图的预编译与准备阶段。在捕获完成后，系统会进行一次性的实例化操作，提前在底层配置好执行该图所需的所有运行时硬件和内存结构，确保后续组件的启动速度达到物理极限。
3. **执行（Execute）**：这是图的高效重播阶段。在后续步骤中，预先实例化好的图可以根据需求被无限次触发。由于繁琐的运行时结构配置已在实例化阶段完成，此时启动整张图的 CPU 开销被降至最低水平。
```cpp
#define N 500000 // tuned such that kernel takes a few microseconds

// A very lightweight kernel
__global__ void shortKernel(float * out_d, float * in_d){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) out_d[idx]=1.23*in_d[idx];
}

bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

// The graph will be executed NSTEP times
for(int istep=0; istep<NSTEP; istep++){
    if(!graphCreated){
        // Capture the graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Launch NKERNEL kernels
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }

        // End the capture
        cudaStreamEndCapture(stream, &graph);

        // Instantiate the graph
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated=true;
    }

    // Launch the graph
    cudaGraphLaunch(instance, stream);

    // Synchronize the stream
    cudaStreamSynchronize(stream);
}
```
