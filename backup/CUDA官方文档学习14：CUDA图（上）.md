CUDA 图提供了 CUDA 中另一种任务提交模型。一张图由内核启动、数据传输等一系列操作构成，操作之间通过依赖关系相连。**图的定义与图的执行相互分离**：图只需定义一次，之后可以反复提交运行。

将图的定义和执行解耦，可以实现多项优化： 第一，相比流机制，CUDA 图降低了 CPU 提交开销，因为大部分准备工作提前一次性完成； 第二，向 CUDA 暴露完整工作流，能够实现基于流的分段任务提交机制无法做到的优化。

为理解 CUDA 图所能实现的优化，我们先看 CUDA 流的工作方式：当向流中放入一个内核时，主机驱动会执行一系列操作，为该内核在 GPU 上的运行做准备。这些用于配置并启动内核的操作属于开销，**每提交一次内核就要付出一次该开销**。对于执行时长短的 GPU 内核，这类开销会占端到端总执行时间相当大的比重。

如果把需要多次执行的工作流封装为 CUDA 图，那么整套开销仅在图实例化阶段一次性支付；后续重复启动该图时，只产生极低的开销。
# 一、图结构
一项操作构成图中的一个**节点**，操作之间的依赖关系则为**边**。这些依赖约束着各项操作的执行顺序。
一旦某操作所依赖的节点全部执行完成，该操作就可以在任意时刻被调度；调度工作由 CUDA 系统负责。
## 1.1 节点类型
图节点可以是以下类型之一：
- 内核（kernel）
- CPU 函数调用
- 内存拷贝（memory copy）
- 内存置零（memset）
- 空节点（empty node）
- 等待 [CUDA 事件](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)
- 记录 [CUDA 事件](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)
- 对[外部资源互操作对象](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#external-resource-interoperability)发送信号
- 等待[外部资源互操作对象](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#external-resource-interoperability)
- 条件图节点
- 图内存节点
- 子图（child graph）：用于执行独立的嵌套图，如下图所示。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261013162955.png)
## 1.2 边数据
CUDA 12.3 在 CUDA 图中引入了**边数据**。在当前版本，非默认边数据唯一用途是启用[可编程依赖启动（Programmatic Dependent Launch）](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)。

一般而言，边数据用于修改一条边所定义的依赖关系，它由三部分构成：**输出端口（outgoing port）**、**输入端口（incoming port）** 与**类型（type）**。

- 输出端口：指定关联边在何时被触发（在上游节点）
>解释：上游节点可以产生多个触发信号，端口就用来选择其中一个信号，默认端口是0，代表等待整个上游节点完整执行完毕之后，才发出触发信号，也就是传统 CUDA Graph 边的行为；内核节点还可以有额外非0输出端口：内核内部可以在运行中途，主动发射信号到这些端口。上游 kernel 不用等整个 kernel 跑完，在 kernel 内部某一点提前发信号，这个信号从非 0 输出端口输出，驱动下游节点。
- 输入端口：指定节点中的哪一部分逻辑依赖于该关联边（在下游节点）
- 类型：修改边两端点之间的依赖关系。
>解释：用来定义这个信号的含义，以及配套的内存同步规则。其中type=0为默认全量依赖，即上游全部完成后，下游才能启动，就是 CUDA Graph 原始边；还有一种是`cudaGraphDependencyTypeProgrammatic`（可编程依赖类型）专门用于 PDL。不再要求上游节点整体执行完成；当上游内核从指定输出端口发出信号，就可以唤醒下游内核。下游内核**可以提前启动**，先执行不依赖上游数据的前置逻辑；运行到设备内等待点阻塞；收到上游端口信号之后，继续往下执行。

端口取值与节点类型、端口方向强相关；边类型也仅能用于特定节点类型。在所有场景下，**零初始化的边数据代表默认行为**：输出端口 0 代表等待上游整个任务执行完毕；输入端口 0 会阻塞下游整个任务；边类型 0 对应完整依赖，并附带内存同步语义。

在各类图 API 中，可以通过与节点平行的数组，可选地指定边数据。如果作为入参时省略边数据，则采用零初始化数据。如果作为查询类输出参数省略边数据：若被忽略的边数据全部为零初始化，API 调用正常；如果调用会丢失有效信息，则返回`cudaErrorLossyQuery`。
>第一句话中的平行数组指的就是节点数组与边数据数组长度相同，下标一一对应

部分流捕获 API 同样支持边数据：`cudaStreamBeginCaptureToGraph()`、`cudaStreamGetCaptureInfo()`以及`cudaStreamUpdateCaptureDependencies()`。这类场景下下游节点尚不存在，边数据会关联到**悬边（半边，dangling edge）**。悬边后续要么连接到未来捕获到的节点，要么在流捕获结束时被丢弃。 请注意：部分边类型**不需要等待上游节点完全执行完毕**。在判断流捕获是否完全合并回原始流时，这类边不会纳入考量，并且捕获结束时无法被丢弃。详细参见[使用流捕获构建 CUDA 图章节](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture)。

目前没有任何节点类型定义额外输入端口；**只有内核节点可以定义额外输出端口**。 存在一种非默认依赖类型：`cudaGraphDependencyTypeProgrammatic`，用于在两个内核节点之间启用[可编程依赖启动](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization)。
# 二、构建并运行图
基于图的任务提交划分为三个独立阶段：**定义、实例化、执行**。
在**定义（创建）阶段**，程序构建图内各项操作的描述，同时定义操作之间的依赖关系。
**实例化**会对图模板生成快照、完成合法性校验，并对任务执行大部分配置与初始化工作，目的是尽可能减少启动阶段需要执行的操作。实例化产出的对象称为**可执行图（executable graph）**。
可执行图可以像其他任意 CUDA 任务一样提交到流中启动，并且能够反复启动多次，无需重复执行实例化流程。
## 2.1 图构建
图可通过两种方式创建：显式图 API、流捕获。
### 2.1.1 图API
下面给出创建对应图的示例（省略变量声明与其他样板代码）。示例中使用`cudaGraphCreate()`创建图，使用`cudaGraphAddNode()`添加内核节点及其依赖关系。所有用于添加节点与依赖的接口可参考[ CUDA 运行时 API 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html)。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261014094701.png)
```cpp
// 创建一张空的 CUDA 图模板，句柄存入`graph`；第二个参数是保留参数，固定填 0。
// 此时图里没有任何节点与边，只是一个空容器。
cudaGraphCreate(&graph, 0);

// 数组保存4个图节点句柄，nodes[0]~nodes[3]
cudaGraphNode_t nodes[4];
// 节点参数结构体，指定节点类型为内核节点
// 这里kParams：复用同一个配置，实际工程每个节点可以独立配置
cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
// 获取内核函数指针
kParams.kernel.func         = (void *)kernelName;
// 设置kernel的网格维度grid=(1,1,1)，线程块block=(1,1,1)
kParams.kernel.gridDim.x    = kParams.kernel.gridDim.y  = kParams.kernel.gridDim.z  = 1;
kParams.kernel.blockDim.x   = kParams.kernel.blockDim.y = kParams.kernel.blockDim.z = 1;

cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);
cudaGraphAddNode(&nodes[1], graph, &nodes[0], NULL, 1, &kParams);
cudaGraphAddNode(&nodes[2], graph, &nodes[0], NULL, 1, &kParams);
cudaGraphAddNode(&nodes[3], graph, &nodes[1], NULL, 2, &kParams);
```
>cudaGraphAddNode函数原型
>```cpp
>cudaGraphAddNode(cudaGraphNode_t* pNode,
                 cudaGraph_t graph,
                 const cudaGraphNode_t* pDependencies,
                 const cudaGraphEdgeData* pEdgeData,
                 size_t numDependencies,
                 const cudaGraphNodeParams* pNodeParams);
>```

上面的示例展示了四个内核节点，节点之间配置了依赖关系，用以演示如何创建一个极简的图。真实应用场景中，通常还会添加内存操作类节点，例如内存拷贝（memcpy）与内存置零（memset）。如需查阅全部用于添加节点的图 API 函数完整参考，请参见 CUDA 运行时 API 文档。

注意：CUDA 图 API 最初为每一类节点单独提供对应的添加接口，例如`cudaGraphAddKernelNode()`、`cudaGraphAddMemcpyNode()`。这些接口现已被统一的多态接口`cudaGraphAddNode()`取代：该接口通过节点参数结构体中的`type`字段，判断待添加节点的类型。这类非多态接口将在未来版本中被弃用。推荐开发者使用`cudaGraphAddNode()`，作为通过 CUDA 图 API 向图中添加节点的标准方式。
### 2.1.2 流捕获
流捕获提供一种机制，可基于现有的流相关 API 来创建图。一段向流中提交任务的代码（包括已有存量代码），可以用`cudaStreamBeginCapture()`与`cudaStreamEndCapture()`将其包裹，示例见下文。
```cpp
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);
```
调用`cudaStreamBeginCapture()`会将流置于**捕获模式**。当流处于捕获状态时，提交到该流的任务不会入队等待执行，而是追加到一张正在逐步构建的内部图中。随后调用`cudaStreamEndCapture()`可返回这张图，同时结束该流的捕获模式。正在通过流捕获构建中的图称为**捕获图（capture graph）**。
除传统遗留流`cudaStreamLegacy`（即 NULL 流）外，流捕获可作用于任意 CUDA 流；该机制支持`cudaStreamPerThread`（每线程默认流）。若程序使用遗留流，可将 0 号流重定义为每线程流，且不会改变原有功能，相关说明参见[异步执行章节](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream)。
可通过`cudaStreamIsCapturing()`查询某个流是否处于捕获模式。
借助`cudaStreamBeginCaptureToGraph()`，可以将任务捕获到用户预先创建好的图中：任务不再捕获到驱动内部自动新建的图，而是存入用户提供的图对象。
#### 2.1.2.1 跨流依赖和事件
流捕获可以处理由`cudaEventRecord()`与`cudaStreamWaitEvent()`表达的跨流依赖，前提是被等待的事件是在**同一张捕获图**中记录生成。
在处于捕获模式的流中记录事件，会生成**捕获事件（captured event）**。捕获事件代表捕获图中的一组节点。
当某个流等待一个捕获事件时：若该流尚未进入捕获模式，则会自动将其切换至捕获模式；并且该流的下一个任务项，会新增对捕获事件所对应节点集合的依赖。此时这两个流会捕获至同一张捕获图。
流捕获场景下存在跨流依赖时，`cudaStreamEndCapture()`**仍必须在调用`cudaStreamBeginCapture()`的原始流上执行**，该流称为**起始流（origin stream）**。因基于事件的依赖而捕获至同一张捕获图的其他流，都必须**合并回起始流**，下文将对此进行说明。当调用`cudaStreamEndCapture()`时，所有捕获至该捕获图的流都会退出捕获模式。若未能将其他流合并回起始流，则整体捕获操作将会失败。
```cpp
// stream1 is the origin stream
cudaStreamBeginCapture(stream1);
kernel_A<<< ..., stream1 >>>(...);
// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);
kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);
// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);
kernel_D<<< ..., stream1 >>>(...);
// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);
// stream1 and stream2 no longer in capture mode
```
上述代码所生成的图，如下图所示：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261014094701.png)
注意：当流退出捕获模式后，该流中下一个**非捕获任务项**（如果存在）仍然会依赖于此前最近的那个非捕获任务项；尽管二者中间的那些任务项已经被移除。
#### 2.1.2.2 禁止与不支持的操作
CUDA流捕获阶段，流内kernel仅被录制为图节点，并未真正调度执行，此时对该捕获流/捕获事件调用同步、查询状态属于非法操作；若句柄覆盖范围包含正在捕获的流，如设备句柄、上下文句柄，执行同步/状态查询同样非法。示例：cudaStreamBeginCapture之后、cudaStreamEndCapture之前调用cudaStreamSynchronize(stream)，或是该设备存在任意流正在捕获时调用cudaDeviceSynchronize()，均会报错；必须等捕获结束、实例化并launch graph之后，才能做同步与状态查询。

当同一上下文内存在任意流正在捕获，且该流创建时**未使用`cudaStreamNonBlocking`**，此时尝试使用遗留流（legacy stream，一般可以认为是默认流）属于非法操作。原因是遗留流句柄始终包含这些其他流；向遗留流入队任务会产生对正在捕获流的依赖，查询或同步遗留流，等价于查询、同步正在捕获的流。

因此在该场景下调用**同步 API**也是非法的。`cudaMemcpy()`就是同步 API 的一例：它会向遗留流提交任务，并且在返回前等待该任务完成。

当一条依赖关系将要连接已捕获的对象与未被捕获、而是入队等待执行的对象时，CUDA 会优先返回错误，而不是忽略该依赖。
将流切换进入捕获模式、或者退出捕获模式时，会切断该模式切换前后，添加到该流中的任务项之间的依赖关系。

> 解释
> 1.Stream Capture 的依赖边界： CUDA Stream 进入 capture 模式后，提交的任务不再直接入队执行，而是被记录为 CUDA Graph 节点。一般情况下，如果一条依赖关系跨越 captured task 与普通 enqueue task，该依赖无法完整表示在 Graph 中，CUDA 为避免静默丢失依赖导致执行错误，会优先报错而不是忽略依赖。
   2.cudaStreamBeginCapture() 与 cudaStreamEndCapture() 本身被定义为明确的模式边界：进入 capture 时，会切断 capture 前普通任务与首个 Graph 节点之间的隐式 stream 依赖；退出 capture 时，也会切断最后一个 Graph 节点与之后普通入队任务之间的隐式依赖。因此 A → BeginCapture → B → EndCapture → C 应理解为普通任务 A、Graph 节点 B、普通任务 C 三个区域，不会自动形成 A→B→C 的跨 capture 边界依赖；如需这类顺序，应在 Graph launch 或通过受支持的显式同步机制中建立。

不能通过等待一个已捕获事件来合并两张相互独立的捕获图：该事件所属捕获图，和当前正在捕获的流所属捕获图不是同一张，此行为非法。
如果在一条正在捕获的流上等待未被捕获的事件，且没有指定 cudaEventWaitExternal 标记，该操作同样非法。

少数用于向流中入队异步操作的API，目前在Graph中不受支持；如果在一条正在捕获的流上调用这类API（例如 cudaStreamAttachMemAsync()），将会返回错误。

#### 2.1.2.3 失效机制
在流捕获过程中，如果执行了非法操作，对应的捕获图就会被置为无效。捕获图失效后，继续使用该图下任何处于捕获状态的流、或是该图关联的已捕获事件都属于非法行为，调用会返回错误；直到通过 cudaStreamEndCapture() 结束本次流捕获。该API调用会把关联的流退出捕获模式，但同时会返回错误码以及空（NULL）图。
### 2.1.3 汇总
图 25 中的示例是一个简化示例，仅用于从概念上展示小规模计算图。在使用 CUDA Graph 的应用程序中，无论是使用 Graph 显式 API，还是使用流捕获（Stream Capture），实际使用都会复杂得多。下面的代码片段并排对比了**Graph API**与**流捕获**两种方式：二者均可构建 CUDA Graph，用于执行一个简单的两阶段归约算法。

图 26 是该 CUDA Graph 的可视化图。它由下方代码调用 `cudaGraphDebugDotPrint` 函数生成，为便于阅读做了小幅调整，再经由 Graphviz 渲染得到。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261026101136.png)
**使用图API构建图**
```cpp
void cudaGraphsManual(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
    cudaStream_t                 streamForGraph;
    cudaGraph_t                  graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t              memcpyNode, kernelNode;
    double                       result_h = 0.0;

    cudaStreamCreate(&streamForGraph);

    // cudaGraphAddNode is the unified node-creation API: one call for every node type,
    // driven by a cudaGraphNodeParams struct (a .type tag + a union of per-type params).
    // It replaces the previously used cudaGraphAddMemcpyNode/cudaGraphAddKernelNode calls. The struct
    // has reserved fields that must be zero, so it is re-zeroed with "= {}" before each node below.
    cudaGraphNodeParams nodeParams   = {};
    cudaMemcpy3DParms   memcpyParams = {0};

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr   = make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr   = make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
    memcpyParams.extent   = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
    memcpyParams.kind     = cudaMemcpyHostToDevice;

    // Create an empty graph; nodes and edges will be added below.
    cudaGraphCreate(&graph, 0);

    // Node 1: H2D memcpy — no dependencies (NULL, 0), so it can start immediately.
    // For a memcpy node the cudaMemcpy3DParms goes into nodeParams.memcpy.copyParams
    // (the .memcpy union member is a wrapper struct, so the copy descriptor nests one level deeper).
    nodeParams                   = {};
    nodeParams.type              = cudaGraphNodeTypeMemcpy;
    nodeParams.memcpy.copyParams = memcpyParams;
    cudaGraphAddNode(&memcpyNode, graph, NULL, /*dependencyData=*/NULL, 0, &nodeParams);

    // Make the next node wait for this memcpy to finish before starting.
    nodeDependencies.push_back(memcpyNode);

    void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize, &numOfBlocks};

    // Node 2: first reduction kernel — depends on the H2D memcpy completing.
    // Kernel params are set directly on the .kernel union member.
    nodeParams                       = {};
    nodeParams.type                  = cudaGraphNodeTypeKernel;
    nodeParams.kernel.func           = (void *)reduce;
    nodeParams.kernel.gridDim        = dim3(numOfBlocks, 1, 1);
    nodeParams.kernel.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
    nodeParams.kernel.sharedMemBytes = 0;
    nodeParams.kernel.kernelParams   = (void **)kernelArgs;
    nodeParams.kernel.extra          = NULL;
    cudaGraphAddNode(
        &kernelNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);

    // Move dependency forward: the next node waits for this kernel.
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    void *kernelArgs2[3] = {(void *)&outputVec_d, (void *)&result_d, &numOfBlocks};

    // Node 3: final reduction kernel — depends on Node 2.
    nodeParams                       = {};
    nodeParams.type                  = cudaGraphNodeTypeKernel;
    nodeParams.kernel.func           = (void *)reduceFinal;
    nodeParams.kernel.gridDim        = dim3(1, 1, 1);
    nodeParams.kernel.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
    nodeParams.kernel.sharedMemBytes = 0;
    nodeParams.kernel.kernelParams   = kernelArgs2;
    nodeParams.kernel.extra          = NULL;
    cudaGraphAddNode(
        &kernelNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    memset(&memcpyParams, 0, sizeof(memcpyParams));

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr   = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr   = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
    memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
    memcpyParams.kind     = cudaMemcpyDeviceToHost;

    // Node 4: D2H memcpy — copies the scalar result back to the host.
    // Again the copy descriptor nests in nodeParams.memcpy.copyParams.
    nodeParams                   = {};
    nodeParams.type              = cudaGraphNodeTypeMemcpy;
    nodeParams.memcpy.copyParams = memcpyParams;
    cudaGraphAddNode(&memcpyNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);
    nodeDependencies.clear();
    nodeDependencies.push_back(memcpyNode);

    cudaGraphNode_t hostNode;
    callBackData_t  hostFnData;
    hostFnData.data    = &result_h;
    hostFnData.fn_name = "cudaGraphsManual";

    // Node 5: host callback — runs on the CPU after the D2H copy completes.
    // The .host member is cudaHostNodeParamsV2 (fn + userData, plus a syncMode field
    // left at 0 by the zero-init above).
    nodeParams               = {};
    nodeParams.type          = cudaGraphNodeTypeHost;
    nodeParams.host.fn       = myHostNodeCallback;
    nodeParams.host.userData = &hostFnData;
    cudaGraphAddNode(&hostNode, graph, nodeDependencies.data(), /*dependencyData=*/NULL, nodeDependencies.size(), &nodeParams);

    size_t numNodes = 0;
    cudaGraphGetNodes(graph, NULL, &numNodes);
    printf("Graph node count: %zu\n", numNodes);

    // Instantiate: compile the graph into an executable form (one-time cost).
    // This is where CUDA optimizes the schedule; repeated launches reuse this.
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Demonstrates cudaGraphClone — in practice, cloning is useful when multiple CPU
    // threads need to launch the same graph concurrently, each with its own independent graphExec.
    cudaGraph_t     clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    cudaGraphClone(&clonedGraph, graph);
    cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0);

    // Refill the host input before each launch so the graph's H2D copy processes
    // new data every time — one instantiated graph reused for different data.
    // The per-iteration sync ensures that copy finishes before we overwrite the buffer.
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(graphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    printf("\nCloned graph:\n");
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(clonedGraphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphExecDestroy(clonedGraphExec);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(clonedGraph);
    cudaStreamDestroy(streamForGraph);
}
```
**使用流捕获构建图**
```cpp
void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                                  float  *inputVec_d,
                                  double *outputVec_d,
                                  double *result_d,
                                  size_t  inputSize,
                                  size_t  numOfBlocks)
{
    cudaStream_t stream1, streamForGraph;
    cudaGraph_t  graph;
    double       result_h = 0.0;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&streamForGraph);

    // Begin capture: operations issued to stream1 are recorded, not executed.
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

    cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1);

    reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

    reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
    cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1);

    callBackData_t hostFnData = {0};
    hostFnData.data           = &result_h;
    hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
    cudaLaunchHostFunc(stream1, myHostNodeCallback, &hostFnData);

    // End capture: the runtime builds a graph from everything recorded above.
    // The resulting graph is structurally identical to the one built manually.
    cudaStreamEndCapture(stream1, &graph);

    size_t numNodes = 0;
    cudaGraphGetNodes(graph, NULL, &numNodes);
    printf("Graph node count: %zu\n", numNodes);

    // Instantiate the captured graph into an executable form.
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Demonstrates cudaGraphClone — in practice, cloning is useful when multiple CPU
    // threads need to launch the same graph concurrently, each with its own independent graphExec.
    cudaGraph_t     clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    cudaGraphClone(&clonedGraph, graph);
    cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0);

    // Refill the host input before each launch so the graph's H2D copy processes
    // new data every time — one instantiated graph reused for different data.
    // The per-iteration sync ensures that copy finishes before we overwrite the buffer.
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(graphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    printf("\nCloned graph:\n");
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        init_input(inputVec_h, inputSize);
        cudaGraphLaunch(clonedGraphExec, streamForGraph);
        cudaStreamSynchronize(streamForGraph);
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphExecDestroy(clonedGraphExec);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(clonedGraph);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(streamForGraph);
}
```
## 2.2 图实例化
无论通过图 API 或是流捕获的方式创建出图之后，都必须对该图执行**实例化**，生成**可执行图（executable graph）**，之后才可以提交运行。 
假设 `cudaGraph_t` 类型的图对象已经创建成功，下面这段代码会对该图进行实例化，生成可执行图 `cudaGraphExec_t graphExec`：
```cpp
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, 0);
```
## 2.3 图执行
在图创建完成并实例化为可执行图之后，就可以发起图的执行了。 假设 `cudaGraphExec_t graphExec` 已经创建成功，下面这段代码片段会将该图提交到指定流上执行：
```cpp
cudaGraphLaunch(graphExec, stream);
```
结合前文，以 4.2.1.2 小节的流捕获示例为基础，下面这段完整代码依次完成：创建图、实例化图、启动执行图：
```cpp
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ... , stream >>>(...);
kernel_B<<< ... , stream >>>(...);
libraryCall(stream);
kernel_C<<< ... , stream >>>(...);

cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, 0);
cudaGraphLaunch(graphExec, stream);
```
# 三、实例化图的更新
当工作流发生变化时，图就会失效，需要进行修改。如果是对图结构的重大改动（例如拓扑结构或节点类型变更），则必须重新实例化，因为和拓扑相关的优化需要重新执行。 但很多场景下，仅仅是节点参数（例如核函数参数、内存地址）发生变化，而图的拓扑保持不变。针对这种场景，CUDA 提供了轻量的**图更新（Graph Update）** 机制：可以就地修改部分节点参数，不需要重建整张图，相比重新实例化效率高得多。

更新操作会在**下一次启动该图时生效**，不会影响此前已经发起的图执行；即便更新发生时旧的图实例正在运行，也不受干扰。一张图可以反复更新、反复启动，因此可以在同一个流上连续排队多次更新与多次图启动。

CUDA 提供两种方式更新已实例化图的参数：**全图更新** 和 **单个节点更新**。
- 全图更新：用户传入一张拓扑完全一致的 `cudaGraph_t` 对象，该对象内的节点携带更新后的参数。
- 单节点更新：用户显式指定并修改单个节点的参数。

当需要修改大量节点，或者调用方不清楚图的拓扑（比如该图是通过库函数流捕获生成）时，使用传入更新后的 `cudaGraph_t` 做全图更新会更方便。 如果改动数量少、并且持有待修改节点的句柄，优先使用单节点更新。单节点更新会跳过对未改动节点的拓扑校验与比对，多数场景下效率更高。

CUDA 还支持单独启用 / 禁用某个节点，且该操作不会改动节点当前的参数值。

后续小节会对这两种方式展开详细说明。
## 3.1 全图更新
`cudaGraphExecUpdate()` 可以使用另一张拓扑完全一致的图（称为**更新源图**）中的参数，对一张已经实例化好的图（称为**原始图**）执行更新。更新源图的拓扑结构，必须和当初用来生成该 `cudaGraphExec_t` 的原始图完全相同。除此之外，依赖关系的指定顺序也必须保持一致。最后，CUDA 要求汇点节点（无后继依赖的节点）的排序具备确定性。CUDA 依靠特定 API 的调用顺序来保证汇点节点排序稳定。

更明确地说，遵循下面这些规则，才能让 `cudaGraphExecUpdate()` 确定性地完成原始图与更新源图之间节点的一一匹配： 对于任意一条正在捕获的流，作用在该流上的所有 API 调用顺序必须完全一致，包括事件等待以及其他不直接对应节点创建的 API。 直接操作某个图节点入边的 API 调用（包含流捕获里的 API、节点添加 API、边增删 API），调用顺序必须一致。并且，当这类 API 通过数组传入依赖项时，数组内部依赖项的排列顺序也必须匹配。

汇点节点的排序必须保持一致。汇点节点指调用 `cudaGraphExecUpdate()` 时，在最终图中没有后继节点、不存在出边的节点。下面这些操作会影响汇点节点的排序（如果存在），**这一组操作合起来的调用顺序必须保持一致**：
1. 添加节点，且该节点最终成为汇点节点；
2. 删除边，使得某个节点转变为汇点节点；
3. `cudaStreamUpdateCaptureDependencies()`，该调用会从捕获流的依赖集合中移除一个汇点节点；
4. `cudaStreamEndCapture()`。
下文示例演示该 API 如何对已实例化图执行更新操作。
```cpp
cudaGraphExec_t graphExec = NULL;  // 可重复执行的图实例；第一次循环时尚未创建

for (int i = 0; i < 10; i++) {
    cudaGraph_t graph;                       // 本轮捕获得到的 CUDA Graph
    cudaGraphExecUpdateResult updateResult;   // 记录更新图实例的结果
    cudaGraphNode_t errorNode;               // 更新失败时，指向导致失败的节点

    // 本例通过流捕获生成 CUDA Graph。
    // 也可以直接调用 CUDA Graph API，手动创建图及其节点。
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // 调用用户定义的、基于 CUDA 流执行的工作负载。
    // 捕获期间，这些操作不会立即按普通方式执行，而是被记录到图中。
    do_cuda_work(stream);

    // 结束流捕获，并将捕获到的操作及其依赖关系保存到 graph 中。
    cudaStreamEndCapture(stream, &graph);

    // 如果之前已经创建过可执行图实例，则先尝试用本轮新捕获的图更新它。
    // 更新成功时，可以复用已有实例，避免再次执行图实例化的开销。
    if (graphExec != NULL) {
        // 更新失败时，errorNode 会标识导致失败的图节点，
        // updateResult 会给出失败原因。
        cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    }

    // 第一次循环时还没有可执行图实例；
    // 如果后续更新失败，也需要重新实例化，才能执行本轮捕获的图。
    if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {

        // 如果已有实例但更新失败，先销毁旧实例。
        // 旧实例无法表示本轮的新图，不能直接用它执行新工作负载。
        if (graphExec != NULL) {
            cudaGraphExecDestroy(graphExec);
        }

        // 根据本轮捕获的 graph 创建新的可执行图实例 graphExec。
        // 此处使用简化形式，未使用错误节点和错误信息输出参数。
        cudaGraphInstantiate(&graphExec, graph, 0);
    }

    // graphExec 已更新或重新实例化，不再需要保留本轮的源图对象。
    cudaGraphDestroy(graph);

    // 在指定流上提交执行可执行图实例。
    cudaGraphLaunch(graphExec, stream);

    // 等待该流上的工作完成，确保本轮图执行结束后再进入下一轮。
    cudaStreamSynchronize(stream);
}
```
典型工作流是：通过流捕获或者图 API 创建初始的 `cudaGraph_t`，然后对该 `cudaGraph_t` 执行实例化并正常启动执行。在第一次启动之后，使用和构建初始图完全相同的方式新建一份 `cudaGraph_t`，再调用 `cudaGraphExecUpdate()`。 如果更新成功（由示例中的 `updateResult` 参数标识），就可以启动这份更新后的 `cudaGraphExec_t`。如果更新因为任何原因失败，则调用 `cudaGraphExecDestroy()` 和 `cudaGraphInstantiate()`：销毁原来的 `cudaGraphExec_t`，并重新实例化一份新的实例。

也可以直接修改 `cudaGraph_t` 中的节点（例如使用 `cudaGraphKernelNodeSetParams()`），再去更新 `cudaGraphExec_t`；但相比之下，使用下一节介绍的**显式单节点更新 API**效率更高。

条件句柄标志与默认值，会作为图更新的一部分同步更新。
更多用法说明以及当前的限制，请查阅 Graph API 文档。
## 3.2 单节点更新
可以直接修改**已实例化图**内部节点的参数。该方式省去了重新实例化、以及新建 `cudaGraph_t` 对象带来的开销。当需要修改的节点数量相对于整张图的总节点数较少时，更适合采用单节点更新。 下面这些 API 用于更新 `cudaGraphExec_t` 中的节点：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261026112149.png)
## 3.3 单节点启用控制
可以使用 `cudaGraphNodeSetEnabled()` API，对已实例化图中的核函数节点、内存置值节点、内存拷贝节点执行启用 / 禁用操作。 借助该能力，可以预先构建一张包含全部所需功能超集的图，在每次启动图的时候按需定制执行逻辑。 可以使用 `cudaGraphNodeGetEnabled()` API 查询节点的启用状态。

被禁用的节点在重新启用之前，行为等价于空节点。节点的参数不会因为启用 / 禁用操作而改变。 无论是单节点更新，还是通过 `cudaGraphExecUpdate()` 执行全图更新，都不会改变节点的启用状态。 在节点处于禁用期间所做的参数更新，会在该节点重新启用时生效。

更多用法说明以及当前存在的限制，请查阅 Graph API 文档。
## 3.4 图更新的限制条件
**核函数节点（Kernel nodes）：**
- 核函数所属的上下文不可变更。
- 如果节点原始绑定的函数未使用 CUDA 动态并行，更新时不能替换为使用 CUDA 动态并行的函数。
**cudaMemset（内存置值）与 cudaMemcpy（内存拷贝）节点：**
- 操作数所分配 / 映射到的 CUDA 设备不可变更。
- 源 / 目标内存，必须和原始的源 / 目标内存属于同一个上下文。
- 仅一维的 `cudaMemset` / `cudaMemcpy` 节点支持参数更新。
**内存拷贝节点额外约束：**
- 不支持修改源或目标的内存类型（例如 `cudaPitchedPtr`、`cudaArray_t` 等），也不支持修改传输类型（`cudaMemcpyKind`）。
**外部信号量等待节点、外部信号量触发节点：**
- 不支持修改信号量的数量。
**条件节点（Conditional nodes）：**
- 句柄的创建与赋值顺序，在两张图之间必须保持一致。
- 不支持修改节点参数（例如条件分支内的图数量、节点上下文等）。
- 条件体子图内部节点的参数修改，需要遵守上面所有规则。
**内存相关约束：**
- 如果一个 `cudaGraph_t` 已经被实例化为另一个 `cudaGraphExec_t`，则不能使用该 `cudaGraph_t` 去更新别的 `cudaGraphExec_t`。
**主机节点、事件记录节点、事件等待节点：** 对这三类节点执行更新没有上述限制。
# 四、条件图节点
条件节点可以实现节点内部子图的条件执行与循环执行。借助该特性，动态、迭代式的工作流可以完全在图内部表达，释放主机 CPU，使其可以并行处理其他任务。

当条件节点的所有依赖都满足后，会在**设备端**完成条件值的判定。条件节点分为下面几类：
- **条件 IF 节点**：执行该节点时，若条件值非零，则执行一次主体子图；还可以可选配置第二个主体子图，当条件值为 0 时，执行该分支子图一次。
- **条件 WHILE 节点**：执行该节点时，若条件值非零，则执行主体子图，并持续循环执行，直到条件值变为 0。
- **条件 SWITCH 节点**：若条件值等于`n`，就执行索引为`n`（从 0 开始计数）的那一份主体子图一次；如果条件值没有匹配对应的子图，则不会启动任何子图。

条件值由**条件句柄（conditional handle）** 访问，句柄必须在创建条件节点之前预先创建。设备侧代码可以调用 `cudaGraphSetConditional()` 设置条件值；创建句柄时还可以指定一个默认值，该默认值会在每次启动图时生效。

创建条件节点时，会生成一张空图，并返回对应的句柄给用户，用于向其中填充内容。条件分支内的主体子图，可以通过图 API 或者 `cudaStreamBeginCaptureToGraph()` 流捕获方式填充节点。

条件节点支持嵌套使用。
## 4.1 条件句柄
条件值由 `cudaGraphConditionalHandle` 表示，通过 `cudaGraphConditionalHandleCreate()` 创建。
一个句柄只能关联**单个条件节点**。该类句柄不支持销毁，因此开发者无需手动管理、追踪它的生命周期。
创建句柄时如果指定了 `cudaGraphCondAssignDefault` 标志，那么**每次图执行开始时**，条件值都会被初始化为设定好的默认值。 如果不设置该标志，则每次图启动时条件值是未定义的；代码不能假设条件值会在多次图执行之间保留上一次的值。
句柄绑定的默认值与标志位，会在**全图更新（whole graph update）** 过程中同步更新。
## 4.2 条件节点主体子图的约束要求
**通用要求：**
- 子图内所有节点必须隶属于同一个设备。
- 子图仅允许包含：核函数节点、空节点、内存拷贝节点、内存置值节点、子图节点、条件节点。
**核函数节点：**
- 子图内的核函数**不允许使用 CUDA 动态并行，也不允许在设备端发起图执行（Device Graph Launch）**。
- 协同启动（cooperative launches）是允许的，但前提是没有启用 MPS。
**内存拷贝 / 内存置值节点：**
- 仅允许操作设备内存、以及**页锁定、设备映射型主机内存**之间的拷贝 / 置值。
- 不支持操作 CUDA 数组（cudaArray）。
- 在实例化时，两个操作数都必须对当前设备可访问。注意：拷贝操作由该子图所在的设备发起执行，即便目标内存位于其他设备上也是如此。
## 4.3 IF条件节点
当执行该节点时，若条件值非零，则会执行一次 IF 节点的主体子图。下图展示了一个包含三个节点的图，其中中间节点 B 是条件节点。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261026114730.png)
下方代码示例演示如何创建一个包含 IF 条件节点的图：条件值由上游核函数设置；条件分支内的主体子图通过图 API 填充。
```cpp
__global__ void setHandle(cudaGraphConditionalHandle handle, int value)
{
    ...
    // 将传入的 value 写入条件句柄，供后续条件节点判断。
    // 该操作必须在条件节点执行之前完成；本例通过图中的依赖关系保证这一点。
    cudaGraphSetConditional(handle, value);
    ...
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // 创建一个空的 CUDA Graph，后续将向其中添加内核节点和条件节点。
    cudaGraphCreate(&graph, 0);

    // 创建条件句柄，供内核设置条件值，并供条件节点读取该值。
    // 本例没有提供默认值，因此每次执行图时，条件值在被内核设置前都未定义。
    // 所以必须确保设置条件值的内核先于条件节点执行。
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // 在条件节点之前添加一个内核节点，用它设置条件句柄的值。
    // 该内核配置为一个线程块、每个线程块一个线程。
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &params);

    // 创建并添加条件节点。
    // 条件类型为 IF：当 handle 中的条件值满足条件时，执行其内部的 IF 子图。
    // 这里将前面设置条件值的内核节点作为条件节点的前驱，
    // 从而保证条件节点读取到的是内核设置后的值。
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 1; // 条件节点只包含一个子图，即 IF 分支的子图
    cudaGraphAddNode(&node, graph, &node, nullptr, 1, &cParams);

    // 获取条件节点的 IF 分支子图。
    // 后续添加到 bodyGraph 的节点，只有在 IF 条件成立时才会执行。
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // 向 IF 分支子图中添加需要按条件执行的工作。
    // 例如，可以在这里添加内核节点或其他图节点。
    ...
    cudaGraphAddNode(&node, bodyGraph, nullptr, nullptr, 0, &params);

    // 将 graph 实例化为可执行图，并立即在默认流（流句柄 0）上启动。
    // 随后等待设备上的工作完成，确保图执行结束后再进行资源清理。
    cudaGraphInstantiate(&graphExec, graph, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // 释放可执行图实例和源图对象。
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```
IF 节点还可以设置可选的第二路主体子图；当节点执行时，如果条件值为 0，该子图会执行一次。
```cpp
void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // 1. 创建一个空的 CUDA Graph
    cudaGraphCreate(&graph, 0);

    // 2. 创建条件句柄 handle，用于存储 IF 条件节点判断时使用的条件值
    // 未指定默认值，因此每次 Graph 执行开始时，条件值都是未定义的
    // 必须在条件节点执行前，通过 Kernel 显式设置条件值
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // 3. 创建 Kernel 节点，用于在 GPU 上设置条件值
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;

    // 设置 Kernel 的两个参数：条件句柄 handle 和条件值 value
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;

    // 将 Kernel 节点添加到 Graph，不设置前驱依赖节点
    // 该节点执行时调用 cudaGraphSetConditional(handle, value)
    cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &params);

    // 4. 创建 IF 条件节点
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;        // 指定用于条件判断的句柄
    cParams.conditional.type   = cudaGraphCondTypeIf; // 条件类型为 IF
    cParams.conditional.size   = 2;             // 包含 if 和 else 两个分支图

    // 将 IF 条件节点添加到 Graph，并依赖前面的 setHandle Kernel 节点
    // 保证先设置条件值，再执行条件判断
    cudaGraphAddNode(&node, graph, &node, nullptr, 1, &cParams);

    // 5. 获取条件节点自动创建的两个分支图
    cudaGraph_t ifBodyGraph = cParams.conditional.phGraph_out[0];   // 条件非零时执行
    cudaGraph_t elseBodyGraph = cParams.conditional.phGraph_out[1]; // 条件为零时执行

    // 6. 分别向 if 和 else 分支图添加需要执行的 Kernel 节点
    ...
    cudaGraphAddNode(&node, ifBodyGraph, nullptr, nullptr, 0, &params);
    ...
    cudaGraphAddNode(&node, elseBodyGraph, nullptr, nullptr, 0, &params);

    // 7. 实例化并执行 Graph
    cudaGraphInstantiate(&graphExec, graph, 0);
    cudaGraphLaunch(graphExec, 0);

    // 等待 GPU 执行完成
    // 本例 value = 1，因此执行 ifBodyGraph，跳过 elseBodyGraph
    cudaDeviceSynchronize();

    // 8. 释放可执行 Graph 和原始 Graph 资源
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```
## 4.4 WHILE条件节点
WHILE 节点的主体子图，只要条件值非零就会持续执行。条件判断会在节点初次执行时、以及每一轮主体子图执行完毕后进行。下图展示了一个三节点图，其中中间节点 B 为条件节点。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261026115955.png)
下面这段代码演示如何创建包含 WHILE 条件节点的图。该句柄通过 `cudaGraphCondAssignDefault` 创建，无需依赖上游核函数。条件节点的主体子图借助图 API 来构建填充。
```cpp
__global__ void loopKernel(cudaGraphConditionalHandle handle, char *dPtr)
{
    // 将 dPtr 指向的值减 1
    // 当减至 0 时，将条件值设置为 0，使 WHILE 循环结束
    if (--(*dPtr) == 0) {
        cudaGraphSetConditional(handle, 0);
    }
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];

    // 1. 分配 1 字节 GPU 显存，用于存储循环计数值
    char *dPtr;
    cudaMalloc((void **)&dPtr, 1);

    // 2. 创建一个空的 CUDA Gr0aph
    cudaGraphCreate(&graph, 0);

    // 3. 创建条件句柄 handle，默认条件值为 1
    // cudaGraphCondAssignDefault 表示每次 Graph 执行开始时，
    // 都将条件值重置为 1，确保 WHILE 循环能够开始执行
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(
        &handle, graph, 1, cudaGraphCondAssignDefault
    );

    // 4. 创建 WHILE 条件节点
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;           // 指定用于条件判断的句柄
    cParams.conditional.type   = cudaGraphCondTypeWhile; // 条件类型为 WHILE
    cParams.conditional.size   = 1;                // WHILE 只有一个循环体子图

    // 将 WHILE 条件节点添加到 Graph，不设置前驱依赖节点
    // 条件值非零时执行循环体，循环体结束后重新判断条件
    cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &cParams);

    // 5. 获取 WHILE 条件节点自动创建的循环体子图
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // 6. 创建 Kernel 节点，用于更新循环计数值和条件值
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)loopKernel;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;

    // 设置 Kernel 的两个参数：条件句柄 handle 和设备指针 dPtr
    kernelArgs[0] = &handle;
    kernelArgs[1] = &dPtr;

    // 将 Kernel 节点添加到 WHILE 循环体子图中
    // 每执行一次循环体，dPtr 的值减 1
    cudaGraphAddNode(&node, bodyGraph, nullptr, nullptr, 0, &params);

    // 7. 将 GPU 中的循环计数值初始化为 10
    // 循环执行 10 次后，dPtr 减至 0，Kernel 将条件值设置为 0
    cudaMemset(dPtr, 10, 1);

    // 8. 实例化并执行 Graph
    cudaGraphInstantiate(&graphExec, graph, 0);
    cudaGraphLaunch(graphExec, 0);

    // 等待 GPU 执行完成
    // 本例循环执行 10 次，最终 dPtr = 0
    cudaDeviceSynchronize();

    // 9. 释放 Graph 及 GPU 显存资源
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(dPtr);
}
```
## 4.5 SWITCH条件节点
SWITCH 节点的各个主体子图采用从零开始编号；当节点执行时，如果条件值等于 n，则执行编号为 n 的主体子图。下图展示了一个三节点图，其中中间节点 B 为条件节点。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261026120346.png)
下面这段代码演示如何创建包含 SWITCH 条件节点的图。条件值由上游核函数设置。该条件分支的各个主体子图通过图 API 进行构建填充。
```cpp
__global__ void setHandle(cudaGraphConditionalHandle handle, int value)
{
    ...
    // 将传入的 value 设置为条件句柄 handle 对应的条件值
    cudaGraphSetConditional(handle, value);
    ...
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // 1. 创建一个空的 CUDA Graph
    cudaGraphCreate(&graph, 0);

    // 2. 创建条件句柄 handle，用于存储 SWITCH 条件节点判断时使用的条件值
    // 未指定默认值，因此每次 Graph 执行开始时，条件值都是未定义的
    // 必须在条件节点执行前，通过 Kernel 显式设置条件值
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // 3. 创建 Kernel 节点，用于在 GPU 上设置条件值
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;

    // 设置 Kernel 的两个参数：条件句柄 handle 和条件值 value
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;

    // 将 Kernel 节点添加到 Graph，不设置前驱依赖节点
    // 该节点执行时调用 cudaGraphSetConditional(handle, value)
    cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &params);

    // 4. 创建 SWITCH 条件节点
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;              // 指定用于条件判断的句柄
    cParams.conditional.type   = cudaGraphCondTypeSwitch; // 条件类型为 SWITCH
    cParams.conditional.size   = 5;                   // 包含 5 个分支子图

    // 将 SWITCH 条件节点添加到 Graph，并依赖前面的 setHandle Kernel 节点
    // 保证先设置条件值，再根据条件值选择分支
    cudaGraphAddNode(&node, graph, &node, nullptr, 1, &cParams);

    // 5. 获取 SWITCH 条件节点自动创建的分支子图数组
    // 条件值为 i 时执行 bodyGraphs[i]（0 <= i < 5）
    // 如果条件值超出 [0, 4]，则不执行任何分支
    cudaGraph_t *bodyGraphs = cParams.conditional.phGraph_out;

    // 6. 分别向 SWITCH 的各个分支子图添加需要执行的 Kernel 节点
    // 这里省略了各分支对应的 Kernel 参数配置
    ...
    cudaGraphAddNode(&node, bodyGraphs[0], nullptr, nullptr, 0, &params);
    ...
    cudaGraphAddNode(&node, bodyGraphs[4], nullptr, nullptr, 0, &params);

    // 7. 实例化并执行 Graph
    cudaGraphInstantiate(&graphExec, graph, 0);
    cudaGraphLaunch(graphExec, 0);

    // 等待 GPU 执行完成
    // 本例 value = 1，因此执行 bodyGraphs[1]，其余分支不执行
    cudaDeviceSynchronize();

    // 8. 释放可执行 Graph 和原始 Graph 资源
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```
# 五、图内存节点
## 5.1 介绍
图内存节点允许 CUDA Graph 在图中创建和管理内存分配。分配得到的内存遵循 GPU 序生命周期语义：它在图中的特定位置变为可访问，并在相应的释放节点之后不再可访问。CUDA 驱动可以据此安全地复用内存。该语义与流序分配 API `cudaMallocAsync` 和流序释放 API `cudaFreeAsync` 一致；这两个 API 也可以在流捕获期间被记录到图中。

在一张图的生命周期内，包括图的多次实例化和多次执行，图分配得到的内存会保持固定的虚拟地址。因此，图中的其他算子可以始终通过该地址访问这块内存，无需因底层物理内存发生变化而更新图。只要同一张图中多个分配对象的 GPU 序生命周期不重叠，CUDA 就可以让它们共用同一块物理内存。

CUDA 也可以在不同图之间复用物理内存。例如，多张图依次提交到同一个流执行时，如果各图中分配对象的 GPU 序生命周期不会同时重叠，CUDA 可以通过虚拟地址映射别名化，让这些对象映射到同一块物理内存。这样既能满足各图的内存需求，也能减少实际占用的物理内存。
## 5.2 API基础
图内存节点用于表示图中的内存分配或释放操作。负责分配内存的节点称为**分配节点**，负责释放内存的节点称为**释放节点**；由分配节点创建的内存称为**图分配内存**。

创建分配节点时，CUDA 会为图分配内存指定一个虚拟地址。只要该分配节点仍然存在，这个虚拟地址就保持固定；但内存中的数据不会因此永久保留。释放操作完成后，原有内容不再有效，后续访问其他分配对象时还可能覆盖这块内存。

图每执行一次，图分配内存都会重新创建。它的实际使用期限从 GPU 执行到分配节点时开始，并在以下任一时刻结束：
- GPU 执行到图中的释放节点；
- GPU 执行到调用 `cudaFreeAsync()` 的流操作；
- 主机调用 `cudaFree()` 时，内存立即被释放。

**销毁一张图不会自动释放仍处于使用状态的图分配内存**。即使图销毁后，分配节点的生命周期已经结束，这块内存仍须通过另一张图中的释放节点，或通过 `cudaFreeAsync()`、`cudaFree()` 显式释放。

图内存节点和其他图节点一样，通过依赖边确定执行顺序。程序必须保证：
- 所有访问这块内存的操作都排在分配节点之后；
- 所有访问操作都排在释放该内存的操作之前。

这里的生命周期通常由 **GPU 实际执行顺序**决定，而不是由主机调用 API 的顺序或图的描述顺序决定。因此，图分配内存可以理解为遵循 **GPU 执行顺序**的内存。
### 5.2.1 图节点API
图内存节点可以通过节点创建 API `cudaGraphAddNode` 显式添加。添加 `cudaGraphNodeTypeMemAlloc` 节点时，CUDA 会分配图内存，并将其地址写入传入的 `cudaGraphNodeParams` 结构体的 `alloc.dptr` 字段。图中使用这块内存的所有操作，都必须通过依赖关系排在分配节点之后；释放节点则必须排在所有使用该内存的操作之后。释放节点同样通过 `cudaGraphAddNode` 添加，节点类型为 `cudaGraphNodeTypeMemFree`。

下图中，内核节点 **a、b、c** 都排在分配节点之后、释放节点之前，因此可以安全访问这块内存。内核节点 **e** 没有被安排在分配节点之后，无法保证它执行时内存已经分配，因此不能安全访问。内核节点 **d** 没有被安排在释放节点之前，无法保证它访问内存时内存尚未释放，因此也不能安全访问。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261026150950.png)
下面的代码片段构建了该图中的计算图：
```cpp
// 创建一张空图，后续将在图中添加内存分配、内核和内存释放节点。
cudaGraphCreate(&graph, 0);

// 设置图内存分配节点的参数。
cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;

// 指定设备 0 为这块内存的驻留设备。
params.alloc.poolProps.location.id = 0;

// 设置要分配的内存大小，单位为字节。
params.alloc.bytesize = size;

// 将内存分配节点添加到图中。
// 该节点创建后，分配得到的设备指针保存在 params.alloc.dptr 中。
cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &params);

// 设置一个会使用图分配内存的内核节点参数。
cudaGraphNodeParams nodeParams = { cudaGraphNodeTypeKernel };
nodeParams.kernel.kernelParams[0] = params.alloc.dptr;
// ……继续设置内核函数、网格大小、线程块大小及其他参数……

// 将内核节点 a 添加到图中，并声明它依赖于分配节点。
// 因此，a 只有在分配节点完成后才能执行，可以安全访问该内存。
cudaGraphAddNode(&a, graph, &allocNode, NULL, 1, &nodeParams);

// 添加内核节点 b 和 c，并分别声明它们依赖于 a。
// 依赖关系保证 b、c 都在 a 之后执行；由于 a 依赖于分配节点，
// b、c 也间接依赖于分配节点，因此可以安全访问该内存。
cudaGraphAddNode(&b, graph, &a, NULL, 1, &nodeParams);
cudaGraphAddNode(&c, graph, &a, NULL, 1, &nodeParams);

cudaGraphNode_t dependencies[2];

// b 和 c 都会使用这块图分配内存，因此释放节点必须等待 b、c 完成。
// 释放节点不必另外依赖 a：b 和 c 已经依赖 a，依赖关系会沿图传递。
dependencies[0] = b;
dependencies[1] = c;

// 设置图内存释放节点的参数，并指定要释放的设备指针。
cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = params.alloc.dptr;

// 将释放节点添加到图中，使其依赖于 b 和 c。
// 这样可以保证 b、c 使用完内存后，才释放这块内存。
cudaGraphAddNode(&freeNode, graph, dependencies, NULL, 2, &freeNodeParams);

// 添加内核节点 d，并让它依赖于 c。
// 但 d 与释放节点之间没有依赖关系，因此无法保证 d 在内存释放前执行。
// 所以 d 不得访问这块图分配内存。
cudaGraphAddNode(&d, graph, &c, NULL, 1, &nodeParams);

// 添加内核节点 e，但不设置任何前驱节点，因此 e 与分配节点之间没有依赖关系。
// 无法保证 e 执行时内存已经分配，所以 e 不得访问这块内存。
// 即使释放节点依赖于 e，也不能补上 e 对分配节点的依赖。
cudaGraphAddNode(&e, graph, NULL, NULL, 0, &nodeParams);
```
### 5.2.2 流捕获
图内存节点也可以通过流捕获创建：在捕获流中调用流序内存分配 API `cudaMallocAsync` 和流序内存释放 API `cudaFreeAsync`，CUDA 会将相应操作记录为图中的内存分配节点和释放节点。

`cudaMallocAsync` 返回的设备虚拟地址可供图中后续操作使用。流捕获会记录流内操作之间的依赖关系；对于正确编写的流代码，`cudaMallocAsync` 和 `cudaFreeAsync` 所要求的执行顺序会确保图中的分配、使用和释放操作按正确顺序排列。因此，通常无需另外手动设置这些节点与流操作之间的依赖关系。原文接下来会通过示例代码说明如何用流捕获构造前图中的依赖结构；为便于说明，该示例省略了内核节点 **d** 和 **e**。
```cpp
cudaMallocAsync(&dptr, size, stream1);
kernel_A<<< ..., stream1 >>>(dptr, ...);

// 在这里将工作分支到 stream2。
// 先在 stream1 中记录 event1，再让 stream2 等待 event1，
// 从而保证 stream2 中后续的工作在此前 stream1 中已提交的工作完成后执行。
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(dptr, ...);

// 在 stream2 中启动 kernel_C。
// stream2 对 event1 的等待会形成图中的依赖关系：
// 捕获 kernel_C 时生成的内核节点会依赖捕获 cudaMallocAsync 时生成的分配节点，
// 因此 kernel_C 执行前，这块内存已完成分配。
kernel_C<<< ..., stream2 >>>(dptr, ...);

// 将 stream2 中的工作汇合回起始流 stream1。
// 先在 stream2 中记录 event2，再让 stream1 等待 event2，
// 以确保 stream1 中后续的工作等待 stream2 中此前提交的工作完成。
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

// 释放 dptr 指向的内存。
// 由于释放操作排在 stream1 的 kernel_B 以及等待 event2 的操作之后，
// 它会依赖所有通过上述流依赖关系先于它执行、且会访问该内存的工作。
cudaFreeAsync(dptr, stream1);

// 在起始流 stream1 中结束捕获，并将捕获到的操作及其依赖关系保存到 graph。
cudaStreamEndCapture(stream1, &graph);
```
### 5.2.3 在分配该内存的图之外访问与释放图内存
图分配内存不一定要由创建它的图释放。如果图执行结束时没有释放这块内存，分配仍然有效，后续 CUDA 操作可以继续访问它。访问操作必须通过 CUDA 事件、流内顺序等机制，明确排在分配操作之后；释放操作则必须排在所有访问操作之后。

这块内存可以通过常规的 `cudaFree` 或 `cudaFreeAsync` 释放，也可以由另一张图中的对应释放节点释放。若创建它的图以 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志实例化，后续再次启动该图时也会释放此前遗留的图分配内存。内存释放后，任何操作都不得再访问它。

由于多个图分配可能共用同一块底层物理内存，释放操作必须等待所有设备端访问操作完成。仅依赖计算内核内部的内存同步（例如基于内存的同步）不足以保证访问与释放之间的正确顺序；必须使用图依赖、CUDA 事件或流顺序等机制建立先后关系。

接下来的示例将展示如何在创建内存的图之外访问这块内存，并通过三种方式保证操作顺序：在同一条流中安排操作、在不同流之间使用事件，以及在分配图和释放图中使用事件建立依赖。

**第一种：通过单个流建立的执行序**
```cpp
// 以下代码构建并实例化一张负责分配内存的图，
// 随后在同一条流中依次启动该图、使用分配得到的内存，并将其释放。
void *dptr;

// 设置图内存分配节点的参数。
cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
params.alloc.bytesize = size;

// 将分配节点添加到 allocGraph。
// 添加节点后，分配得到的设备指针保存在 params.alloc.dptr 中。
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &params);
dptr = params.alloc.dptr;

// 将包含分配节点的图实例化为可执行图。
cudaGraphInstantiate(&allocGraphExec, allocGraph, 0);

// 在 stream 上启动分配图。
// 后续提交到同一条流的操作会按流顺序排在图执行之后。
cudaGraphLaunch(allocGraphExec, stream);

// 在同一条流中启动内核，使用图分配得到的设备内存。
// 由于内核排在分配图之后，执行时该内存已完成分配。
kernel<<< ..., stream >>>(dptr, ...);

// 在同一条流中异步释放设备内存。
// 流顺序保证释放操作排在此前使用该内存的内核之后。
cudaFreeAsync(dptr, stream);
```
**第二种：通过记录并等待 CUDA 事件建立的执行序**
```cpp
// 以下展示跨图、跨流使用图分配内存的方式：
// allocGraph 负责分配内存，stream2 上的内核使用该内存，
// freeGraph 负责释放内存。通过 CUDA 事件建立执行依赖，
// 保证内存先分配、再使用，最后释放。

// allocGraph 的构建过程。
void *dptr;

// 将内存分配节点添加到 allocGraph。
// 分配节点创建后，分配得到的设备指针保存在 allocNodeParams.alloc.dptr 中。
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
dptr = allocNodeParams.alloc.dptr;

// freeGraph 中使用该指针的内核节点，以及释放该内存的节点。
// 此处将设备指针作为内核参数；添加节点时还应按实际图结构设置依赖关系。
kernelNodeParams.kernel.kernelParams[0] = allocNodeParams.alloc.dptr;
cudaGraphAddNode(&freeNode, freeGraph, NULL, NULL, 1, dptr);

// 分别将分配图和释放图实例化为可执行图。
cudaGraphInstantiate(&allocGraphExec, allocGraph, 0);
cudaGraphInstantiate(&freeGraphExec, freeGraph, 0);

// 在 allocStream 上启动分配图。
cudaGraphLaunch(allocGraphExec, allocStream);

// 通过事件建立 stream2 对分配操作的依赖。
// allocEvent 在 allocStream 中记录；stream2 等待该事件后，
// 才会执行后续操作，从而确保内存分配已完成。
// 也可以通过同步 allocStream 等方式建立相应的先后顺序。
cudaEventRecord(allocEvent, allocStream);
cudaStreamWaitEvent(stream2, allocEvent);

// 在 stream2 上启动内核，使用刚才分配的设备内存。
kernel<<< ..., stream2 >>>(dptr, ...);

// 再通过事件建立 stream3 对内存使用操作的依赖。
// 事件记录在 stream2 中，因此它会在该流此前提交的内核完成后触发；
// stream3 等待该事件后，才会执行后续操作。
cudaStreamRecordEvent(streamUseDoneEvent, stream2);
cudaStreamWaitEvent(stream3, streamUseDoneEvent);

// 此时 stream3 已等待内核完成，因此可以启动释放图。
// 释放图中的操作也可以在内存仍有效时访问它；
// 图中的释放节点则必须排在所有内存访问操作之后。
cudaGraphLaunch(freeGraphExec, stream3);
```
**第三种：通过图外部事件节点建立的执行序**
```cpp
// 分配图产生的设备内存指针。该指针会传给其他流上的内核，以及后续的释放图。
void *dptr;

// allocEvent 表示分配图中的内存分配操作已经完成，
// 等待该事件后，其他流才可以安全使用这块内存。
cudaEvent_t allocEvent;

// streamUseDoneEvent 表示 stream2 中使用这块内存的操作已经完成，
// 释放图等待该事件后，才能安全释放内存。
cudaEvent_t streamUseDoneEvent;

// 在分配图中添加内存分配节点，并取得分配得到的设备指针。
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
dptr = allocNodeParams.alloc.dptr;

// 创建事件记录节点，并将它添加到分配图。
// 该节点依赖于 allocNode，因此只有内存分配完成后才会记录 allocEvent。
cudaGraphNodeParams allocEventNodeParams = { cudaGraphNodeTypeEventRecord };
allocEventNodeParams.eventRecord.event = allocEvent;
cudaGraphAddNode(&recordNode, allocGraph, &allocNode, NULL, 1, allocEventNodeParams);

// 将包含分配节点和事件记录节点的图实例化为可执行图。
cudaGraphInstantiate(&allocGraphExec, allocGraph, 0);

// 在消费和释放图中添加事件等待节点。
// 该节点等待 streamUseDoneEvent，表示释放图中的后续操作需要等到
// stream2 中使用这块内存的操作完成。
cudaGraphNodeParams streamWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
streamWaitEventNodeParams.eventWait.event = streamUseDoneEvent;
cudaGraphAddNode(&streamUseDoneEventNode, waitAndFreeGraph, NULL, NULL, 0, streamWaitEventNodeParams);

// 添加另一个事件等待节点，等待分配图记录的 allocEvent。
// 这样，消费和释放图可以在执行内核前确认内存已经分配完成。
cudaGraphNodeParams allocWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
allocWaitEventNodeParams.eventWait.event = allocEvent;
cudaGraphAddNode(&allocReadyEventNode, waitAndFreeGraph, NULL, NULL, 0, allocWaitEventNodeParams);

// 将分配图产生的设备指针设置为消费图中内核的参数。
kernelNodeParams->kernelParams[0] = allocNodeParams.alloc.dptr;

// 添加使用该内存的内核节点，并声明它依赖于 allocReadyEventNode。
// 因此，内核只有在 allocEvent 已完成、内存可用后才能执行。
cudaGraphAddNode(&kernelNode, waitAndFreeGraph, &allocReadyEventNode, NULL, 1, &kernelNodeParams);

// 释放节点必须排在所有使用该内存的操作之后。
// 它既要等待消费图内部的 kernelNode 完成，
// 也要等待 stream2 上的操作通过 streamUseDoneEventNode 确认完成。
dependencies[0] = kernelNode;
dependencies[1] = streamUseDoneEventNode;

// 创建内存释放节点，并指定要释放的设备指针。
cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = dptr;

// 将释放节点添加到消费和释放图，并设置它依赖于上述两个节点。
// 这样可以避免内存仍被图内内核或 stream2 上的操作使用时就被释放。
cudaGraphAddNode(&freeNode, waitAndFreeGraph, &dependencies, NULL, 2, freeNodeParams);

// 将消费和释放图实例化为可执行图。
cudaGraphInstantiate(&waitAndFreeGraphExec, waitAndFreeGraph, 0);

// 在 allocStream 上启动分配图。
// 图中的 allocEvent 记录节点会在分配完成后记录事件。
cudaGraphLaunch(allocGraphExec, allocStream);

// 让 stream2 等待 allocEvent，确保分配图中的分配操作已完成。
// 随后在 stream2 上启动内核，使用这块图分配内存。
cudaStreamWaitEvent(stream2, allocEvent);
kernel<<< ..., stream2 >>>(dptr, ...);

// 在 stream2 中记录 streamUseDoneEvent。
// 该事件会在此前提交到 stream2 的内核完成后触发，
// 供释放图中的等待节点确认内存使用已经结束。
cudaStreamRecordEvent(streamUseDoneEvent, stream2);

// 在 stream3 上启动消费和释放图。
// 图中的事件等待节点会等待 streamUseDoneEvent，确保 stream2 上的内核完成；
// 内核节点依赖 allocReadyEventNode，确保分配已完成。
// 释放节点则依赖图内核和事件等待节点，避免内存访问与释放发生冲突。
cudaGraphLaunch(waitAndFreeGraphExec, stream3);
```
### 5.2.4 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志
通常，如果一张图仍有尚未释放的图分配内存，CUDA 会阻止它再次启动。否则，重复执行分配节点可能会在同一虚拟地址创建新的分配，导致原有分配无法释放并造成内存泄漏。

实例化图时指定 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志后，即使图中仍有未释放的分配，也可以再次启动。每次启动时，CUDA 会自动异步释放该图上一次执行遗留的未释放分配。

这个功能适用于“单个生产者、多个消费者”的算法：每轮由生产者图创建若干分配，运行时再根据条件让不同的消费者访问这些内存。由于后续消费者可能仍需访问某些分配，消费者不适合负责释放它们。启用自动释放后，生产者图的启动逻辑可以统一处理上一次执行遗留的分配，算法无需在每次重新启动前逐一跟踪并释放它们。

**注意：** 该标志只影响图再次启动时的行为，不会改变图销毁时的行为。销毁图时，应用程序仍须显式释放尚未释放的图分配内存，否则可能发生内存泄漏。
```cpp
// 创建生产者图：分配两块设备内存，并由生产者内核向其中写入数据。
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
cudaMallocAsync(&data1, blocks * threads, cudaStreamPerThread);
cudaMallocAsync(&data2, blocks * threads, cudaStreamPerThread);
produce<<<blocks, threads, 0, cudaStreamPerThread>>>(data1, data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);

// 实例化生产者图，并启用“启动时自动释放”标志。
// 再次启动该图时，CUDA 会自动异步释放上一次执行遗留的图分配内存。
cudaGraphInstantiateWithFlags(&producer,
                              graph,
                              cudaGraphInstantiateFlagAutoFreeOnLaunch);
cudaGraphDestroy(graph);  // 实例化完成后，销毁用于构建图的源图对象

// 创建第一个消费者图：捕获一个使用 data1 的异步库调用。
// 该库调用需要支持 CUDA 流捕获。
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consumerFromLibrary(data1, cudaStreamPerThread);
cudaStreamEndCapture(cudaStreamPerThread, &graph);

// 按普通方式实例化消费者图，不启用启动时自动释放功能。
cudaGraphInstantiateWithFlags(&consumer1, graph, 0);
cudaGraphDestroy(graph);  // 销毁源图对象，保留可执行图实例

// 创建第二个消费者图：捕获使用 data2 的内核操作。
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consume2<<<blocks, threads, 0, cudaStreamPerThread>>>(data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&consumer2, graph, 0);
cudaGraphDestroy(graph);  // 销毁源图对象，保留可执行图实例

// 循环启动生产者图和消费者图。
// 所有图都提交到 myStream，因此按提交顺序执行：生产者先分配并填充数据，
// 消费者随后读取相应数据。是否启动第二个消费者由运行时条件决定。
bool launchConsumer2 = false;
do {
    cudaGraphLaunch(producer, myStream);
    cudaGraphLaunch(consumer1, myStream);
    if (launchConsumer2) {
        cudaGraphLaunch(consumer2, myStream);
    }
} while (determineAction(&launchConsumer2));

// 循环结束后，在 myStream 中异步释放两块内存。
// 流顺序保证此前提交的图操作先完成对内存的使用，然后再执行释放。
cudaFreeAsync(data1, myStream);
cudaFreeAsync(data2, myStream);

// 销毁可执行图实例，释放相应的图资源。
cudaGraphExecDestroy(producer);
cudaGraphExecDestroy(consumer1);
cudaGraphExecDestroy(consumer2);
```
### 5.2.5 子图中的内存节点
CUDA 12.9 支持将子图的所有权转移给父图。转移后，子图可以包含内存分配节点和内存释放节点。这样，开发者可以先独立构建包含内存操作的子图，再将其加入父图。
子图所有权转移到父图后，该子图受到以下限制：
- 不能再单独实例化或销毁；
- 不能再作为子图加入另一张父图；
- 不能作为参数传给 `cuGraphExecUpdate`；
- 不能再向其中添加内存分配节点或内存释放节点。
```cpp
// 创建一张独立的子图，稍后将在其中添加内存分配、使用和释放节点。
cudaGraphCreate(&child, 0);

// 设置图内存分配节点的参数。
cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;

// 指定设备 0 为这块内存的驻留设备。
allocNodeParams.alloc.poolProps.location.id = 0;

// 设置要分配的内存大小，单位为字节。
allocNodeParams.alloc.bytesize = size;

// 将内存分配节点添加到子图。
// 分配节点创建后，分配得到的设备指针保存在 allocNodeParams.alloc.dptr 中。
cudaGraphAddNode(&allocNode, child, NULL, NULL, 0, &allocNodeParams);

// 可在此处向子图添加使用该内存的其他节点。
// 这些节点应依赖于分配节点，并且应先于释放节点执行。

// 设置内存释放节点的参数，并指定要释放的设备指针。
cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = allocNodeParams.alloc.dptr;

// 将释放节点添加到子图，并声明它依赖于分配节点。
// 如果子图中还添加了其他访问该内存的节点，释放节点也必须依赖于这些节点，
// 以确保所有访问完成后再释放内存。
cudaGraphAddNode(&freeNode, child, &allocNode, NULL, 1, freeNodeParams);

// 创建父图，作为容纳子图的外层图。
cudaGraphCreate(&parent, 0);

// 将子图作为图节点加入父图，并把子图所有权转移给父图。
// 转移后，子图由父图管理，不能再作为独立图单独使用或销毁。
cudaGraphNodeParams childNodeParams = { cudaGraphNodeTypeGraph };
childNodeParams.graph.graph = child;
childNodeParams.graph.ownership = cudaGraphChildGraphOwnershipMove;
cudaGraphAddNode(&parentNode, parent, NULL, NULL, 0, &childNodeParams);
```
## 5.3 优化内存复用
CUDA 主要通过两种方式复用内存：
- **图内复用：** 类似流序内存分配器，CUDA 可以将同一段虚拟地址和底层物理内存分配给生命周期不重叠的不同分配对象。
- **图间复用：** CUDA 可以通过虚拟地址别名映射，让不同图的虚拟地址映射到同一段物理内存。
### 5.3.1 图内地址复用
在同一张图中，如果不同分配对象的生命周期不重叠，CUDA 可以把相同的虚拟地址范围分配给它们。因此，生命周期互不重叠的不同分配对象，其指针不一定是唯一的。

图中的示例展示了这一机制：新分配节点（节点 2）可以复用依赖节点（节点 1）释放的地址。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261027121502.png)

下图中新增的分配节点 **4** 没有依赖释放节点 **2**，因此 CUDA 无法保证释放节点 2 先于节点 4 执行。基于图中的依赖关系，节点 4 不能复用由释放节点 2 释放的地址。
节点 4 是否能使用其他地址，取决于图中其他分配与释放节点之间的依赖关系及其生命周期是否重叠；仅凭此图，无法判断它是否必须使用全新的地址。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261027122051.png)
### 5.3.2 物理内存的管理与共享
在 GPU 按执行顺序到达分配节点之前，CUDA 会为图分配内存建立虚拟地址到物理内存的映射。为减少物理内存占用和映射开销，如果多个图不会同时执行，CUDA 可以让它们的不同分配共用同一段物理内存。

但如果某段物理内存同时映射给多个正在执行的图，或者仍被尚未释放的图分配占用，就不能被其他分配复用。

在图实例化、启动或执行期间，CUDA 都可能调整物理内存映射。为避免不同图中仍然有效的分配指向同一段物理内存，CUDA 也可能在后续图启动之间加入同步。

程序必须遵守图分配内存的生命周期：只有在分配之后、释放之前才能访问该内存。若在生命周期之外访问指针，即使其虚拟地址唯一，也可能读到或改写已分配给其他对象的实时数据。使用 Compute Sanitizer 等工具可以检测这类错误。

下图示说明：多张图依次在同一条流中启动，因而不会并发执行；并且每张图都会释放自己分配的全部内存。在这种情况下，CUDA 可以让这些分配复用同一段物理内存。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261027122706.png)
## 5.4 性能注意事项
当多张图依次提交到同一条流时，CUDA 会尝试让它们复用同一段物理内存，因为这些图不会并发执行。为避免反复建立映射带来的开销，CUDA 会在图的多次启动之间保留已有的物理内存映射。

如果之后某张图可能与其他图并发执行，例如被提交到另一条流，CUDA 就必须重新映射内存，使并发执行的图使用不同的物理内存，避免数据相互覆盖。

以下操作可能触发图内存重新映射：
- 改变图的启动流；
- 修剪图内存池。修剪操作会显式释放未使用的内存；
- 重新启动一张图，而另一张图中尚未释放的分配仍映射到同一段物理内存。此时，CUDA 会在重新启动前调整映射。

重新映射必须按执行顺序进行，并且要等到该图之前的执行完成，避免解除仍在使用的内存映射。映射操作可能需要调用操作系统，因此开销相对较高。若应用始终将包含内存分配节点的图提交到同一条流，通常可以减少这类重新映射开销。
### 5.4.1 首次启动与cudaGraphUpload
图实例化时，CUDA 尚不知道该图将在哪条流中执行，因此无法提前分配或映射物理内存。内存映射会推迟到图启动时进行。

调用 `cudaGraphUpload` 可以提前完成该图首次启动所需的内存映射，并将图与指定流关联，从而把映射开销从后续启动路径中分离出来。如果之后仍在同一条流中启动该图，通常无需再次进行映射。

如果上传图和启动图时使用不同的流，其效果类似于切换图的启动流，可能触发内存重新映射。此外，其他内存池管理操作也可能从空闲流中回收内存，抵消预上传带来的效果。
## 5.5 物理内存占用
异步内存分配采用内存池管理。即使销毁了包含图内存节点的图，CUDA 也不一定会立即把相关物理内存归还给操作系统，哪怕图中的分配都已经释放。若要明确将这部分内存归还给操作系统，应用程序可以调用 `cudaDeviceGraphMemTrim`。
>注意：图中的释放节点执行后，该分配对象的生命周期结束，程序不应再访问它；CUDA 可以把对应的物理内存留在图内存池中，以便后续图分配复用。这样可以避免之后重新向操作系统申请内存以及重新建立映射的开销。所以，释放节点释放的是**图分配对象**，不一定会立即把底层物理内存归还给操作系统。只有调用 `cudaDeviceGraphMemTrim`，CUDA 才会尝试解除映射并归还图内存池中当前未使用的物理内存。

`cudaDeviceGraphMemTrim` 会解除映射并释放图内存池中当前未被使用的物理内存。尚未释放的分配，以及已安排执行或正在执行的图，仍被视为正在使用内存，因此不会受到修剪影响。修剪后，物理内存可供其他 CUDA 分配或其他进程使用；但被修剪的图下次启动时，CUDA 可能需要重新分配并映射内存，从而产生额外开销。

图内存池与流序内存分配器使用的内存池不同，因此 `cudaDeviceGraphMemTrim` 修剪的不是 `cudaMemPoolTrimTo()` 所管理的内存池。CUDA 提供 `cudaDeviceGetGraphMemAttribute`，供应用程序查询图内存池的物理内存占用：
- `cudaGraphMemAttrReservedMemCurrent`：当前进程中，驱动为图分配预留的物理内存总量。
- `cudaGraphMemAttrUsedMemCurrent`：当前至少映射给一张图的物理内存总量。
应用程序可以通过监测这些属性，了解图分配导致的物理内存增长，并评估图间内存共享机制节省了多少物理内存。
