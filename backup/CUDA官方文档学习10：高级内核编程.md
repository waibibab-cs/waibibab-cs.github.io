本章将首先深入讲解NVIDIA GPU的硬件模型，随后介绍CUDA核代码中用于提升核函数性能的若干高级特性。本章会引入与线程作用域、异步执行以及配套同步原语相关的一系列概念。这些理论阐述，是理解核代码内各类高级性能特性所必需的基础。

其中部分特性的详细说明，将在本编程指南后续章节的专题板块展开。

本章介绍的高级同步原语，完整内容见官当文档4.10节与4.11节。
异步数据拷贝（含张量内存加速器TMA）在本章进行引入，完整内容见官方文档4.12节。
# 一、使用PTX
并行线程执行（PTX）是CUDA用于对硬件指令集架构（ISA）进行抽象的虚拟机指令集架构，已于官方文档1.3.3节介绍。直接编写PTX代码属于高度进阶的优化手段，大多数开发者并不需要使用，仅应作为最后的备选方案。尽管如此，在某些场景下，直接编写PTX所带来的细粒度控制能力，能够为特定应用提升性能。这类场景通常出现在应用中对性能极度敏感的代码段——哪怕只有零点几个百分点的性能提升，也能带来显著收益。全部可用的PTX指令可查阅《PTX指令集架构文档》。

**cuda::ptx 命名空间**
在代码中直接使用PTX的一种方式，是借助libcu++库提供的cuda::ptx命名空间。该命名空间提供了与PTX指令一一对应的C++函数，简化了在C++程序内调用PTX指令的操作。更多详情，请参考cuda::ptx命名空间文档。

**内联PTX**
在代码中嵌入PTX的另一种方式是使用内联PTX。该方法在对应文档中有详细描述，其原理与在CPU上编写汇编代码十分相似。
# 二、硬件实现
流式多处理器（Streaming Multiprocessor，简称SM，参见GPU硬件模型）被设计为可并发执行数百个线程。为管理如此大规模的线程，它采用一种独特的并行计算模型——单指令多线程（Single-Instruction, Multiple-Thread，SIMT），相关介绍见《SIMT执行模型》章节。指令采用流水线方式执行，既利用单个线程内部的指令级并行，也通过并发硬件多线程挖掘大规模的线程级并行，详见《硬件多线程》章节。与CPU核心不同，SM按序发射指令，不做分支预测，也不执行推测执行。

《SIMT执行模型》与《硬件多线程》章节描述了所有GPU设备共有的SM架构特性。《计算能力》章节则针对不同计算能力的设备给出具体细节。

NVIDIA GPU架构采用小端序存储表示。
## 2.1 SIMT执行模型
每个流式多处理器（SM）以warp（线程束）为单位创建、管理、调度和执行线程，一个warp包含32个并行线程。构成warp的所有线程从同一个程序地址开始执行，但每个线程拥有独立的指令地址计数器与寄存器状态，因此可以独立地进行分支、执行不同代码。warp一词源自纺织工艺，是最早的并行线程相关术语。半线程束（half-warp）指一个warp的前16个或后16个线程；四分之一线程束（quarter-warp）则对应一个warp四等分后的其中一组。

> 同warp的每个线程拥有独立PC（指令地址计数器）+独立寄存器。刚启动的时候，32个线程PC相同，从同一个代码地址开始跑。但是每个线程都有自己单独的PC、单独的寄存器栈。理论上，每个线程可以跑到完全不同的代码地址，独立分支。
> 既然每个线程都有独立PC，为什么还会有分支发散？答案在下一段：虽然线程PC独立，但是warp的硬件指令发射单元，一次只能广播一条指令给warp内所有线程。

一个warp每次执行同一条公共指令。因此，当warp内全部32个线程的执行路径一致时，硬件可以达到最高执行效率。如果warp内线程因数据相关的条件分支产生路径分歧，warp会依次执行每一条被选中的分支路径，同时禁用不在该路径上的线程。分支发散仅发生在同一个warp内部；不同warp之间相互独立执行，无论它们执行的代码路径相同还是完全无关。

> CUDA的一个warp硬件在同一时刻仅能广播发射单条指令；若warp内32个线程程序计数器PC不一致产生分支分化时，硬件会串行执行不同分支路径：先执行if分支，将走else的线程标记为非活跃，这类线程不参与计算但仍占用指令发射周期，随后再切换执行else分支并禁用if侧线程。当warp出现分支 divergence，原本一轮指令发射就能完成的任务需要多轮发射，直接降低硬件执行效率，极端场景下性能会减半；只有warp内所有线程走完全相同分支时，全部32线程都处于活跃状态，硬件才能跑满100%效率。

SIMT架构与SIMD（单指令多数据）向量体系结构有相似之处：二者均由单条指令控制多个处理单元。但二者存在关键区别：SIMD向量架构会向软件暴露SIMD向量宽度；而SIMT指令定义的是单个线程的执行与分支行为。
与SIMD向量机不同，SIMT允许程序员编写两种代码：面向独立标量线程的线程级并行代码，以及面向协同线程的数据并行代码。从代码正确性角度来说，程序员基本可以忽略SIMT机制；但如果编写代码时尽量避免warp内线程发生分支发散，能够获得显著的性能提升。
这一点在实践中类似于高速缓存行：设计代码保证正确性时，可以完全不用关心缓存行大小；但想要达到峰值性能，就必须在代码结构上将其纳入考量。与之相对，向量架构需要软件手动将加载操作合并为向量，并自行处理分支发散问题。

> SIMD（如AVX/AVX512）与SIMT的共同点是单条指令控制多个计算单元；二者核心差异在于：SIMD的向量宽度对软件显式暴露，需要程序员手动打包向量数据、用软件mask处理分支，硬件无独立PC，所有向量lane必须同步执行，而SIMT以标量线程为编程单元，开发者编写单线程逻辑，由编译器和硬件自动将32个线程打包为warp，每个线程具备独立PC，支持自由分支；从编码角度，SIMT代码正确性不受warp分支发散影响，仅会带来性能损耗，这一点类似CPU缓存行——保证结果正确时无需关心warp，但追求峰值性能就要减少分支发散；相比SIMD，SIMT免去程序员手动合并访存与处理向量mask的负担，访存合并、线程打包交由硬件与编译器自动完成。

### 2.1.1 独立线程调度
在计算能力低于7.0的GPU上，线程束（warp）使用由该warp全部32个线程共享的单个程序计数器，同时搭配一个活动掩码，标记warp内处于活动状态的线程。因此，处于分歧代码区域或不同执行状态的同warp线程，无法互相发送信号或交换数据；依赖锁/互斥量实现细粒度数据共享的算法，可能会发生死锁，死锁是否出现取决于竞争线程所属的warp。

> 上述计算能力的GPU中，SIMT的硬件核心是：一个warp内32个线程共享同一个PC，依靠32位active mask控制线程是否参与运算，同一时刻PC只能指向一处代码地址，被mask禁用的线程停留在这个公共PC，仅不执行计算，并非持有独立PC等待。若warp线程发生分支分化，不同分支里的线程无法跨分支通信或互相等待，例如if分支线程自旋等待else分支线程写flag时，任一时刻只有一侧线程能运行，另一方被屏蔽，永远无法完成该同步逻辑。

在计算能力7.0及更高版本的GPU中，独立线程调度实现了线程间完全并发，不再受warp限制。启用独立线程调度后，GPU为每个线程维护独立的执行状态，包括程序计数器与调用栈，并且能够以单线程粒度让出执行权——目的或是为了更好利用执行资源，或是让某个线程等待其他线程产生的数据。调度优化器负责将同一个warp内的活动线程编组为SIMT执行单元。该机制保留了前代GPU SIMT执行的高吞吐特性，但灵活性大幅提升：线程现在可以在子warp粒度发生分支发散与重新汇聚。

> 该硬件升级核心为：打破传统SIMT架构下整个warp共用PC、必须整体调度的限制，为每个线程配备独立PC与独立调用栈，支持单线程粒度yield让出执行权，线程阻塞等待时仅暂停自身而不冻结整个warp；调度器动态将处于相同PC地址的活跃线程打包为子warp粒度的SIMT单元发射指令，同一warp内不同线程子集可各自组成SIMT组独立执行，既保留SIMT批量发射带来的高吞吐，又大幅提升分支发散场景下的执行灵活性。

独立线程调度会破坏那些依赖旧GPU架构隐式warp同步行为的代码。warp同步代码假设：同一个warp内的线程在每一条指令处都是锁步执行。但如今线程可以在子warp粒度发散、重新汇聚，使得该假设不再成立。这会导致实际参与代码执行的线程集合与预期不符。任何面向CC7.0之前GPU开发的warp同步代码（例如无需同步的warp内规约操作），都需要重新审查以保证兼容性。开发者应当使用__syncwarp()对这类代码做显式同步，确保代码在所有代际GPU上行为正确。

> 【⬆️讲解】
> 什么是隐式warp同步行为？老CUDA程序员写代码时，依赖一个旧硬件带来的隐性假设：同一个warp内所有线程，每一条指令都锁步执行（lockstep）。只要warp到达某一行代码，所有线程一定都到达这一行。
> 典型例子：无同步的warp内规约求和（intra-warp reduction）老代码写法（CC<7.0）：
> ```cpp
> // 旧代码，依赖隐式warp锁步，Volta之后会出错
> if (lane < 16) val += shfl_xor(val,16);
> if (lane < 8)  val += shfl_xor(val,8);
> if (lane < 4)  val += shfl_xor(val,4);
> ```
> 旧硬件：warp所有线程锁步，执行完第一行所有线程全部完成，才会进入下一行if，不需要同步。
> Volta独立线程调度摧毁了这个假设！
> 现在warp内线程是独立调度，部分线程可能已经跑到第2个if，还有部分线程停留在第1个if。
> shfl_xor（warp内寄存器交换）读取别的lane寄存器时，对方线程可能还没更新寄存器，读出脏数据，结果随机错误。
> 解决方案是使用__syncwarp()做显式warp内同步，强制所有warp线程汇合，保证执行到这一行时，全部线程都完成前面操作。
> ```cpp
> if (lane < 16) val += shfl_xor(val,16);
> __syncwarp();
> if (lane < 8)  val += shfl_xor(val,8);
> __syncwarp();
> ...
> ```

注意⚠️
线程束（warp）中参与当前指令执行的线程称为活动线程；不在当前指令执行路径上的线程则处于非活动状态（被禁用）。线程变为非活动线程存在多种原因：比warp内其他线程更早退出程序、选择了与warp当前正在执行的分支不同的分支路径，或是线程块总线程数不是warp大小整数倍时，块末尾多出的那些线程。
若一个warp执行非原子指令，且warp内多个线程向全局内存或共享内存的同一地址执行写操作：写入该地址的串行写操作次数会随设备计算能力不同而变化。但无论任何计算能力，最终由哪一个线程完成最后的写入，结果都是未定义的。
若一个warp执行原子指令，且warp内多个线程对全局内存的同一地址执行读-改-写（RMW）操作：每一次读-改-写操作都会完整执行，并且所有操作会被串行化；但这些操作的执行顺序是未定义的。

## 2.2 硬件多线程
当流式多处理器（SM）收到一个或多个待执行的线程块时，会将这些线程块切分为若干线程束（warp），每个warp由warp调度器负责调度执行。线程块切分为warp的方式固定不变：每个warp内的线程拥有连续递增的线程ID，第一个warp包含线程0。《线程层级》章节介绍了线程ID与线程块内线程索引之间的对应关系。

一个线程块内的warp总数定义如下：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261006211236.png)

T 为每个线程块内的线程数量，Wsize 是线程束大小，固定等于32，ceil(x, y) 表示将 x 向上取整至 y 的最近整数倍
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261006211249.png)
流式多处理器（SM）处理的每个线程束，其执行上下文（程序计数器、寄存器等）会在该warp的整个生命周期内保存在片上。因此，warp之间的切换没有开销。在每一个指令发射周期，warp调度器会挑选一个存在就绪线程、可以执行下一条指令的warp（即该warp内的活动线程），并向这些线程发射指令。

每个SM拥有一组32位寄存器，这些寄存器在各个warp之间进行划分；同时SM配备共享内存，共享内存在线程块之间划分。对于给定的核函数，SM上能够驻留并并发处理的线程块与warp数量，取决于核函数占用的寄存器与共享内存大小，以及该SM可用的寄存器、共享内存总量。每个SM还存在可驻留线程块和warp的上限。这些限制，以及SM可用寄存器与共享内存容量，由设备的计算能力决定，详见《计算能力》章节。如果单个SM没有足够资源来至少处理一个线程块，核函数将会启动失败。一个线程块所分配的寄存器与共享内存总量，可以通过《占用率》章节中记载的多种方式确定。
## 2.3 异步执行特性
新一代NVIDIA GPU内置了异步执行能力，可在GPU内部实现数据搬运、计算与同步操作之间更多的重叠。借助这些能力，GPU代码中发起的部分操作，可以与同一个线程块内的其他GPU代码异步执行。不要将此种异步执行，与2.5节介绍的CUDA异步API相混淆：后者用于让GPU核函数启动或内存操作之间、或是与CPU之间异步运行。

计算能力8.0（NVIDIA Ampere GPU架构）引入了硬件加速的全局内存到共享内存异步数据拷贝以及异步屏障（参见《NVIDIA A100 Tensor Core GPU架构》）。

计算能力9.0（NVIDIA Hopper GPU架构）通过张量内存加速器（TMA）单元扩展了异步执行特性。TMA单元可在全局内存与共享内存之间双向传输大块数据与多维张量；同时还新增异步事务屏障以及异步矩阵乘累加运算（详情参见深度解读Hopper架构的技术博客）。

CUDA提供可在设备代码中由线程调用的API，用于使用上述特性。异步编程模型定义了异步操作相对于CUDA线程的行为。

异步操作由CUDA线程发起，但实际执行是异步的，如同由另一个线程执行，我们称这个“线程”为异步线程（async thread）。在规范编写的程序中，一个或多个CUDA线程会与该异步操作进行同步；发起异步操作的CUDA线程不一定需要参与同步。异步线程始终与发起该操作的CUDA线程绑定。


异步操作依靠同步对象通知任务完成，同步对象可以是屏障（barrier）或者流水线（pipeline）。这类同步对象在《高级同步原语》章节详细说明，《异步数据拷贝》章节演示了它们在异步内存操作中的作用。
> CUDA线程：SM上跑通用代码的真实硬件线程；异步线程async thread：CUDA文档抽象名词，用来指代TMA/copy引擎后台执行异步拷贝/异步矩阵运算的硬件任务，不是warp线程，不能跑通用计算。

### 2.3.1 异步线程与异步代理
异步操作的内存访问方式可能与普通操作不同。为区分这些不同的内存访问方式，CUDA引入了异步线程（async thread）、通用代理（generic proxy）与异步代理（async proxy）概念。普通操作（加载与存储指令）经由通用代理执行。部分异步指令（例如 LDGSTS、STAS/REDAS）在模型上视作运行于通用代理中的异步线程。另一些异步指令（例如使用TMA的批量异步拷贝、部分张量核心指令 tcgen05.、wgmma.mma_async.）在模型上视作运行于异步代理中的异步线程。

运行在通用代理中的异步线程：发起异步操作时，会关联一个异步线程，该异步线程与发起本次操作的CUDA线程相互独立。对同一地址，在异步操作之前执行的、经由通用代理的普通加载/存储操作，保证发生在该异步操作之前。但是，异步操作之后对同一地址执行的普通加载/存储，不保证维持顺序；在异步线程完成之前，可能产生数据竞争。

运行在异步代理中的异步线程：发起异步操作时，会关联一个异步线程，该异步线程与发起本次操作的CUDA线程相互独立。对同一地址，无论是异步操作之前还是之后的普通加载/存储，均不保证内存顺序。需要使用代理栅栏（proxy fence）在不同代理之间做同步，保证内存序正确。《使用张量内存加速器（TMA）》章节演示了在TMA异步拷贝场景下，如何利用代理栅栏保证程序正确性。

> 【⬆️详解】
> CUDA引入proxy（代理）模型，本质是：GPU存在多条独立的内存访问硬件通路，不同通路有各自独立的内存顺序/可见性规则。
> • generic proxy：通用代理，SM普通load/store默认走这条通路。
> • async proxy：异步代理，TMA、wgmma.mma_async这类专用硬件引擎走这条独立通路。
> 异步操作会绑定一个抽象async thread，这个async thread跑在哪种proxy，决定了内存序规则。
> ① 运行在 Generic Proxy（通用代理）上的异步线程
> 指令例子：LDGSTS、STAS、REDAS
> • 硬件通路：仍然走SM的通用访存通路，不是独立硬件引擎；只是操作本身被建模为async thread。
> • 内存序规则：对同一个内存地址✅在发起异步操作之前的普通load/store（generic proxy）：保证排在异步操作前面执行，顺序不被重排。❌ 在发起异步操作之后的普通load/store：没有顺序保证，可能和异步操作乱序执行，产生数据竞争。
> ② 运行在 Async Proxy（异步代理）上的异步线程
> 指令例子：TMA批量异步拷贝、tcgen05.、wgmma.mma_async.（Hopper张量核心异步指令）
> • 硬件通路：完全独立的硬件引擎（TMA单元、张量核心硬件），不和SM普通load/store共享访存流水线。
> • 内存序规则：对同一个内存地址❌发起异步操作之前、之后的普通load/store，全都不保证内存顺序。✅ 想要保证可见性，必须显式插入 proxy fence（代理栅栏），打通 async proxy 和 generic proxy 两条通路。
> 示例场景（TMA拷贝全局内存到共享内存）：
> 1. CUDA线程在generic proxy里写全局内存缓冲区；
> 2. 发起TMA异步拷贝（async proxy通路）；
> 3. 如果不加proxy fence：TMA硬件引擎可能读到旧数据，哪怕你在TMA启动之前已经写好了全局内存。哪怕你用barrier等待TMA任务完成，barrier只能确认拷贝硬件动作结束；barrier不会自动同步两个proxy之间的内存可见性。这是Hopper架构代码最容易踩的大坑。
# 三、线程作用域
CUDA线程构成一套线程层级结构，用好该层级结构，是编写正确且高性能CUDA核函数的关键。在这套层级结构中，内存操作的可见性与同步作用域可以是不同的。为了描述这种非一致特性，CUDA编程模型引入线程作用域（thread scope）的概念。线程作用域定义：哪些线程能够观察到某个线程的加载/存储操作，同时规定哪些线程之间可以借助原子操作、屏障等同步原语进行互相同步。每一个作用域在内存层级中都对应一个一致性点（point of coherency）。

线程作用域在CUDA PTX中开放使用，同时也作为扩展功能在libcu++库中提供。下表列出可用的线程作用域：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261006211310.png)
# 四、高级异步原语
本节介绍三类同步原语：
作用域原子操作（Scoped Atomics）：将C++内存序与CUDA线程作用域相结合，用以在线程块、集群、设备或系统作用域内安全地完成线程间通信（参见《线程作用域》章节）。
异步屏障（Asynchronous Barriers）：将同步拆分为到达阶段与等待阶段，可用于跟踪异步操作的执行进度。
流水线（Pipelines）：对任务进行分段处理，协调多缓冲区生产者-消费者模型；常用于将计算与异步数据拷贝进行重叠执行。
## 4.1 作用域原子操作
本节重点介绍支持C++标准原子内存语义的作用域原子操作，这类原子可通过libcu++库或编译器内置函数使用。作用域原子操作提供了在CUDA线程层级的合适粒度下实现高效同步的工具，能够让复杂并行算法同时保证正确性与性能。
### 4.1.1 线程作用域与内存序
作用域原子操作结合了两大核心概念：
线程作用域（Thread Scope）：定义哪些线程能够观察到该原子操作产生的效果
内存排序（Memory Ordering）：定义该原子操作相对于其他内存操作的顺序约束
```cpp
#include <cuda/atomic>
__global__ void block_scoped_counter() {
    // Shared atomic counter visible only within this block
    __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;
    // Initialize counter (only one thread should do this)
    if (threadIdx.x == 0) {
        counter.store(0, cuda::memory_order_relaxed);
    }
    __syncthreads();
    // All threads in block atomically increment
    int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);
    // Use old_value...
}
```
本示例实现了一个线程块作用域原子计数器，用以演示作用域原子操作的基础概念：
• 共享变量：使用__shared__内存，在块内所有线程之间共享唯一一个计数器。
• 原子类型声明：cuda::atomic<int, cuda::thread_scope_block> 创建一个在线程块范围内可见的原子整型变量。
• 单次初始化：仅由0号线程完成计数器初始化，避免初始化阶段产生数据竞争。
• 线程块同步：syncthreads() 保证所有线程在继续执行前，都能看到已经初始化完成的计数器。
• 原子自增：每个线程对计数器执行原子自增，并获取自增前的旧值。
此处选用cuda::memory_order_relaxed，是因为我们仅需要原子性（不可分割的读-改-写操作），而不需要不同内存地址之间的顺序约束。由于这是简单的计数操作，自增的先后顺序不会影响程序正确性。
对于生产者-消费者模型，则需要通过获取-释放（acquire-release）语义保证正确的内存顺序：
```cpp
__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;
    if (threadIdx.x == 0) {
        // Producer: write data then signal ready
        data = 42;
        ready.store(true, cuda::memory_order_release);  // Release ensures data write is visible
    } else {
        // Consumer: wait for ready signal then read data
        while (!ready.load(cuda::memory_order_acquire)) {  // Acquire ensures data read sees the write
            // spin wait
        }
        int value = data;
        // Process value...
    }
}
```

> 【cuda/cpp内存序介绍】
> 内存序本质：控制编译器 + GPU硬件能否对内存读写指令重排，以及内存写操作何时对其他线程可见。
> 只有原子变量（cuda::atomic / std::atomic）才可以指定内存序；普通__shared__/全局内存变量没有内存序。
> CUDA libcu++ 完全复用 C++ 标准 5 种内存序，同时搭配 thread_scope（block/cluster/device/system）。
> 1. memory_order_relaxed 松弛序
> 只保证一件事：原子操作本身不可分割（原子性）。
> ✅ 原子操作不会被其他线程的原子操作打断。
> ❌ 不提供任何内存顺序约束，不保证其他内存读写的可见性。
> 适用场景：单纯计数，只关心原子变量本身的值，原子操作和其他地址的数据没有依赖关系。
> ⚠️ 绝对不能用于生产者消费者！
> ```cpp
> // 错误写法！
> data = 42;
> ready.store(true, relaxed); // relaxed不能保证data写在ready前面
> ```
> 编译器/GPU硬件可能重排：先写ready，再写data。消费者读到ready=true的时候，data还没写完。
> 2. memory_order_release 释放（只用于原子store写操作）
> 写屏障语义：release store之前的所有内存读写，不能被重排到这个store之后。
> 通俗：在release原子写之前完成的所有普通写，都会“打包”跟着这个原子写对外可见。
> ```cpp
> data = 42;                     // 普通写
> ready.store(true, release);   // release：保证data写一定在ready=true之前完成
> ```
> data=42 不会被重排到ready.store后面。
> 3. memory_order_acquire 获取（只用于原子load读操作）
> 读屏障语义：acquire load之后的所有内存读写，不能被重排到这个load之前。通俗来说，只要读到acquire load的结果，那么这个load之后的代码，一定能看到release打包过来的数据。
> ```cpp
> while (!ready.load(acquire)) {} // acquire
> int val = data;                 // 这行一定在load成功之后执行，能看到data=42
> ```
> ✅ Release-Acquire 配对（生产者消费者经典模型）
> 生产者：普通数据写入 → atomic.store(release)
> 消费者：atomic.load(acquire)成功 → 读取普通数据
> 规则：A线程release写，B线程acquire读到A写入的值，则A中release之前所有写，对B中acquire之后全部可见。这就是前面那段data + ready代码的原理。

### 4.1.2 性能考虑
尽可能使用最小范围的作用域：线程块作用域原子操作远快于系统作用域原子操作。
优先选用更弱的内存序：仅在保证程序正确性必需时，才使用更强的内存序。
考虑内存位置：共享内存原子操作比全局内存原子操作速度更快。
## 4.2 异步屏障
异步屏障与典型的单阶段屏障（syncthreads()）有所不同：线程通知自身已抵达屏障（到达操作），和等待其他线程抵达屏障（等待操作）这两个行为相互分离。这种分离机制允许线程执行与屏障无关的额外运算，更充分地利用等待时间，从而提升执行效率。异步屏障可用于实现CUDA线程间的生产者-消费者模型；也可让拷贝操作在完成时向屏障发送“到达”信号，以此在内存层次结构中实现异步数据拷贝。

> 【⬆️讲解】
> 1. syncthreads()：单阶段屏障
> 当block内所有线程执行到__syncthreads()这一行时：线程必须停在这里（原地阻塞）；等block里全部线程都抵达这一行；全部到齐之后，才统一放行，继续往下跑。
> 👉 到达arrive 和 等待wait是绑定在一起的同一个动作。线程到达屏障就必须原地等待，不能做别的计算。
> 2. 异步屏障 cuda::barrier：arrive 和 wait 分离
> • arrive（到达登记）：线程调用arrive()，只是告诉屏障：我这边任务做完了。调用arrive之后，线程不会阻塞，可以继续执行后续计算！
> • wait（等待）：线程后续在某个位置调用wait()，此时线程才会阻塞，直到屏障收集到预设数量的arrive信号。

异步屏障可在计算能力7.0及以上的设备上使用。计算能力8.0及以上的设备为共享内存中的异步屏障提供硬件加速，并在同步粒度上实现重大改进：支持对线程块内任意CUDA线程子集进行硬件加速同步。更早的架构仅支持在整个线程束（syncwarp()）或整个线程块（syncthreads()）级别做硬件加速同步。

CUDA编程模型通过cuda::std::barrier提供异步屏障，这是libcu++库中符合ISO C++标准的屏障。除实现标准std::barrier之外，该库还提供CUDA专属扩展，可选择屏障的线程作用域以提升性能，并暴露底层cuda::ptx接口。cuda::barrier能够与cuda::ptx互操作：通过友元函数cuda::device::barrier_native_handle()获取屏障的原生句柄，再将句柄传入cuda::ptx系列函数。CUDA还提供一套原语API，用于线程块作用域下共享内存中的异步屏障。

> ① 高层API：cuda::std::barrier（libcu++）
> • 兼容C++标准std::barrier，写法贴近C++，可读性好；
> • CUDA扩展：可以指定thread_scope（block/cluster/device），缩小同步范围，提升性能。
> ② 底层PTX互操作
> 通过barrier_native_handle()拿到barrier底层原生句柄，传入cuda::ptx内置函数，直接操作PTX barrier指令。
> 适合极致性能调优，直接控制硬件原语，一般工程代码优先用高层cuda::std::barrier。

下表概述了可在不同线程作用域执行同步的异步屏障。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261006211345.png)


同步的时间分离（Temporal Splitting of Synchronization）
如果不使用支持到达-等待机制的异步屏障，在线程块内部实现同步，需要借助__syncthreads()；若使用协作组（Cooperative Groups），则通过block.sync()完成。
```cpp
#include <cooperative_groups.h>
__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();
    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */
         // Wait for all threads to arrive here.
        block.sync();
        /* code after wait */
    }
}
```
线程会在同步点（block.sync()）处阻塞，直到所有线程都抵达该同步点。此外，同步点之前发生的内存更新，保证在同步点之后对块内所有线程可见。
该模式分为三个阶段：
1. 同步前代码：执行内存更新，这些更新将在同步之后被读取。
2. 同步点。
3. 同步后代码：可以看到同步点之前完成的内存更新。
如果改用异步屏障，则这种时间分离式同步模式如下。
```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);
__global__ void split_arrive_wait(int iteration_count, float *data)
{
  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  __shared__ barrier_t bar;
  auto block = cooperative_groups::this_thread_block();
  if (block.thread_rank() == 0)
  {
    // Initialize barrier with expected arrival count.
    init(&bar, block.size());
  }
  block.sync();
  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // This thread arrives. Arrival does not block a thread.
    barrier_t::arrival_token token = bar.arrive();
    compute(data, i);
    // Wait for all threads participating in the barrier to complete bar.arrive().
    bar.wait(std::move(token));
    /* code after wait */
  }
}
```
在该模式下，同步点被拆分为到达点（bar.arrive()）与等待点（bar.wait(std::move(token))）。线程首次调用bar.arrive()时，就开始参与这个cuda::barrier。当线程调用bar.wait(std::move(token))，线程将被阻塞，直到所有参与线程完成指定次数的bar.arrive()调用；该次数是屏障初始化时传入的预期到达计数参数。
参与线程在调用bar.arrive()之前完成的内存更新，能够保证：参与线程在调用bar.wait(std::move(token))之后，可以看到这些内存更新。请注意：bar.arrive()调用不会阻塞线程，线程可以继续执行其他工作，只要这些工作不依赖其他参与线程在bar.arrive()之前产生的内存更新。
到达-等待模式分为五个阶段：
1. 到达点之前的代码：执行内存更新，这些更新将在等待操作之后被读取。
2. 到达点：附带隐式内存栅栏（等价于 cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)）。
3. 到达点与等待点之间的代码。
4. 等待点。
5. 等待点之后的代码：可以看到到达点之前完成的内存更新。
关于异步屏障使用的完整指南，请参阅《异步屏障》章节。

> 【⬆️详解】
> 1.barrier 参与机制与 token（令牌）
> 线程第一次调用 bar.arrive()，就代表加入这个barrier的同步组。
> arrive() 会返回一个 token（令牌），这个令牌必须通过 std::move 移动，传给 bar.wait()。
> ◦ 语义：这个token代表本次arrive对应的同步事件；
> ◦ 移动语义：token所有权转移，不能重复使用，防止错误调用。
> barrier初始化时需要传入预期到达计数(expected arrival count)。
> 2.内存可见性规则
> ✅ 规则：参与线程在**arrive()之前完成的内存写操作，在所有参与线程执行完wait()之后**，保证全部可见。
> ❗ 不是arrive之后立刻可见；可见性保证生效的节点是 wait() 返回之后。
> 3.arrive() 不阻塞 + 中间代码的限制
> bar.arrive() 调用完，线程不会停下阻塞，可以继续跑代码。arrive 和 wait 中间写的代码，不能依赖其他线程 arrive 之前写入的共享数据。
> 原因：此时barrier还没有集齐所有arrive，同步没有完成，其他线程的内存写还没有全局可见。
> 中间这段代码只能做独立计算，不读取本次同步依赖的共享数据。
> 这正是性能优化点：利用原本原地阻塞等待的时间，执行无关计算，实现计算与等待重叠。

## 4.3 流水线
CUDA编程模型提供流水线同步对象作为一种协同机制，用于将异步内存拷贝编排为多个阶段，便于实现双缓冲或多缓冲的生产者-消费者模式。流水线是一个拥有头部与尾部的双端队列，按照先进先出（FIFO）顺序处理任务。生产者线程向流水线头部提交任务，消费者线程从流水线尾部取出任务。

流水线可通过 libcu++ 库中的 cuda::pipeline API 以及一套原语API对外提供能力。下表介绍这两套API的主要功能。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261006211408.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261006211417.png)
cuda::pipeline API 接口更丰富，限制更少；而原语 API 仅支持跟踪从全局内存到共享内存的异步拷贝，且有特定的大小与对齐要求。原语 API 的功能等价于作用域为 cuda::thread_scope_thread 的 cuda::pipeline 对象。

有关详细使用范式与示例，请参见《Pipelines》章节。
# 五、异步数据拷贝
在存储层次结构内实现高效的数据搬运，是GPU计算获得高性能的基础。传统同步内存操作会迫使线程在数据传输期间空闲等待。GPU本身依靠并行机制隐藏内存延迟：当内存操作执行时，流式多处理器（SM）会切换去执行另一个线程束。即便依靠并行机制实现了延迟隐藏，内存延迟依然可能成为瓶颈，限制内存带宽利用率与计算资源效率。为解决这类瓶颈，现代GPU架构提供硬件加速的异步数据拷贝机制，允许内存传输独立执行，同时线程继续执行其他任务。

异步数据拷贝将启动内存传输与等待传输完成两个动作解耦，实现计算与数据搬运的重叠。如此一来，线程可以在内存延迟周期内执行有效任务，从而提升整体吞吐量与资源利用率。

尽管本节所述的底层概念与原理和前面“异步执行”章节内容相近，但那一章介绍的是内核启动与内存传输（例如由cudaMemcpyAsync发起的传输）这类异步操作，属于应用程序不同组件之间的异步。本节所描述的异步，指在GPU的DRAM（即全局内存）与SM片上存储器（如共享内存、张量存储器）之间传输数据时，不会阻塞GPU线程。该异步发生在单次内核调用的执行过程内部。

想要理解异步拷贝如何提升性能，可以先研究一种常见的GPU计算范式。CUDA应用经常采用拷贝-计算（copy and compute）范式，流程如下：
1.从全局内存读取数据；
2.将数据存入共享内存；
3.在共享内存的数据上执行计算，并可能将结果写回全局内存。
该范式中的拷贝阶段通常写作 `shared[local_idx] = global[global_idx]`。编译器会把这行全局内存到共享内存的拷贝，展开为两条操作：先从全局内存读入寄存器，再从寄存器写入共享内存。

当该范式出现在迭代算法中时，每个线程块在执行完 `shared[local_idx] = global[global_idx]` 赋值操作后，都需要执行同步。目的是确保共享内存的所有写入操作全部完成，才可以进入计算阶段。在计算阶段结束后，线程块还需要再次同步，防止在所有线程完成计算前覆盖共享内存。下面的代码片段展示了该范式。
```cpp
#include <cooperative_groups.h>

__device__ void compute(int* global_out, int const* shared_in) {
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory.
}

__global__ void without_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // Compute the index of the current batch for this block in global memory.
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    size_t global_idx = block_batch_idx + threadIdx.x;
    shared[local_idx] = global_in[global_idx];

    // Wait for all copies to complete.
    block.sync();

    // Compute and write result to global memory.
    compute(global_out + block_batch_idx, shared);

    // Wait for compute using shared memory to finish.
    block.sync();
  }
}
```
> 【⬆️代码详解】
> 这是不使用异步拷贝的经典CUDA分块（tiled）拷贝计算内核，基于cooperative_groups（协同组），对应文档前面说的 copy and compute pattern，用来对比后面的TMA/pipeline异步版本。
> 具体来说，代码将全局内存global_in的数据分批次搬运到共享内存shared，在共享内存上做计算，最后写结果回全局内存。每一轮循环：拷贝 → 块同步 → 计算 → 块同步，一轮做完再进入下一批次。
> ```cpp
> __device__ void compute(int* global_out, int const* shared_in) {
>     // Computes using all values of current batch from shared memory.
>     // Stores this thread's result back to global memory.
> }
> ```
> 设备端函数：从共享内存shared_in读取当前批次数据做计算，结果写回全局内存global_out。这里只是函数声明，不关心具体计算逻辑。
> ```cpp
> __global__ void without_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
>   auto grid = cooperative_groups::this_grid();
>   auto block = cooperative_groups::this_thread_block();
>   assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size
> ```
> • this_grid()：当前网格（所有线程块）
> • this_thread_block()：当前线程块
> • assert：仅用于示例说明，保证输入总大小 = 批次数量 × grid线程块数量，属于演示用约束，实际代码一般删掉。
> ```cpp
>   size_t local_idx = block.thread_rank();
> ```
> thread_rank()：线程在当前block内的索引，等价threadIdx.x。
> 随后循环遍历每一个批次（batch），每一轮处理一块数据。
> 1.计算当前线程在全局内存中的地址global_idx
> 2.`shared[local_idx] = global_in[global_idx];`
> 这就是前面文档提到的普通全局→共享内存拷贝：编译器展开为：全局内存load进寄存器 → 寄存器store到共享内存。
> 剩下就是进行块同步-计算-在进行一次块同步

借助异步数据拷贝，全局内存到共享内存的数据搬移可以异步执行，从而在等待数据传输完成期间，更高效地利用流式多处理器（SM）。
```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__device__ void compute(int* global_out, int const* shared_in) {
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory.
}

__global__ void with_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // Compute the index of the current batch for this block in global memory.
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;

    // Whole thread-group cooperatively copies whole batch to shared memory.
    cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, block.size());

    // Compute on different data while waiting.

    // Wait for all copies to complete.
    cooperative_groups::wait(block);

    // Compute and write result to global memory.
    compute(global_out + block_batch_idx, shared);

    // Wait for compute using shared memory to finish.
    block.sync();
  }
}
```
cooperative_groups::memcpy_async 函数将 block.size() 个元素从全局内存拷贝至共享内存。该操作的执行效果等价于由另一个线程完成拷贝；拷贝完成后，该线程会与当前线程调用的 cooperative_groups::wait 操作同步。在拷贝操作完成前，修改源全局内存数据，或是读写目标共享内存，都会引发数据竞争。

本示例阐释了所有异步拷贝操作背后的核心思想：将内存传输发起动作与传输完成等待动作解耦，让线程在后台搬运数据期间执行其他任务。CUDA编程模型提供多套API来使用该能力：包括协同组与libcu++库中提供的memcpy_async系列函数，以及底层的cuda::ptx原语API。这些API拥有相近语义：它们把对象从源地址拷贝到目标地址，行为等价于另一个独立线程执行拷贝；拷贝完成后，可借助不同的完成机制进行同步。

现代GPU架构提供多种用于异步数据搬移的硬件机制：
• LDGSTS（计算能力8.0及以上）：支持高效的小规模全局内存到共享内存的异步传输。
• 张量内存加速器（TMA，计算能力9.0及以上）：对上述能力进行扩展，提供针对大规模多维数据传输优化的批量异步拷贝操作。
• STAS指令（计算能力9.0及以上）：支持小规模从寄存器到集群内分布式共享内存的异步传输。

这些机制拥有不同的数据通路、传输大小与对齐要求，开发者可根据自身特定的数据访问模式选择最合适的方案。下表概述了GPU内部异步拷贝所支持的数据通路。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261006211452.png)

# 六、L1缓存与共享内存的权衡配置
正如L1数据缓存章节所述，SM上的L1缓存与共享内存复用同一份物理资源，该资源称为统一数据缓存（unified data cache）。在大多数架构上，如果内核很少使用甚至不使用共享内存，可将统一数据缓存配置为架构所允许的最大L1缓存容量。

为共享内存预留的统一数据缓存空间可按单个内核进行配置。应用程序可以在内核启动前调用cudaFuncSetAttribute函数，设置预留份额（carveout），即期望分配的共享内存容量。
```cpp
cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
```

应用程序可将预留份额设置为该架构最大共享内存容量的整数百分比。除百分比数值外，还提供三个便捷枚举值作为预留份额参数：
• cudaSharedmemCarveoutDefault：默认配置
• cudaSharedmemCarveoutMaxL1：最大化L1缓存
• cudaSharedmemCarveoutMaxShared：最大化共享内存
不同架构支持的最大共享内存容量与可用预留份额各不相同；详细信息请参见《各计算能力对应的共享内存容量》章节。

当所选的整数百分比预留份额无法精确匹配硬件支持的共享内存容量时，系统会选用大于该值的下一档可用容量。例如，计算能力12.0的设备最大共享内存容量为100KB；若将预留份额设为50%，最终得到的共享内存是64KB而非50KB。原因是计算能力12.0设备支持的共享内存档位为：0、8、16、32、64、100KB。

传给cudaFuncSetAttribute的函数必须使用__global__限定符声明。cudaFuncSetAttribute对驱动而言只是提示（hint）；如果内核执行有需要，驱动可以选择其他预留份额。

还有另一项CUDA API：cudaFuncSetCacheConfig，应用程序同样可以使用它来调整内核的L1缓存与共享内存之间的资源分配比例。但该API会对内核启动时的共享内存/L1缓存配比设置硬性要求。因此，交替启动具有不同共享内存配置的内核时，会因共享内存的重新配置而产生不必要的启动串行化。
推荐使用cudaFuncSetAttribute，因为驱动可根据执行函数的需要，或是为了避免资源抖动，自行选用其他配置。

若内核需要每个线程块的共享内存分配量超过48KB，则该内核具备架构相关性。这类内核必须使用动态共享内存，而非静态大小数组，并且需要通过cudaFuncSetAttribute显式启用，方式如下。
```cpp
// Device code
__global__ void MyKernel(...)
{
  extern __shared__ float buffer[];
  ...
}

// Host code
int maxbytes = 98304; // 96 KB
cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
MyKernel <<<gridDim, blockDim, maxbytes>>>(...);
```
> 【⬆️解释】
> 当单个线程块需要 >48KB 的共享内存（Ampere/Hopper等架构默认静态shared上限是48KB），就不能用静态__shared__ float buf[xxx]，必须采用动态共享内存，并且主机端调用cudaFuncSetAttribute声明该内核允许使用更大的动态共享内存上限。
> 两个配套API：
>1.cudaFuncAttributePreferredSharedMemoryCarveout：设置统一SRAM在Shared/L1之间的划分偏好（前面讲的carveout）
>2.cudaFuncAttributeMaxDynamicSharedMemorySize：声明这个kernel最多可以使用多大的动态共享内存（本示例代码）