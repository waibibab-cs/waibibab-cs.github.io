# GPU基础
GA100 (A100) GPU 由 128 个独立 SM（流式多处理器）与全局共享 L2 缓存构成，硬件层级与 CUDA 编程模型一一对应：整块 GPU 对应 Kernel 的 Grid，SM 负责独立执行 CUDA Block，Block 不可跨 SM；每个 SM 内部包含大量 SP 流处理器、寄存器堆、Tensor Core 张量核心、SFU 特殊功能单元、Warp 调度器以及 L1/Shared Memory 统一片上存储，32 个 SP 组成硬件最小调度单元 Warp，对应 CUDA Thread 线程，遵循 SIMT 执行模式；SM 内寄存器与共享内存仅对本 SM 内线程可见，SM 间通信依赖 L2 或显存，SM 的寄存器、共享内存资源会限制可驻留的 Block/Warp 数量，线程束分化会带来性能损耗，优化时应尽量利用片上存储降低显存访问开销。
同一个 warp 内的 32 个线程，硬件同一时刻只能发射同一条指令，分支不同会发生线程束分化；该限制仅作用于 warp 内部，同一个线程块下不同 warp、不同 block 的 warp 互相独立，可以同时执行不同指令。Warp 调度器是 SM 内部硬件单元，以 warp 为调度粒度，在就绪 warp 之间切换，当 warp 因访存阻塞时调度其他 warp 执行，以此掩盖内存延迟，充分利用计算单元；SM 上可驻留 warp 数量受寄存器与共享内存资源约束。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831101318.png)
GA100/A100 GPU 内存层级遵循 “离 SM 越近速度越快、容量越小、成本越高”：Shared Memory 与 L1 Cache 位于 SM 内部，属于每个 SM 私有 SRAM；L2 Cache 是 GPU 片上共享 SRAM；Global Memory (HBM) 是 GPU 芯片外部 DRAM 显存。访问延迟：Shared Memory (19‑23 周期) < L1 (33 周期) < L2 (200 周期) < Global Memory (290 周期)。SRAM 相比 DRAM 成本高约 100 倍，速度约快 8 倍。Shared Memory 供 Block 内部线程显式高速交互；L1、L2 由硬件自动缓存数据；跨 SM 的数据通信只能走 L2 与全局显存，显存访问延迟最高，是程序主要延迟来源，优化应尽可能使用片上存储，降低对全局显存访问。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831102507.png)
下图为 CUDA 软件内存模型，定义各类内存的访问权限与作用域：寄存器是单线程私有最快存储；Local Memory 同样线程私有，但物理落在全局显存，用于寄存器溢出场景，性能差；Shared Memory 为 Block 范围内共享，位于 SM 片上 SRAM，仅本 Block 线程可见；Global Memory 对整个 Grid 所有线程可读可写，跨 Block 数据交换只能通过 Global Memory（包含被L1/L2缓存命中的情况）；Constant Memory 全局线程只读，由 CPU 主机负责写入初始化。CPU Host 仅可与 Global、Constant 内存传输数据，无法直接访问寄存器、Shared、Local 内存。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831102926.png)
# TPU基础
GPU、TPU 以及众多 AI 加速器高层思想是相似的：轻量化控制单元 + 大规模矩阵乘计算单元 + 高带宽存储；但是执行模型、硬件组织有重要差异。
TPU的核心计算单元主要分为以下三个：
* Scalar Unit：派发指令、调度VPU/MXU、处理少量标量运算
* VPU：做逐元素向量运算，如激活函数ReLU/GeLU、加减、归一化；负责数据搬运，把数据预处理之后送入矩阵单元 MXU
* MXU：矩阵乘法单元，专门做矩阵乘法
此外还包含两类片上存储：
* Smem：给Scalar Unit使用的片上高速存储
* Vmem：VPU 向量单元的片上本地存储

![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831103210.png)
对比图：

| 项目   | NVIDIA GPU(A100)                  | Google TPU                    |
| ---- | --------------------------------- | ----------------------------- |
| 控制单元 | SM 内 Warp Scheduler，大量 SP 标量线程    | Scalar Unit，轻量调度控制器           |
| 矩阵硬件 | Tensor Core，分散在每一个 SM 内部          | MXU 大矩阵单元，单块矩阵加速器             |
| 普通计算 | SP 流处理器，线程粒度                      | VPU 向量单元，向量粒度                 |
| 片上存储 | Register + Shared/L1 per‑SM       | Smem + Vmem per‑TC            |
| 主存   | HBM                               | HBM                           |
| 执行模型 | SIMT，有 Warp 线程束，Grid‑Block‑Thread | 向量 / 矩阵指令，**无 Warp，只有 Block** |
| 擅长   | 矩阵乘 + 各类通用不规则并行任务                 | 矩阵运算极强；非矩阵任务相对吃亏              |
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831104034.png)
# 缓解memory bound的策略
屋顶线 Roofline 模型：横轴运算强度 FLOPS/Byte，纵轴实际算力。斜线段为不同内存层级的带宽限制，水平段为 CPU/GPU 计算单元算力上限。运算强度低时为内存受限，如稀疏矩阵乘；强度足够高达到屋顶线则为计算受限，稠密矩阵乘可打满算力；ML 优化核心就是提高运算强度，规避内存瓶颈。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831104315.png)

**0.控制发散**
控制流分支发散 (Control divergence) 是 GPU‑SIMT 执行模型带来的性能问题，不属于内存瓶颈。同一个 warp 内所有线程同一时刻只能执行同一条指令；若 warp 内部线程走不同 if‑else 分支，硬件会串行依次执行各个分支路径，不走该分支的线程计算单元闲置空转，引入额外开销；分支发散仅发生在 warp 内部，不同 warp 之间条件分支不会产生发散开销。
其中，AXBYZ的执行顺序是谓词执行 (predicated execution) 示意，编译器把分支消除，每条指令附带线程掩码，指令流交错；真实硬件一般来说，大块分支会完整跑完 if 再跑 else（ABXYZ），小块执行谓词执行（AXBYZ）
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831105248.png)

**1.低精度计算**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831105339.png)
举例子说明降低数据精度可以提升算术运算强度，以向量逐元素 ReLU 操作为例：
1. FP32 单精度：每个元素读 4 字节、写 4 字节，共 8 字节内存访问，完成 1 次浮点运算，运算强度为 **8 字节 / FLOP**。
2. FP16 半精度：每个元素读 2 字节、写 2 字节，共 4 字节内存访问，同样只完成 1 次浮点运算，运算强度变为 **4 字节 / FLOP**。

现代 GPU 依靠**低精度 / 混合精度**加速计算，张量核 (Tensor Core) 是典型硬件：输入使用 16 比特（FP16/BF16）做乘法，**累加器用 FP32 高精度做求和**，最终输出 FP32，兼顾速度与累加精度。
1. **适合 16 比特存储的运算**：矩阵乘法、大部分逐元素运算（ReLU、tanh、加减乘）。
2. **需要更高计算精度 (FP32) 的运算**：大数累加小数容易产生舍入误差；规约类操作，如求和、softmax、归一化。
3. **需要更大数值范围 (FP32/BF16) 的运算**：指数、对数、幂这类输出和输入差距很大的逐元素算子，还有损失函数。
> 混合精度思想：输入乘法用低精度提速省带宽，**累加保持高精度避免精度丢失**；不同算子根据数值范围、累加特性选择合适精度。

FP8 通过把每个数压缩到 8 bit 来显著降低存储和带宽成本，但必须在动态范围与表示精度之间做权衡，因此出现了 E4M3 和 E5M2 两种主要格式。E4M3 尾数更多、精度更好，E5M2 指数更多、范围更大。
由于 FP8 本身范围有限，实际训练中通常需要对数据进行 scaling。普通 FP8 往往整块数据共享一个 scale，而 Blackwell 的 MXFP8 采用更细粒度的多 scaling factor（例如每 32 个值一个 scale），从而更好适配局部数据分布，并使 E4M3 这样的高精度 FP8 格式更可用。
但 MXFP8 的代价是数据布局和转置变复杂：scale 与局部 block 绑定后，矩阵转置不再只是交换行列，还涉及 block 重组与 scale 重解释，因此 forward、dgrad、wgrad 等不同 GEMM 可能需要 rowwise / columnwise 的不同低精度表示。

![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831110253.png)

**2.算子融合**
将多个连续小算子合并为单个 GPU kernel，中间结果保留在片上存储，不写回全局显存，减少显存读写与 kernel 启动开销，提高算术强度，缓解内存带宽瓶颈。多用于矩阵乘 / 卷积后接逐元素激活；受寄存器、共享内存大小约束，并非越多算子融合越好。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831111900.png)

**3.重计算**
Recomputation（Activation Recomputation / Gradient Checkpointing）的核心，是利用“中间激活可以由前面的 checkpoint 重新计算”这一性质，在前向传播时只保存少量关键 activation，丢弃其余中间结果；反向传播需要这些值时，再执行部分前向计算重新生成它们，从而以额外计算换取显著的显存节省。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831112842.png)
下面举一个实际的简单例子，通过重计算s1与s2，将反向传播计算需要读s1、s2、out变成只需要读取x（用于重计算）与out
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831113126.png)

**4.合并访存**
GPU 显存 DRAM 采用突发 Burst 传输模式，受硬件行存储物理结构限制，打开一整行存储单元开销很高，因此硬件每次会搬运一整块连续数据，而不是只取线程需要的少量字节。

GPU 以 warp（32 线程）为单位发起内存访问，内存合并访问（memory coalescing）就是把同一个 warp 内线程的内存请求合并成少数大内存事务：当 warp 线程访问连续、对齐的内存地址时，每次搬运的数据几乎全部被利用，带宽利用率高；如果地址离散、跨步大，就会触发大量独立内存事务，硬件搬运大量程序用不到的数据，产生读放大，有效带宽大幅下降，程序更容易变成内存受限。注意，在二维矩阵中，同一个 warp 的线程如果沿行访问，通常容易 coalesce；如果沿列访问，通常容易 stride 很大，导致不合并。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831144638.png)

**5.tiling**
原始矩阵乘法计算过程如下图所示，假设每个线程t(x,y)负责输出矩阵的每个元素P(x,y)的结果，在计算的过程中会发生大量重复从全局内存访问数据的现象：t(0,0)与t(0,1)都访问了元素M(0,0)、t(0,0)与t(1,0)都访问了元素N(1,0)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831151328.png)

Tiling（分块）是一种通过重组线程和数据访问顺序，减少 global memory 访问、提高数据复用率的 GPU 优化方法。其核心思想是：将大规模计算划分为若干小块（tile），由一个 thread block 协作把当前计算所需的数据块从 global memory 加载到更快的 shared memory 中，然后多个线程反复复用这些数据完成局部计算，最后再加载下一组 tile。

一个例子如下：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901111958.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112025.png)
优势分析：传统方法通常是每个线程独立计算一个输出矩阵元素，并自行从全局内存读取所需的A、B，例如计算C00与C01时，两者都需要A00和A01，此时同一数据被多个线程重复从global memory读取。Tiling后C00与C01属于同一个线程块处理的目标，重复元素的访存发生在速度更快地shared memory上，实现部分的数据复用
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831152851.png)

Tiling 分块技术存在实际复杂性：当矩阵维度不能被 Tile 尺寸整除时，会生成部分仅负责边界残缺区域的线程块，块内大量线程处于闲置状态，造成硬件利用率下降；因此选择 Tile 大小时，需要综合多方面因素权衡，包括要保证全局内存访问可以合并（Coalesced memory access）、受限于每个线程块可用的 Shared memory 容量，同时还要考虑矩阵维度能否被 Tile 尺寸整除带来的边界利用率问题。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260831153118.png)

分块 tiling 的性能好坏，很大程度取决于数据能不能对齐；一旦不对齐，性能直接掉一大截，下图的锯齿就是对齐 / 不对齐交替带来的真实性能波动。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901084825.png)
例如：当矩阵尺寸从 1792 增大到 1793 时，分块 tile 总数从 98 暴涨至 120。A100 仅有 108 个 SM，120 个 tile 超出硬件并行单元数量，引发周期性性能波动，这就是 wave quantization 现象。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901084926.png)
# FlashAttention
先从普通Attention开始：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112054.png)
假设：$`Q,\,K,\,V \in \mathbb R^{N\times d}`$，那么$`QK^T`$会得到$`N \times N`$的Attention Score矩阵
上述产生的一个问题是大量N×N矩阵需要反复在HBM与GPU计算单元之间搬运：
```text
Q,K->GEMM->S=QK->写回HBM->重新读S->softmax->P->写回HBM->重新读P和V->GEMM->O
```
FlashAttention就是利用tiling的思想解决上述问题
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112112.png)
于是整个过程就变成：
```text
HBM:Q_i,K_j,V_j->加载到片上(shared memory)->S_ij=Q_i K_j^T->局部softmax->直接乘V_j->更新O_j
```
这样的话Sij就根本不用写回HBM，计算完这个tile就可以丢掉，但最大的问题是如何做局部softmax？标准soft需要知道整行元素，如果Attention score的某行是S1 S2 S3 S4，在tiling中只能看到前两个S1 S2，没办法直接得到最终softmax。
解决办法：**Online Softmax**
Online‑Softmax 不一次性对全部 key 做 softmax，迭代维护三组运行状态：每行最大值 $m$、指数累加和 $l$、部分输出 $z$。每读入一块分块的 $K_i, V_i$，计算局部得分 $$S_i = \frac{QK_i^T}{\sqrt{d_k}}$$ 求出本块局部的 $\tilde m_i,\tilde l_i,\tilde z_i$，再利用数值稳定的合并公式，将旧状态 $(m^\mathrm{old},l^\mathrm{old},z^\mathrm{old})$ 和本块局部结果融合更新得到新状态：
$m^\mathrm{new} = \max\big(m^\mathrm{old},\tilde m_i\big)$
$l^\mathrm{new} = e^{m^\mathrm{old}-m^\mathrm{new}} l^\mathrm{old} + e^{\tilde m_i - m^\mathrm{new}} \tilde l_i$
$z^\mathrm{new} = \frac{e^{m^\mathrm{old}-m^\mathrm{new}} l^\mathrm{old}\cdot z^\mathrm{old} + e^{\tilde m_i - m^\mathrm{new}} \tilde z_i}{l^\mathrm{new}}$
## FlashAttention计算实例
**step1、初始化阶段**
接下来举一个完整的例子，仍设: 
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112342.png)
**step2、处理K1、V1**
计算: 
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112437.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112516.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112555.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112614.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112635.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112653.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112711.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112726.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112753.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260901112808.png)
标准Attention访存复杂度 $`O(N^2)`$，反复读写 $`S=QK^\top`$，$`P`$ 至HBM； FlashAttention将 $`K,V`$ 切分为tile，小块计算在SRAM完成，不在HBM保存完整 $`S,P`$； 仅维护状态 $`m,l,z`$，规模为 $`O(Nd)`$； 通过 $`l^{old} \leftarrow l^{old} e^{m^{old}-m^{new}},\ z^{old} \leftarrow z^{old} e^{m^{old}-m^{new}}`$ 在SRAM内缩放历史状态，无需重读历史 $`K,V`$； HBM访存复杂度由 $`O(N^2)`$ 降至 $`O(Nd)`$，缓解GPU显存带宽瓶颈。