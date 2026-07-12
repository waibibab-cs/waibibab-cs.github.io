本文通过介绍Megatron系列的三篇论文来讲解大模型推理的一些并行策略，主要包含以下三篇：
v1: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism  
v2: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM  
v3: Reducing Activation Recomputation in Large Transformer Models
上述三篇论文主要通过模型并行训练优化显存，提高训练效率，而本文主要记录上述三篇论文与推理优化有关的核心思路。
# megatron-v1
## 论文介绍
Megatron-v1 主要提出了面向 Transformer 结构的层内模型并行方法，即后续常说的张量并行（Tensor Parallelism, TP）。该方法的核心动机是：当大语言模型参数量持续增大时，单张 GPU 难以同时容纳完整模型参数、优化器状态以及中间激活，因此需要将一个 Transformer 层内部的矩阵计算拆分到多张 GPU 上执行。
论文分别针对 MLP、Self-Attention 以及 Embedding / LM Head 设计了切分方式，使每张 GPU 只保存部分权重并执行部分 GEMM 计算，同时通过少量 All-Reduce 通信保证最终计算结果与单卡执行一致。
虽然该论文主要面向训练场景，但其张量并行思想对推理系统同样重要。现代大模型推理框架中的 ColumnParallelLinear、RowParallelLinear、按 Head 切分 Attention、Vocab Parallel Embedding 等设计，本质上都延续了 Megatron-v1 的核心思路。对于推理优化而言，Megatron-v1 最值得关注的不是训练数据、优化器或下游任务结果，而是它如何在 Transformer 层内部划分计算、如何减少同步点，以及张量并行带来的通信开销与扩展性限制。
## MLP张量并行
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260624105856.png)
Transformer 中的 MLP 通常包含两个线性层：
$$
X \rightarrow Linear(H \rightarrow 4H) \rightarrow GeLU \rightarrow Linear(4H \rightarrow H)  
$$
Megatron-v1 对这两个线性层采用不同的切分方式。第一层按列切分权重矩阵，使每张 GPU 只计算一部分中间隐藏维度：
$$
A = [A_1, A_2]  
$$
于是各 GPU 分别计算：
$$
Y_i = GeLU(XA_i)  
$$
第二层线性层则按行切分权重矩阵，使每张 GPU 直接消费自己上一层产生的局部结果。各 GPU 得到部分输出后，再通过一次 All-Reduce 求和，得到完整的 MLP 输出。

因此，MLP 张量并行的核心是：
- 第一层 Linear：Column Parallel；
- 第二层 Linear：Row Parallel；
- GeLU 在各 GPU 本地计算；
- MLP 前向只需要一次 All-Reduce，节省了一次中间的All-Gather
这种设计减少了中间同步点，是现代推理框架中 `ColumnParallelLinear` 和 `RowParallelLinear` 的基础。
## Self-Attention张量并行
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260624111719.png)
Self-Attention 的天然并行点在于多头注意力。不同 Attention Head 之间相对独立，因此可以将不同 Head 分配到不同 GPU 上。
Megatron-v1 将 Q、K、V 三个投影矩阵按列切分，使每张 GPU 只负责一部分 attention heads：
$$
Q = [Q_1, Q_2], \quad K = [K_1, K_2], \quad V = [V_1, V_2]  
$$
这样，每张 GPU 可以在本地完成自己负责的 Q/K/V 投影、attention score 计算、softmax 以及 attention value 加权，不需要在 attention 计算过程中跨 GPU 通信。

Attention 结束后，输出投影层按行切分。各 GPU 先基于自己的局部 attention 输出计算部分结果，然后通过一次 All-Reduce 得到完整的 hidden states。
因此，Self-Attention 张量并行的核心是：
- Q/K/V 投影：Column Parallel；
- Attention Head：按 Head 分配到不同 GPU；
- Attention 计算过程本地完成；
- 输出投影：Row Parallel；
- Attention 前向只需要一次 All-Reduce，节省了一次All-Gather
对于推理而言，这意味着 KV Cache 通常也会按 Head 维度分布在不同 TP rank 上，每张 GPU 只保存和计算自己负责的部分 heads。

MLP和多头自注意力的原始过程和TP过程的示意图如下所示，其中x代表着做了矩阵运算：
![4126aa3420b3742a5497f080adb87f6f.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/4126aa3420b3742a5497f080adb87f6f.png)
## 每个Transformer Layer的通信开销
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260624111945.png)
在 Megatron-v1 的张量并行设计中，一个 Transformer Layer 包含两个主要并行区域：
1. Self-Attention；
2. MLP。
每个区域的结构都遵循：
$$
ColumnParallelLinear \rightarrow 本地计算 \rightarrow RowParallelLinear \rightarrow AllReduce  
$$
因此，在前向传播中：
- Self-Attention 输出投影后需要 1 次 All-Reduce；
- MLP 第二个线性层后需要 1 次 All-Reduce。
所以，一个 Transformer Layer 前向总共需要：
$$
2 \times AllReduce  
$$
训练时还需要反向传播，因此完整训练过程通常是：
$$
2 \times AllReduce_{forward} + 2 \times AllReduce_{backward}  
$$
而推理阶段没有反向传播，所以重点关注前向的两次 All-Reduce。
这说明张量并行并不是“免费”的：它可以降低单卡显存占用，并把矩阵计算分摊到多张 GPU 上，但每一层都会引入固定的跨 GPU 通信。推理时可以近似理解为：
$$
T_{layer} = T_{compute} + 2T_{allreduce}  
$$
因此，TP 更适合参数量较大、单卡放不下或单卡计算压力很高的模型；如果模型较小或通信带宽不足，All-Reduce 反而可能成为推理延迟瓶颈。
## Vocab Parallel Embedding
语言模型的输入 embedding 和输出 LM Head 通常共享同一个词表矩阵。该矩阵大小为：
$$
H \times V  
$$
其中 (H) 是 hidden size，(V) 是词表大小。由于词表通常有几万甚至十几万个 token，因此 embedding / LM Head 也适合做并行切分。

Megatron-v1 将词表维度切分到不同 GPU 上：
$$
E = [E_1, E_2, ..., E_n]  
$$
每张 GPU 只保存一部分词表对应的 embedding 参数。

输入阶段，每张 GPU 只能查到自己负责词表范围内的 token embedding（id转embedding），其他位置置零，随后通过一次 All-Reduce 得到完整输入表示。
## 总结
Megatron-v1 虽然主要面向训练，但它的张量并行思想是现代大模型推理系统的重要基础。
对推理系统而言，主要启发有三点。
第一，Linear 层可以通过 Column Parallel 和 Row Parallel 拆分，使模型参数和计算量分布到多张 GPU 上。
第二，Attention 可以按 Head 切分。推理时，每张 GPU 只负责部分 attention heads，因此 KV Cache 也通常按 Head 维度分布在不同 GPU 上。
第三，张量并行会引入固定通信开销。每个 Transformer Layer 前向通常需要两次 All-Reduce，因此 TP 更适合大模型或单卡放不下的场景；对于较小模型，通信开销可能抵消并行收益。
因此，Megatron-v1 对推理优化的核心意义不是提出了完整推理系统，而是奠定了多 GPU 推理中张量切分、通信设计和计算-通信权衡的基础。
# megatron-v2
## 论文介绍
Megatron-v2 在 Megatron-v1 张量并行的基础上，进一步引入流水线并行（Pipeline Parallelism, PP）和数据并行（Data Parallelism, DP），形成由 TP、PP 和 DP 组成的三维并行方案，论文将其称为 PTD-P。

该方法的核心动机是：仅使用张量并行虽然可以拆分单层计算，但当并行范围扩展到多机时，频繁的 All-Reduce 会经过较慢的跨节点链路，通信开销显著增加；同时，过高的张量并行度还会使单个 GPU 上的 GEMM 规模变小，降低计算效率。因此，Megatron-v2 采用节点内张量并行、节点间流水线并行，并使用数据并行进一步扩展训练规模。

论文还提出了交错式流水线调度（Interleaved Pipeline Schedule）。该方法让一张 GPU 负责多个不连续的模型块，通过更细粒度的流水线执行减少 pipeline bubble，但会增加流水线阶段之间的通信次数。为此，论文进一步设计了 scatter/gather 通信优化，减少相邻流水线阶段之间重复的跨节点数据传输。

虽然 Megatron-v2 主要面向大模型训练，但其核心思想同样适用于分布式推理。现代推理系统中的 TP+PP 混合并行、节点内外分层通信、流水线气泡控制以及 microbatch 调度，都可以在该论文中找到基础思路。对于推理优化而言，最值得关注的是流水线并行、交错式调度、TP 与 PP 的组合原则，以及跨节点通信优化。
## pipeline model parallelism
流水线并行（Pipeline Parallelism，PP）沿模型的层维度进行切分，将连续的 Transformer Layers 分配到不同设备上。每个设备只保存并计算自己负责的模型层，相邻流水线阶段之间通过点对点通信传递激活或梯度。

假设模型共有 (L) 层，使用 (p) 个流水线阶段，则理想情况下每个阶段负责约：
$$[  
\frac{L}{p}  
]$$
层。例如：
```text
Stage 0 / GPU 0：Layer 1  - 6
Stage 1 / GPU 1：Layer 7  - 12
Stage 2 / GPU 2：Layer 13 - 18
Stage 3 / GPU 3：Layer 19 - 24
```
前向传播时，数据按照 Stage 0 到 Stage (p-1) 的顺序执行；反向传播时，梯度按照相反方向传递。
与张量并行不同：
- 张量并行在每个 Transformer Layer 内切分矩阵，并频繁执行 All-Reduce；
- 流水线并行在层之间切分模型，主要在相邻阶段之间传输 hidden states，通信方式通常是点对点发送与接收。
因此，PP 更适合将超大模型扩展到多个节点，而 TP 更适合部署在节点内部的高速互联 GPU 上。

**Microbatch**
如果将一个完整 batch 作为整体依次经过所有流水线阶段，那么某一时刻通常只有一个阶段在工作，其他设备处于等待状态，无法发挥多设备的并行能力。

为提高设备利用率，PP 将一个 batch 划分为 (m) 个较小的 microbatch：
$$
Batch={Microbatch_1,Microbatch_2,\ldots,Microbatch_m}  
$$
不同 microbatch 可以同时在不同流水线阶段执行。例如：
```text
时刻 1：Stage 0 处理 MB1
时刻 2：Stage 0 处理 MB2
        Stage 1 处理 MB1
时刻 3：Stage 0 处理 MB3
        Stage 1 处理 MB2
        Stage 2 处理 MB1
```
当流水线被填满后，所有阶段都可以同时工作，只是处理不同的 microbatch。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260711221006.png)
**Pipeline Bubble**
流水线开始时，后面的阶段必须等待前面的阶段产生输出；流水线结束时，前面的阶段已经完成任务，但后面的阶段仍在处理剩余数据。
这些设备空闲时间称为流水线气泡（Pipeline Bubble）。
设：
- (p)：流水线阶段数量（一般就是GPU的数量）；
- (m)：一个 batch 中的 microbatch 数量；
- (t_f)：一个 microbatch 在单个阶段上的前向时间；
- (t_b)：一个 microbatch 在单个阶段上的反向时间（这里谈论的是训练场景，所以除了前向传播，还有反向传播的过程）
默认调度下，气泡时间为：
$$
t_{pb}=(p-1)(t_f+t_b)  
$$
理想计算时间为：
$$
t_{id}=m(t_f+t_b)  
$$
因此，气泡占理想计算时间的比例为：
$$
\frac{t_{pb}}{t_{id}}=\frac{p-1}{m}  
$$
由此可知：
- 流水线阶段 (p) 越多，气泡越大；
- microbatch 数量 (m) 越多，气泡占比越低；
- 为使气泡足够小，通常需要满足：
$$
m\gg p  
$$
但增加 microbatch 数量会增加需要同时保存的激活，导致更高的显存占用（原因：举一个例子，对于device4，microbatch1的激活在前向传播之后不能丢弃，需要保存到显存中直到microbatch1-8全部前向传播完成，并完成microbatch1的反向传播阶段之后再能丢弃）。因此，PP 需要在流水线利用率与激活显存之间权衡。

**GPipe：默认流水线调度**
GPipe 使用一种简单的“All-Forward-All-Backward”调度：
1. 先执行所有 microbatch 的前向传播；
2. 所有前向完成后，再依次执行所有 microbatch 的反向传播；
3. 整个 batch 完成后统一更新参数。
其执行过程可见Figure3
这种调度能够保证严格的同步训练语义：同一个 batch 内的所有 microbatch 都使用相同版本的模型参数，参数只在整个 batch 完成后更新。
GPipe 的气泡比例为：
$$
\frac{p-1}{m}  
$$
因此，需要较多的microbatch才能摊薄流水线气泡。
GPipe 的主要问题是激活显存开销较大。由于所有 microbatch 都要先完成前向传播，在开始反向传播之前，必须保存全部 (m) 个 microbatch 的中间激活，因此激活显存开销与 (m) 成正比。

可以概括为：
- 优点：调度简单，保证严格同步语义；
- 缺点：需要保存所有 microbatch 的激活；
- 减少气泡需要增大 (m)，但增大 (m) 又会进一步增加显存占用。

**Pipeline Flush**
为了保证同步训练语义，流水线在每个 batch 的末尾需要执行一次 Pipeline Flush。
Flush 期间：
- 不再向流水线注入新的 microbatch；
- 等待当前流水线中的所有前向和反向任务执行完成；
- 所有阶段完成后，才统一执行优化器更新；
- 下一个 batch 使用更新后的参数重新启动流水线。
Pipeline Flush 会产生设备空闲时间，是流水线气泡的重要来源。
一些异步流水线方法可以取消 Flush，例如早期的 PipeDream，但这可能导致同一个 microbatch 的前向传播和反向传播使用不同版本的参数，即出现 Weight Staleness。
Megatron-v2 选择保留 Pipeline Flush，从而保证严格的同步优化器语义。

**PipeDream-Flush：1F1B 调度**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260711222710.png)
为减少 GPipe 的激活显存开销，Megatron-v2 采用 PipeDream-Flush 调度。
PipeDream-Flush 仍然保留每个 batch 末尾的 Flush，因此不会引入参数版本不一致问题。它将执行过程划分为三个阶段：（具体过程可见Figure4的上图）
1.Warm-up 阶段
流水线逐步填充。不同阶段执行不同数量的前向传播，使足够多的 microbatch 进入流水线。位于流水线前部的设备需要执行更多次前向传播，后部设备执行的前向次数较少。
2.Steady State 阶段
流水线填满后，每个设备交替执行1F1B(一次forward、一次backward、一次forward、一次backward......)
前向传播将新的 microbatch 注入流水线，反向传播则及时释放旧 microbatch 的激活。
3.Cool-down 阶段
停止注入新的 microbatch，并完成流水线中剩余 microbatch 的反向传播。
PipeDream-Flush 的气泡大小与 GPipe 基本相同，仍然为：
$$
\frac{p-1}{m}  
$$
它的核心优化不是减少气泡，而是减少需要同时保存的激活数量。
GPipe 需要保存最多 (m) 个 microbatch 的激活，而 PipeDream-Flush 中同时处于 in-flight 状态的 microbatch 数量最多约为流水线深度 (p)。因此，激活存储由：O(m) 降低为：O(p)  
其核心特点是：
- 使用 1F1B 调度；
- 及时执行反向传播并释放激活；
- 最多保存约 (p) 个 microbatch 的激活；
- 保留 Pipeline Flush；
- 不引入 Weight Staleness；
- 气泡大小本身没有明显减少。

**交错式流水线调度**
虽然 PipeDream-Flush 降低了激活显存，但其气泡比例仍然为：
$$  
\frac{p-1}{m}  
$$
为了进一步降低流水线气泡，Megatron-v2 基于原有的1F1B调度方法提出交错式流水线调度（Interleaved Pipeline Schedule）（具体可见：Figure4的下图）
默认 PP 中，每个设备负责一个连续的模型分块。例如：
```text
GPU 0：Layer 1  - 4
GPU 1：Layer 5  - 8
GPU 2：Layer 9  - 12
GPU 3：Layer 13 - 16
```
交错式 PP 将每个设备负责的层进一步拆分为多个较小的 Model Chunk，并将不连续的 chunk 分配到同一设备。
>个人理解：正因为同一个设备的v个不同chunk是不连续的layer，才能保证一个microbatch可以拆分成独立的v份，例如原来的GPU0方Layer1-4的话，每个microbatch都必须跑完Layer1-4才能传到下一个GPU中处理Layer5-8。如果不连续的话，就可以：每个microbatch先处理Layer1-2直接单独地传到下一个GPU中做Layer3-4，同理，当GPU3昨晚Layer8的时候再传回GPU0去做接下来的Layer9。
>因此我觉得这种方法实际上是一种变相地去将原来气泡比例中的分母m提高为了原来的v倍（因为每个microbatch可以分拆成独立调度的v个部分）

例如，每个设备拥有两个 chunk：
```text
GPU 0：Layer 1 - 2，Layer 9  - 10
GPU 1：Layer 3 - 4，Layer 11 - 12
GPU 2：Layer 5 - 6，Layer 13 - 14
GPU 3：Layer 7 - 8，Layer 15 - 16
```
从逻辑上看，一个物理设备承担了多个虚拟流水线阶段。不同 chunk 的计算在设备之间交错执行，因此又称 Virtual Pipeline Parallelism。
设每个设备拥有 (v) 个 Model Chunk。拆分后，每个 chunk 的计算时间约为：
$$  
\frac{t_f}{v},\qquad \frac{t_b}{v}  
$$
于是交错式调度的气泡时间变为：
$$
t_{pb}^{interleaved}=\frac{(p-1)(t_f+t_b)}{v}  
$$
气泡比例为：
$$  
\frac{1}{v}\cdot\frac{p-1}{m}  
$$
此外，该调度要求 microbatch 数量是流水线并行度的整数倍，以保证各虚拟阶段能够按照预定顺序调度。

**交错式调度的代价**
交错式调度减少了气泡，但不是没有代价。
每个设备拥有 (v) 个 chunk 后，相邻 chunk 之间需要更频繁地传输激活和梯度，因此流水线通信次数也增加约 (v) 倍。
因此，交错式调度本质上是在：
- 更小的 Pipeline Bubble；
- 更高的阶段间通信开销；
之间进行权衡。
只有当额外通信开销能够被高速网络或通信优化有效控制时，交错式调度才能带来实际收益。Megatron-v2 后续提出的 Scatter/Gather Communication Optimization，正是为了降低交错式流水线增加的跨节点通信成本。

**几种调度方法对比**

|调度方法|执行方式|气泡比例|激活显存|参数语义|
|---|---|--:|--:|---|
|GPipe|所有前向完成后再执行所有反向|((p-1)/m)|(O(m))|严格同步|
|PipeDream-Flush|Warm-up + 1F1B + Cool-down|((p-1)/m)|(O(p))|严格同步|
|Interleaved 1F1B|每个设备负责多个 Model Chunk|((p-1)/(mv))|与 1F1B 接近|严格同步|

因此，三种方法的演进关系可以概括为：
```text
GPipe
  ↓ 通过1F1B及时释放激活
PipeDream-Flush
  ↓ 通过一个设备承担多个虚拟阶段降低气泡
Interleaved 1F1B
```

综上，流水线并行优化不是单纯减少气泡，而是在设备利用率、显存占用和通信开销之间寻找平衡。
## TP&&PP
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260711233518.png)
在固定 GPU 数量和全局 Batch Size 的情况下，可以选择不同的张量并行度 (t)、流水线并行度 (p) 和数据并行度 (d)。不同配置会同时影响：
- 单卡显存占用；
- GEMM 计算效率；
- GPU 空闲时间；
- 跨 GPU 通信量。
因此，并行度并不是越大越好。Megatron-v2 的目标是分析不同并行方式之间的相互影响，为 TP、PP 和 DP 的组合提供配置原则。
由于集群通常采用分层互联结构，例如节点内使用高速 NVLink，节点间使用相对较慢的 InfiniBand，因此相同的数据通信量在不同链路上的耗时可能相差很大。论文主要建立通信量模型，而没有直接建立通信时间模型。

**符号定义**
论文使用以下符号：
- (p)：Pipeline Parallel Size；
- (t)：Tensor Parallel Size；
- (d)：Data Parallel Size；
- (n)：GPU 总数；
- (B)：Global Batch Size；
- (b)：Microbatch Size；
- (m)：每条流水线处理的 Microbatch 数量。
三种并行度满足：$p \times t \times d=n$
每个数据并行副本负责的 Batch Size 为：$\frac{B}{d}$
因此，每条流水线中的 Microbatch 数量为：$m=\frac{B}{bd}$ 

**TP 与 PP 的基本关系**
TP 和 PP 都可以将一个模型拆分到多张 GPU 上：
- TP 在单个 Transformer Layer 内切分矩阵计算；
- PP 沿模型层数切分，不同阶段(设备）负责不同层。
为了单独分析 TP 与 PP，论文暂时令：d=1  
此时：$p \times t=n$
即：$p=\frac{n}{t}$
在 GPU 总数 (n) 固定时，提高 TP 并行度会减少 PP 阶段数；降低 TP 并行度则需要更多 PP 阶段。
默认流水线调度的气泡比例为：$\frac{p-1}{m}$
将 (p=n/t) 代入后得到：$\frac{n/t-1}{m}$
在 (B)、(b) 和 (d) 固定时，(m) 不变。因此，增大 (t) 会减少流水线阶段数 (p)，从而减小 Pipeline Bubble。
从流水线气泡的角度看，提高 TP 并行度是有利的。但 TP 会引入更高的集合通信开销，因此不能只根据气泡大小选择并行配置。

**PP的通信开销**
PP 只在相邻流水线阶段之间传输激活或梯度。
设：
- (b)：Microbatch Size；
- (s)：Sequence Length；
- (h)：Hidden Size。
一个阶段输出的激活大小为：$bsh$
因此，对于每个 Microbatch，相邻两个流水线阶段之间：
- 前向传播需要传输 (bsh) 个元素；
- 反向传播需要传输 (bsh) 个梯度元素。
PP 通信具有以下特点：
- 只发生在相邻阶段之间；
- 使用 Point-to-Point 通信；
- 每经过一个阶段边界通信一次；
- 不需要所有 GPU 同时参与。
因此，PP 通信通常比 TP 的 All-Reduce 更容易扩展到多个节点。

**TP 的通信开销**
Megatron 的 TP 在每个 Transformer Layer 中包含：
- 前向传播 2 次 All-Reduce；
- 反向传播 2 次 All-Reduce。
Ring All-Reduce 中，每张 GPU 对一个大小为 (bsh) 的张量产生的数据传输量约为：(具体可以了解ring all-reduce的过程，底层包含了reduce-scatter+all-gather两个阶段)
$$
2bsh\frac{t-1}{t}  
$$
每层共有 4 次 All-Reduce，因此，每张 GPU 在一个 Transformer Layer、一个 Microbatch 上的 TP 通信量为：
$$
8bsh\frac{t-1}{t}  
$$
如果一个流水线阶段包含 $l^{stage}$个 Transformer Layer，则该阶段每张 GPU 的 TP 通信量为：
$$
l^{stage}  
\cdot  
8bsh\frac{t-1}{t}  
$$
与 PP 相比，TP 通信具有以下特点：
- 每个 Transformer Layer 都要通信；
- 使用所有 TP Rank 参与的 All-Reduce；
- 通信频率较高；
- 对互联带宽和通信延迟更敏感。
在推理阶段没有反向传播，因此每层只有前向的 2 次 All-Reduce，对应通信量约为：
$$
4bsh\frac{t-1}{t}  
$$
但在 Decode 阶段，单次计算的 Batch 和 Sequence 维度通常较小，计算量下降后，All-Reduce 延迟更容易成为瓶颈。

综上，提高 TP 并行度可以：
- 减少每张 GPU 保存的模型参数；
- 分摊每层的计算；
- 减少 PP 阶段数；
- 降低 Pipeline Bubble。
但同时也会带来两个问题。
第一，TP 的 All-Reduce 通信可能跨越节点。当t大于单个服务器内的 GPU 数量时，All-Reduce 需要经过较慢的节点间网络，通信开销显著增加。
第二，矩阵被切分到更多 GPU 后，每张 GPU 执行的 GEMM 规模变小。过小的 GEMM 难以充分利用 GPU，可能降低计算效率。
因此，单纯扩大 TP 并行度虽然能够降低流水线气泡，却可能因为通信和计算效率下降而损失整体吞吐。

**TP 与 PP 的核心权衡**

|并行方式|主要优点|主要开销|
|---|---|---|
|TP|分摊单层参数与计算，降低 PP 阶段数|每层频繁 All-Reduce，GEMM 规模变小|
|PP|通信频率较低，适合跨节点扩展|存在 Pipeline Bubble|
|增大 TP|减少 PP 气泡|增加集合通信压力|
|增大 PP|避免跨节点 TP|增加流水线气泡|

因此，最优配置通常不是只使用 TP 或只使用 PP，而是将两者结合：
```
节点内部：使用 TP
节点之间：使用 PP
```
节点内 GPU 通常通过 NVLink 或 NVSwitch 连接，可以较高效地完成频繁的 All-Reduce；节点间则使用通信频率更低的 PP，只在相邻流水线阶段之间传输激活。
## DP&&TP&&PP
Megatron-v2 分析了数据并行（DP）与流水线并行（PP）、张量并行（TP）之间的关系。
设：
- (n)：GPU 总数；
- (p)：流水线并行度；
- (t)：张量并行度；
- (d)：数据并行度；
- (B)：全局 Batch Size；
- (b)：Microbatch Size。
三种并行度满足：$p \times t \times d=n$  
**DP 与 PP**
暂时令 (t=1)，则：$p=\frac{n}{d}$
定义：$b'=\frac{B}{b}$
每条流水线处理的 Microbatch 数量为：$m=\frac{B}{bd}=\frac{b'}{d}$
流水线气泡比例为：$\frac{p-1}{m}=\frac{n-d}{b'}$
因此，在 GPU 总数和 Batch Size 固定时，增大数据并行度 (d) 会减少流水线阶段数，从而降低 Pipeline Bubble。(可见Figure6)
同时，Ring All-Reduce 的通信量随数据并行度近似按照：$\frac{d-1}{d}$变化，并不会随 (d) 线性增长。因此，如果模型能够放入较少的流水线阶段，增加 DP 通常能够提高吞吐。
增大全局 Batch Size (B) 也可以增加 Microbatch 数量，进一步降低 Pipeline Bubble，并减少数据并行通信相对于计算的占比。

**DP 与 TP**
TP 需要对每个 Microbatch、每个 Transformer Layer 执行 All-Reduce，因此通信频率较高，跨节点执行时开销尤其明显。
DP 则只需在一个 Batch 的反向传播完成后同步一次梯度，因此通信频率低于 TP。
此外，增大 TP 会使每张 GPU 执行的子矩阵变小，可能降低 GEMM 效率；而 DP 中每个模型副本仍执行完整规模的矩阵计算，更容易保持较高的 GPU 利用率。
因此，论文给出的配置原则是：$M=t\times p$
模型并行度 (M) 只需要足够大，使模型参数和中间状态能够放入 GPU 显存。满足显存要求后，应优先使用数据并行扩展到更多 GPU，而不是继续增大 TP 或 PP。
例如有64张GPU，一个完整的模型副本需要16张GPU（比如t=8，p=2），此时就可以把d设置为4，以提高吞吐
## Microbatchsize权衡
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260712112619.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260712112607.png)
Microbatch Size (b) 会同时影响 GPU 的计算效率和流水线气泡，因此并不是越大或越小越好。
设：
- (B)：Global Batch Size；
- (d)：Data Parallel Size；
- (b)：Microbatch Size；
- (p)：Pipeline Parallel Size；
- (t_f(b))：单个 Microbatch 的前向计算时间；
- (t_b(b))：单个 Microbatch 的反向计算时间。
每个数据并行副本处理的 Batch Size 为：$B'=\frac{B}{d}$
每条流水线中的 Microbatch 数量为：$m=\frac{B'}{b}=\frac{B}{bd}$
忽略通信开销时，一个 Batch 的总计算时间可以近似表示为：$\left(m+p-1\right)×\left(t_f(b)+t_b(b)\right)$

增大 (b) 可以增大单次 GEMM 的规模，提高算术强度和 GPU 利用率。论文中的实验表明，在单 GPU 上，增大 Microbatch Size 可使每卡吞吐提高约 (1.3×)。
但 (b) 增大后，Microbatch 数量会减少，从而使流水线气泡比例增大，此外，更大的 Microbatch 通常也会占用更多激活显存。
减小 (b) 可以增加流水线中的 Microbatch 数量，从而更充分地填充流水线、降低 Pipeline Bubble。但 Microbatch 过小时，每张 GPU 执行的矩阵规模变小，GPU 利用率下降，Kernel 启动和通信开销所占比例也会增大。
因此，Microbatch Size 存在以下权衡：

|Microbatch Size|计算效率|Pipeline Bubble|显存占用|
|---|---|---|---|
|较大|较高|较大|较高|
|较小|较低|较小|较低|

核心结论是：
> Microbatch Size 需要在 GPU 计算效率、Pipeline Bubble 和显存占用之间进行权衡，应通过实际性能测试选择，而不能使用固定值。
## Communication Optimizations
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260712114129.png)
在流水线并行中，相邻 Pipeline Stage 需要在前向传播时传递激活，在反向传播时传递梯度。由于这些通信采用 Point-to-Point Send/Recv，单次通信通常只发生在两张 GPU 之间，难以充分利用一个计算节点中的多张网卡。
Megatron-v2 利用 TP 和 PP 组合时的数据复制特征，提出了 Scatter/Gather 通信优化。

**优化前：重复进行跨节点传输**
在张量并行中，每个 Transformer Layer 的输出经过 Row Parallel Linear 和 All-Reduce 后，会在同一 TP Group 的所有 GPU 上形成完整且相同的激活张量。

假设相邻的两个 Pipeline Stage 分别位于两个节点中，且 TP 并行度为 (t)。由于发送端的 (t) 张 GPU 都拥有相同的激活，每张 GPU 都会将完整张量发送给下一个节点中对应的 GPU：
```text
发送节点                     接收节点
TP Rank 0：完整激活  ─────→  TP Rank 0
TP Rank 1：完整激活  ─────→  TP Rank 1
...
TP Rank t-1：完整激活 ────→  TP Rank t-1
```
这意味着同一个激活张量会通过节点间网络重复发送 (t) 次，造成大量冗余通信。
设激活张量大小为：$bsh$
优化前，每对对应的 GPU 都需要跨节点传输一个大小为 (bsh) 的完整张量。

**Scatter/Gather 优化**
Megatron-v2 在发送端先将通过scatter完整激活沿某一维度切分为 (t) 个分块：
$$
X=[X_0,X_1,\ldots,X_{t-1}]  
$$
每个 TP Rank 只负责发送其中一个分块：
```text
发送节点                     接收节点
TP Rank 0：X0  ───────────→  TP Rank 0
TP Rank 1：X1  ───────────→  TP Rank 1
...
TP Rank t-1：Xt-1 ────────→  TP Rank t-1
```

因此，每张 GPU 的跨节点传输量由bsh，降低为原来的t分之一
接收端获得各自的数据分块后，再通过节点内的 All-Gather 恢复完整激活：
由于节点内通常使用带宽更高的 NVLink 或 NVSwitch，而节点间通信使用相对较慢的 InfiniBand，因此该方法本质上是：
> 用高速节点内 All-Gather，替代重复的低速跨节点传输。

当t=8时，每张 GPU 只需跨节点发送原张量的 (1/8)，然后在接收节点内部通过 All-Gather 恢复完整张量。
这一优化对交错式流水线调度尤其重要。交错式调度虽然能够减少 Pipeline Bubble，但会增加约 (v) 倍的阶段间通信。Scatter/Gather 可以降低额外跨节点通信的成本，使交错式流水线在实际系统中获得性能收益。
## 总结
Megatron-v2 虽然主要面向大模型训练，但其并行配置和通信优化方法同样适用于分布式推理。
第一，推理系统应根据硬件拓扑组合 TP 和 PP。TP 每层都需要执行 All-Reduce，对通信带宽要求较高，因此更适合放在 NVLink、NVSwitch 等高速互联的节点内部；当模型需要扩展到多个节点时，应优先使用通信频率较低的 PP。
第二，PP 主要解决模型容量问题，而不一定能够降低单请求延迟。一个请求仍需依次经过所有 Pipeline Stage，只有通过并发请求或 Batch 填满流水线，才能减少设备空闲时间并提高吞吐。因此，PP 更适合大模型部署和高并发推理。
第三，流水线切分需要在气泡和通信之间权衡。将每个设备划分为多个虚拟 Stage 可以减小 Pipeline Bubble，但也会增加阶段间通信。推理系统不能只追求更细粒度的流水线，还需要考虑请求数量、阶段负载和网络开销。
第四，通信优化应充分利用节点内外带宽差异。Scatter/Gather 优化将完整张量的重复跨节点传输，转化为分片跨节点传输和节点内 All-Gather。其本质是减少较慢的节点间通信，将更多通信放到更快的节点内链路上。
第五，Batch 或 Microbatch 大小需要根据实际负载选择。较大的 Batch 能提高 GEMM 计算效率，但会增加排队时间和显存占用；较小的 Batch 延迟更低，但 GPU 利用率可能下降。因此，推理系统需要在吞吐、延迟和显存之间动态权衡。
第六，推理中的数据并行可以理解为模型副本并行。当一个 TP+PP 模型实例已经能够容纳模型后，可以复制多个实例分别处理不同请求，从而扩展服务吞吐。
总体来看，Megatron-v2 对推理系统最重要的启发是：分布式并行不能只考虑模型能否放入显存，还必须同时考虑硬件拓扑、通信模式、流水线气泡、计算效率和请求负载。
# megatron-v3
## 论文介绍
Megatron-v3 主要研究大模型训练中的激活显存问题，并提出了序列并行（Sequence Parallelism，SP）和选择性激活重计算（Selective Activation Recomputation）两种优化方法。
其中，选择性激活重计算主要服务于反向传播，对推理系统意义较小；而序列并行则具有一定的推理参考价值。传统张量并行只切分 Attention 和 MLP 中的大型矩阵计算，LayerNorm、Dropout 和 Residual 等区域的激活仍会在各个 TP Rank 上重复保存。Megatron-v3 将这些非张量并行区域沿序列维度切分，使每张 GPU 只保存和处理部分序列数据。
为了在序列并行区域与张量并行区域之间转换，论文使用 All-Gather 和 Reduce-Scatter 替代原有的 All-Reduce，在不显著增加通信量的情况下减少激活显存。
对于推理优化而言，Megatron-v3 最值得关注的是：如何沿序列维度划分中间张量，以及如何通过集合通信在不同张量布局之间高效转换。这些思想对长序列推理、Prefill 阶段的显存优化以及后续 Context Parallelism 的理解具有参考价值。
## sequence parallelism
**传统张量并行的问题**
Megatron-v1 的张量并行主要切分 Transformer 中计算量较大的部分：
- Self-Attention；
- MLP 中的两个线性层。
这些区域的权重和部分中间激活会按照张量并行度 (t) 分布到不同 GPU 上，因此显存占用可以随 (t) 增大而降低。
但是，Transformer 中仍有一部分算子没有被 TP 切分，例如：
- LayerNorm；
- Attention 和 MLP 后的 Dropout；
- Residual Add 等逐元素操作。
这些算子的计算量较小，不适合沿 hidden dimension 切分，但它们的输入激活仍然以完整形式保存在每个 TP Rank 上。

论文发现，LayerNorm、Dropout 和 Residual 等非 TP 区域中的操作，对不同 token 通常是相互独立的。
例如，LayerNorm 对每个 token 的 hidden dimension 独立归一化，并不需要同时访问其他 token：因此，可以将这些区域的激活沿序列维度 (s) 切分到不同 TP Rank 上。

对于 (t) 张 GPU：于是，LayerNorm、Dropout 和 Residual 等操作都可以在各 GPU 上独立执行，每张 GPU 只需要处理约 (1/t) 的 token。

Sequence Parallelism 并不是建立一组新的 GPU，而是复用原来的 TP Group：
```text
同一组 GPU：
大矩阵计算区域       → Tensor Parallel
LayerNorm等轻量区域  → Sequence Parallel
```
因此，一个 Transformer Layer 会在两种张量布局之间转换：
```text
Sequence Parallel Layout
        ↓
Tensor Parallel Layout
        ↓
Sequence Parallel Layout
```
**两种张量布局**
Sequence Parallel Layout 沿序列维度切分：
$$
X_i^s  
\in  
\mathbb{R}^{\frac{s}{t}\times b\times h}  
$$
每张 GPU 拥有部分 token，但拥有这些 token 的完整 hidden dimension。
Tensor Parallel Layout 则主要沿 hidden dimension 切分：
$$
Z_i^h  
\in  
\mathbb{R}^{s\times b\times\frac{4h}{t}}  
$$
每张 GPU 拥有完整序列，但只拥有部分 hidden features。
因此，SP 和 TP 的区别可以概括为：

|并行区域|切分维度|每张 GPU 保存的内容|
|---|---|---|
|Sequence Parallel|Sequence (s)|部分 token、完整 hidden|
|Tensor Parallel|Hidden (h)|完整 token、部分 hidden|

论文使用集合通信在这两种布局之间进行转换
**g与 $\bar g$通信算子**
论文将传统 TP 中的通信算子重新组合，定义了两个新的算子：

| 算子       | 前向传播           | 反向传播           |
| -------- | -------------- | -------------- |
| g        | All-Gather     | Reduce-Scatter |
| $\bar g$ | Reduce-Scatter | All-Gather     |
其中：
- g将 Sequence Parallel Layout 转换为 Tensor Parallel Layout；
- $\bar g$将 Tensor Parallel Layout 转换回 Sequence Parallel Layout。
二者互为共轭算子，即前向中的 All-Gather，在反向中对应 Reduce-Scatter；前向中的 Reduce-Scatter，在反向中对应 All-Gather。
一个 Transformer 子模块的执行流程为：
```text
Sequence Parallel
LayerNorm
      ↓
g：All-Gather
      ↓
Tensor Parallel
Attention 或 MLP
      ↓
ḡ：Reduce-Scatter
      ↓
Sequence Parallel
Dropout + Residual
```
**MLP 中的具体执行过程**
论文以 LayerNorm 和 MLP 为例说明 SP 与 TP 如何组合。
未并行时，MLP 可以表示为：
$$
Y=\operatorname{LayerNorm}(X),
Z=\operatorname{GeLU}(YA),
W=ZB,
V=\operatorname{Dropout}(W)  
$$
其中：$X,Y,W,V\in\mathbb{R}^{s\times b\times h}$
两个线性层的权重分别为：$A\in\mathbb{R}^{h\times4h}$和$B\in\mathbb{R}^{4h\times h}$

step1. Sequence Parallel LayerNorm
首先将输入沿序列维度切分：$X=[X_1^s,X_2^s,\ldots,X_t^s]$
每个 Rank 独立执行 LayerNorm：$Y_i^s=\operatorname{LayerNorm}(X_i^s)$
因为 LayerNorm 在每个 token 的 hidden dimension 上独立执行，所以不需要跨 GPU 通信。

step2. All-Gather 进入 Tensor Parallel 区域
MLP 第一层采用 Column Parallel，每个 Rank 保存一部分输出列：$A=[A_1^c,A_2^c,\ldots,A_t^c]$
其中：
$$  
A_i^c\in  
\mathbb{R}^{h\times\frac{4h}{t}}  
$$
标准 Column Parallel Linear 要求每个 Rank 都拥有完整序列的输入，因此先通过 (g) 执行 All-Gather,现在每个 TP Rank 都拥有完整的 (Y)，但只保存权重的一部分。

step3. Column Parallel Linear 与 GeLU
每个 Rank 独立计算：
$$
Z_i^h=
\operatorname{GeLU}(YA_i^c)  
$$
其中：
$$
Z_i^h  
\in  
\mathbb{R}^{s\times b\times\frac{4h}{t}}  
$$
这里的上标 (h) 表示张量沿 hidden dimension 被切分。
由于 GeLU 是逐元素操作，各 Rank 可以直接对局部结果执行 GeLU，不需要通信。

step4. Row Parallel Linear
第二个权重矩阵按行切分,每个 Rank 计算一个局部输出，传统 TP 会通过 All-Reduce 求和，使每个 Rank 都得到完整的 (W)。

step5. Reduce-Scatter 返回 Sequence Parallel 区域
SP 不需要让所有 Rank 都保存完整的 (W)。因为接下来的 Dropout 和 Residual 可以沿序列维度独立执行。
因此，论文将“求和”和“沿序列切分”合并为一次 Reduce-Scatter：
每个 Rank 最终只得到：
$$
W_i^s  
\in  
\mathbb{R}^{\frac{s}{t}\times b\times h}  
$$
然后在本地执行：
$$
V_i^s=\operatorname{Dropout}(W_i^s)  
$$
Residual Add 也可以在相同的序列分片上完成。
因此，完整流程为：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260712123729.png)
Attention 部分采用相同思想：LayerNorm 和输出后的轻量操作使用 Sequence Parallel，Attention 内部的大型矩阵计算继续使用 Tensor Parallel。

**通信量分析**
传统 Tensor Parallel 在一个 Transformer Layer 的完整前向和反向传播中需要：$4\times\operatorname{AllReduce}$
其中 Attention 和 MLP 分别在前向、反向各产生一次 All-Reduce。
加入 Sequence Parallel 后，通信变为：
$$
4\times\operatorname{AllGather}  
+  
4\times\operatorname{ReduceScatter}  
$$
表面上看，通信操作数量从 4 次增加到 8 次。但 Ring All-Reduce 本身可以分解为：allreduce+reducescatter因此，最终传输的数据量相同。
>Ring All-Reduce过程示意图
>分为reduce scatter+all gather，这里reduce scatter比较难理解，我们给出示意图
>![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260712130606.png)

所以论文认为，Sequence Parallelism 不会增加理论通信带宽开销，只是将原来的 All-Reduce 拆成了具有不同输出布局的 Reduce-Scatter 和 All-Gather。

需要注意的是，通信数据量相同并不代表实际延迟完全相同。All-Gather 和 Reduce-Scatter 的启动次数更多，具体性能还会受到通信库、消息大小和硬件拓扑影响。

**核心总结**
Sequence Parallelism 并不是取代 Tensor Parallelism，而是补充 TP 未覆盖的区域：
```text
LayerNorm、Dropout、Residual
        ↓ 沿 sequence 维度切分
Sequence Parallel

Attention、MLP GEMM
        ↓ 沿 hidden 维度切分
Tensor Parallel
```
通过AllGather和ReduceScatter在两种布局之间转换，并将传统 All-Reduce 重组为 Reduce-Scatter 与 All-Gather，在理论通信量不增加的情况下，使非 TP 区域的激活也能按照并行度 (t) 分摊。