# 前言
Transformer模型在不同参数量规模下，计算量（FLOPs）在注意力与MLP模块的分配变化，说明了对于不同规模的模型，我们优化侧重点也应该有所不同。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260722104520.png)

**课程大纲**
* basics
	* 目标：能够训练一个基础的语言模型
	* 内容：
		* **tokenization**：是关于“模型操作的基本单元是什么？”；形式上来说，分词器（tokenizer）负责在原始输入与序列（token序列）之间进行转换；流行的分词器：BPE（2015）；作用包括缩短上下文长度（例如：1000字节->250个token）、自适应计算（对输入中关键部分分配更多建模能力）；终极愿景是无分词器的模型架构可以直接以字节作为输入，这类方案前景广阔但尚未在前沿模型中落地
		* **模型架构**：从最原始的Transformer开始介绍；然后介绍后续的优化与改进，例如激活函数的改进（ReLU、SwiGLU）、位置编码的改进（sinusoidal、RoPE）、归一化方法改进（LayerNorm、RMSNorm、QK norm）、注意力机制改进（full/sparse/local attention、GQA)、Recurrence/state-space models/linear attention（Mamba、Gated DeltaNet）、MLP改进（dense,MOE)、Shape（隐藏层维度、专家数）
		* **训练**：是关于“如何设置模型的参数？“；包含损失函数（比如multi-token prediction）、优化器（AdamW、SOAP、Muon）、initialization scale（Xavier init、muP）、学习率调度（cosine、WSD）、正则化（dropout、weight decay）、批次大小（critical batch size）、MoE specific：load balancing（aux-free）
* systems
	* 目标：最大化利用硬件（GPU or TPU）
	* 内容：
		* **基础**：资源核算（分析一个模型方寸与计算的特征，例如在1T tokens训练一个70B的模型需要的计算量为6×70e9×1e12=4.2e23 FLOPS）；模型参数必须从HBM移动到SMs去计算；实例：B200能够使用8TB/sec的带宽执行2.25 PFLOPS/sec的计算；屋顶线分析：理解计算受限与访存受限；Benchmarking and profiling（nsight）
		* **kernels**：内核是运行在GPU上的一个函数；在使用 PyTorch 时，每个基础运算都会启动一个标准核函数；可以编写自定义核函数来让 GPU 跑得飞快；核心原则：组织计算流程，最小化数据搬运开销；朴素实现：读取高带宽显存（HBM）→ 计算 A → 写回 HBM → 读取 HBM → 计算 B → 写回 HBM；融合实现：读取 HBM → 同时计算 A 和 B → 一次写回 HBM；优化策略：算子融合（如矩阵乘法 + 激活函数）、分块计算（如 FlashAttention 中的 tiling 技术）；关键优化点：线程束分化（warp divergence）、内存合并访问（memory coalescing）、存储体冲突（bank conflicts）、占用率（occupancy）、批量异步内存传输；核函数编写工具：CUDA / Triton / CUTLASS / ThunderKittens
		* **并行化**：如果我们有 1024 张 GPU，该怎么办？GPU 之间的数据传输速度会更慢，但 “最小化数据移动” 的核心原则依然适用；使用经典的集合通信操作（例如：gather、reduce、all-reduce）；将内存数据（参数、激活值、梯度、优化器状态）分片存储在不同 GPU 上；计算拆分方式：{数据并行、张量并行、流水线并行、序列并行、专家并行}
		* **推理**：目标是根据提示词生成词元；推理同样是强化学习、测试时计算与模型评估等场景的必要环节；分为两个阶段预填充（prefill）和解码（decode）；prefill：所有词元已知，可一次性并行处理（计算密集型阶段）；decode：需要逐一生成词元（内存密集型阶段）；decode加速方法包含使用更轻量的模型（通过模型剪枝、量化、知识蒸馏实现）/投机解码（Speculative decoding）/系统级优化：算子融合核函数、连续批处理（continuous batching）
* scaling_laws
	* 目标：在给定计算量（FLOPs）预算下，通过小规模实验建立 “计算量→超参数→损失” 的预测关系，指导最优模型规模与训练策略，避免大规模调参的高昂成本
	* 内容：缩放定律的核心思想是从 “单一规模调优” 转向 “缩放配方（scaling recipe）”：先在不同小规模计算量（如最高 1e24 FLOPs）下进行实验，测量不同模型参数规模、训练数据量对应的损失，拟合出缩放规律；再用该规律外推到目标大规模计算量（如 1e25 FLOPs），预测最优超参数（模型大小 N / 训练 token 数 D）和最终损失，甚至在训练前就能评估方案效果。经典的计算最优缩放定律（Kaplan 2020、Hoffmann 2022）通过 ISOFLOP 曲线，确定不同计算量下的最优模型规模，而后续工作（如 Yang+2022 的超参数迁移、Delphi Scaling Suite）进一步验证了该方法的有效性，最终训练结果与缩放定律的预测高度吻合。
* data
	* 目标：通过高质量数据构建与全面评测体系，使大语言模型具备多语言理解、对话交互、代码生成及智能体任务执行等通用能力。
	* 内容：大语言模型的能力提升依赖于数据构建与评估体系的协同设计。在数据构建方面，需要从网页、书籍、论文、代码等多源数据中进行数据采集，并通过数据清洗、格式转换、质量筛选、去重、数据配比调整以及合成数据增强等流程构建高质量训练数据。根据训练阶段不同，数据可分为预训练数据（大规模、多样化语料）、中期训练数据（高质量长上下文数据）和后训练数据（对话数据、工具调用轨迹等）。在模型评估方面，需要针对模型目标能力设计多维度测试指标，包括困惑度评估基础语言建模能力，以及通过GPQA、HLE、SWE-Bench、Terminal-Bench等高级任务评测模型在推理、代码和智能体场景中的实际表现，从而指导模型优化并衡量其真实应用能力。
* alignment
	* 目标：通过对模型进行偏好对齐训练，使其生成更加符合人类意图、安全性要求和任务需求的高质量响应
	* 内容：在完成基础预训练（预测下一个token）后，需要进一步通过弱监督方法提升模型行为能力。其核心流程包括：首先利用模型生成多个候选回答；随后通过人工标注、验证器或语言模型评估器对回答质量进行评分；最后根据偏好数据更新模型，使其倾向于生成更优响应。常见方法包括基于强化学习的PPO（Proximal Policy Optimization）、直接偏好优化DPO（Direct Policy Optimization）以及去除价值函数的GRPO（Group Relative Preference Optimization）。相比直接生成高质量答案，对模型输出进行评价通常更容易，因此偏好学习能够有效利用弱监督信号。但强化学习方法仍面临训练不稳定、参数调节困难，以及大规模部署时需要额外推理基础设施和效率权衡等挑战
# Tokenization
tokens：通常用indices表示，比如`[15496,11,995,0]`
encodes：将string转换为tokens的过程
decodes：将tokens转换为string的过程
Tokenizer（分词器）：可以执行encodes与decodes的类

压缩率：每个tokens代表的字节数，一种提高压缩率的方法是增加vocabulary size，但是会陷入稀疏性。
vocabulary size与分词的关系：举一个例子，对于一个文本`"I love eating strawberries and kiwis."`，在小词表场景中，词表容量有限，无法收录长词和低频词，只能用基础子词拆分：`["I", " love", " eating", " straw", "berries", " and", " ki", "wis", "."]`；大词表场景下，词表进一步收录了更多变体词、多语言词，甚至完整短语：`["I", " love eating", " strawberries", " and", " kiwis", "."]`

character_tokenizer：直接把文本按单个字符拆分，然后分别按照unicode进行编码
vocabulary_size=max(indices)+1
问题：1.词表太大；2.压缩率极低

byte_tokenizer：用字节作为最小分词单元，直接把文本转为UTF-8，压缩率为1
文本：你好
UTF-8 转字节拆分：`0xE4 0xBD 0xA0 0xE5 0xA5 0xBD`

word_tokenizer：先使用某种规则（例如空格或某种正则表达式）将字符串分成若干chunks，然后将每个chunk映射成一个整数
好处是：通常每个token是有意义的（比如映射前就是一个个单词）
压缩率一般较高但词表会很大
注意：1.通常词表大小不固定；2.对于一个在训练阶段没有见过的新word，将会映射成一个UNK token，这对后续计算会有一定影响

bpe_tokenizer：Byte Pair Encoding，核心思想是通过不断合并训练语料中高频出现的字符/字节组合，构建适合数据分布的 token 词表。其核心思想是让常见序列使用少量 token 表示，而罕见序列通过多个 token 表示，从而在词表规模和序列长度之间取得平衡。
# Pytorch
 张量（tensor）：支持 GPU 加速、可自动求导的多维数组，是深度学习所有数据运算的基础载体。几乎所有内容（参数、梯度、激活值、优化器状态）可以作为浮点数存储至其中，
 
 通常浮点数指float32，为了进一步降低内存占用，可使用float16、bfloat16、float8、float4、。结论：1.精度越大通常效果越好，但时间/空间开销大；2.一种方法是采用混合训练，例如对参数/激活值/梯度采用bf16，对优化器状态采用fp32（pytorch有自动的混合精度库AMP）

einops：PyTorch 简洁的维度重排库，包含einsum、reduce、rearrange
einsum：用下标表达式简洁实现张量任意线性运算
```python
import torch
# 定义维度
b, n, h, d, k = 2, 5, 2, 4, 3
# 初始化张量
q = torch.randn(b, n, h, d)
k = torch.randn(b, n, k, d)
v = torch.randn(b, n, k, d)
# 1. 普通原生写法
attn1 = q @ k.transpose(-1, -2)
out\_normal = attn1 @ v
# 2. 合并一行einsum写法
out\_einsum = torch.einsum('bnhd,bnkd,bnkd->bnhd', q, k, v)
```
reduce：对一个tensor使用某种规则（eg,sum,mean...）进行规约
```python
x = torch.ones(2, 3, 4)
# old way
y = x.sum(dim=-1)#(2,3,4)->(2,3)
# new way
y = reduce(x, "... hidden -> ...", "sum")
```
rearrange：对一个 tensor 进行维度重排、拆分或合并
```python
x = torch.ones(2, 3, 4)  # (batch, seq, hidden)
# old way
y = x.permute(1, 0, 2).reshape(3, -1)  # (2,3,4)->(3,2,4)->(3,8)
# new way
y = rearrange(x, "b s h -> s (b h)")
```
一个浮点数操作（FLOP）是一个基本的操作，例如加法或乘法，有两种表示方法：
* FLOPs：FLOP操作量
* FLOP/s（FLOPS）：每秒的FLOP操作量，衡量硬件性能
例子：
* 训练GPT-4(2023)大概需要2e25 FLOPs
* `H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without`，H100支持结构化稀疏，让模型中的一般权重归零，因此不启用稀疏时，其性能只有一半：`h100_flop_per_sec = 1979e12 / 2`

每种GPU都有一个承诺（promised）的峰值性能（FLOP/s）的参数，该参数会随着不同数据类型而发生变化，然而实际（actual）的FLOP/s往往与承诺的不同，可以使用MFU（Model FLOPs utilization）来衡量（忽略通信及其他开销）：
`mfu = actual_flop_per_sec / promised_flop_per_sec`

以relu操作介绍算术强度(arithmetic_intensity)概念：
```python
n = 1024 * 1024
x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
y = torch.relu(x)
# 传输的字节数，包含read x和write y两个过程，其中每个bf16是2个字节
bytes = (2 * n) + (2 * n) 
flops = n
communication_time = bytes / h100_bytes_per_sec
computation_time = flops / h100_flop_per_sec
total_time = max(communication_time, computation_time)
# 若通信时间大于计算时间为memory-bound，反之为compute-bound
```
另一种判断memory-bound or compute-bound的方法：
硬件的算术强度a：`h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec = 295.3731`
负载的算术强度b：`arithmetic_intensity = flops / bytes`
若a > b为带宽受限，反之计算受限

roofline model：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260724115045.png)
x轴是算术强度，y轴是硬件在一个负载在x轴对应的算术强度下的理论性能峰值

## 一些pytorch的例子展示
**1.deep_linear_network**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260724152432.png)
现在有如图所示的一个网络，层数L=3，维度D=8，则其参数量为`(D * D) * L`，输入批次B=4，对于一个输入x，会依次进行以下运算：
第一层：`x = F.relu(x @ w1)`
第二层：`x = F.relu(x @ w2)`
第三层：`x = F.relu(x @ w3)`

**2.gradients_basics**
模型公式：`loss = 0.5 * (x·w - 5)²`，目标是让预测值 `x·w` 逼近目标值 5
其中：
`x = torch.tensor([1., 2, 3]) `
`w = torch.tensor([1., 1, 1], required_grad = True) # 要求权重的梯度` 
前向传播：
`pred_y = x @ w `
`loss = 0.5 * (pred_y - 5).pow(2)`
反向传播：
`loss.backward() # 自动计算梯度`
求导：`d(loss)/dw = (x·w - 5) * x`，代入后得:
`assert torch.equal(w.grad, torch.tensor([1, 2, 3]))`

**3.gradients_flops**
该例子用一个简单的模型展示计算梯度所需的计算量
我们定义一个简单的模型，类似上述“1.deep_linear_network”，只不过是两层的linear network，其中B=1024，D=256
输入与权重：
```python
x = torch.ones(B, D, device=cuda_if_available())
w1 = torch.randn(D, D, device=cuda_if_available(), requires_grad=True)
w2 = torch.randn(D, D, device=cuda_if_available(), requires_grad=True)
```
前向传播：
```python
h1 = einsum(x, w1, "batch in, in out -> batch out") 
h2 = einsum(h1, w2, "batch in, in out -> batch out")
loss = (h2.mean() - 0)**2
```
反向传播：
```python
h1.retain_grad()
h2.retain_grad()
loss.backward()
```
部分解释：
- `retain_grad()`：默认情况下，PyTorch 只会保留叶子节点（如 `w1, w2`）的梯度，中间变量 `h1, h2` 的梯度会被释放。这里显式调用，是为了后续查看 / 验证它们的梯度。
- `loss.backward()`：触发自动求导，PyTorch 会从 `loss` 开始，沿计算图反向计算所有 `requires_grad=True` 张量的梯度：
    1. 对 `w2` 的梯度：由 `loss` 对 `h2` 求导，再乘以 `h2` 对 `w2` 的导数得到
    2. 对 `w1` 的梯度：由 `loss` 对 `h1` 求导，再乘以 `h1` 对 `w1` 的导数得到
    3. 梯度会被自动存入 `w1.grad` 和 `w2.grad`

接下来**聚焦于网络的第二层**，也就是h1--乘以w2--h2的一层：
前向传播过程，假设使用bf16量化：
`num_forward_flops = 2 * B * D * D`
反向传播过程，我们需要计算：
1.h1.grad=dloss/dh1（为什么？h1是第一层的输出、第二层的输入，它不是参数，只是中间结果，但第一层参数w1的梯度需通过h1传递：`dloss/dw1 = dloss/dh1 * dh1/dw1`，所以h1.grad必须算出来，才能继续往前传播，计算第一层的梯度）
2.w2.grad=dloss/dw2（为什么？w2是模型的可训练参数，我们的目标就是用梯度下降来更新它，让loss变小，公式是`w2 = w2 - lr * w2.grad`）
其中计算过程为：
```python
# 解释：dloss/dh1 = dloss/dh2 * dh2/dh1 = h2.grad @ w2
h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
assert torch.allclose(h1.grad, h1_grad)
# 解释：dloss/dw2 = dloss/dh2 * dh2/dw2 = h2.grad @ h1
w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")
assert torch.allclose(w2.grad, w2_grad)
```
总计算量：
`num_backward_flops = (2 * B * D * D) + (2 * B * D * D)`

**4.optimizer**
使用之前的网络，其中：B=2,D=4,L=3
>优化器介绍：
>1.基础：SGD（随机梯度下降）
>所有优化器的起点，更新公式是：w = w - lr×g
>优点：简单直接；缺点：学习率固定，对稀疏梯度、不同参数更新效率差，易震荡
>2.Momentum（带动量的SGD）
>给梯度加了指数移动平均（EMA），模拟物理中的 “惯性”
>![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260724162548.png)
   加快收敛速度，减少震荡，更容易跳出局部鞍点
   3.AdaGrad
   用**梯度平方的累积和**来调整每个参数的学习率
   ![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260724162647.png)
   给高频更新的参数降低学习率，给低频更新的参数提高学习率，解决稀疏梯度问题，缺点是：梯度平方会一直累积，后期学习率会越来越小，容易提前 “僵住”
   4.RMSProp
   把 AdaGrad 的 “累积和” 改成了**指数移动平均**，解决梯度平方一直累积的问题
   ![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260724162746.png)
   学习率自适应调整，不会提前衰减，训练更稳定
   5.Adam
   同时结合了**Momentum 的梯度平滑**和**RMSProp 的学习率自适应**
   ![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260724162826.png)
   优点：收敛快、训练稳定、参数更新更合理，是现在深度学习中最常用的优化器

这里使用AdaGrad：
```python
optimizer = AdaGrad(model.parameters(). lr=0.01)
state = model.state_dict()
# 计算梯度
x = torch.randn(B, D, device=cuda_if_available())
y = torch.tensor([4., 5.], device=cuda_if_available())
pred_y = model(x).mean()
loss = F.mse_loss(input_pred_y, target=y)
loss.backward()
# 进行一步优化,具体过程忽略
optimizer.step()
# 释放内存
optimizer.zero_grad(set_to_none=True)
```
**5.上述例子的内存占用**
参数（2 bytes for bf16）
`parameter_memory = 2 * (D * D * L)`
激活（bf16）
`activation_memory = 2 * B * D * L`
梯度（bf16）
`gradient_memory = 2 * number_parameter`
优化器状态（fp32）
`optimizer_state_memory = 4 * number_parameter`
优化器上为了稳定通常用fp32，对于adagrad，通常用4 bytes/parameter存二阶矩，对于adam需要用8bytes/parameter存储一阶矩与二阶矩

## 梯度累积与激活检查点
为了降低内存占用，通常用这两个方法
梯度累积（Gradient Accumulation）是一种在显存有限时模拟大批次训练的技术：比如你想用 batch size=1024 训练模型，但显存只能装下 256，就可以设置 micro_batch_size=256、累积步数 = 4，依次处理 4 个微批次并累加梯度，直到完成 1024 个样本的梯度计算后再更新参数，这样既能获得大批次训练的稳定性，又能把单步显存占用控制在小批次水平。

激活检查点（Activation Checkpointing）核心思路是：在前向传播时，不保存所有层的激活值，只选择性保存部分关键层的激活；在反向传播时，再重新执行一次部分前向传播，计算出缺失的中间激活值来完成梯度计算。举个例子：若全保留激活（例如bf16），显存占用约为 `2×B×L×D`；使用激活检查点后，我们只保存每 4 层的激活值，反向传播时按需重新计算中间层的激活，显存峰值占用能大幅降低，代价只是多了少量重复的前向计算步骤