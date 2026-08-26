# 朴素LinearAttention
上一节我们分析过，注意力运算基础的形式是：$`(b,n,k)@(b,k,n)->(b,n,n)`$与$`(b,n,n)@(b,n,k)->(b,n,k)`$，操作数的规模是$`bn^2d`$，而FFN层的基础计算形式是$`(b,n,d)@(d,d)`$，操作数规模是$`bnd^2`$，因此随着上下文长度n的增长，注意力运算量呈现二次增长，FFN运算量线性增长

有两个基础方法缓解上述问题，第一个是算法层面的（Sparse Attention），这种方法通常将多头划分为Local Attention（局部注意力）与Global Attention（全局注意力），前者每个token仅关注附近一小段窗口内的 token，后者指定部分 token 为全局 token，能够访问全部历史 token；第二个是系统层面的（FlashAttention）不改变注意力数学公式，依靠显存读写优化、分块重计算（tiling+recompute），降低 GPU 显存开销，提升计算吞吐速度，在长序列下显著加速前向 / 反向传播，但这两个方法并未从根本上改变注意力运算虽上下文长度呈平方增长的情况。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825103853.png)

对于常规的注意力计算：$`Q \in \mathbb R^{n\times d_k},\ K \in \mathbb R^{n\times d_k},\ V \in \mathbb R^{n\times d_v}`$，可以表达为：
$`\text{Attn}(Q,K,V) = \rho(QK^\top)V`$，其中$`\rho(\cdot) = \text{softmax}\left(\frac{\cdot}{\sqrt{d_k}}\right)`$
QK之间的计算复杂度为$`(n^2d)`$，是长上下文的计算瓶颈来源，如果能够通过乘法结合率使得：$`\big(QK^\top\big)V = Q\big(K^\top V\big)`$，则计算复杂度将会变为$`(nd^2)`$，而阻碍这一变化的主要原因就是$`\rho`$这个操作

《Transformer are RNNs》中提出了一种替换softmax的方法，先把attention写成：
$$O_i = \frac{\sum_{j=1}^{N} \big(sim(Q_i, K_j) * V_j\big)}{\sum_{j=1}^{N} sim(Q_i, K_j)}$$
替换softmax的关键就是：把公式6的相似度函数写成拆分的形态： $$O_i = \frac{\sum_{j=1}^{N} \big(\varphi(Q_i)^\top \varphi(K_j)\big) * V_j}{\sum_{j=1}^{N} \varphi(Q_i)^\top \varphi(K_j)} $$
常规 Attention 使用点积 $`Q_i K_j^\top`$ 衡量 Query 与 Key 的相关性，但点积幅值会随特征维度增大而增大，直接输入 softmax 容易造成分布过于尖锐和梯度饱和，因此先除以 $`\sqrt{d_k}`$ 控制数值尺度；随后通过 softmax 将相似度归一化为非负且和为 1 的注意力权重，用于对 Value 进行加权聚合。 其中softmax中的exp操作就是上述的相似度函数 $`sim`$。总结：
- 常规 attention： $`sim(a,b)=\exp\left(\frac{a^\top b}{\sqrt{d_k}}\right)`$ 
- 线性注意力替换后的可分离相似度函数： $`sim(a,b)=\varphi(a)^\top \varphi(b)`$
$`\varphi(\cdot)`$ 为**核函数**，对原始 Query、Key 向量做逐元素非线性变换。 为保证相似度 $`sim(a,b)=\varphi(a)^\top\varphi(b)\ge 0`$，映射后的向量每一维的取值必须非负。 常用可选核映射：$`\text{elu}(\cdot)+1`$、$`\text{relu}(\cdot)+1`$，经过变换后向量全部元素 $`\ge0`$，点积结果自然满足非负约束。

有了核函数拆分，KV就能先计算了，公式可以变为：
$$V_i' = \frac{\varphi(Q_i)\sum_{j=1}^{N}\varphi(K_j)*V_j}{\varphi(Q_i)\sum_{j=1}^{N}\varphi(K_j)} $$
上述公式可以通过并行转换成一个RNN的表达形式:
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825111419.png)
即当求解第i个token的输出时，通过前面i-1个token的计算迭代来求解结果。其中S和Z是每一步迭代过程需要传递的信息，而S与Z时常量值，因此，相比标准attention中的KV Cache，存储与访存开销降低了许多
其流程可以表示成如下形式：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825111545.png)
其基本递推公式表达如下：$`S_t = S_{t-1} + \phi(x_t W_K)(x_t W_V)^\top`$、$`z_t = z_{t-1} + \phi(x_t W_K)`$，其中 $`t`$ 表示当前token的索引，$`S`$ 和 $`z`$ 表示记忆值，$`S`$ 的大小是 $`\mathbb R^{d_k \times d_v}`$。输出对应的一般公式为：$`o_t = \frac{S_t \phi(q_t)}{z_t \phi(q_t)}`$
总之，Linear Attention相较于常规attention的不同点：1.Linear Attention去掉了softmax计算；2.因果掩膜由一个Mask矩阵乘法运算转变成了递推运算。

应用实例
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825111945.png)
# 扩展LinearAttention
下图将 Mamba‑2 简化类比为带动态门控衰减的广义线性注意力，通过输入依赖的$`\gamma_t`$对历史记忆做选择性遗忘，解决朴素线性注意力只累加不遗忘的缺陷。 即便引入逐位置门控，Mamba‑2 依旧保有对偶性质，推理走 RNN 递推，训练阶段仍可以实现并行计算。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825114939.png)

下图对比了 Mamba‑2 与 Gated Delta Net 的记忆更新机制。 Mamba‑2 的记忆更新为 $`S_t = \gamma_t S_{t-1} + k_t v_t^\top,\quad y_t = q_t^\top S_t + v_t^\top D,\quad \gamma_t = f(x_t)`$ 依靠标量门控 $`\gamma_t`$ 对全部记忆做**全局统一衰减**，仅能整体淡忘历史，无法局部修改记忆。 Gated Delta Net 引入额外输入依赖门控 $`\beta_t=f(x_t)`$： $`S_t = \gamma_t\big(I-\beta_t k_t k_t^\top\big)S_{t-1} + \beta_t k_t v_t^\top\\ y_t = q_t^\top S_t`$ 借助投影算子 $`I-\beta_t k_t k_t^\top`$，定向擦除记忆中当前 $`k_t`$ 方向的旧信息，再写入新的 $`\beta_t k_t v_t^\top`$；当 $`\beta_t=0`$ 时可直接跳过本轮记忆更新。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825115121.png)
实例：Qwen‑Next 采用 $`\textbf{3‑1混合堆叠}`$：连续3层 Gated DeltaNet(GDN)，后接1层 Gated Attention，循环排布。 其中GDN为线性复杂度记忆模块，具备全局衰减与key方向定向擦除机制，并非传统简单线性注意力； Gated Attention 仍是带Softmax的二次复杂度注意力，增加Sigmoid输出门控，用来弥补GDN在精细长距离匹配上的不足。 所有层支持外层MoE封装，统一使用Zero‑Centered RMSNorm。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825115216.png)

下图来自对混合线性注意力的系统性研究，表格梳理了多种矩阵状态线性模型的状态更新公式；基于 RULER 长上下文基准，实验改变线性记忆层与标准 Softmax 注意力层的堆叠比例，发现全部使用线性记忆会造成长距离事实召回显著下降，混入少量完整注意力（如 3‑1 配比）即可恢复接近 Transformer 基线的性能
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825145629.png)
# Dynamic Sparse Attention
DSA（Dynamic Sparse Attention）作为GDN‑Attention混合架构之外的备选长上下文方案，不对全部历史token执行注意力计算；其由轻量Lightning索引器与细粒度token选择机制构成，索引器以少量索引头、ReLU激活、支持FP8低精度运算的方式快速计算query与历史token间的索引分数，依据分数筛选top‑$`k`$关键token，仅针对选出的KV条目完成标准注意力计算。该模块可在短上下文预训练结束后外挂接入实现$`\textit{post‑hoc}`$适配，无需重新开展长上下文预训练，即可扩展模型长上下文能力。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825151022.png)
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825151132.png)
# Mixture of experts
MoE（Mixture‑of‑Experts，混合专家）对比稠密Transformer，仅改造FFN前馈子模块，自注意力层保持不变； 稠密模型中所有token共享同一个完整FFN，全部参数参与运算； MoE将单个FFN替换为多个独立专家网络，通过门控选择器为每个token路由，仅激活少数专家完成计算，其余专家休眠。 因此可以提升专家总数来扩大模型总参数量，而不显著增加每个token的实际FLOPs，实现参数量与计算量解耦。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825153723.png)
下图解释MoE流行的核心原因：MoE实现了参数量与计算量FLOPs的解耦。 左图在固定训练FLOPs的条件下，增大MoE总参数量（增加专家数目），测试损失持续下降； 右图将MoE(Switch‑Base)与稠密T5‑Base对齐单步FLOPs，MoE取得更优的负对数困惑度，且专家数量越多性能越好。 即在相同训练算力预算下，MoE依靠大量休眠专家扩充模型总容量，获得优于稠密模型的训练效果
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825164651.png)
除此之外，另一个很大的好处是MOE提供了额外的并行化维度（专家并行；EP）
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260825164928.png)

MoE 虽然性能收益很大，但有两个现实问题。第一，它非常吃多机集群，大量专家参数要分散在很多 GPU 上，单机很难跑起来；第二训练不稳定，负载均衡损失是启发式的，loss 容易震荡，比稠密模型难训练调参，这也是 MoE 过去没有大范围普及的原因（左下图为MOE训练过程）
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260826095704.png)

**路由函数**
大体分为“token选择专家”、“专家选择token”以及全局决策，而通常采用的方法是“token选择专家”
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260826100201.png)
MoE 路由负责将 token 分配给对应专家，主流分为 Top-k路由与哈希路由。
Top-k Routing：通过可训练的 Router 网络计算 token 对各个专家的概率分数，选取概率最高的 Top-k专家参与计算，输出按概率加权融合；该方式感知 token 语义，为绝大多数 MoE 模型采用，但容易出现负载倾斜，需要引入负载均衡损失做约束。
Hash Routing：无可训练路由参数，依靠固定哈希函数直接将 token 映射至专家；天然实现负载均衡，但不感知语义，专家难以学习特定语义模式，一般作为实验基线。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260826101409.png)
MoE 的经典 Top-$`k`$ 路由的计算过程如下：
设第 $`l`$ 层输入第t个 token 向量为 $`\boldsymbol u_t^l`$，专家总数为 $`N`$，每个 token 激活专家数量为 $`K`$，$`\text{FFN}_i`$ 代表第 $`i`$ 个专家的前馈网络模块
首先将输入向量与路由权重向量 $`\boldsymbol e_i^l`$ 做矩阵点积，经 $`\text{Softmax}`$ 得到 token $`t`$ 分配至专家 $`i`$ 的打分 $`s_{i,t}=\text{Softmax}_i\left({\boldsymbol u_t^l}^T\boldsymbol e_i^l\right)`$
随后执行 $`\text{Top-}k`$ 筛选得到门控权重 $`g_{i,t}`$：若专家 $`i`$ 属于分数最高的 $`K`$ 个专家，则 $`g_{i,t}=s_{i,t}`$，否则 $`g_{i,t}=0`$；
层输出为各选中专家输出按门控加权求和并叠加残差连接，即 $`\boldsymbol h_t^l=\sum_{i=1}^N \big(g_{i,t}\text{FFN}_i(\boldsymbol u_t^l)\big)+\boldsymbol u_t^l`$
上述实现为先全局$`\text{Softmax}`$再$`\text{Top-}k`$，代表模型包括DeepSeek V1‑2、Qwen‑MoE与Grok；而Mixtral、DBRX、DeepSeek V3采用先$`\text{Top-}k`$后局部$`\text{Softmax}`$的变体，即先基于原始logit选出$`K`$个专家，仅对选中的$`K`$个专家内部执行$`\text{Softmax}`$归一化，保证选中专家的门控权重之和严格为1。路由网络本质等价于不带激活的多分类逻辑回归，仅用于专家打分，不输出特征，后续依靠$`\text{Top-}k`$实现稀疏计算。

**MOE训练**
混合专家模型MoE依靠$`\text{Top-}k`$稀疏门控实现每个token仅激活少量专家，以此在扩大模型参数量的同时控制单次前向计算开销，但该离散选择操作属于非光滑阶跃函数，在专家分数发生排名交换处不存在有效导数，造成反向传播时主任务损失的梯度无法穿过$`\text{Top-}k`$算子回传到路由网络Router；被选中专家的FFN参数可正常接收主损失梯度完成更新，而路由网络得不到来自下游任务的梯度信号，无法更新。
针对该梯度断裂问题存在三类主流解决思路：第一，将路由分配视作强化学习的采样动作，借助策略梯度与模型任务奖励更新路由策略，但存在梯度方差大、训练不稳定的缺陷，较少用于大规模预训练；第二，采用Gumbel‑Softmax等随机扰动方法，以连续可微分布近似离散$`\text{Top-}k`$选择，打通主损失到Router的梯度通路，推理阶段切回硬选择，但其性能对温度超参数较为敏感；第三也是工业界大规模MoE实际采用的启发式辅助损失方案，前向传播依旧保留硬$`\text{Top-}k`$稀疏计算以维持算力效率，总损失构造为$`\mathcal L_\text{total}= \mathcal L_\text{main}+\lambda \mathcal L_\text{aux}`$，主任务损失$`\mathcal L_\text{main}`$用于更新Transformer主干与专家FFN，负载均衡辅助损失$`\mathcal L_\text{aux}`$直接作用于$`\text{Top-}k`$截断之前的路由输出概率$`\boldsymbol s`$，整条数据流完全可微，独立为Router提供梯度，约束各专家接收token尽量均衡，该方案实现简单、训练收敛稳定，被Mixtral、Qwen‑MoE、DeepSeek‑MoE等模型广泛使用。
# 参考资料
https://zhuanlan.zhihu.com/p/1969419528065773811