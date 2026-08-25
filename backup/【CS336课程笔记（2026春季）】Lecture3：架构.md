# post-norm与pre-norm
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824094001.png)
假设当前隐藏状态为 xl​，Transformer 子层为 F(⋅)，它可以是 Attention，也可以是 MLP。
原始Transformer用的是post-norm，表达式为：
$·x_{l+1}=LN(x_l+F(x_l))·$
Pre-Norm 则把 LayerNorm 放到子层之前：
$·x_{l+1}=x_l+F(LN(x_l))·$

二者最核心的区别是残差路径不一样，post-norm虽然存在xl到xl+1的残差，但中间必经过LayerNorm和子层结构，并不是一条真正的“原样直通”的路径；而pre-norm存在一条真正的恒等残差路径，无论F这一支多么复杂，xl仍然可以直接传给下一层，多层以后，x0到xl始终存在直接的信息通路，这就是pre-norm对深层Transformer很重要的原因。

梯度传播时，pre-norm一般更加稳定，$x_{l+1} = x_l + F\left(\mathrm{LN}(x_l)\right)$求导之后是：
$$ \frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F\left(\mathrm{LN}(x_l)\right)} {\partial x_l} $$
其中I是单位矩阵，因此即使F分支梯度很小，仍然存在一条恒等路径直接向前传播
而对于post-norm，也就是$x_{l+1} = \mathrm{LN}\left(x_l + F(x_l)\right)$而言，求导大致是：
$$ \frac{\partial x_{l+1}}{\partial x_l} = J_{\mathrm{LN}} \left( I + J_F \right) $$
其中$J_{\mathrm{LN}}$表示子层F的雅可比矩阵，如果堆很多层：$J_{\mathrm{LN},L} \cdots J_{\mathrm{LN},2} J_{\mathrm{LN},1}$梯度就会不断受影响
总之，post-norm可以理解成，如果想把layern的梯度传回layer1，就必须经过一长串变换，pre-norm相当于存在一条稳定的残差流，每层只需不断添加一点信息

pre-norm最大问题是：随着层数越来越深，后面的某些层可能只是在残差流上做非常小的修正，从$x_L = x_0 + \sum_{l=0}^{L-1} F_l\left(\mathrm{LN}(x_l)\right)$可以看出，每层只是在原始残差流上添加增量，于是对于非常深的pre-norm网络，趋势往往是
```text
Layer 1 + 大量更新 
Layer 2 + 更新 
...
Layer 60 + 小更新 
Layer 61 + 小更新 
... 
Layer 100 + 很小更新
```
名义上有100层，但有效深度可能没有达到真正的100层

总结：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824101056.png)
# LayerNorm与RMSNorm
**LayerNorm定义**：
设一个token的隐藏向量为：$x = [x_1, x_2, \dots, x_d]$，Layernorm先计算均值：$\mu=\frac{1}{d}\sum_{i=1}^{d} x_i$，再计算方差：$\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i-\mu)^2$，然后归一化：$\hat{x}_i = \frac{x_i-\mu} {\sqrt{\sigma^2+\epsilon}}$，最后进行可学习的仿射变换：$y_i = \gamma_i \hat{x}_i + \beta_i$，总之Layernorm的完整作用就是：先中心化到均值 0，再缩放到方差约为 1，最后通过 γ,β 恢复可学习的尺度和平移。

**RMSNorm定义**：
RMSNorm不计算均值，它只计算平方根：$`\mathrm{RMS}(x) = \sqrt{ \frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon }`$，然后：$`\hat{x}_i = \frac{x_i} {\mathrm{RMS}(x)}`$，最后乘一个可学习的缩放参数：$`y_i = \gamma_i \frac{x_i} { \sqrt{ \frac{1}{d} \sum_{j=1}^{d}x_j^2 + \epsilon } }`$，总之，RMSNorm只控制向量整体尺度，不强制均值变成0

RMSNorm相对于LayerNorm少了一次均值相关的计算和数据操作，理论上总体少了一半操作，但计算快了不止一倍，因为这部分计算时memory-bound的，就比如归一化这个操作虽然操作量远低于其他，但却占了25.5%的时间
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824103851.png)

此外，RMSNorm可学习参数只有一个，相较于LayerNorm来说，少了一半，这是另一个优势
# GLU
**GLU（Gated Linear Unit，门控线性单元）** 是现代 Transformer，尤其是大语言模型 MLP/FFN 中非常重要的一类结构。可以把它理解成：不是直接对线性变换结果做激活，而是让一条分支产生“内容”，另一条分支产生“门”，再逐元素相乘，由门决定哪些特征通过、通过多少，这也是理解SwiGLU、GEGLU、ReGLU的基础

标准的Transformer的前馈网络通常是：$\mathrm{FFN}(x) = W_2\,\phi(W_1x)$，其中W1将输入维度扩展（一般是扩展为原来的4倍），经过激活函数（如ReLU、GeLU）之后再映射回原来的维度

原始 GLU 的核心形式是：$\mathrm{GLU}(x) = (xW_a+b_a) \odot \sigma(xW_b+b_b)$，其中：$\sigma(x)=\frac{1}{1+e^{-x}}$，第一条分支产生真正内容，第二条分支产生门，例如：内容分支产生`a=[2, -1, 5]`，门控分支经过Sigmoid得到`g=[0.9, 0.1, 0.6]`，那么输出得到`y=[1.8, -0.1, 3]`，因此GLU是在学习根据当前输入，这个特征到底应该通过多少？当把sigmoid换成其他激活函数的时候，就产生了：ReGLU、GEGLU、SwiGLU

此外由于多了一个新矩阵，GLU 类 FFN（SwiGLU/GeGLU/ReGLU）一共包含 **3 个权重矩阵**，而标准 FFN 仅 2 个权重矩阵。为保证替换前后**总参数量、计算量保持一致**，需要将 GLU 的中间隐层维度缩放为原标准 FFN 的 $\boldsymbol{\dfrac{2}{3}}$。注意：该缩放只是**等参数对比的工程约定**，并非数学强制约束；若不做缩放，GLU 的参数量与 FLOPs 会上涨 50%，训练推理成本随之增加。
# Positional Encoding与RoPE
self-attention的公式为：
$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
公式本身只关心token之间的内容关系，并不知道谁在前、谁在后。如果没有位置编码，self-attention看到的只是同一组token，只是排列顺序不同，所以必须额外告诉模型某个token在第几个位置，这就是位置编码的作用

一种最简单的方法是$h_i=x_i+p_i$，例如：
```text
我      = token_embedding(我)   + position_embedding(0)
喜欢    = token_embedding(喜欢) + position_embedding(1)
AI      = token_embedding(AI)   + position_embedding(2)
```
这样同一个 token 放在不同位置时，最终表示就不同。

Sinusoidal Position Encoding 是原始 Transformer 使用的位置编码，其特点是位置编码不是训练出来的，而是通过固定的 sin/cos 函数计算出来。
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
最终：$h_{pos} = x_{pos} + PE(pos)$
也就是直接把位置编码加到输入embedding上，然后再计算QK矩阵，Attention最终还是要自己从QK计算中把相对位置关系学出来，也就是说Sinusoidal 提供了位置，但没有直接把“相对距离”嵌入Attention 的内积结构中。

RoPE 则更进一步，它在探究如何用嵌入向量qkv向量的同时加入位置信息，函数表达如下：$q_m=f_q(x_m,m); k_n=f_k(x_n,n); v_n=f_v(x_n,n)$，其中$q_m$表示第m个token对应的词向量$x_m$集成位置信息m之后的query向量。而$k_n$和$v_n$则表示第n个token对应的词向量$x_n$集成位置信息n之后的key和value向量。

Sinusoidal函数的做法是在计算QKV之前，用Sinusoidal公式计算一个位置编码向量$p_i$加到词嵌入$x_i$上，$p_i$同样是d维向量，然而再乘以对应的变换矩阵W：
$$f_{t:t\in\{q,k,v\}}(x_i, i) := W_{t:t\in\{q,k,v\}}(x_i + p_i)$$

为了能利用token之间的相对信息，假定query向量$q_m$和key向量$k_n$之间的内积操作可以被一个函数g表示，该函数g的输入是词嵌入向量$x_m$，$x_n$和它们之间的相对位置m-n：
$$\big\langle f_q(x_m, m), f_k(x_n, n) \big\rangle = g(x_m, x_n, m-n)$$
接下来的目标就是找到一个等价的位置编码方式，从而使得上述关系成立。假设词嵌入向量的维度d=2，这样就可以利用二维平面上向量的性质推出（证明过程此处忽略）:
$$\begin{align*} f_q(x_m, m) &= (W_q x_m) e^{i m \theta} \\ f_k(x_n, n) &= (W_k x_n) e^{i n \theta} \\ g(x_m, x_n, m-n) &= \mathrm{Re}\left[ (W_q x_m) (W_k x_n)^* e^{i(m-n)\theta} \right] \end{align*}$$
进一步可以表示为：
$$\begin{equation} \begin{aligned} g(x_m, x_n, m-n) &= \begin{pmatrix} q_m^{(1)} & q_m^{(2)} \end{pmatrix} \begin{pmatrix} \cos((m-n)\theta) & -\sin((m-n)\theta) \\ \sin((m-n)\theta) & \cos((m-n)\theta) \end{pmatrix} \begin{pmatrix} k_n^{(1)} \\ k_n^{(2)} \end{pmatrix} \end{aligned} \end{equation}$$
将2维进行扩展，可以表示为：$f_{\{q,k\}}(x_m, m) = R_{\Theta,m}^d W_{\{q,k\}} x_m$
其中：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824155755.png)
总结来说，RoPE 的 self-attention 操作的流程是：对于 token 序列中的每个词嵌入向量，首先计算其对应的 query 和 key 向量，然后对每个 token 位置都计算对应的旋转位置编码，接着对每个 token 位置的 query 和 key 向量的元素按照 **两两一组** 应用旋转变换，最后再计算 query 和 key 之间的内积得到 self-attention 的计算结果，可以表示为下图：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824155913.png)
# hyperparameter
**1.前馈比（model dimension ratio）**：在FFN结构当中，输入的维度一般会从$d_{model}$膨胀到$d_{ff}$，经过激活函数之后再恢复到$d_{model}$，那么其中前馈比就是：
$$r_{ff} = d_{ff} / d_{model}$$
一般来说门控FFN的前馈比设置在4×2/3=8/3≈2.67可以保证计算量与普通FFN设置在常用值（4）一致
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824200428.png)
一个反常的例子是T5模型，它将前馈比设置为高达64的值，目的是提高硬件的利用率，然而这不一定能够带来较好的效果，如图所示，一般来说这个值设置在1-10效果较优：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824200605.png)

**2.(单头维度×头数量）/总维度**
一般来说这个值为1，不会增加总计算量，当然也有一些例外
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824200903.png)

**3.aspect ratio（模型长宽比）**
是指：大模型宽度（维度）与深度与深度的比值
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824201036.png)
一般来说对于这个参数的设置比较宽松

**4.词表尺寸**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824201209.png)

**5.dropout与weight decay**
dropout：训练时，以设定概率随机把一部分神经元输出置 0，阻断部分前向传播通路；推理阶段不做随机屏蔽。每次训练相当于训练多个子网络，避免神经元过度互相依赖，**抑制过拟合**。weight decay：属于优化器层面的正则手段，等价于 L2 权重惩罚。每一步更新时，让权重向 0 做小幅收缩

大模型预训练数据集规模极大，数据量远大于模型参数量，理论上看大预训练模型好像不需要正则来抑制过拟合。在实践中：老一代模型预训练开启Dropout=0.1，搭配weight deca，新一代主流大模型，关闭dropout，仅靠weight decay做正则，Qwen是少数例外依然保留dropout。一个很重要反直觉结论：LLM 里 weight decay 主要目的**不是防止过拟合**，它和 cosine 学习率调度相互作用，起到约束权重范数、稳定优化的效果。dropout 更多保留在微调阶段使用。

总之，目前的趋势是预训练过程（开weight-decay、关dropout）、训练一般可以都开，推理都关
# Stability trick
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824202343.png)
通常什么会导致不稳定（loss尖峰）？softmaxes！主要的原因是其中包含的指数运算与除法运算
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824202456.png)
softmax通常存在于两个地方：1.输出端（概率分布）；2.注意力运算内部
对于输出端，一种解决方法是采用z-loss：
常规softmax计算见左图（主loss），主loss的一个问题是，如果将输出的所有logits增加一个相同的值C，最终输出结果不会变化，这也导致主loss无法反映Z值的膨胀问题，会导致loss尖峰、训练崩溃
而右图是z-loss，增加了一个惩罚项，Z过大会使这个惩罚项变得很大，从而在后续训练中修正，使Z值平稳下来，避免了Z的过大或过小
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824202802.png)

解决注意力内的softmax问题是QK norm，输入经过层 LN → QKV 投影 → Split 拆分出 Q、K、V；在 Q、K 送入 BMM（QK 矩阵相乘）之前，分别对 Q、K 各自再做一次 RMS‑Norm（LayerNorm），V 不做归一化，这就是 QK‑Norm。
主要目的是防止QK的点积数值过大，送入softmax后同样出现梯度消失或者数值溢出
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824203802.png)
# attention heads
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824204329.png)
首先分析常规多头注意力机制下的操作数与访存数：
首先，矩阵乘法FLOPs的估算规则为：对于$A_{M×K}×B_{K×N}$，FLOPs=2MKN（乘加）
**一、算术操作数**
输入：$X∈R^{b×n×d}$
QKV投影：$X_{b,n,d}@W_{d,3d}$，总FLOPs为$6bnd^2$
注意力运算：第一个矩阵乘：$QK^T$：$(b,n,k) @ (b,k,n) \rightarrow (b,n,n)$ 单头 FLOPs：$2 \cdot n \cdot k \cdot n = 2n^2k$ 2. 第二个矩阵乘：$\mathit{Score} \cdot V$：$(b,n,n) @ (b,n,k) \rightarrow (b,n,k)$ 单头 FLOPs：$2 \cdot n \cdot n \cdot k = 2n^2k$ ，单头合计：$2n^2k + 2n^2k = 4n^2k$ ；$h$个头：$4 \cdot b  n^2 d$。
由于此处n<d，总体计算量应该由$bnd^2$主导
输出投影：$(b,n,d) @ (d,d)$，FLOPs为$2bnd^2$
综上，FLOPs为($bnd^2$)

**二、内存访问量**
1.$bnd$输入/输出张量访存
2.$bhn^2$：注意力分数矩阵访存（主要部分）：多头注意力的每个头都会生成一份n×n的注意力分数矩阵，每个头分数矩阵大小都是n×n，共h个头，batch为b，总大小为$bhn^2$
3.权重矩阵访存，QKV权重形状均为$d^2$

KV Cache示意图
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824210859.png)
带来的问题：极大降低了算术强度，总体的FLOPs不变，但访存量变化为：$(bn^2d+nd^2)$
分析：（decode阶段）
1.$bn^2d$：读取KV-Cache的访存开销，其中K,V∈$R^{b×n×d}$
每解码一步，必须把全部历史的K、V从HBM读出来，用于和新的单步Q做矩阵乘
2.$nd^2$：K、V投影权重的访存
每次step都需要读取K、V投影的矩阵$W_K$,$W_V$∈$R^{d×d}$

![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824213231.png)
如上图所示，想要提升算术强度就需要降低n/提高b

对于上述问题，引出MQA：KV的单头维度保留d/h，但头的数量降低为原来的1/h，计算的时候将这一个头通过广播分别做矩阵乘以满足Q计算的维度需求（但没有显式拷贝数据）
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824213427.png)
但这种方法会明显降低系统表达能力，为了进行这个权衡，进一步提出GQA
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260824213821.png)

