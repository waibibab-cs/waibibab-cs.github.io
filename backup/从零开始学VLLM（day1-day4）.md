本文基于的vllm commit hash为：`c16bb6068f70878fb8a2f7c4d6cda95cd03a778b`
# day1
在学习之前，想到的一个问题是vllm与pytorch这种通用深度学习框架的关系是什么？实际上，vllm正式基于pytorch构建的，后者解决的是“模型的一次前向计算”，而前者解决的是“如何让大量高并发请求持续、高效地使用模型”。具体关系可以总结为下图：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260914220225.png)
## token与tokenizer
大模型看不懂字符串（英文单词、中文），所以把请求中的prompt输入模型前，需要把字符串转换为其能看懂的数字，tokenizer就干这件事，具体来说包含了两个步骤：step1.把字符串切分成更小的单元（token）；step2.把切好的token映射成整数，也就是token id。一般来说，词表大小（vocab size）=最大token id +1。这部分详情也可见：[【CS336课程笔记（2026春季）】Lecture1&2：Tokenization与Pytorch](https://waibibab-cs.github.io/post/%E3%80%90CS336-ke-cheng-bi-ji-%EF%BC%882026-chun-ji-%EF%BC%89%E3%80%91Lecture1%262%EF%BC%9ATokenization-yu-Pytorch.html)

得到token id之后，会再经过embedding步骤将id转换为输入向量，一般来说输入向量的维度也就是隐藏层的维度，故embedding权重形状：`[vocab_size, hidden_size]`
综上，对于一个字符串，假设被切成x个token，则经过tokenization与embedding之后，会以形状为`[x, hidden_size]`的输入矩阵送入大模型。

## transformer
最初的transformer是为机器翻译设计的，采用encoder-decoder架构，简单理解就是：encoder用于理解需要被翻译的源文本，decoder依赖encoder的理解结果生成目标文本。在此过程中，encoder对源文本的理解是双向的，后面的token可以看到前面的token，前面的token也可以看到后面的。
OpenAI在提出GPT的时候认为所有任务都可以转换为“预测下一个token”，故将上述架构简化为decoder-only架构，也是目前几乎所有大模型采用的架构。虽然decoder-only架构下的prefill阶段与encoder处理的目标都是整个输入上下文，但前者采用掩码限制了“前面的token无法看见后面的token”。
一层transformer大体干两件事：**Attention**与**MLP(FFN)**

Attention干什么？例如我们读上面我刚说的那句话：`虽然decoder-only架构下的prefill阶段与encoder处理的目标都是整个输入上下文，但前者采用掩码限制了“前面的token无法看见后面的token”`。读到“前者”的时候，我们的大脑会不由自主地将注意力放到`decoder-only架构下的prefill阶段`上，Attention就是对这种现象进行数学建模，每当模型处理序列中某一个token时，都能兼顾句子中的其它token，并为这些token分配一个权重，权重越大，说明这个token与当前处理的token越相关。为了实现该功能，Attention为每个token向量引入了三个可学习的角色：
* Q：当前token主动查询其它token发出的信息
* K：其它token暴露出的用于被查询关联的特征
* V：每个token固有的信息或内容
Attention的具体公式如下：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260915135844.png)
其中：$QK^T$（形状：`[token个数,token个数]`）为每个token与其它token的相似度矩阵，也可以理解为token之间的影响程度，如下图所示，对于句子“Agent learns because it is smart.”，其相似性矩阵就可以用下图定性解释，第四行第一列代表了”Agent“对应的token对”it“对应的token影响巨大。此外，正方形表示自己对自己的影响，但至于这个影响到底多大可以忽略不计，右上角的掩码意味着禁止前面token受到后面token的影响。
![82b49ff92e1ca10d7bf2300b17ed7832_720.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/82b49ff92e1ca10d7bf2300b17ed7832_720.png)
Softmax用于把上述矩阵转换为概率表示，由于该运算涉及指数算子，所以除以$\sqrt{d_k}$防止数值爆炸。例如对上图的第四行，可能会转换为：`[a,b,c,d,0,0]`，然后用V进行加权求和。为什么要加权求和？注意力的目的就是让每个token融入其它token的信息（换句话说叫：携带上下文信息），这里的“其它token信息”就是每个token自己固有的V，而具体融入多少就是由相似度矩阵决定。例如对于“it”这个token，原本自己携带的信息为V4，通过相似度矩阵的加权求和，就变为了：aV1+bV2+cV3+dV4（a+b+c+d=1），这样就融入了前三个token的信息，得到形状`[token个数，hidden_size]`的输出
假设字符串通过tokenizer变成了tokenid矩阵（形状为：`[token个数,1]`），再经过embedding变成输入矩阵X（形状为：`[token个数,hidden_size]`），那么X与QKV的关系如下为：$Q=XW_Q，K=XW_K，V=XW_V$，其中三个W矩阵都属于要学习的**模型权重**，形状均为：`[hidden_size,hidden_size]`（三个W矩阵每层各有一份）

上述例子token之间的关联度偏向于一种指代关系（"it"指代了"agent"），但语言中的词间可能有更多类型的影响，比如从属关系，时态的影响等，一次注意力计算只能让语言学习到一种关联，多头注意力为解决这个问题应运而生。
其思想很简单：把输入矩阵X按hidden_size维度拆成多份X1、X2...Xn，对应存在多份的WQ1/WK1/WV1，WQ2/WK2/WV2，...，WQn/WKn/WVn（形状变为：`[hidden_size/n,hidden_size/n]`，因此总参数量不变）分别通过上述公式计算之后得到形状为`[token个数,hidden_size/n]`的输出，最后将n个输出在第1个维度上拼接（concat），恢复`[token个数,hidden_size]`的形状。
>在VLLM里，每层Attention默认走PagedAttention

MLP是什么？attention计算只做不同token之间信息的交互与融合，MLP（FFN，前馈网络）会为每一个独立的token做非线性映射，提升其表征能力。一般来说，对于attention计算后形状为`[token个数,hidden_size]`的输出，MLP会先对其进行升维操作，将其转换为`[token个数，hidden_size*4]`，然后再降维恢复原来的形状，从而更好地提取语义。

总之，一个transformer layer大体可以表示为如下结构：（第二个模块为MLP模块，其中第一个Linear用于升维，第二个用于降维）
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260624111945.png)

在输入经过多层transformer layer之后，设输出张量的形状为`[token个数，hidden_size]`，通过如下几步预测下一个token：
1. 取最后一个token的向量，形状为：`[1,hidden_size]`
2. 送入投影层LM Head（形状：`[hidden_size,Vacab_size]`），得到logits（形状：`[1,Vacab_size]`
3. 使用Softmax将logits转换为概率分布，然后从概率分布中挑选下一个token
4. （循环自回归），将新生成的token追加到输入序列末尾，重新送入起点，继续预测下一个token，直到超过最大长度或遇到结尾符
为什么第一步只取最后一个token用于预测？可以简单地理解为这是源于自回归语言建模的任务定义，本身模型训练的过程也是以“预测下一个token”为导向。
>从第四步可以看出LLM生成是逐token的串行过程，如果要生成x个token就需要forward x次，这x个迭代循环过程在VLLM的`EngineCore.run_busy.loop()`里

## KV Cache
为什么需要kvcache？直接看下面的例子：
假设输入prompt是“今天很”，共计3个token，送入注意力计算最终得到`[3,5]`的注意力矩阵（假设隐藏层维度为5）
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260915155109.png)
在第一轮的推理过程中，假设使用注意力矩阵（蓝色）经过一些操作预测下一个token为“高”，然后进入自回归的下一轮迭代：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260328232117.png)
如上图所示，第二轮推理仅仅是为了得到注意力矩阵的最后一行数据，反推到前面可知只需要相似度矩阵的最后一行，继续反推可知，只需要Q矩阵的最后一行，而KV矩阵的前三行保持不变，所以可以将其存储下来，在QKV投影阶段各自只需要计算一行而非`token个数`行，（KVCache使得KV只需计算一行、只需要新算出attention矩阵的最后一行反推出Q矩阵也只需要计算一行）这是典型的空间换时间的思路。

**KVCache**存储问题：
一个token在一个attention占的KV大小：
`per_token_layer_bytes=2*hidden_size*dtype_bytes`
对于Llama-3-70B：80层、模型维度8192、BF16，每个token大概有2.5MB
一个4K（输入+输出）的请求->10GB KV 空间占用极大！
>在VLLM里：
>KVCache由BlockPool（`vllm/v1/core/block_pool.py`）管理
>每个请求一张BlockTable，把逻辑序列映射到物理block
## Prefill&Decode
一个请求会分为两个阶段：prefill与decode，可以理解为前者是把输入prompt的所有token（假设n个）做个一次性的forward预测下一个的token（$t_{n+1}$），后者会进行x次循环（x为最终输出token数-1），每次循环会将上一轮预测得到的token加入输入序列中重新进行一次forward并得到新的预测token。
具体对比为：（B为batchsize；S与token数量；H为hidden_size）

| 方面        | Prefill                               | Decode                             |
| --------- | ------------------------------------- | ---------------------------------- |
| 处理对象      | 整个Prompt                              | 每轮一个新Token/请求                      |
| 执行次数      | 每个请求通常一次                              | 每生成一个Token执行一次                     |
| 输入形状      | [B,S,H]                               | [B,1,H]                            |
| 并行度       | Token间高度并行                            | 单请求Token间串行                        |
| Attention | 多个Query对多个K/V                         | 一个Query读取全部历史K/V                   |
| KV Cache  | 批量生成并写入                               | 读取历史KV，并追加一个KV                     |
| FFN       | 大矩阵乘法                                 | 小批量矩阵乘法                            |
| 主要瓶颈      | 通常偏计算                                 | 通常偏显存带宽                            |
| GPU利用率    | 较高                                    | 单请求时较低                             |
| 主要指标      | TTFT                                  | TPOT、ITL                           |
| 优化重点      | 高效GEMM、FlashAttention、Chunked Prefill | Batching、PagedAttention、CUDA Graph |
## GPU内存层级
GPU设备的内存空间可详细参考：[CUDA官方文档学习3：CUDA SIMT核函数](https://waibibab-cs.github.io/post/CUDA-guan-fang-wen-dang-xue-xi-3%EF%BC%9ACUDA%20SIMT-he-han-shu.html)
典型显卡的规格可见下图
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260915170248.png)
PCIe/NVLink带宽规格参数可参见：[PCIe & NVLink 带宽速查表](https://github.com/ForceInjection/AI-fundamentals/blob/main/01_hardware_architecture/performance/05_pcie_nvlink_speed_reference.md)
## 计算/内存受限与batching
具体的定义与roofline相关的内容可见：[【CS336课程笔记（2026春季）】Lecture1&2：Tokenization与Pytorch](https://waibibab-cs.github.io/post/%E3%80%90CS336-ke-cheng-bi-ji-%EF%BC%882026-chun-ji-%EF%BC%89%E3%80%91Lecture1%262%EF%BC%9ATokenization-yu-Pytorch.html#%E4%B8%80%E4%BA%9Bpytorch%E7%9A%84%E4%BE%8B%E5%AD%90%E5%B1%95%E7%A4%BA)的Pytorch小节
在这里只需要强调的是prefill一般为计算受限、decode为内存受限，continuous batching是一个加速decode的重要方法，原理就是其一定程度上缓解了带宽瓶颈

对于同时到来的多个请求：
若采用串行处理的方法，在decode阶段：加载一次权重--->算一个token
若采用批处理，例如batchsize=32：加载一次权重--->算32个请求各自的一个token
批处理具有多种形式：
1. static batching（朴素方法）：所有请求同时进入，最终性能取决于最慢那个。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260329105304.png)
2. continuous batching（vllm默认）：每有一个请求完成就重新组batch。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260329105812.png)
# day2
## 并行性
这部分内容可以详见：
1.[大模型推理入门学习2：大模型推理并行策略简介](https://waibibab-cs.github.io/post/da-mo-xing-tui-li-ru-men-xue-xi-2%EF%BC%9A-da-mo-xing-tui-li-bing-xing-ce-lve-jian-jie.html)(入门，必看)
2.[大模型推理入门学习4：Megatron系列推理并行策略](https://waibibab-cs.github.io/post/da-mo-xing-tui-li-ru-men-xue-xi-4%EF%BC%9AMegatron-xi-lie-tui-li-bing-xing-ce-lve.html)（进阶，选看）
## 推理服务的核心指标
**TTFT**：time to first token：从请求发出到生成第一个token的用时，主要由prefill阶段决定
**TPOT**：time per output token：首token生成之后，后续每个token之间的用时间隔，主要由decode阶段决定
**TTLT**：time to last token：从请求发出到生成最后一个token的总用时
**吞吐**：包含每秒生成的token数或每秒完成的请求数
# day3
## llm.py
**路径**：/home/dongmingzhe/vllm/vllm/entrypoints/llm.py
**简介**：这个文件包含了`LLM`类，它是vLLM暴露给用户的离线推理API封装类，本身并不是底层引擎，而是对`LLMEngine`的上层包装，给离线推理提供方便的入口。
**本小节阅读的类或函数**：
* `class LLM（全部）`
### LLM类
**作用**：根据输入提示词与采样参数生成文本，包含分词器、语言模型（支持分布式部署），以及为中间状态（KV缓存）分配的GPU显存空间。
>**1.什么是采样参数**？
>在大模型decode阶段，当模型算出每个候选token的概率之后，采样参数可以控制如何从概率分布当中挑选下一个token。常用的采样参数包含：
>- temperature：用于缩放logits，控制随机性；默认值为1.0；t=0时贪婪采样，永远选择概率最高的token，结果固定，适合代码/抽取类任务；0<t<1时，更保守，输出稳定，适合回答、RAG等；t>1时，放大随机性，脑洞更大，适合创意写作。
>- top_p（核采样）：只保留累积概率 ≥ top_p的候选 token，剩下直接丢弃；默认 `1.0`，范围 `(0,1]`；例如：`top_p=0.9`意为把概率累加起来，直到总和达到 0.9，只在这批 token 里采样；1.0 = 不截断，全部 token 参与采样
>- top_k：只保留概率最高的前 k 个 token；`-1` （默认值）代表不限制，全部 token 都参与；例如：`top_k=20`，仅从概率最高 20 个 token 中选
>- 除此之外，还有：min_p、repetition_penalty、frequency_penalty、presence_penalty、seed等

#### 构造函数
##### part1.参数列表
```python
class LLM(BeamSearchOfflineMixin, PoolingOfflineMixin, OfflineInferenceMixin):
    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template: Path | str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.92,
        cpu_offload_gb: float = 0,
        offload_group_size: int = 0,
        offload_num_in_group: int = 1,
        offload_prefetch_step: int = 1,
        offload_params: set[str] | None = None,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        return_sampling_mask: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        structured_outputs_config: dict[str, Any]
        | StructuredOutputsConfig
        | None = None,
        profiler_config: dict[str, Any] | ProfilerConfig | None = None,
        attention_config: dict[str, Any] | AttentionConfig | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        quantization_config: dict[str, Any] | QuantizationConfigArgs | None = None,
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
        spec_method: str | None = None,
        spec_model: str | None = None,
        spec_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
```
**构造函数阅读**
1.model：HuggingFace Transformers模型的名称（vLLM自动去HF Hub下载）或本地路径。

>裸星号的作用：Python语法，是一个参数分隔屏障，使前面的参数可以用位置传参或关键字传参，但后面的参数必须使用关键字传参

2.runner：runner 用来选择 vLLM 使用哪一套**模型运行器（Runner）** ，决定当前实例是做文本生成、向量池化 (embedding)、还是草稿模型（推测解码）。其中`RunnerOption` 是字面量类型：`Literal["auto", "generate", "pooling", "draft"]`，区别如下：
* auto（默认）：自动识别模型类型，选择对应的 runner。例如普通大语言模型（Qwen/Llama，做对话、续写）就选generate、Embedding / 分类 / 奖励模型就选pooling、草稿小模型（Speculative Decoding推测解码）选draft
* generate：文本生成Runner，就是我们最常用的，auto-regressive 自回归 token 生成，对话、续写、问答都用这个。
* pooling：池化Runner，用于Embedding向量模型等，不做token生成，只输出向量/打分
* draft：草稿模型Runner，用于推测解码里的 draft 小模型，配合目标大模型加速推理。
注意：一个vLLM实例只能同时跑一种runner

3.convert：在 `runner="pooling"` 时，指定如何把模型适配成向量提取（`"embed"`）或分类（`"classify"`）模型。

4.tokenizer：HuggingFace Transformers分词器的名称或本地路径。
>注意区分：embedding 负责把 token id 转换为向量；而 tokenizer 是字符与 token id 之间的转换；关于tokenizer的部分可见：[【CS336课程笔记（2026春季）】Lecture1&2：Tokenization与Pytorch](https://waibibab-cs.github.io/post/%E3%80%90CS336-ke-cheng-bi-ji-%EF%BC%882026-chun-ji-%EF%BC%89%E3%80%91Lecture1%262%EF%BC%9ATokenization-yu-Pytorch.html#tokenization)

5.tokenizer_mode: 分词器模式。"auto"：优先使用快速分词器（若可用）；"slow"：始终使用慢速分词器。
>fast（快速分词器）：Rust 实现，底层是 `tokenizers` 库，速度快、支持批量，vLLM 默认 `auto` 优先选它； slow（慢速分词器）：纯 Python 实现，`tokenization_xxx.py`，速度慢；但兼容性强，适合特殊模型

6.skip_tokenizer_init: 若为True，则跳过分词器与反分词器的初始化。此时输入需要提供合法的prompt_token_ids，prompt字段传入None。例如：
```python
llm = LLM(model="你的模型", skip_tokenizer_init=True)

outputs = llm.generate([
    {
        "prompt": None,
        "prompt_token_ids": [101, 2023, 2003, 1037, 2742],
    }
])
```
7.trust_remote_code：下载模型与分词器时，信任HuggingFace等来源的远程代码。
8.allowed_local_media_path: 允许API请求从服务器文件系统指定目录读取本地图片或视频。存在安全风险，仅可在可信环境开启。例如指定：
```bash
vllm serve your-vision-model \
  --allowed-local-media-path /data/vllm-images
```
之后的请求中就可以引用：
```json
{
  "type": "image_url",
  "image_url": {
    "url": "file:///data/vllm-images/cat.jpg"
  }
}
```
9.allowed_media_domains：若设置，多模态输入仅允许使用属于该域名的媒体URL。控制的输入为网络媒体URL，典型地址例如`https://images.example.com/cat.jpg`，限制URL的主机名必须在允许列表中。例如：
```bash
vllm serve your-vision-model \
  --allowed-media-domains images.example.com cdn.example.org
```
之后的请求就可以引用：
```text
https://images.example.com/cat.jpg   ✓
https://other.example.com/cat.jpg    ✗
```
10.tensor_parallel_size：张量并行分布式推理所使用的GPU数量。
>推理并行策略可见：
>1.[大模型推理入门学习2：大模型推理并行策略简介](https://waibibab-cs.github.io/post/da-mo-xing-tui-li-ru-men-xue-xi-2%EF%BC%9A-da-mo-xing-tui-li-bing-xing-ce-lve-jian-jie.html)
>2.[大模型推理入门学习4：Megatron系列推理并行策略](https://waibibab-cs.github.io/post/da-mo-xing-tui-li-ru-men-xue-xi-4%EF%BC%9AMegatron-xi-lie-tui-li-bing-xing-ce-lve.html)

11.dtype: 模型权重与激活值的数据类型。当前支持`float32`、`float16`、`bfloat16`等。若设为`auto`，则读取Transformers模型配置中的dtype；
12.quantization:模型权重量化方式。当前支持"awq"、"gptq"以及实验性的"fp8"。 若为None，会优先读取模型配置文件中的`quantization_config`；若该项也为空，则认为模型权重未量化，使用dtype确定权重数据类型。
>**AWQ**:Activation-aware Weight Quantization；用校准数据观察激活值，找出对输出影响较大的权重通道，通过缩放减小这些权重量化后的误差
>**GPTQ**：Generative Pre-trained Transformer Quantization；逐步量化权重，并利用输入数据估计的二阶信息，调整尚未量化的权重来补偿误差

13.revision: 使用的模型特定版本，可以是分支名、标签名或提交哈希ID。
14.tokenizer_revision: 使用的分词器特定版本，可以是分支名、标签名或提交哈希ID。
15.chat_template：要应用的对话模板。
16.seed: decode采样阶段随机数生成器的初始化种子。
17.gpu_memory_utilization: GPU显存预留比例（取值0~1），用于存放模型权重、激活值与KV缓存。取值越高，可分配的KV缓存空间越大，模型吞吐越高；但取值过高会引发显存溢出（OOM）。
18.cpu_offload_gb: 用于模型权重卸载的CPU内存大小，单位GiB。该机制相当于扩充GPU可存放模型权重的显存容量，但每一轮前向计算都需要承担CPU-GPU的数据传输开销。
19.offload_group_size：预取式权重卸载：将每offload_group_size层划分为一组，卸载每组最后的`offload_num_in_group`层。默认0（关闭该功能）。
20.offload_num_in_group：预取式权重卸载：每组需要卸载的层数。默认为1。
21.offload_prefetch_step: 预取式权重卸载：提前预取的层数。数值越大，延迟掩盖效果越好，但占用更多GPU显存。默认为1。
22.offload_params: 预取式权重卸载：需要选择性卸载的参数名片段集合。只有名称包含集合内片段的参数才会被卸载（例如MLP权重填{"gate_up_proj", "down_proj"}；MoE专家权重填{"w13_weight", "w2_weight"}）。若为None或空集合，则卸载全部参数。
23.enforce_eager: 是否强制使用Eager执行模式。若为True，则禁用CUDA Graph，全程使用Eager模式执行模型；若为False，则混合使用CUDA Graph与Eager模式。 24.enable_return_routed_experts: 是否返回路由选中的专家信息。
25.return_sampling_mask:开启后会返回每个采样 token 对应的后处理支持集，也就是采样阶段允许参与候选的 token 掩码。依赖 Model Runner V2 和对数概率计算，多用于结构化生成、推测解码等场景，用于记录每一步哪些 token 允许被采样；普通推理场景默认关闭，避免额外开销。
26.disable_custom_all_reduce：参见`[ParallelConfig][vllm.config.ParallelConfig]`
27.hf_token: 访问远程文件时用作HTTP Bearer授权的HuggingFace令牌。若设为True，则使用`hf auth login`登录生成的令牌（存储在~/.cache/huggingface/token）。
28.hf_overrides: 如果传入字典，则字典参数会转发至HuggingFace配置；如果传入可调用对象，则调用该对象更新HuggingFace配置。
29.pooler_config：为池化模型初始化非默认池化配置，例如 `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`
30.compilation_config: 整数或字典类型。整数代表编译优化模式；字典则可完整指定编译相关配置
31.attention_config: 注意力机制配置。可为字典或AttentionConfig实例；字典会自动转换为AttentionConfig对象。用于指定注意力后端以及其他注意力相关设置。
##### part2
```python
"""LLM constructor."""
if "disable_log_stats" not in kwargs:
	kwargs["disable_log_stats"] = True

if "worker_cls" in kwargs:
	worker_cls = kwargs["worker_cls"]
	# if the worker_cls is not qualified string name,
	# we serialize it using cloudpickle to avoid pickling issues
	if isinstance(worker_cls, type):
		kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

if "kv_transfer_config" in kwargs and isinstance(
	kwargs["kv_transfer_config"], dict
):
	from vllm.config.kv_transfer import KVTransferConfig

	raw_config_dict = kwargs["kv_transfer_config"]
	try:
		kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
	except ValidationError as e:
		logger.error(
			"Failed to convert 'kv_transfer_config' dict to "
			"KVTransferConfig object. Dict: %s. Error: %s",
			raw_config_dict,
			e,
		)
		# Consider re-raising a more specific vLLM error or ValueError
		# to provide better context to the user.
		raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e
	if hf_overrides is None:
		hf_overrides = {}
```
**主要作用**：LLM构造函数开头的参数预处理，在把`**kwargs`交给`EngineArgs`之前，先修正几个特殊情况：
1. 注入`disable_log_stats`：离线`LLM`类没有持续运行的调度器，统计日志没意义且会拖慢，故默认关掉
2. `worker_cls`序列化：`isinstance(x, type)`判断这是不是一个类，类对象无法安全地用标准 `pickle` 跨进程传递（函数/类定义在 `__main__` 或局部作用域时尤其会失败），所以用 `cloudpickle` 序列化成字节。注释里的 "qualified string name" 指 `"vllm.v1.worker.gpu_worker.Worker"` 这种全限定名——那种情况不需要处理，原样透传。
>- `worker_cls`是vLLM用来指定worker工作进程类的参数，有两种传入方式：a.全限定字符串（qualified string name），例如`worker_cls="vllm.v1.worker.gpu_worker.GPUWorker"`；b.直接传类对象本身
>- `pickle`是Python内置序列化库，序列化就是把内存的Python对象变成一串二进制字节，反序列化就是拿到这串二进制，在另一个Python环境重建出一模一样的对象。
>- 跨进程传递：vLLM中主进程（主LLM实例）启动多个GPU worker子进程、主进程要把配置参数（`worker_cls`、各种config）传给子进程，而将类传给子进程时，pickle只保存类名，子进程得不到类的定义
>- cloudpickle：增强pickle，可以把类/函数的定义一起打包，解决`__main__`自定义类跨进程反序列化失败问题

3. `kv_transfer_config`的dict转换为对象：`kv_transfer_config`是 vLLM 控制多 worker 间 KVCache 传输的配置；这段代码检测到传入字典类型的该参数时，把字典转为`KVTransferConfig`强类型对象并校验，校验失败则打印日志并抛出异常。
##### part3.make config函数
```python
def _make_config(value: Any, cls: type[_R]) -> _R:
	"""Convert dict/None/instance to a config instance."""
	if value is None:
		return cls()
	if isinstance(value, dict):
		return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})  # type: ignore[arg-type]
	return value
```
**主要作用**：把用户传进来的配置参数——不管是没传（`None`）、传了个字典、还是已经传了个配置对象——统统转成一个标准的配置对象，好让后面的 `EngineArgs` 拿到手就能直接用。字典转对象时会顺手把那些不能当构造参数用的字段（运行时才填的状态字段）过滤掉，避免报错。
##### part4
```python
if isinstance(compilation_config, int):
	compilation_config_instance = CompilationConfig(
		mode=CompilationMode(compilation_config)
	)
else:
	compilation_config_instance = _make_config(
		compilation_config, CompilationConfig
	)

structured_outputs_instance = _make_config(
	structured_outputs_config, StructuredOutputsConfig
)
profiler_config_instance = _make_config(profiler_config, ProfilerConfig)
attention_config_instance = _make_config(attention_config, AttentionConfig)

# warn about single-process data parallel usage.
_dp_size = int(kwargs.get("data_parallel_size", 1))
_distributed_executor_backend = kwargs.get("distributed_executor_backend")
if (
	_dp_size > 1
	and not _distributed_executor_backend == "external_launcher"
	and not current_platform.is_tpu()
):
	raise ValueError(
		f"LLM(data_parallel_size={_dp_size}) is not supported for single-"
		"process usage and may hang. Please use "
		"the explicit multi-process data-parallel example at "
		"'examples/features/data_parallel/data_parallel_offline.py'."
	)
```
**主要作用**：继续在真正搭建推理引擎**之前**做的两件事：先把几个配置参数（你可能传了简写数字、字典、或现成的配置对象）统一加工成标准配置对象，再检查一下数据并行的用法是否合法。
##### part5.创建引擎
```python
engine_args = EngineArgs(
	model=model,
	runner=runner,
	convert=convert,
	tokenizer=tokenizer,
	tokenizer_mode=tokenizer_mode,
	skip_tokenizer_init=skip_tokenizer_init,
	trust_remote_code=trust_remote_code,
	allowed_local_media_path=allowed_local_media_path,
	allowed_media_domains=allowed_media_domains,
	tensor_parallel_size=tensor_parallel_size,
	dtype=dtype,
	quantization=quantization,
	revision=revision,
	tokenizer_revision=tokenizer_revision,
	seed=seed,
	gpu_memory_utilization=gpu_memory_utilization,
	kv_cache_memory_bytes=kv_cache_memory_bytes,
	cpu_offload_gb=cpu_offload_gb,
	offload_group_size=offload_group_size,
	offload_num_in_group=offload_num_in_group,
	offload_prefetch_step=offload_prefetch_step,
	offload_params=offload_params or set(),
	enforce_eager=enforce_eager,
	enable_return_routed_experts=enable_return_routed_experts,
	return_sampling_mask=return_sampling_mask,
	disable_custom_all_reduce=disable_custom_all_reduce,
	hf_token=hf_token,
	hf_overrides=hf_overrides,
	mm_processor_kwargs=mm_processor_kwargs,
	pooler_config=pooler_config,
	structured_outputs_config=structured_outputs_instance,
	profiler_config=profiler_config_instance,
	attention_config=attention_config_instance,
	compilation_config=compilation_config_instance,
	quantization_config=quantization_config,
	logits_processors=logits_processors,
	spec_method=spec_method,
	spec_model=spec_model,
	spec_tokens=spec_tokens,
	**kwargs,
)

log_non_default_args(engine_args)

self.llm_engine = LLMEngine.from_engine_args(
	engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
)
self.model_config = self.llm_engine.model_config
self.engine_class = type(self.llm_engine)

self.request_counter = Counter()
self.default_sampling_params: dict[str, Any] | None = None

supported_tasks = self.llm_engine.get_supported_tasks()
self.supported_tasks = supported_tasks

self.runner_type = self.model_config.runner_type
self.renderer = self.llm_engine.renderer
self.chat_template = load_chat_template(chat_template)
self.input_processor = self.llm_engine.input_processor

self.renderer.warmup(ChatParams(chat_template=self.chat_template))

# The renderer thread pool is only consumed by the async renderer
# path; the synchronous `LLM` entrypoint runs multimodal
# preprocessing serially. Warn so the setting is not a silent
# no-op. See vllm-project/vllm#42901.
if self.model_config.renderer_num_workers > 1 and self.runner_type != "pooling":
	logger.warning_once(
		"`renderer_num_workers=%d` was set, but the offline `LLM` "
		"entrypoint uses the synchronous renderer path and runs "
		"multimodal preprocessing serially across prompts. The "
		"renderer thread pool is only consumed by the async "
		"renderer path used by `vllm serve` / `AsyncLLM`, so this "
		"setting has no effect here.",
		self.model_config.renderer_num_workers,
	)

PoolingOfflineMixin.__init__(self)

# Cache for __repr__ to avoid repeated collective_rpc calls
self._cached_repr: str | None = None
```
**主要作用**：把上一步整理好的参数打包成 `EngineArgs`，用它**真正创建推理引擎**（加载模型、占显存），然后把引擎里后续要用到的部件挂到 `self` 上，`LLM` 对象才算能用。主要包含以下内容：
1. `EngineArgs(...)` 打包配置：把散落的参数收进一个 `EngineArgs` 对象。左侧是 `LLM` 的参数名，右侧有 4 个是上一步转换过的 `_instance`，最后 `**kwargs` 把剩余参数一并灌进去。
>`EngineArgs` 是`@dataclass`定义的数据类，专门承载 vLLM 引擎的全部启动配置，实现命令行自动解析、前置参数校验，充当配置缓冲层。
>dataclass 自动生成`__init__`、`__repr__`等样板代码，适合**以保存数据为主**的类；普通 class 需要手动编写这些基础方法，适合包含复杂业务逻辑的对象。
2. `log_non_default_args` ：打印"用户改过的"参数（默认值不刷屏），并自动脱敏 `hf_token` 这类敏感字段。
3. `self.llm_engine = LLMEngine.from_engine_args(engine_args, UsageContext.LLM_CLASS)`：前面都只是"填表"，这一行才真正干活：加载模型权重到 GPU、建 KV cache 等。`UsageContext.LLM_CLASS` 告诉引擎"现在是离线 `LLM` 模式"（区别于 `vllm serve` 的在线模式）。
4. `request_counter` 统计发过多少请求；`default_sampling_params` 先占位 `None`，等用户第一次设默认采样参数时才填。
5. `supported_tasks`问引擎"你支持哪些任务"（生成 / 分类 / 打分 / 嵌入等），存下来后续做合法性检查。
6. 挂载四个核心部件：`runner_type`（引擎跑的是哪种模式【生成/池化】）、`renderer`（把用户输入【含聊天模板、多模态图片】加工成模型能识别的格式）、`chat_template`（对话模板，把`[{role, content}]`拼成模型认识的字符串）、`input_processor`（输入的预处理流水线）
7. `renderer.warmup(...)` 预热 跑一次空转换，把模板编译、缓存等一次性开销提前付掉，免得第一个请求特别慢。
8. 对 `renderer_num_workers` 的警告：这参数只在在线服务（`vllm serve` / `AsyncLLM`）里开线程池；离线 `LLM` 是串行处理，设了也没用。为防止"设了却静默失效"，这里主动 `warning_once` 提醒（`warning_once` = 只警告一次，不刷屏）。
#### from engine args
```python
@classmethod
def from_engine_args(cls, engine_args: EngineArgs) -> "LLM":
	"""Create an LLM instance from EngineArgs."""
	return cls(**vars(engine_args))
```
**主要作用**：接收已经填好、检查完的 EngineArgs 参数盒子，把盒子里所有参数全部拿出来，一股脑传给 LLM，直接造出一个能跑模型的 LLM 对象。不用我们手动一个个抄写参数，把配置盒子和模型启动两件事分开。
**输入（参数）**：`EngineArgs`对象
**输出（改变了什么）**：实例化后的LLM对象
#### get_world_size
```python
def get_world_size(self, include_dp: bool = True) -> int:
	"""Get the world size from the parallel config.

	Args:
		include_dp: If True (default), returns the world size including
			data parallelism (TP * PP * DP). If False, returns the world
			size without data parallelism (TP * PP).

	Returns:
		The world size (tensor_parallel_size * pipeline_parallel_size),
		optionally multiplied by data_parallel_size if include_dp is True.

	"""
	parallel_config = self.llm_engine.vllm_config.parallel_config
	if include_dp:
		return parallel_config.world_size_across_dp
	return parallel_config.world_size
```
**主要作用**：计算这次推理一共有多少个进程/设备在参与
**输入（参数）**：`include_dp: bool = True`，默认 `True` 表示"算上数据并行的那些副本"，传 `False` 表示"只算一个副本内部切了几份"。
**输出（改变了什么）**：返回一个 计算结果
```text
include_dp=True   →  PP × TP × PCP × DP   # 所有进程，含 DP 副本
include_dp=False  →  PP × TP × PCP        # 单个副本内部的并行度
```
#### reset_mm_cache
```python
def reset_mm_cache(self) -> None:
	self.renderer.clear_mm_cache()
	self.llm_engine.reset_mm_cache()
```
**主要作用**：清空多模态（图片/音频）预处理结果的缓存——你反复喂同一张图时 vLLM 会缓存处理结果避免重复计算，这个函数就是把这份缓存扔掉，把内存还回来。
**输出（改变了什么）**：清空前段侧/引擎侧的两处的缓存并重置统计
#### get_default_sampling_params
```python
def get_default_sampling_params(self) -> SamplingParams:
	if self.default_sampling_params is None:
		self.default_sampling_params = self.model_config.get_diff_sampling_param()
	if self.default_sampling_params:
		return SamplingParams.from_optional(**self.default_sampling_params)
	return SamplingParams()
```
**主要作用**：回答"用户没传 `sampling_params` 时该用什么默认值"——优先采用模型作者在 `generation_config.json` 里推荐的采样设置（有些模型会写 `temperature=0.6` 之类的建议值），模型没给就用 vLLM 的通用默认值。
**输出**：返回一个 `SamplingParams` 对象，同时会改变 `self.default_sampling_params` 这一个值——第一次调用时把查到的结果缓存起来，以后直接复用，不再重查。
#### generate
```python
def generate(
	self,
	prompts: PromptType | Sequence[PromptType],
	sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
	*,
	use_tqdm: bool | Callable[..., tqdm] = True,
	lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
	priority: list[int] | None = None,
	tokenization_kwargs: dict[str, Any] | None = None,
	mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[RequestOutput]:
	runner_type = self.model_config.runner_type
	# 池化模型（embedding那类）不能调用，会直接报错！
	if runner_type != "generate":
		raise ValueError(
			"LLM.generate() is only supported for generative models. "
			"Try passing `--runner generate` to use the model as a "
			"generative model."
		)

	if sampling_params is None:
		sampling_params = self.get_default_sampling_params()

	return self._run_completion(
		prompts=prompts,
		params=sampling_params,
		output_type=RequestOutput,
		use_tqdm=use_tqdm,
		lora_request=lora_request,
		tokenization_kwargs=tokenization_kwargs,
		priority=priority,
		mm_processor_kwargs=mm_processor_kwargs,
	)
```
**主要作用**：`LLM` 类最常用的方法——把一批 prompt 扔给引擎跑完生成，自动做批处理和排队（受显存限制），等全部跑完再一次性返回结果。
**输入（参数）**：
* `prompts`：单个 prompt 或一批（推荐传一批，性能最好）
* `sampling_params`：可省；单个则对所有 prompt 生效，传列表则必须与 `prompts` 等长、一一对应
* `use_tqdm`：进度条：`True` 显示、`False` 关掉、也可以传一个函数定制
* `lora_request`：指定用哪个 LoRA 微调权重
* `priority`：请求优先级，仅当引擎开了优先级调度时有效，且必须与 `prompts` 等长
* ``tokenization_kwargs` / `mm_processor_kwargs``：透传给分词 / 多模态处理器的额外参数
**输出（改变了什么）**：返回 `list[RequestOutput]`，顺序跟你传入的 `prompts` 完全一致（内部会做排序优化，返回前再还原）。没有改变 `self` 上的任何配置，但副作用很实在：这是一次真实的推理——会占 GPU、耗时、消耗 KV cache；如果没传 `sampling_params`，还会顺手把 `self.default_sampling_params` 缓存填上。
#### enqueue
```python
def enqueue(
	self,
	prompts: PromptType | Sequence[PromptType],
	sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
	lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
	priority: list[int] | None = None,
	use_tqdm: bool | Callable[..., tqdm] = True,
	tokenization_kwargs: dict[str, Any] | None = None,
	mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[str]:
	"""Enqueue prompts for generation without waiting for completion.

	This method adds requests to the engine queue but does not start
	processing them. Use wait_for_completion() to process the queued
	requests and get results.

	Args:
		prompts: The prompts to the LLM. See generate() for details.
		sampling_params: The sampling parameters for text generation.
		lora_request: LoRA request to use for generation, if any.
		priority: The priority of the requests, if any.
		use_tqdm: If True, shows a tqdm progress bar while adding requests.
		tokenization_kwargs: Overrides for `tokenizer.encode`.
		mm_processor_kwargs: Overrides for `processor.__call__`.

	Returns:
		A list of request IDs for the enqueued requests.

	"""
	runner_type = self.model_config.runner_type
	if runner_type != "generate":
		raise ValueError("LLM.enqueue() is only supported for generative models.")

	if sampling_params is None:
		sampling_params = self.get_default_sampling_params()

	return self._add_completion_requests(
		prompts=prompts,
		params=sampling_params,
		use_tqdm=use_tqdm,
		lora_request=lora_request,
		priority=priority,
		tokenization_kwargs=tokenization_kwargs,
		mm_processor_kwargs=mm_processor_kwargs,
	)
```
**主要作用**：`generate()` 的"半程版"——只把请求登记进引擎的等待队列就立刻返回，不等模型生成一个 token；等你之后再调 `wait_for_completion()` 才真正开跑。所以它本身不触发任何推理，毫秒级返回。
**输入（参数）**：和 `generate()` 几乎一样，但没有 `*`，所以后面几个也能按位置传
**输出（改变了什么）**：返回 `list[str]`——请求 ID 列表，与传入的 `prompts` 顺序一一对应。它改变的是引擎队列的状态（多了一批 pending 请求），同时 `self.request_counter` 被推进；若没传 `sampling_params`，还会顺带填充 `self.default_sampling_params` 缓存。
**细节**：
1. 与generate的关系：
```text
generate()   = _add_completion_requests + _run_engine      # 一步到位，阻塞
enqueue()    = _add_completion_requests                    # 只登记，立即返回
wait_for_completion() = _run_engine                        # 才真正跑
```
2. 为什么要有它？`generate()` 是阻塞的——一批必须跑完才能提交下一批，批与批之间引擎会空转。`enqueue()` 允许你连续登记好几批，最后统一 `wait_for_completion()` 一次收完。
3. `generate()` 能保证"输出顺序 = 输入顺序"，是因为它一次性把整批登记完再收；而 `enqueue()` 允许你分多次登记，此时最终顺序 = 登记顺序，跟你某一次调用的输入顺序未必一致。
#### wait_for_completion
```python
@overload
def wait_for_completion(
	self,
	*,
	use_tqdm: bool | Callable[..., tqdm] = True,
) -> list[RequestOutput | PoolingRequestOutput]: ...

@overload
def wait_for_completion(
	self,
	output_type: type[_O] | tuple[type[_O], ...],
	*,
	use_tqdm: bool | Callable[..., tqdm] = True,
) -> list[_O]: ...

def wait_for_completion(
	self,
	output_type: type[Any] | tuple[type[Any], ...] | None = None,
	*,
	use_tqdm: bool | Callable[..., tqdm] = True,
) -> list[Any]:
	"""Wait for all enqueued requests to complete and return results.

	This method processes all requests currently in the engine queue
	and returns their outputs. Use after enqueue() to get results.

	Args:
		output_type: The expected output type(s). If not provided, accepts
			both RequestOutput and PoolingRequestOutput.
		use_tqdm: If True, shows a tqdm progress bar.

	Returns:
		A list of output objects for all completed requests.

	"""
	if output_type is None:
		output_type = (RequestOutput, PoolingRequestOutput)

	return self._run_engine(output_type, use_tqdm=use_tqdm)
```
这三个不是三个函数，而是同一个函数的三张"脸"——前两个 `@overload` 只给类型检查器看，运行时会被第三个（真正的实现）完全覆盖。
**主要作用**：把 `enqueue()` 登记进队列的请求**全部跑完并收结果**——它是 `generate()` 拆解后的后半截，真正耗时烧 GPU 的部分（`_run_engine` 里的 `while ... step()` 循环）就在这儿。
**输入（参数）**：
* 第一个overload：只有`use_tqdm`
* 第二个overload：`output_type`+`use_tqdm`
* 真身：`output_type=None`+`use_tqdm`
**输出（改变了什么）**：返回 `list[...]`，元素是 `output_type` 指定的那些类的实例。改变的是引擎队列状态——把队列里所有 pending 请求消费掉变成已完成，队列变空。
#### collective_rpc
```python
def collective_rpc(
	self,
	method: str | Callable[..., _R],
	timeout: float | None = None,
	args: tuple = (),
	kwargs: dict[str, Any] | None = None,
) -> list[_R]:
	"""Execute an RPC call on all workers.

	Args:
		method: Name of the worker method to execute, or a callable that
			is serialized and sent to all workers to execute.

			If the method is a callable, it should accept an additional
			`self` argument, in addition to the arguments passed in `args`
			and `kwargs`. The `self` argument will be the worker object.
		timeout: Maximum time in seconds to wait for execution. Raises a
			[`TimeoutError`][] on timeout. `None` means wait indefinitely.
		args: Positional arguments to pass to the worker method.
		kwargs: Keyword arguments to pass to the worker method.

	Returns:
		A list containing the results from each worker.

	Note:
		It is recommended to use this API to only pass control messages,
		and set up data-plane communication to pass data.

	"""
	return self.llm_engine.collective_rpc(method, timeout, args, kwargs)
```
**主要作用**：手动把一条指令广播给所有 worker 进程（每张 GPU 一个），让它们各自执行同一个方法，再把各进程的返回值收回来。这是给高级场景（权重在线更新、RLHF 交换权重、自定义模型改写）留的"后门"——常规推理接口做不到这些。
**输入（参数）**：
* `method`：要执行的方法，字符串（方法名）或者一个函数对象
* `timeout`：超时秒数，`None`是无限等，超时抛`TimeoutError`
* `args`：传给该方法的位置参数，元组
* `kwargs`：传给该方法的关键字参数，字典
**输出（改变了什么）**：返回 `list[_R]`——每个 worker 一个返回值，列表长度 = worker 数量（world size）。不改变 `LLM` 自身的状态，但它执行的那个 `method` 可以在 worker 上干任何事（改权重、清缓存、打印信息），所以实际影响完全取决于你传了什么。
**细节**：
1. 写的函数要这样定义：
```python
def my_fn(worker, extra_arg):     # worker 会被自动填进来
    ...
llm.collective_rpc(my_fn, args=(extra_arg,))
```
2. 只用它传"指令"，别用它传"数据"。因为 RPC 走的是控制通道，传大块数据（比如几百 MB 的权重张量）会把控制通道堵住，影响正常调度——真正的大数据传输应该另开数据通道（data-plane），比如代码里 `apply_model`、weight transfer 那些用法就是只传一个函数或一小段 init 信息（`kwargs={"init_info": ...}`），数据本身走别的路。
>**`RPC` = Remote Procedure Call，远程过程调用。** 这里的"远程"指的是**另一个进程**（每个 GPU 一个 worker 进程），不一定是另一台机器。调用链是：
>`LLM.collective_rpc → llm_engine → engine_core → executor → 各 worker`
#### apply model
```python
def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
	"""Run a function directly on the model inside each worker,
	returning the result for each of them.

	!!! warning
		To reduce the overhead of data transfer, avoid returning large
		arrays or tensors from this method. If you must return them,
		make sure you move them to CPU first to avoid taking up additional
		VRAM!
	"""
	return self.llm_engine.apply_model(func)
```
**主要作用**：`apply_model` 是 vLLM 基于 `collective_rpc` 封装的高阶 API，它广播一个用户函数至全部 GPU worker 进程，在每个 worker 上把模型`nn.Module`对象传入该函数执行，最后收集所有 worker 的返回结果；它仅开放模型对象给用户操作（最小权限），适合权重查看、在线修改、权重统计等模型相关操作，但**不支持超时控制**，若传入函数死循环会导致引擎永久阻塞。
**输入（参数）**：只有一个参数 `func: Callable[[nn.Module], _R]`——一个"接收模型对象、返回任意结果"的函数。
**输出（改变了什么）**：返回 `list[_R]`，每个 worker 一个结果，长度 = worker 数量。不改变 `LLM` 自身状态，但 `func` 内部可以对模型做任何事。
#### chat
```python
def chat(
	self,
	messages: list[ChatCompletionMessageParam]
	| Sequence[list[ChatCompletionMessageParam]],
	sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
	use_tqdm: bool | Callable[..., tqdm] = True,
	lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
	chat_template: str | None = None,
	chat_template_content_format: ChatTemplateContentFormatOption = "auto",
	add_generation_prompt: bool = True,
	continue_final_message: bool = False,
	tools: list[dict[str, Any]] | None = None,
	chat_template_kwargs: dict[str, Any] | None = None,
	tokenization_kwargs: dict[str, Any] | None = None,
	mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[RequestOutput]:
	model_config = self.model_config
	runner_type = model_config.runner_type
	if runner_type != "generate":
		raise ValueError(
			"LLM.chat() is only supported for generative models. "
			"Try passing `--runner generate` to use the model as a "
			"generative model."
		)

	if sampling_params is None:
		sampling_params = self.get_default_sampling_params()

	return self._run_chat(
		messages=messages,
		params=sampling_params,
		output_type=RequestOutput,
		use_tqdm=use_tqdm,
		lora_request=lora_request,
		chat_template=chat_template,
		chat_template_content_format=chat_template_content_format,
		chat_template_kwargs=chat_template_kwargs,
		add_generation_prompt=add_generation_prompt,
		continue_final_message=continue_final_message,
		tools=tools,
		tokenization_kwargs=tokenization_kwargs,
		mm_processor_kwargs=mm_processor_kwargs,
	)
```
**主要作用**：用"对话消息列表"而不是裸文本作为输入——自动套用模型的聊天模板把 `[{role, content}]` 拼成模型认识的字符串，然后走和 `generate()` 完全一样的生成流程。`chat`实际上就是`generate`的上层封装。
**输入（参数）**：
* message：一个对话，或一批对话（代表多个独立请求）。每个对话 = message 的列表，每条 message = 含 role / content 的字典
>对话就是一轮完整的多轮聊天的全部历史，例如：
>`[ {"role":"system", "content":"你是一个助手"}, {"role":"user", "content":"1+1等于几？"}, {"role":"assistant", "content":"等于2"}, {"role":"user", "content":"那再加1呢？"} ]`
>role包含system（系统提示）、user（用户）、assistant（模型上一轮自己输出的回答）
* chat_template：用哪个聊天模板，不传就用模型自带的
* add_generation_prompt：是否在末尾加“轮到助手说话”的起始标记
* continue_final_message：让最后一条消息未写完，模型接着写
* tools：提供给模型的工具
* chat_template_kwargs：透传给模板的额外参数
**输出（改变了什么）**：`list[RequestOutput]`，顺序与输入的 messages 一致。不改变 `LLM` 自身的状态（除了可能填充那份默认采样参数缓存）。
#### enqueue chat
```python
def enqueue_chat(
	self,
	messages: list[ChatCompletionMessageParam]
	| Sequence[list[ChatCompletionMessageParam]],
	sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
	use_tqdm: bool | Callable[..., tqdm] = True,
	lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
	priority: list[int] | None = None,
	chat_template: str | None = None,
	chat_template_content_format: ChatTemplateContentFormatOption = "auto",
	add_generation_prompt: bool = True,
	continue_final_message: bool = False,
	tools: list[dict[str, Any]] | None = None,
	chat_template_kwargs: dict[str, Any] | None = None,
	tokenization_kwargs: dict[str, Any] | None = None,
	mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[str]:
	model_config = self.model_config
	runner_type = model_config.runner_type
	if runner_type != "generate":
		raise ValueError(
			"LLM.enqueue_chat() is only supported for generative models. "
			"Try passing `--runner generate` to use the model as a "
			"generative model."
		)

	if sampling_params is None:
		sampling_params = self.get_default_sampling_params()

	return self._add_chat_requests(
		messages=messages,
		params=sampling_params,
		use_tqdm=use_tqdm,
		lora_request=lora_request,
		priority=priority,
		chat_template=chat_template,
		chat_template_content_format=chat_template_content_format,
		chat_template_kwargs=chat_template_kwargs,
		add_generation_prompt=add_generation_prompt,
		continue_final_message=continue_final_message,
		tools=tools,
		tokenization_kwargs=tokenization_kwargs,
		mm_processor_kwargs=mm_processor_kwargs,
	)
```
**主要作用**：`chat()` 的"半程版"——把对话消息渲染成 prompt、登记进引擎队列就立刻返回，不等生成；之后由 `wait_for_completion()` 统一收结果。等于 `enqueue()`（登记不跑）的对话输入版本。
**输入（参数）**：大部分与`chat`一致，外加了特有的`priority`
**输出（改变了什么）**：`list[str]`——请求 ID 列表，与 `messages` 顺序一一对应。改变引擎队列状态（多了一批 pending 请求），并推进 `self.request_counter`。
#### start（stop） profile
```python
def start_profile(self, profile_prefix: str | None = None) -> None:
	"""Start profiling with optional custom trace prefix.

	Args:
		profile_prefix: Optional prefix for the trace file names. If provided,
					   trace files will be named as "<prefix>_dp<X>_pp<Y>_tp<Z>".
					   If not provided, default naming will be used.

	"""
	self.llm_engine.start_profile(profile_prefix)
def stop_profile(self) -> None:
	self.llm_engine.stop_profile()
```
**主要作用**：打开性能分析开关——从这一刻起，各 GPU worker 上的 `torch.profiler` 开始逐算子记录 CPU/GPU 各花了多少时间。用来回答"到底慢在哪里"。必须和 `stop_profile()` 配对使用。
**输入（参数）**：只有一个可选参数 `profile_prefix: str | None = None`——trace 文件名的前缀。
**输出（改变了什么）**：返回 `None`。它改变的是 worker 内部的状态（把 profiler 从"关闭"切到"记录中"，对应 `engine_core.profile(True, prefix)`），真正产出的是磁盘上的 trace 文件——而且是在你调 `stop_profile()` 时才落盘。
#### reset prefix cache
```python
def reset_prefix_cache(
	self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool:
	return self.llm_engine.reset_prefix_cache(
		reset_running_requests, reset_connector
	)
```
**主要作用**：清空 KV prefix cache（前缀缓存）——vLLM 会把"开头相同的 prompt"已经算好的 KV 缓存存下来复用（比如同一个 system prompt），这个函数就是把这份缓存全部作废、把显存还回来。
**输入（参数）**：
* `reset_running_requests`：是否**强行清**。`False` = 只在引擎空闲时才清（清不掉就返回 `False`）；`True` = 把正在跑的请求全部抢占、踢回等待队列来腾地方
* `reset_connector`：是否连外部 KV 连接器（跨实例 KV 传输、LMCache 那类）的缓存也一起清
**输出（改变了什么）**：返回 `bool`——"是否清成功了"，不是"清了多少"。改变的是缓存状态：把前缀缓存里保存的 KV 块释放掉。副作用（当 `reset_running_requests=True`）：所有 running 请求被抢占并重新排队——它们不会被丢弃，但已有进度作废、要重算；已经产出但还没发出去的部分会被丢掉（源码里的 `drop_stale_output=True`）。
#### sleep
```python
def sleep(self, level: int = 1, mode: PauseMode = "abort"):
	self.llm_engine.sleep(level=level, mode=mode)
```
**主要作用**：让引擎"睡着"——暂停推理，并按档位决定要不要把显存里的东西（模型权重、KV cache）卸下去，好把这块显卡资源让给别人（换模型、跑训练）。配 `wake_up()` 唤醒。
**输入（参数）**：
* `level`：睡多深？如果是0:只暂停调度。请求还收得进，但一条都不处理；显存原封不动（连前缀缓存都不清）；如果是1：把模型权重卸到 CPU 内存，丢掉 KV cache；如果是2：权重与KV cache全丢掉。
* `mode`：已经在跑的请求怎么办？`"abort"`（默认）：立即中止所有在途请求；`"wait"`：等它们跑完再睡；`"keep"`：把它们冻结在队列里，醒了继续
**输出（改变了什么）**：返回 `None`。改变的是引擎的运行状态（运行中 → 睡眠中）和显存/CPU 内存占用（按 level 卸载）。
#### release kv cache memory
```python
def release_kv_cache_memory(self) -> None:
	self.llm_engine.release_kv_cache_memory()
```
**主要作用**：把 KV cache 占的显存物理释放掉，但保留模型权重——相当于 `sleep(level=1)` 的"减配版"（level 1 会连权重一起卸到 CPU，它只丢 KV cache）。
**输出（改变了什么）**：返回 `None`。改变的是显存占用（KV cache 那块被释放）+ 引擎的睡眠状态（内部按 level 1 记录）。恢复用 `wake_up(tags=["kv_cache"])`。
#### wake up
```python
def wake_up(self, tags: list[str] | None = None):
	self.llm_engine.wake_up(tags)
```
**主要作用**：`sleep()` 的逆操作——把之前卸载或丢弃的显存资源按 tag 重新分配回来，让引擎恢复干活。
**输入（参数）**：`tags: list[str] | None = None`，各值代表的含义：
* `None`：全部恢复
* `["scheduling"]`：从 level 0睡眠恢复（level 0 只暂停调度，没有显存需要恢复）
* `["weights"]` / `["kv_cache"]`：只恢复其中一项（部分唤醒）
**输出（改变了什么）**：在 `LLM` 这层没有写返回类型，实际返回 `None`——引擎那层算出的 `bool`（是否完全清醒）被丢掉了。改变的是：worker 上重新分配显存 + 内部记账集合 `sleeping_tags` 移除对应项；如果全醒了，调度器会被恢复。
#### get metrics
```python
def get_metrics(self) -> list["Metric"]:
	return self.llm_engine.get_metrics()
```
**主要作用**：读取 vLLM 当前的运行统计信息，就像查看程序的“监控面板”，例如处理了多少请求、推理速度和延迟等。
**输出（改变了什么）**：返回 `list["Metric"]`，也就是由多个 `Metric` 对象组成的列表；其中可能包含 `Counter`、`Gauge`、`Histogram` 和 `Vector`，每项指标通常具有名称、标签和值。这个函数只读取当前快照，不会修改或清零统计数据，也不会改变模型参数、请求状态或 KV Cache。
**细节补充**：
1. Prometheus：一种程序监控工具。vLLM 在运行时会记录请求数、生成 token 数、推理速度、等待时间和缓存使用率等数据，Prometheus 负责保存和读取这些监控指标。
2. Metric：表示一项监控指标。每个指标通常包含名称、标签和值，例如某个指标可以表示“当前正在运行的请求数量”。
3. Counter：计数器，一般只会不断增加，例如累计处理的请求数、累计生成的 token 数。
4. Gauge：仪表值，可以增加也可以减少，例如当前等待的请求数、正在运行的请求数或 KV Cache 使用率。
5. `Vector`：一组相互关联的统计值；当前代码主要用它记录投机解码中不同 token 位置被接受的数量，入门阶段知道它表示“一组数值”即可。
#### init weight transfer engine
```python
def init_weight_transfer_engine(
	self, request: WeightTransferInitRequest | dict
) -> None:
	init_info_dict = (
		request["init_info"] if isinstance(request, dict) else request.init_info
	)

	self.llm_engine.collective_rpc(
		"init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
	)
```
**主要作用**：为强化学习训练场景初始化“模型权重传输引擎”，并通过 `collective_rpc` 通知所有 vLLM worker 准备接收训练端之后传来的新模型权重。
**输入（参数）**：接收一个 `request`，它可以是 `WeightTransferInitRequest` 对象，也可以是普通 `dict`；函数会从中取出 `init_info`，其中保存了所选传输后端需要的初始化信息。
**输出（改变了什么值）**：函数返回 `None`，但会改变所有 worker 的内部状态，使其权重传输机制完成初始化；它此时还不会真正传输或更新模型权重。
**细节补充**：
1. 权重传输引擎：专门负责在训练端和 vLLM 推理端之间传递模型权重的组件。这个函数只完成该组件的初始化，为后续的权重更新建立通信条件。
在强化学习训练中，流程大致是：
```text
vLLM 用当前权重生成回答
        ↓
训练端根据奖励评价回答
        ↓
训练端更新模型权重
        ↓
把最新权重传给 vLLM
        ↓
vLLM 使用新权重生成下一批回答
```
#### start weight update
```python
def start_weight_update(self) -> None:
	"""Start a new weight update."""
	self.llm_engine.collective_rpc("start_weight_update")
```
**主要作用**：通知所有 vLLM worker 开始一次新的“权重更新会话”，让它们进入准备接收新模型权重的状态；它对应 `初始化 → 开始更新 → 传输权重 → 结束更新` 流程中的“开始更新”阶段。
**输入（参数）**：没有需要用户手动传入的参数；`self` 表示当前 `LLM` 对象，函数通过其内部的 `llm_engine` 发出通知。
**输出（改变了什么值）**：函数返回 `None`，但所有 worker 会把内部的权重更新状态标记为“正在进行”；某些传输后端还会准备逐层加载模型权重。此时通常还没有收到新权重，因此不会直接把模型参数更新成新值。
####  start draft weight update
```python
def start_draft_weight_update(self) -> None:
	self.llm_engine.collective_rpc("start_draft_weight_update")
```
**主要作用**：通知所有 vLLM worker 开始一次针对“投机解码草稿模型”的权重更新会话，将后续收到的新权重写入草稿模型，而不是主模型。
**输出（改变了什么值）**：函数返回 `None`，但会把 worker 的权重更新目标切换为草稿模型，并将内部状态标记为“草稿模型权重更新正在进行”。此时只是做好准备，还没有真正接收和修改权重值。
**细节补充**：
1. 投机解码：一种加速大模型生成的方法。它先让较小、速度较快的草稿模型预测若干 token，再由较大、结果更可靠的主模型一次性检查这些 token；验证通过的 token 可以直接使用，从而减少主模型的推理次数。
2. 草稿模型：投机解码中负责快速提出候选 token 的小模型。它通常速度更快，但预测结果不一定完全可靠，因此还需要主模型验证。
#### update weights
```python
def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
	update_info_dict = (
		request["update_info"] if isinstance(request, dict) else request.update_info
	)

	self.llm_engine.collective_rpc(
		"update_weights", kwargs={"update_info": update_info_dict}
	)
```
**主要作用**：在已经开始的权重更新会话中，把训练端提供的一个权重更新块发送给所有 vLLM worker，并加载到本次会话选定的主模型或草稿模型中；一个完整模型可以通过多次调用该函数分块更新。
**输入（参数）**：接收一个 `request`，它可以是 `WeightTransferUpdateRequest` 对象，也可以是普通 `dict`；函数从中取出后端专用的 `update_info`，其形式可以是一个字典，也可以是为不同 worker 准备的字典列表。
**输出（改变了什么值）**：函数返回 `None`，但会通过 `collective_rpc` 让所有 worker 接收并应用本次权重更新，从而改变目标模型的部分权重值；某些后端可能先将更新放入处理队列，并在 `finish_weight_update()` 时确保全部完成。
**细节补充**：
1. 分块更新：多次调用该函数分别传输模型的不同部分，例如一次传输某一层或一组参数；全部传输完成后，再调用 `finish_weight_update()` 结束本次会话。
2. 传输后端：真正负责权重通信和加载的实现，例如使用 NCCL 通信或 CUDA IPC 共享 GPU 数据。`update_info` 的具体结构由所选后端决定。
#### finish weight update
```python
def finish_weight_update(self, weight_version: str | None = None) -> None:
	self.llm_engine.collective_rpc("finish_weight_update")
	if weight_version is not None:
		self.llm_engine.set_weight_version(weight_version)
```
**主要作用**：通知所有 vLLM worker 结束当前权重更新会话，等待并完成尚未处理完的权重更新、恢复默认更新目标；如果提供了 `weight_version`，还会给这批新权重记录一个版本标签。
**输入（参数）**：`weight_version` 是可选的字符串参数，默认值为 `None`；可以传入 `"step-100"`、`"v2"` 等字符串标识本次更新，也可以不传。
**输出（改变了什么值）**：函数返回 `None`；它会结束 worker 的“正在更新”状态，并保留已经加载的新权重。若提供了版本号，EngineCore 中记录的当前权重版本也会被更新，但版本号本身不会再次修改模型权重。
#### update weight version
```python
def update_weight_version(self, new_version: str) -> None:
	self.llm_engine.set_weight_version(new_version)
```
**主要作用**：只修改 vLLM 记录的“当前模型权重版本标签”，用于标识现在使用的是哪一版权重；它不会传输或修改真正的模型权重。
**输入（参数）**：接收一个字符串 `new_version`，例如 `"v2"`、`"step-100"` 或 `"checkpoint-500"`。
**输出（改变了什么值）**：函数返回 `None`；它会将 EngineCore 内部保存的 `_weight_version` 替换为 `new_version`，但模型参数、worker 状态和正在进行的推理请求都不会被改变。
#### repr
```python
def __repr__(self) -> str:
	# Cache the result to avoid repeated collective_rpc calls
	if self._cached_repr is None:
		results = self.llm_engine.collective_rpc("get_model_inspection")
		# In distributed settings, we get results from all workers
		# Just return the first one (they should all be the same)
		if results:
			self._cached_repr = results[0]
		else:
			self._cached_repr = f"LLM(model={self.model_config.model!r})"
	return self._cached_repr
```
**主要作用**：生成当前 vLLM 模型的可读字符串说明，以类似 Transformers 的树形结构展示模型由哪些层和模块组成；调用 `repr(llm)`、`print(llm)` 或在交互环境中查看 `llm` 时可能触发它。
**输出（改变了什么值）**：返回一个表示模型结构的 `str`；第一次调用时会通过 `collective_rpc` 获取所有 worker 的检查结果，并把第一个结果保存到 `_cached_repr`，之后直接返回缓存。它只改变缓存，不会修改模型结构、模型权重或推理状态。
**细节补充**：
1. `__repr__`：Python 的特殊方法，用于生成一个对象的字符串表示。调用 `repr(llm)`、使用 `f"{llm!r}"`，或者在交互式环境中直接查看对象时，Python 会调用这个方法。
2. 树形结构：大模型由很多嵌套模块组成，例如模型中包含多层 Transformer，每一层又可能包含注意力层、归一化层和前馈网络。返回的字符串会按照这种父子关系逐层展示。
3. `_cached_repr`：专门缓存模型结构字符串的成员变量。它在 `LLM` 初始化时为 `None`，表示还没有获取过模型结构。
# day4
## offline_utils.py
**路径**：/home/dongmingzhe/vllm/vllm/entrypoints/offline_utils.py
**简介**：负责组织 vLLM 的离线推理流程，将用户输入转换并加入 `LLMEngine`，再循环驱动引擎执行，直到收集并返回全部推理结果。
**本小节阅读的类或函数**：
* `Class OfflineInferenceMixin（部分）`
### OfflineInferenceMixin类
**作用**：`OfflineInferenceMixin` 为 `LLM` 类提供可复用的离线推理功能，负责预处理普通提示词和聊天消息，并整理采样参数、LoRA、多模态输入及请求优先级。  
它还会把请求加入 `LLMEngine`，循环调用引擎完成推理，最后收集、排序并返回所有请求的最终结果。
#### run completion
```python
def _run_completion(
	self,
	prompts: PromptType | Sequence[PromptType],
	params: SamplingParams
	| PoolingParams
	| Sequence[SamplingParams | PoolingParams],
	output_type: type[_O],
	*,
	use_tqdm: bool | Callable[..., tqdm] = True,
	lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
	priority: list[int] | None = None,
	tokenization_kwargs: dict[str, Any] | None = None,
	mm_processor_kwargs: dict[str, Any] | None = None,
):
	self._add_completion_requests(
		prompts=prompts,
		params=params,
		use_tqdm=use_tqdm,
		lora_request=lora_request,
		priority=priority,
		tokenization_kwargs=tokenization_kwargs,
		mm_processor_kwargs=mm_processor_kwargs,
	)
	return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)
```
**主要作用**：组织一次完整的离线 completion 推理：先把一个或多个 `prompts` 预处理并加入 `LLMEngine`，再循环驱动引擎，直到所有请求完成并返回结果。
**输入（参数）**：主要输入是提示词 `prompts`、推理参数 `params` 和期望的结果类型 `output_type`；还可以设置进度条 `use_tqdm`、LoRA 适配器 `lora_request`、请求优先级 `priority`、分词参数 `tokenization_kwargs` 以及多模态处理参数 `mm_processor_kwargs`。
**输出（改变了什么值）**：返回一个按请求编号排序的最终结果列表；执行过程中会增加请求编号计数器、把请求加入并运行 `LLMEngine`，还会把生成任务的 `SamplingParams.output_kind` 设置为 `FINAL_ONLY`，但不会修改模型权重。
**细节补充**：
1. 完整流程：该函数本身没有实现分词、调度或模型计算，而是把流程分成“添加请求”和“运行引擎”两个阶段。
2. `_add_completion_requests()`：负责整理输入数量、推理参数、LoRA 和优先级，然后预处理每个 prompt，并将请求加入 `LLMEngine`。
3. `_run_engine()`：持续检查引擎中是否还有未完成请求，并反复调用 `llm_engine.step()`。所有请求结束后，它会收集并返回最终结果。
#### add request
```python
def _add_request(
	self,
	prompt: EngineInput,
	params: SamplingParams | PoolingParams,
	lora_request: LoRARequest | None = None,
	priority: int = 0,
) -> str:
	if isinstance(params, SamplingParams):
		# We only care about the final output
		params.output_kind = RequestOutputKind.FINAL_ONLY

	request_id = str(next(self.request_counter))

	return self.llm_engine.add_request(
		request_id,
		prompt,
		params,
		lora_request=lora_request,
		priority=priority,
	)
```
**主要作用**：为一个已经预处理好的输入创建请求编号，将离线生成任务设置为“只返回最终结果”，然后把该请求加入 `LLMEngine` 的等待队列；这个函数只负责入队，不会立即执行模型推理。
**输入（参数）**：
- `prompt`：已经预处理成 `EngineInput` 的模型输入。
- `params`：生成参数 `SamplingParams` 或池化参数 `PoolingParams`。
- `lora_request`：可选的 LoRA 适配器请求，默认不使用。
- `priority`：请求优先级，默认是 `0`。
**输出（改变了什么值）**：返回 `LLMEngine` 实际登记的请求 ID 字符串；执行时会递增 `request_counter`，可能把 `SamplingParams.output_kind` 修改为 `FINAL_ONLY`，并在引擎中新增一个等待处理的请求，但不会修改模型权重。
**细节补充**：
1. 外部请求 ID：当前函数生成的简单编号，例如 `"0"`。最终返回给用户的 `RequestOutput.request_id` 通常使用这个外部编号。
2. 内部请求 ID：`LLMEngine.add_request()` 会为外部编号附加随机字符，例如 `"0-a1b2c3d4"`，降低不同请求发生 ID 冲突的风险。
3. 当前函数的返回 ID：因为它直接返回 `LLMEngine.add_request()` 的结果，所以通常得到引擎实际登记的内部请求 ID，而不是最初那个简单编号。
**定位**：
```text
用户调用 LLM.generate()
        ↓
OfflineInferenceMixin._run_completion()
        ↓
_add_completion_requests()
整理 prompts、params、LoRA 和 priority
        ↓
_preprocess_cmpl_one()
把原始 prompt 渲染为 EngineInput
        ↓
_render_and_add_requests()
逐个处理批量输入
        ↓
_add_request()                    ← 当前函数
设置 FINAL_ONLY、生成编号并提交请求
        ↓
LLMEngine.add_request()
处理输入并登记前端输出状态
        ↓
InputProcessor.process_inputs()
生成 EngineCoreRequest
        ↓
InputProcessor.assign_request_id()
生成带随机后缀的内部请求 ID
        ↓
OutputProcessor.add_request()
登记用于接收输出的 RequestState
        ↓
EngineCoreClient.add_request()
        ↓
EngineCore.add_request()
        ↓
Scheduler.add_request()
请求进入 waiting 队列
```
请求全部添加完成后，执行链路转向：
```text
OfflineInferenceMixin._run_engine()
        ↓
while 引擎中还有未完成请求
        ↓
LLMEngine.step()
        ↓
EngineCore：调度 → 模型执行 → 状态回写
        ↓
OutputProcessor：反分词并组装 RequestOutput
        ↓
_run_engine() 收集并排序最终结果
```
因此，`_add_request()` 位于“输入准备”和“真正运行引擎”之间，是单个离线请求进入 vLLM 核心引擎的入口：上游把 prompt 和参数准备好，下游负责调度、模型计算、采样和输出处理。
#### run engine
```python
def _run_engine(
	self,
	output_type: type[_O] | tuple[type[_O], ...],
	*,
	use_tqdm: bool | Callable[..., tqdm] = True,
) -> list[_O]:
	# Initialize tqdm.
	if use_tqdm:
		num_requests = self.llm_engine.get_num_unfinished_requests()
		tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
		pbar = tqdm_func(
			total=num_requests,
			desc="Processed prompts",
			dynamic_ncols=True,
			postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
		)
	# Run the engine.
	outputs: list[_O] = []
	total_in_toks = 0
	total_out_toks = 0
	while self.llm_engine.has_unfinished_requests():
		step_outputs = self.llm_engine.step()
		for output in step_outputs:
			assert isinstance(output, output_type)
			if output.finished:
				outputs.append(output)  # type: ignore[arg-type]
				if use_tqdm:
					if isinstance(output, RequestOutput):
						# Calculate tokens only for RequestOutput
						n = len(output.outputs)
						assert output.prompt_token_ids is not None
						total_in_toks += len(output.prompt_token_ids) * n
						in_spd = total_in_toks / pbar.format_dict["elapsed"]
						total_out_toks += sum(
							len(stp.token_ids) for stp in output.outputs
						)
						out_spd = total_out_toks / pbar.format_dict["elapsed"]
						pbar.postfix = (
							f"est. speed input: {in_spd:.2f} toks/s, "
							f"output: {out_spd:.2f} toks/s"
						)
						pbar.update(n)
					else:
						pbar.update(1)
					if pbar.n == num_requests:
						pbar.refresh()
	if use_tqdm:
		pbar.close()
	# Sort the outputs by request ID.
	# This is necessary because some requests may be finished earlier than
	# its previous requests.
	return sorted(outputs, key=lambda x: int(x.request_id))
```
**主要作用**：这是 vLLM 离线推理的核心驱动循环，它不断调用 `LLMEngine.step()` 推进连续批处理，直到所有请求完成，同时可显示处理进度和 token 速度。
**输入（参数）**：
- `output_type`：允许返回的输出类型，可以是一个类型，也可以是多个类型组成的元组。
- `use_tqdm`：是否显示进度条；也可以传入自定义的进度条函数，默认显示。
**输出（改变了什么值）**：返回 `list[_O]`，即按照数字请求 ID 排序的全部最终结果；执行过程中会不断改变请求的调度和生成状态，并更新进度条及局部 token 统计值，但不会修改模型权重。
**细节补充**：
1. 核心驱动循环：引擎中已经存在待处理请求，但请求不会自己完成，需要外层代码反复调用 `step()` 推动它们向前运行。这里的 `while` 循环就承担这个作用。
2. 离线推理：一次提交一个或多个请求，然后当前函数一直等待，直到所有请求结束后统一返回结果。
3. 连续批处理：每次 `step()` 都会重新选择本轮要处理的请求。已经完成的请求可以退出，新请求可以加入，长短不同的请求不必互相等待。
4. `num_requests`：调用循环开始时，引擎中尚未完成的请求数量，用作进度条的总数。
5. `LLMEngine.step()`：推动整个引擎完成一轮“调度、模型计算、结果处理”。这是循环中最重要的一行。
6. `RequestOutput`：文本生成任务的结果，通常包含原始 prompt、生成文本、生成 token ID 和结束原因。
**定位**：
`_run_engine()` 位于离线 API 的最外层驱动位置，上游负责准备和添加请求，它负责反复驱动引擎，下游负责调度、模型计算和输出处理。
```text
用户调用 LLM.generate()
        ↓
OfflineInferenceMixin._run_completion()
        ↓
_add_completion_requests()
预处理并加入全部生成请求
        ↓
_run_engine()                         ← 当前函数
while 还有未完成请求：
    调用 LLMEngine.step()
        ↓
LLMEngine.step()
        ├─ 从 EngineCore 获取本轮结果
        ├─ OutputProcessor 处理输出
        ├─ 处理停止字符串产生的结束请求
        └─ 记录统计信息
        ↓
EngineCore.step()
        ├─ Scheduler.schedule()
        │  选择本轮请求并分配 KV Cache
        ├─ ModelExecutor.execute_model()
        │  执行模型前向计算和采样
        └─ Scheduler.update_from_output()
           写回新 token 并检查请求是否结束
        ↓
OutputProcessor.process_outputs()
反分词并生成 RequestOutput
        ↓
_run_engine() 收集 finished 的结果
        ↓
全部请求完成后排序并返回
```
它有几个主要上游入口：
```text
_run_completion()
普通生成或 pooling 请求
        └─ 调用 _run_engine()

_run_chat()
聊天请求完成模板处理和入队
        └─ 调用 _run_engine()

_render_and_run_requests()
输入已经由其他流程提供
        └─ 调用 _run_engine()

LLM.wait_for_completion()
先通过 enqueue() 添加请求，之后等待完成
        └─ 调用 _run_engine()
```
因此，`_run_engine()` 可以看作离线推理的“发动机开关和循环控制器”：它自己不实现 Transformer、PagedAttention 或采样算法，而是持续调用 `LLMEngine.step()`，让底层的调度器、KV Cache 管理器、模型执行器和输出处理器共同完成所有请求。
## day4-day5阶段性自检
**一、`LLM.generate()` 里，参数校验、默认采样参数、转发——哪个是它真正做的事？它自己有没有创建任何 GPU 资源？**
`LLM.generate()` 三件事都会做，但它最核心的职责是“转发”：先检查模型是否支持生成任务，在用户没有提供 `SamplingParams` 时补上默认值，然后把请求交给 `_run_completion()`。
它自己不会创建任何 GPU 资源，也不直接执行模型计算；模型权重和 KV Cache 等 GPU 资源主要在 `LLM.__init__()` 创建 `LLMEngine` 时加载或分配，真正的调度与计算则由后续的 `LLMEngine`、`EngineCore` 和模型执行器完成。

**二、为什么 `_run_engine` 的返回值要 `sorted(key=lambda x: int(x.request_id))`？如果不排序会发生什么？**
因为 vLLM 使用连续批处理，不同请求的输入长度和生成长度不同，所以完成顺序不一定等于提交顺序。`sorted(key=lambda x: int(x.request_id))` 会按照离线模式生成的数字请求 ID（`"0"`、`"1"`、`"2"`……）重新排序，让返回结果与用户传入 prompts 的顺序一致。
```text
提交顺序：请求 0、请求 1、请求 2
完成顺序：请求 1、请求 2、请求 0
排序之后：请求 0、请求 1、请求 2
```

**三、`_run_completion` 里"先全部 add_request 再 run_engine"这个顺序，对 GPU 利用率有什么影响？如果改成"加一个跑一个"会怎样？**
“先全部 `add_request` 再 `run_engine`”能让调度器在第一轮调度时就看到所有请求，从中选出多个请求组成较大的 batch，交给 GPU 并行处理，因此通常能提高 GPU 利用率和整体吞吐量。
在 prefill 阶段，一轮可以批量处理多个请求的输入 token；进入 decode 阶段后，一轮通常可以同时为多个请求生成下一个 token。某些请求提前完成后，调度器还会重新组合剩余请求，这就是连续批处理。
如果改成“添加一个请求，把它完全跑完，再添加下一个”：
会产生以下影响：
- GPU 每次只能处理一个请求，batch 较小，许多计算单元可能处于空闲状态。
- 无法利用不同请求之间的并行计算。
- kernel 启动、调度等固定开销需要为每个请求重复承担。
- 整体吞吐量通常明显下降，完成全部请求所需的总时间更长。
- 第一个请求可能更早开始并返回，但这是用整体吞吐量换取单个请求的更低等待时间。
需要注意，全部 `add_request` 只是让请求进入等待队列，并不代表马上为所有请求分配完整的 KV Cache。调度器仍会根据 token 预算、KV Cache 空间和 `max_num_seqs` 等限制，决定每一轮实际让哪些请求进入 GPU。

**四、分配KV Cache发生在什么时候？**
有两种不同层面的“分配 KV Cache”，需要区分：
phase1.`LLM.__init__()` 阶段：预先申请整块 KV Cache 显存池
```text
加载模型
  ↓
profiling 测量可用显存
  ↓
计算可以创建多少个 KV Cache block
  ↓
在 GPU 上创建 KV Cache 张量
```
这一步相当于提前建好一个“公共仓库”。显存已经被 vLLM 预留，但还没有指定其中哪些 block 属于哪个请求。
严格来说，不是 `LLM.__init__()` 自己直接分配，而是它在创建引擎的过程中触发底层 worker 完成分配。

phase2.`Scheduler.schedule()` 阶段：给具体请求分配 KV Cache block
当请求已经入队并被调度器选中执行时，会调用：
```text
Scheduler.schedule()
        ↓
KVCacheManager.allocate_slots()
        ↓
BlockPool.get_new_blocks()
```
这一步从启动时创建的公共 KV Cache 显存池中，挑选若干空闲 block 分配给具体请求，并记录在该请求的 `block_table` 中。
因此，`add_request()` 本身一般只是让请求进入等待队列，并不会立即给它分配 KV Cache block。只有请求真正被调度执行时，才会获得 block。

phase3.模型 forward 阶段：真正把 K、V 数据写进去
请求得到 block 之后，模型开始计算：
```text
GPUModelRunner.execute_model()
        ↓
模型 forward
        ↓
Attention
        ↓
把新 token 的 Key、Value 写入已分配的 KV Cache block
```
这时 KV Cache 才真正保存某个请求的 K、V 数据。

phase4.请求结束时：归还 block，但通常不释放整个显存池
请求生成结束后：
```text
Scheduler.update_from_output()
        ↓
发现请求 finished
        ↓
KVCacheManager.free()
        ↓
block 回到 BlockPool
```
这里是把 block 归还给 vLLM 的空闲块池，供其他请求复用；通常不是把这部分 GPU 显存直接还给操作系统或 CUDA。

**五、`output_kind = FINAL_ONLY` 和流式输出（`CUMULATIVE` / `DELTA`）的区别是什么？离线模式为什么可以只要 FINAL_ONLY？**
三者的区别在于：每次引擎产生可返回结果时，`RequestOutput` 中应该包含多少内容。
假设模型依次生成“你”“好”“呀”：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260924110420.png)
- `CUMULATIVE`：每次都返回从开始到当前为止的完整结果。客户端使用方便，但前面的内容会被重复传输。
- `DELTA`：每次只返回相对于上次新增的部分。重复数据少，更适合网络流式输出，但客户端需要自己把片段拼接起来。
- `FINAL_ONLY`：中间过程不向用户构造和返回结果，只在请求结束后返回一次完整结果。
离线模式之所以使用 `FINAL_ONLY`，是因为 `LLM.generate()` 是阻塞式调用：
程序本来就要等所有请求结束后才能拿到 `outputs`，调用者无法在生成过程中消费中间片段。因此，构造 `CUMULATIVE` 或 `DELTA` 中间结果没有实际价值，反而会增加对象创建、文本复制和输出处理的开销。
需要注意，`FINAL_ONLY` 只控制“什么时候把结果返回给用户”，不会改变模型内部的逐步生成过程：
```text
第 1 轮：生成 token 1，不返回给用户
第 2 轮：生成 token 2，不返回给用户
第 3 轮：生成 token 3，请求结束
         ↓
一次性返回完整结果
```
vLLM 仍然会反复执行 `LLMEngine.step()`，逐步调度、计算、采样和检查停止条件；它不会因为设置了 `FINAL_ONLY`，就一次 forward 生成全部 token。
在线聊天服务则不同：用户希望像 ChatGPT 一样实时看到文字出现，所以通常使用：
- `DELTA`：每次向客户端发送新增内容，最适合流式传输。
- `CUMULATIVE`：每次发送当前完整内容，客户端不必自己拼接，但重复传输更多。

**六、一次`LLMEngine.step()` 是阻塞完成一批请求还是一批token生成？**
一次 `LLMEngine.step()` 是阻塞完成“一轮批量 token 计算”，不是阻塞到一批请求全部生成完毕。
更准确地说，一轮 `step()` 会：
```text
从多个请求中选择本轮要计算的 token
        ↓
把这些 token 组成一个 batch
        ↓
等待 GPU 完成本轮模型计算和采样
        ↓
更新各请求的状态
        ↓
返回本轮产生的输出
```
例如当前有三个请求：
```text
请求 A：处于 decode，本轮生成 1 个 token
请求 B：处于 decode，本轮生成 1 个 token
请求 C：处于 prefill，本轮处理 100 个 prompt token
```
调度器可能把它们放进同一次 `step()`：
```text
一次 step：
A 的 1 个 token
+ B 的 1 个 token
+ C 的 100 个 token
= 本轮批量处理 102 个 token
```
调用者会阻塞等待这 102 个 token 的本轮计算完成，但不会等待 A、B、C 全部结束。下一轮还需要再次调用 `step()`：
```text
while llm_engine.has_unfinished_requests():
    step_outputs = llm_engine.step()
```
不同阶段的情况是：
- prefill 阶段：一个请求在一轮中可以处理多个输入 token。
- 普通 decode 阶段：一轮通常为每个被选中的请求生成 1 个 token。
- 投机解码：一个请求一轮可能接受并生成多个 token。
- 某些请求可能恰好在本轮遇到 EOS 或达到 `max_tokens`，因此会在本轮完成。
- 没完成的请求会继续参加后续 `step()`。

**七、`request_id` 为什么是 `"0"`、`"1"` 这样的数字字符串，而不是 UUID？**
UID（通用唯一标识符）是一种几乎不会重复的 128 位编号，例如：
```
550e8400-e29b-41d4-a716-446655440000
```

离线模式中，请求都来自同一个 `LLM` 实例，因此递增计数器 `"0"`、`"1"` 已能保证不重复，而且更短、更直观，还能反映输入顺序。

当前源码内部还会追加随机后缀，例如 `"0-a1b2c3d4"`，进一步防止 ID 冲突；最终返回给用户的仍是 `"0"`。

**八、LLM类的 `runner` 参数支持 `"auto"`，说它会"自动识别模型类型"选出对应的 runner，但紧接着又强调"一个 vLLM 实例只能同时跑一种 runner"。这两句话矛盾吗？如果加载的是一个 embedding 模型、不显式写 `runner`，然后直接调 `llm.generate(...)`，会在哪一步、以什么形式失败？**
不矛盾——`auto` 是启动时选一次（选种），"只能跑一种"说的是选完之后不能中途切换（不能改）。一个是"怎么定"，一个是"定了能不能变"。
失败过程：`auto` 识别出是 embedding 模型 → 定成 `runner_type = "pooling"` → 你调 `generate()` 时，函数体第一行的守门员就拦下：

```python
runner_type = self.model_config.runner_type
if runner_type != "generate":
    raise ValueError("LLM.generate() is only supported for generative models...")
```

失败时机：调用 `generate()` 的瞬间，还没进 `_run_completion`、没分词、没入队、没碰引擎。**形式**：`ValueError`，报错信息提示你改用 `--runner generate`。

关键点：`runner_type` 是在 `LLMEngine.from_engine_args()` 建引擎时就已经定死的，所以这个错误不需要等推理，入口处立刻就能报出来——这正是把守门员放在最外层而非引擎里的原因。

**九、 `cpu_offload_gb` 和 `offload_group_size` / `offload_num_in_group` / `offload_prefetch_step` / `offload_params` 这两组参数看起来都在"把权重挪到 CPU 内存"，它们解决的是同一个问题吗？请说明两者的机制差异、以及各自要付出的代价——为什么后者要引入"分组"和"预取"这些概念？**
不是同一个问题：一个管"扩容量"，一个管"省容量的同时不拖慢速度"。
为什么后者要引入"分组"和"预取"：
- 预取：Transformer 是逐层顺序执行的。算第 N 层时，第 N+`prefetch_step` 层的权重就可以异步从 CPU 搬到 GPU——等真的算到那层，数据已经到了，搬运延迟被计算时间掩盖。这是它比 `cpu_offload_gb` 快的核心原因。
- `offload_params`则是进一步取舍：只卸 MLP / MoE 专家这类大块权重，把有限显存留给更影响速度的部分。
一句话：`cpu_offload_gb` 是"显存不够就粗放地扩到 CPU，慢就慢"；`offload_*` 是"精细决定卸哪些、并提前搬运来掩盖延迟"——代价是调参复杂、而且要额外占一点显存做预取。
为什么要分组？（或者说为什么不直接这样：假设总共10层，卸载后4层，然后预取长度为2的话，就是计算第5层的时候，传输第7层）
分组的目的不是"少搬几次"，而是让预取调度变成编译期可静态推导的。
核心：静态缓冲池 + CUDA Graph
这个 offloader 的整个设计目标写在文件开头（[prefetch.py:1-8](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/model_executor/offloader/prefetch.py#L1-L8)）：
> Uses **static buffers** and event-based stream forking for **torch.compile + CUDA graph compatibility**.

要能被 CUDA Graph 捕获，所有显存地址必须在编译期固定，运行时不能动态分配、不能有依赖数据的分支。于是预取缓冲是一次性预分配的静态槽位池：
```python
self.buffer_pool = StaticBufferPool(
    param_infos=param_infos,
    slot_capacity=self.prefetch_step,     # ← 槽位数 == 预取步长
    device=device,
)
for idx, offloader in enumerate(self.module_offloaders):
    slot_idx = idx % self.prefetch_step    # ← 第 idx 个卸载层固定用第 idx%step 个槽
    offloader.assign_buffer_slot(self.buffer_pool, slot_idx)
```

以及 `_get_next_prefetch_index()` 的注释（[prefetch.py:131](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/model_executor/offloader/prefetch.py#L131)）：

> Return a refill target that **preserves static-buffer slot ownership**.

"槽位归属必须在整个运行期间保持不变"—— 这是硬约束。所以：
分组做了两件事
① 把"卸载哪些层"压缩成一条周期规则
```python
if module_index % self.group_size >= self.group_size - self.num_in_group:
    # 卸载这层
```
`group_size=8, num_in_group=2` → 卸载第 6,7,14,15,22,23,... 层（[offload.py:59-63](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/offload.py#L59-L63)）。这是一个周期为 8 的规则模式，任何时刻的结构都是同构的，槽位 `idx % prefetch_step` 的循环因此可以静态推导、永不冲突。
② 把调参空间从"组合数"降到两个整数
如果允许任意指定层号，参数就是一张列表（如 `[3, 7, 8, 9]`）。80 层的模型有 `C(80, K)` 种选法——你不可能手工调。而 `group_size` + `num_in_group` 两个数就能自动铺满任意层数的模型。
回到你的反例
你的方案（10 层卸后 4 层、预取 2）单独看是能跑的，问题在两点：
1. 它不是一个可推广的规则。换成 80 层模型，你得重新手工决定卸哪几层——这就是"任意集合"方案的根本麻烦。
2. 它的预取节奏无法写成固定步进公式。代码里的 `_get_next_prefetch_index` 用的是 `(index + prefetch_step) % module_count`，即按 module 序号固定步进——这只有在卸载模式是周期规则时才成立。任意集合下，连续两个卸载层的间距忽大忽小，就得做运行时判断，静态性一破，CUDA Graph 就进不去了。
所以：分组不是"为了少搬运"，而是把一个自由组合的调度问题，降维成一条周期规则——代价是牺牲选择自由度，换来的是编译期可静态化。
