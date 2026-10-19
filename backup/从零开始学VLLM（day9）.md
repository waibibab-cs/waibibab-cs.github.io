本文基于的vllm commit hash为：`c16bb6068f70878fb8a2f7c4d6cda95cd03a778b`
# day9-Scheduler(一)
调度器在**每一个推理步**回答一个问题：
**"这一步（一次 GPU 前向）应该让哪些请求参与，每个请求算多少个 token？"**
这个问题的答案决定了：
- GPU 利用率（batch 越大越划算）
- 延迟（长 prompt 会不会把短请求堵住）
- 显存占用（KV Cache 块够不够分）
其中包含了三条核心机制：
1. **连续批处理（Continuous Batching）**
传统静态批处理的问题：一批请求必须同生共死。
5 个请求里 4 个都生成完了，第 5 个还要生成 500 个 token，
GPU 就得陪跑 500 步，大部分算力浪费在 padding 上。
连续批处理解决方式：每一步都重新组 batch。
```
静态批处理（浪费）：
步 1: [A B C D E]   ← A~D 在第 3 步就结束了
步 2: [A B C D E]
步 3: [A B C D E]
步 4: [E _ _ _ _]   ← 只有 E 还在生成，其余位置空转 😱
步 5: [E _ _ _ _]
...

步 500: [E _ _ _ _]
连续批处理（vLLM）：
步 1: [A B C D E]
步 2: [A B C D E]
步 3: [A B C D E]   ← A~D 结束，立刻腾出位置
步 4: [E F G H I]   ← ★ waiting 队列里的新请求马上补进来
步 5: [E F G H I]
```
在代码里的体现：`Scheduler.schedule()` 分两个阶段——
* 先遍历 `running` 队列（已经在跑的）
* 再从 `waiting` 队列提拔新请求上来填补预算

2. **分块预填充（Chunked Prefill）**
问题：一个 8000 token 的长 prompt，prefill 一次算完会占满一步的全部预算，导致同一时刻的 decode 请求全部卡住，延迟飙升。
解决：把长 prompt 切成几块，分几步算。
```python
# V1 默认 enable_chunked_prefill=True
# 一个 8000 token 的 prompt，token_budget=2048 时：
步 1: 算 prompt[0:2048]      + 其他请求的 decode
步 2: 算 prompt[2048:4096]   + 其他请求的 decode
步 3: 算 prompt[4096:6144]   + 其他请求的 decode
步 4: 算 prompt[6144:8000]   → prefill 完成，进入 decode
步 5: 生成第 1 个 token
```
这样长 prompt 和短请求可以混在同一批里，互相不阻塞。

3. **抢占（Preemption）**
问题：显存不够了。running 里的请求越来越多，KV Cache 块不够分。
解决：把某个"受害者"请求踢出去
```python
# ★ 重要：V1 只有 RECOMPUTE（重算）一种抢占，没有 SWAP（换出到 CPU）
#   V0 有 SWAP，V1 删掉了。老教程讲 SWAP 的，在这个版本里是错的。
_preempt_request(request):
    1. 释放它所有的 KV Cache block
    2. num_computed_tokens = 0        # ← 清零！之前的计算白算了
    3. status = PREEMPTED
    4. 塞回 waiting 队列的**队头**（优先重新调度）
```
代价：重算。被抢占的请求下次要从头 prefill。
所以抢占是"不得已而为之"，vLLM 会尽量选影响最小的受害者。
> 💡 **这解释了为什么"显存不够"的报错在 vLLM 里不常见，但"跑得慢"很常见**：
> 显存不够时 vLLM 不会崩，而是默默抢占 + 重算，表现为吞吐下降。

## request.py
**路径**：/home/dongmingzhe/vllm/vllm/v1/request.py
**简介**：`request.py` 定义了 vLLM 核心调度器使用的“请求状态模型”：它把输入 token、生成参数、已生成 token、计算进度、KV Cache 信息和生命周期状态集中保存在一个对象中，供 Scheduler 在多轮推理中持续更新。主要包含以下几个类：
* `Request`：核心请求对象，保存输入 token、输出 token、采样/池化参数、优先级、多模态特征、已计算 token 数、抢占次数和缓存哈希等运行状态。**信息最完整，会在调度和生成过程中不断变化**。
* `StreamingUpdate`：可恢复流式会话的新一段输入，只保留追加 token、多模态特征、到达时间和生成长度等必要字段。**是对已有 `Request` 的轻量增量更新，不代表一个完整新请求**。
* `RequestStatus`：请求状态枚举，描述请求处于等待、运行、抢占或各种结束状态。**只表示生命周期状态，不保存请求内容**。
请求的主要状态流转是：
```
WAITING
   ↓
RUNNING
   ├─→ PREEMPTED → WAITING
   └─→ FINISHED_STOPPED / FINISHED_LENGTH_CAPPED
       / FINISHED_ABORTED / FINISHED_ERROR
```
此外，`_FINISHED_REASON_MAP` 负责把内部的 `RequestStatus` 转换成用户最终看到的 `stop`、`length`、`abort`、`error` 等结束原因。`Request` 是 EngineCore 和 Scheduler 内部使用的对象，不同于前端传入的 `EngineCoreRequest`，也不同于最终返回用户的 `RequestOutput`。

**本小节阅读的内容**：
* 看`Request`类
* 略看`RequestStatus`类
### Request类
**作用**：`Request` 是 vLLM Scheduler 内部使用的核心可变请求对象，它把一次推理所需的输入 token、采样或池化参数、LoRA、多模态信息、优先级与缓存哈希，以及运行过程中不断变化的请求状态、已计算 token 数、已生成 token 和抢占次数集中保存起来，使调度器能够跨多轮 `step()` 持续调度、更新并最终结束该请求。
#### 构造函数
##### part1.参数列表
```python
class Request:
    def __init__(
        self,
        # 请求的内部唯一编号，用于调度、查找、更新和取消请求
        request_id: str,
        # prompt 分词后的 token ID；使用纯 embedding 输入时可以是 None
        prompt_token_ids: list[int] | None,
        # 采样参数，例如 max_tokens、温度和停止条件
        sampling_params: SamplingParams | None,
        # embedding、分类等池化任务的参数；通常与 sampling_params 二选一
        pooling_params: PoolingParams | None,
        # 请求所属的前端客户端编号，多客户端模式下用于把结果送回正确客户端
        client_index: int = 0,
        # 请求到达引擎的时间；未提供时使用当前时间，可用于调度和性能统计
        arrival_time: float | None = None,
        # 已经计算好的 prompt 向量，可以代替或配合 token ID 输入
        prompt_embeds: torch.Tensor | None = None,
        # 混合输入的位置标记；每个布尔值表示该位置来自 token ID 还是 embedding
        prompt_is_token_ids: list[bool] | None = None,
        # 图片、视频等多模态输入经过预处理后的特征及位置信息
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        # 加入前缀缓存哈希的额外字符串，用于隔离本来不应共享缓存的请求
        cache_salt: str | None = None,
        # 请求的调度优先级；调度器可据此决定请求执行顺序
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        # 计算 token block 哈希的函数，用于判断请求之间能否复用前缀缓存
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        # 请求所属的会话编号，用于关联同一流式会话中的多段请求
        session_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        # 是否在请求加入 Scheduler 后立即取消，主要用于触发资源清理流程
        abort_immediately: bool = False,
    ) -> None:
```
需要重点记住的是：
```
request_id              标识请求
prompt_token_ids        输入内容
sampling/pooling_params 决定任务类型和行为
priority                影响调度顺序
block_hasher             服务于前缀缓存
resumable/session_id     服务于流式会话
```
##### part2.保存请求身份和任务配置
```python
self.request_id = request_id
self.client_index = client_index
self.priority = priority
self.sampling_params = sampling_params
self.pooling_params = pooling_params
self.lora_request = lora_request
self.structured_output_request = 
self.arrival_time = arrival_time if arrival_time is not None else time.time()
```
这一部分保存请求最基本的信息：
- 用 ID 区分请求。
- 用优先级和到达时间辅助调度。
- 保存生成、池化和 LoRA 参数。
- 记录请求来自哪个客户端。
##### part3.初始化结构化输出和请求状态
```python
self.structured_output_request = StructuredOutputRequest.from_sampling_params(
	sampling_params
)
if self.structured_output_request is not None:
	self.structured_output_request.reasoning_ended = reasoning_ended
	self.structured_output_request.reasoning_parser_kwargs = (
		reasoning_parser_kwargs
)
self.status = RequestStatus.WAITING
self.events: list[EngineCoreEvent] = []
self.stop_reason: int | str | None = None
```
如果生成参数要求 JSON、语法约束等结构化输出，就创建对应的约束对象。
普通请求初始状态是 `WAITING`；如果需要先编译结构化语法，状态会变成：
```
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```
##### part4.区分生成任务与池化任务
```python
if pooling_params is not None:
	# Pooling models.
	self.max_tokens = 1
elif sampling_params is not None:
	# Generative models.
	assert sampling_params.max_tokens is not None
	self.max_tokens = sampling_params.max_tokens
	if self.structured_output_request is not None:
		self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

	if sampling_params.extra_args is not None:
		self.kv_transfer_params = sampling_params.extra_args.get(
			"kv_transfer_params"
		)
		self.ec_transfer_params = sampling_params.extra_args.get(
			"ec_transfer_params"
		)
		self.kv_cache_report_mode = sampling_params.extra_args.get(
			"kv_cache_report_mode", "incremental"
		)
	else:
		self.kv_cache_report_mode = "incremental"
else:
	raise ValueError("sampling_params and pooling_params can't both be unset")
```
这里确定请求属于哪类任务：
- pooling 请求只需要一次模型结果，所以 `max_tokens = 1`。
- 生成请求使用 `SamplingParams.max_tokens`。
- 两种参数都没有时，请求无法执行，直接报错。
同时还会从生成参数中读取 KV Cache 和 encoder cache 的传输配置。
##### part5.初始化prompt和token状态
```python
self.prompt_token_ids = prompt_token_ids
self.prompt_embeds = prompt_embeds
# Per-position mask used in mixed-mode (chat completion with
# prompt_embeds). `None` except when both `prompt_token_ids` and
# `prompt_embeds` are set and their positions are interleaved.
self.prompt_is_token_ids = prompt_is_token_ids
# Cache per-block prompt-embed hashes to avoid rehashing the same
# tensor slices when generating extra keys.
self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
	prompt_token_ids, prompt_embeds
)
self._output_token_ids: list[int] = []
```
输入可以是：
- `prompt_token_ids`：分词后的 token ID。
- `prompt_embeds`：已经计算好的输入向量。
- 混合输入：部分位置使用 token，部分位置使用 embedding。
随后建立两个重要列表：
```
_output_token_ids：只保存模型新生成的 token
_all_token_ids：保存 prompt token 和生成 token
```
##### part6.初始化调度进度
```python
self.spec_token_ids = []
self.num_computed_tokens = 0
self.num_output_placeholders = 0
self.num_in_flight_tokens = 0
self.num_preemptions = 0
```
这些字段记录请求在多轮调度中的进度：
- `num_computed_tokens`：已经完成计算的 token 数。
- `spec_token_ids`：投机解码产生的候选 token。
- `num_in_flight_tokens`：已经提交计算但结果尚未处理的 token。
- `num_preemptions`：请求被调度器抢占的次数。
其他 stale token、下一次可解码步骤等字段，主要服务于异步调度和流水线并行，入门阶段知道它们用于防止状态错乱即可。
##### part7.保存多模态、只读视图和运行统计
```python
self.mm_features = mm_features or []

self.output_token_ids = ConstantList(self._output_token_ids)
self.all_token_ids = ConstantList(self._all_token_ids)

self.trace_headers = trace_headers
self.session_id = session_id
self.prefill_stats = PrefillStats()
self.spec_decode_metrics = None
```
这里保存：
- 图片等多模态特征。
- 请求追踪和会话信息。
- prefill 与投机解码统计。
`ConstantList` 提供只读视图，避免外部直接修改 token 列表，导致 `_output_token_ids` 和 `_all_token_ids` 不一致。
##### part8.初始化前缀缓存和流式请求状态
```python
self.block_hashes = []
self._block_hasher = block_hasher
self.update_block_hashes()

self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

self.resumable = resumable
self.streaming_queue = None
self.abort_immediately = abort_immediately
```
`block_hashes` 用于判断当前 prompt 的完整 block 是否与其他请求相同，从而支持前缀缓存复用。
流式相关字段表示：
- 请求是否允许追加输入或恢复。
- 后续输入更新存放在哪里。
- 请求加入 Scheduler 后是否需要立即取消。
整体初始化过程可以概括为：
```
保存请求身份
    ↓
确定生成或池化任务
    ↓
初始化请求状态
    ↓
整理 prompt、token 和 embedding
    ↓
初始化调度进度
    ↓
准备多模态和统计信息
    ↓
计算前缀缓存哈希
    ↓
准备流式请求状态
```

最需要掌握的核心字段是：
```
request_id             请求身份
status                 当前生命周期状态
prompt_token_ids       输入 token
output_token_ids       已生成 token
num_computed_tokens    已计算进度
max_tokens             最大生成量
block_hashes           前缀缓存依据
```
#### from engine core request函数
```python
@classmethod
def from_engine_core_request(
	cls,
	request: EngineCoreRequest,
	block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
	return cls(
		request_id=request.request_id,
		client_index=request.client_index,
		prompt_token_ids=request.prompt_token_ids,
		prompt_embeds=request.prompt_embeds,
		prompt_is_token_ids=request.prompt_is_token_ids,
		mm_features=request.mm_features,
		sampling_params=request.sampling_params,
		pooling_params=request.pooling_params,
		arrival_time=request.arrival_time,
		lora_request=request.lora_request,
		cache_salt=request.cache_salt,
		priority=request.priority,
		trace_headers=request.trace_headers,
		block_hasher=block_hasher,
		resumable=request.resumable,
		session_id=request.session_id,
		reasoning_ended=request.reasoning_ended,
		reasoning_parser_kwargs=request.reasoning_parser_kwargs,
		abort_immediately=request.abort_immediately,
	)
```
1. **主要作用**
`from_engine_core_request()` 把前端传来的 `EngineCoreRequest` 转换成 Scheduler 内部使用的 `Request`，并初始化请求状态、token 进度、结构化输出信息和前缀缓存哈希等运行数据。

2. **输入（参数）**
- `cls`：当前类，这里就是 `Request`，由 `@classmethod` 自动传入。
- `request`：待转换的 `EngineCoreRequest`，包含输入 token、采样参数、多模态特征和请求元数据。
- `block_hasher`：可选的 block 哈希计算函数，用于前缀缓存。

1. **输出（改变了什么值）**
返回一个新的 `Request` 对象，不修改原来的 `EngineCoreRequest`。新对象会复制或引用其中的请求数据，并额外初始化 `WAITING` 状态、已生成 token 列表、已计算 token 数和 block 哈希等内部运行状态；它还没有进入 Scheduler。

2. **术语解释**
- `@classmethod`：类方法装饰器。调用时 Python 自动传入类本身 `cls`，因此可以通过 `cls(...)` 创建对象，而不必把类名写死为 `Request(...)`。
- `cls`：当前类。在这里通常就是 `Request`，所以：
```
return cls(...)
```
相当于：
```
return Request(...)
```
- `EngineCoreRequest`：前端与 EngineCore 之间传递请求的数据对象，重点是描述“用户提交了什么”。
- `Request`：Scheduler 内部使用的可变状态对象，除了保存输入，还会记录“当前执行到哪里”。
- 转换：不是简单修改对象类型，而是读取 `EngineCoreRequest` 的字段，调用 `Request` 构造函数创建一个新的对象。
- `request_id`：请求的内部唯一编号。转换后保持不变，方便前端、EngineCore 和 Scheduler 对应同一个请求。
- `client_index`：请求所属的前端客户端编号，用于将结果发回正确的客户端。
- `prompt_token_ids`：prompt 分词后得到的 token ID。
- `prompt_embeds`：已经计算好的 prompt 向量，可用于 embedding 输入或 token 与 embedding 混合输入。
- `prompt_is_token_ids`：混合输入中每个位置的数据来源标记，用来区分 token ID 和 prompt embedding。
- `mm_features`：图片、视频等多模态内容经过处理后的特征。
- `arrival_time`：请求到达时间，Scheduler 可以结合它和优先级确定请求顺序。
- `lora_request`：当前请求需要使用的 LoRA 适配器。
- `cache_salt`：加入缓存哈希的额外内容，用于防止不应共享缓存的请求错误命中同一前缀。
- `priority`：请求优先级。
- `trace_headers`：链路追踪信息，用于统计请求经过各组件的时间。
- `resumable`：表示请求是否支持后续追加输入或恢复执行。
- `session_id`：流式请求所属的会话编号。
- `reasoning_ended`：记录 reasoning 阶段是否结束，供结构化输出逻辑使用。
- `reasoning_parser_kwargs`：reasoning 解析器的额外配置。
- `abort_immediately`：表示请求加入 Scheduler 后是否应立即取消，主要用于清理已经准备的传输资源。
- `block_hasher`：根据请求 token 计算完整 block 哈希的函数。
- `BlockHash`：某段 token block 的指纹。两个请求拥有相同 block 哈希时，可能可以复用同一份前缀 KV Cache。
- 前缀缓存：多个请求拥有相同 prompt 前缀时，复用已经计算好的 KV Cache，减少重复 prefill 计算。
- 新增的内部状态：调用 `Request` 构造函数后，会产生原 `EngineCoreRequest` 中没有的动态字段，例如：
```
status = WAITING
num_computed_tokens = 0
output_token_ids = []
num_preemptions = 0
block_hashes = [...]
```
- 还没有入队：返回 `Request` 只完成对象转换，真正加入 waiting 队列是在后续 `Scheduler.add_request()` 中。

5. 核心流程
```
接收 EngineCoreRequest
        ↓
读取请求 ID、输入、参数和元数据
        ↓
把 block_hasher 一起传给 Request 构造函数
        ↓
调用 cls(...)
        ↓
Request.__init__()
    ├─ 保存输入和参数
    ├─ 初始化状态与 token 进度
    ├─ 初始化多模态和流式信息
    └─ 计算初始 block 哈希
        ↓
返回新的 Request
```
该函数本质上是一组字段映射：
```
EngineCoreRequest.request_id
    → Request.request_id

EngineCoreRequest.prompt_token_ids
    → Request.prompt_token_ids

EngineCoreRequest.sampling_params
    → Request.sampling_params
```

6. **在整个项目中的定位**
它位于“请求跨过 EngineCore 边界”和“请求进入 Scheduler”之间：
```
用户调用 LLM.generate()
        ↓
InputProcessor.process_inputs()
        ↓
生成 EngineCoreRequest
        ↓
EngineCoreClient.add_request()
        ↓
EngineCore.preprocess_add_request()
        ↓
Request.from_engine_core_request()       ← 当前函数
EngineCoreRequest → Request
        ↓
初始化结构化输出语法
        ↓
EngineCore.add_request()
        ↓
Scheduler.add_request()
Request 进入 waiting 队列
        ↓
Scheduler.schedule()
持续读取和修改 Request 的运行状态
        ↓
ModelExecutor.execute_model()
```
因此，它是前端请求数据模型 `EngineCoreRequest` 与调度器运行状态模型 `Request` 之间的转换桥梁。
### RequestStatus类
```python
class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()
    def __str__(self) -> str:
        return self.name
    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED
    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)

# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
```
`RequestStatus` 是 vLLM 请求的“状态标签集合”。每个 `Request` 对象都会通过 `request.status` 保存当前状态，调度器根据它判断请求是在等待、运行、暂停，还是已经结束。
**1. 为什么继承 `enum.IntEnum`**
```
class RequestStatus(enum.IntEnum):
```
`IntEnum` 的成员既是枚举值，也可以像整数一样比较大小。这里使用 `IntEnum` 的关键原因是代码通过大小关系判断请求是否结束：
```
return status > RequestStatus.PREEMPTED
```
`enum.auto()` 会按照定义顺序自动赋值。因此这些状态大致对应：
```
WAITING                                   1
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR     2
WAITING_FOR_REMOTE_KVS                    3
WAITING_FOR_STREAMING_REQ                 4
RUNNING                                   5
PREEMPTED                                 6
---------------- 结束状态分界线 ----------------
FINISHED_STOPPED                          7
FINISHED_LENGTH_CAPPED                    8
...
```
这里“定义顺序”具有实际含义：`PREEMPTED` 后面的状态全部被视为真正结束。因此以后增加新状态时必须谨慎，不能随便放在 `PREEMPTED` 后面。
**2. 各个状态的含义**

|状态|是否结束|核心含义|
|---|---|---|
|`WAITING`|否|请求已经进入调度器，等待分配计算资源|
|`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`|否|等待结构化输出语法准备完成|
|`WAITING_FOR_REMOTE_KVS`|否|等待从远端加载 KV Cache|
|`WAITING_FOR_STREAMING_REQ`|否|当前一段流式输入已经处理完，等待用户继续输入|
|`RUNNING`|否|请求正在被调度和执行|
|`PREEMPTED`|否|请求因资源不足被抢占，之后还会重新排队运行|
|`FINISHED_STOPPED`|是|正常遇到停止条件|
|`FINISHED_LENGTH_CAPPED`|是|达到了长度上限|
|`FINISHED_ABORTED`|是|被用户或系统主动取消|
|`FINISHED_IGNORED`|是|请求无法处理而被忽略，通常是 prompt 过长|
|`FINISHED_ERROR`|是|执行过程中发生错误|
|`FINISHED_REPETITION`|是|检测到重复生成，因此提前停止|

下面具体介绍所有状态：
**`WAITING`**
这是普通排队状态。请求已经进入 `Scheduler`，但可能还没有获得 KV Cache block、token 预算或模型执行位置。
```
请求进入 Scheduler
        ↓
     WAITING
        ↓ 获得资源并被调度
     RUNNING
```

**`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`**
结构化输出要求模型按照指定格式生成，例如 JSON Schema、正则表达式或有限选项。vLLM 需要先把这些规则编译成生成时可使用的 grammar；编译完成前，请求不能运行。
```
创建带结构化输出的请求
        ↓
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        ↓ grammar 编译完成
     WAITING
        ↓
     RUNNING
```

**`WAITING_FOR_REMOTE_KVS`**
表示请求需要的部分 KV Cache 正在从远端传输过来，常见于 Prefill/Decode 分离或分布式 KV Cache 场景。
此时请求不是普通排队，而是被外部数据依赖阻塞：
```
等待远端 KV Cache
        ↓ 传输完成
WAITING 或 PREEMPTED
        ↓
     RUNNING
```

**`WAITING_FOR_STREAMING_REQ`**
它用于可恢复的流式输入会话。当前输入片段已经处理结束，但整个会话没有销毁，仍在等待用户提交下一段输入。
例如：
```
用户输入第一段
    ↓
模型生成完第一段回答
    ↓
WAITING_FOR_STREAMING_REQ
    ↓ 用户继续输入
WAITING → RUNNING
```
它不是内部的最终状态，所以：
```
RequestStatus.is_finished(
    RequestStatus.WAITING_FOR_STREAMING_REQ
)
```
结果是 `False`。
但它在 `_FINISHED_REASON_MAP` 中被映射为 `FinishReason.STOP`，表示当前这一轮输出已经正常停止，可以向前端交付完成信号；底层会话仍可保留，等待下一段输入。“当前轮输出结束”和“底层请求永久结束”在这里不是同一件事。

**`PREEMPTED`**
抢占表示请求原本正在运行，但由于 KV Cache 空间或调度资源不足，被暂时移出运行队列。
在当前实现中，抢占时通常会：
```
释放请求占用的 KV Cache block
        ↓
status = PREEMPTED
        ↓
num_computed_tokens 重置
        ↓
重新放回等待队列
        ↓
以后重新计算并继续执行
```
所以 `PREEMPTED` 不是失败，也不是结束，只是暂时让出资源。

**`FINISHED_STOPPED`**
表示正常遇到停止条件，例如：
- 生成了 EOS token；
- 生成了用户指定的 stop token；
- pooling 请求已经得到结果；
- 编码器任务完成。
对应前端原因：
```
FinishReason.STOP
```

**`FINISHED_LENGTH_CAPPED`**
表示生成长度达到了限制，常见的限制包括：
- `sampling_params.max_tokens`；
- 模型最大上下文长度 `max_model_len`。
对应：
```
FinishReason.LENGTH
```

**`FINISHED_ABORTED`**
请求被外部主动终止，例如用户取消请求、客户端断开连接或引擎关闭。
对应：
```
FinishReason.ABORT
```

**`FINISHED_IGNORED`**
表示请求没有真正进入正常执行流程，通常是输入 prompt 超出模型长度限制。虽然内部叫 `IGNORED`，但对外仍映射为：
```
FinishReason.LENGTH
```
这样可以和 OpenAI 风格 API 的结束原因保持一致。

**`FINISHED_ERROR`**
请求因模型执行、结构化输出或其他内部异常而结束，对应：
```
FinishReason.ERROR
```

**`FINISHED_REPETITION`**
启用重复检测后，如果输出出现符合配置条件的重复序列，请求会被提前终止，对应：
```
FinishReason.REPETITION
```

**3. 为什么需要两套状态**
`RequestStatus` 和 `FinishReason` 服务于不同层次：
```
RequestStatus
引擎内部使用，描述详细生命周期
例如 WAITING、RUNNING、PREEMPTED
              ↓ 映射
FinishReason
输出给前端，解释为什么本轮输出结束
例如 stop、length、abort、error
```
多个内部状态可以映射到同一个外部原因。例如：
```
FINISHED_LENGTH_CAPPED ─┐
                        ├─→ FinishReason.LENGTH
FINISHED_IGNORED ───────┘
```
这样调度器能够保留详细信息，而 API 用户只需要理解较少、较稳定的结束原因。

**4. 典型生命周期**
普通请求：
```
WAITING
   ↓ 被 Scheduler 选中
RUNNING
   ├─→ FINISHED_STOPPED
   ├─→ FINISHED_LENGTH_CAPPED
   ├─→ FINISHED_REPETITION
   ├─→ FINISHED_ERROR
   └─→ PREEMPTED
          ↓ 重新排队
       RUNNING
```
带结构化输出或远端 KV 的请求：
```
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
                    ↓
                 WAITING
                    ↓
WAITING_FOR_REMOTE_KVS
                    ↓
                 RUNNING
                    ↓
              某个结束状态
```
需要注意：这个类本身只定义状态和辅助判断，并不会主动推动状态变化。真正修改 `request.status` 的主要是 `Request` 初始化逻辑、`Scheduler` 的调度与输出处理逻辑，以及停止条件检查函数。
## scheduler.py（Scheduler类）
**路径**：/home/dongmingzhe/vllm/vllm/v1/core/sched/scheduler.py
**简介**：`scheduler.py` 是 vLLM V1 引擎的调度中心：它管理请求的等待、运行、抢占和结束状态，在每轮模型执行前决定“本轮让哪些请求计算多少个 token”，并在模型返回结果后更新请求状态、KV Cache 占用和最终输出。它负责组织计算，但不直接执行模型的神经网络运算。
当前文件实际上只直接定义了一个类：
- 管理 `waiting`、`running` 等请求队列。
- 根据 token 预算、请求优先级和 KV Cache 空间选择请求。
- 通过 `KVCacheManager` 给请求分配或释放逻辑 KV Cache block。
- 资源不足时抢占请求，稍后重新调度。
- 支持分块 Prefill、前缀缓存、推测解码和结构化输出。
- 支持远程 KV Cache、编码器缓存及多模态输入。
- 接收 `ModelRunnerOutput`，检查停止条件并生成 `EngineCoreOutputs`。
- 收集调度、缓存和推测解码等统计信息。
最核心的两个方法是：
- `schedule()`：模型运行前，生成本轮的 `SchedulerOutput`。
- `update_from_output()`：模型运行后，根据结果更新请求并生成输出。
下面展示了一些与Scheduler类有关的类，这些类虽然不是在本文件中定义，但与 `Scheduler` 紧密协作：

| 类                     | 核心作用                   | 与 `Scheduler` 的区别          |
| --------------------- | ---------------------- | -------------------------- |
| `SchedulerInterface`  | 规定调度器必须提供哪些方法          | 是抽象接口；`Scheduler` 是具体实现    |
| `Request`             | 保存单个请求的 token、参数、状态等   | 是被调度的数据，不负责调度              |
| `RequestQueue`        | 按 FCFS 或优先级保存请求        | 只负责请求排序，决策由 `Scheduler` 完成 |
| `KVCacheManager`      | 管理和分配逻辑 KV Cache block | 管缓存资源，不决定整批请求如何执行          |
| `EncoderCacheManager` | 管理图片等多模态编码结果的缓存        | 只处理编码器缓存                   |
| `SchedulerOutput`     | 描述本轮要执行哪些请求和多少 token   | 是调度结果，不执行模型                |
| `ModelRunnerOutput`   | 保存模型本轮生成的 token 等结果    | 来自模型执行端，交回调度器处理            |
| `EngineCoreOutputs`   | 按客户端整理的最终引擎输出          | 是调度器处理模型结果后产生的输出           |
整体调用关系：
```
EngineCore.add_request()
        ↓
Scheduler.add_request(Request)
        ↓
等待队列 / 运行队列
        ↓
Scheduler.schedule()
        ↓
SchedulerOutput
        ↓
ModelExecutor / ModelRunner 执行模型
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ├─ 更新 Request 状态
        ├─ 检查停止或抢占
        ├─ 分配、保留或释放缓存
        └─ 生成 EngineCoreOutputs
```

简而言之，`Scheduler` 是连接“请求管理、缓存管理和模型执行”的控制中心，而 `ModelExecutor/ModelRunner` 才是真正完成 GPU 模型计算的组件。
**本小节阅读的内容**：
* 重点看`Scheduler`类及相关的重要函数
### 构造函数
#### part1.参数列表
```python
def __init__(
    self,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    structured_output_manager: StructuredOutputManager,
    block_size: int,
    hash_block_size: int | None = None,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    include_finished_set: bool = False,
    log_stats: bool = False,
) -> None:
```
- `self`：当前创建的 `Scheduler` 对象，由 Python 自动传入。
- `vllm_config`：vLLM 的总配置，包含模型、调度、并行、缓存、LoRA、推测解码和监控等配置。
- `kv_cache_config`：已经根据 GPU 显存和模型结构计算好的 KV Cache 布局与容量配置。调度器据此管理缓存 block。
- `structured_output_manager`：结构化输出管理器，负责约束模型按照 JSON、正则表达式等指定格式生成内容。
- `block_size`：调度器管理 KV Cache 时，一个 block 对应的 token 数量。
- `hash_block_size`：计算前缀缓存哈希时使用的 token 分块大小；不传时默认等于 `block_size`。
- `mm_registry`：多模态模型注册表，用于判断模型是否支持图片、视频等输入，并获取对应的多模态处理能力。
- `include_finished_set`：是否额外按客户端记录已完成的请求 ID，主要用于多引擎场景。
- `log_stats`：是否收集和记录调度、缓存、性能等统计数据。
#### part2.保存基础配置和模型特征
```python
self.scheduler_config = vllm_config.scheduler_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
...
self.is_encoder_decoder = ...
self.is_mm_encoder_only = ...
```
这一部分从总配置中提取调度、缓存、并行和监控等子配置，同时记录模型是否为编码器—解码器模型、多模态编码器模型以及是否使用 MRoPE。
如果启用了统计功能，还会创建 `KVCacheMetricsCollector`，用于记录 KV Cache 的使用情况。
#### part3.设置调度容量限制
```python
self.max_num_running_reqs = ...
self.max_num_active_reqs = ...
self.max_num_scheduled_tokens = ...
self.max_model_len = ...
```
这一部分确定调度器每轮最多可以处理：
- 多少个运行请求；
- 多少个活跃请求；
- 多少个 token；
- 单个请求允许使用的最大模型上下文长度。
这些值是 `schedule()` 选择请求和组建批次时的核心限制。
#### part4.初始化远程缓存连接器
```python
self.connector = ...
self.ec_connector = ...
self.kv_event_publisher = ...
```
这一部分处理分布式缓存传输：
- `connector`：负责跨节点传输 KV Cache，例如 Prefill/Decode 分离。
- `ec_connector`：负责传输图片等编码器输入产生的缓存。
- `kv_event_publisher`：发布 KV Cache 分配、释放等事件。
如果没有配置远程缓存传输，这些连接器保持为空，不影响普通本地推理。
>远程缓存传输主要用于 Prefill/Decode 分离、跨节点推理、KV Cache 卸载或多个实例复用缓存等场景：例如一台机器完成 prompt 的 Prefill 并生成 KV Cache，另一台机器直接加载这些缓存继续 Decode，从而避免重复计算；多模态场景也可通过 `ec_connector` 传输图片编码器产生的特征缓存。实现时，Scheduler 侧的 Connector 负责判断哪些缓存需要加载或保存、分配本地逻辑 block并生成传输元数据，Worker 侧 Connector 根据元数据把缓存数据在本地 GPU、CPU或远端节点之间推送/拉取；异步加载期间请求进入 `WAITING_FOR_REMOTE_KVS`，传输完成后通知 Scheduler 恢复调度。`kv_event_publisher` 只负责发布缓存分配、释放和传输事件，便于外部缓存系统同步与监控，本身不传输缓存数据。
#### part5.创建请求容器和队列
```python
self.requests = {}
self.policy = SchedulingPolicy(...)
self.waiting = create_request_queue(...)
self.skipped_waiting = create_request_queue(...)
self.running = []
```
这是请求管理的核心部分：
- `requests`：通过 `request_id` 查找请求。
- `waiting`：保存可以参与后续调度的等待请求。
- `skipped_waiting`：保存暂时不能调度的请求，例如正在等待远程 KV。
- `running`：保存已经进入运行状态的请求。
- `policy`：决定等待请求的选择顺序，例如先来先服务或优先级调度。
随后创建的若干集合用于记录已完成、被抢占、缓存传输完成或语法编译失败的请求。
#### part6.初始化多模态编码器缓存
```python
supports_mm_inputs = ...
mm_budget = ...
self.encoder_cache_manager = ...
```
如果模型支持图片、视频或编码器—解码器输入，这一部分会计算编码器计算预算和缓存容量，并创建 `EncoderCacheManager`。
它管理的是图片特征、编码器输出等缓存，与文本自注意力使用的 KV Cache 不同。
#### part7.初始化推测解码配置
```python
self.num_spec_tokens = ...
self.num_lookahead_tokens = ...
self.dynamic_sd_lookup = ...
self.use_eagle = ...
```
这一部分读取推测解码配置，确定：
- 每轮最多生成多少个候选 token；
- 需要预留多少前瞻位置；
- 是否使用 EAGLE；
- 是否根据批次大小动态调整候选 token 数量。
这些配置会影响每轮 token 调度量以及 KV Cache 的预留空间。

**上述内容详细解释**
推测解码的核心思想是：先让一个更快的“草稿模块”一次预测多个候选 token，再由主模型在一次 forward 中并行验证这些候选 token。被主模型接受的 token 才会成为最终输出；从第一个被拒绝的候选开始，后面的候选通常都不能继续采用。
例如草稿模块预测：
```
候选 token：A → B → C → D
主模型验证：✓   ✓   ✗
最终接受：  A、B
```
这样主模型一次计算可能确认多个 token，而普通自回归解码一次通常只能前进一个 token。
1. `num_spec_tokens`
```
self.num_spec_tokens = vllm_config.num_speculative_tokens
```
它表示每轮最多使用多少个推测候选 token，也常记为 `K`。
例如：
```
num_spec_tokens = 4
```
表示草稿模块每轮最多提出 4 个候选 token，主模型随后批量验证它们。
需要注意：
- 这是候选数量，不是保证生成的最终 token 数量。
- 主模型可能接受 0～4 个候选。
- 普通自回归解码通常还可能在验证后得到一个额外的主模型 token，因此一轮涉及的位置数量可能接近 `1 + K`。
- 设置为 `0` 通常表示不使用推测解码。
数值越大并不一定越快：
```
K 较小
├─ 草稿计算少
├─ KV Cache 额外占用少
└─ 一次主模型调用前进的 token 可能较少
K 较大
├─ 有机会一次接受更多 token
├─ 草稿和验证开销更大
├─ 需要更多 KV Cache 位置
└─ 候选接受率低时可能浪费更多计算
```
因此最合适的 `K` 与模型、硬件、批次大小和候选接受率有关。

2. `num_lookahead_tokens`
```
self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
```
它表示在主模型本轮明确调度的 token 位置之外，还需要额外预留多少个 KV Cache 槽位。
草稿模块可能需要读取或写入比主模型当前查询范围更靠后的 token 位置，因此 Scheduler 在调用：
```
self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```
时，会为这些未来位置提前保留空间。
它和 `num_spec_tokens` 的区别是：
```
num_spec_tokens
    = 本轮最多提出多少个候选 token
    = 生成/验证层面的数量
num_lookahead_tokens
    = KV Cache 还要额外预留多少个未来位置
    = 缓存容量层面的数量
```
预留槽位不表示对应 token 已经生成或一定会被接受，只是保证后续草稿计算有安全的缓存空间。
当前配置的大致规则是：
```
没有推测解码
    num_lookahead_tokens = 0
EAGLE、MTP、DSpark 或普通 draft model
    num_lookahead_tokens = num_spec_tokens
DFlash
    num_lookahead_tokens = num_spec_tokens + 1
```
DFlash 多预留一个位置，是因为它使用类似 in-fill 的计算方式，需要额外的查询位置。

3. `dynamic_sd_lookup`
```
self.dynamic_sd_lookup: list[int] | None = None
```
它用于动态推测解码，即根据当前批次中的请求数量，动态决定这一轮使用多少个候选 token，而不是始终使用固定的 `num_spec_tokens`。
如果配置了：
```
num_speculative_tokens_per_batch_size
```
构造函数会生成一个按批次大小直接查询的数组：
```
self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(...)
```
例如配置逻辑可能是：
```
批次大小 1～4    → K = 6
批次大小 5～16   → K = 4
批次大小 17～64  → K = 2
```
运行时可以直接查询：
```
batch_size = len(num_scheduled_tokens)
K = self.dynamic_sd_lookup[batch_size]
```
通常采用这种策略的原因是：
- 批次小时，GPU 可能没有充分利用，可以使用更多候选 token，提高单个请求每轮前进的距离。
- 批次大时，主模型本身已经有较高的并行度，过多候选会扩大验证批次、增加 KV Cache 和显存压力，因此可以降低 `K`。
- 实际最优值还取决于接受率和硬件性能，并不是固定规律。
`dynamic_sd_lookup` 中的值不会超过全局 `num_spec_tokens`，后者仍然是候选数量的上限。
如果没有配置动态策略：
```
self.dynamic_sd_lookup = None
```
每轮都使用固定的：
```
num_spec_tokens_to_schedule = self.num_spec_tokens
```
需要注意，动态策略调整的是本轮实际候选数；缓存容量规划通常仍需要考虑全局最大候选数，避免某一轮增大 `K` 后空间不足。

4. `use_eagle`
```
self.use_eagle = speculative_config.use_eagle()
```
它表示当前是否使用 EAGLE 系列的推测解码方式。
普通 draft model 的基本形式是：
```
独立的小模型
    ↓
根据 token 预测未来 token
    ↓
主模型验证
```
EAGLE 类方法更偏向：
```
利用主模型产生的隐藏状态
    ↓
较轻量的草稿模块预测未来表示/token
    ↓
主模型批量验证候选 token
```
隐藏状态比单纯的 token ID 包含更多上下文信息，因此 EAGLE 类方法希望获得更高的候选接受率，同时让草稿部分保持较低成本。
在当前代码中，`use_eagle()` 不只代表最原始的 EAGLE，还被用作“利用主模型隐藏状态进行推测解码”的统一判断，包括：
```
eagle
eagle3
mtp
dflash
dspark
```
它会影响：
- KV Cache 需要预留多少未来位置；
- Prefill 边界需要保留多少前瞻 token；
- 前缀缓存命中后如何恢复草稿模型状态；
- 尾部 KV block 是否可以直接复用；
- 推测候选位置如何分配。

5. `use_eagle_block_drop`
虽然示例中没有单独列出，但构造函数同时设置了：
```
self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()
```
EAGLE 的尾部缓存 block 可能包含仍会变化的候选位置，这些位置不是稳定的已确认前缀。如果把它们当作普通前缀缓存复用，可能使用错误的草稿状态。
因此 block drop 会丢弃这种不稳定的尾部 block，只保留已经稳定确认的缓存部分：
```
稳定、已确认的前缀 block
    → 可以进入前缀缓存并复用

包含未确认候选的尾部 block
    → 丢弃或重新计算
```
这会牺牲少量缓存复用，但可以保证推测解码状态正确。关闭该功能属于实验性行为，也可能影响候选 token 的接受率。

6. 四者之间的关系
```
speculative_config
        ↓
use_eagle
决定采用哪类草稿生成方法
        ↓
num_spec_tokens
确定本轮最多提出多少个候选 token
        ↓
dynamic_sd_lookup
根据实际批次大小进一步选择本轮真正使用的 K
        ↓
num_lookahead_tokens
根据推测方法和最大候选数预留额外 KV Cache 槽位
        ↓
Scheduler.schedule()
安排候选验证所需的 token 和 KV block
        ↓
草稿模块生成候选
        ↓
主模型批量验证
        ↓
update_from_output()
统计接受/拒绝数量，并回滚被拒绝候选对应的计算进度
```
简单地说：
```
num_spec_tokens      决定“最多猜几个”
dynamic_sd_lookup    决定“这一轮实际猜几个”
use_eagle            决定“用什么类型的方法猜”
num_lookahead_tokens 决定“为了这些猜测要多留多少缓存位置”
```
#### part8.创建 KV Cache 管理器
```python
self.kv_cache_manager = KVCacheManager(...)
```
`KVCacheManager` 根据已经传入的 `kv_cache_config`，负责：
- 给请求分配和释放逻辑 KV Cache block；
- 管理前缀缓存；
- 判断缓存空间是否足够；
- 为推测解码和并行模式预留空间。
这里创建的是缓存的管理器，不是在重新创建 GPU 上的 KV Cache 张量池；实际张量池已在创建 `Scheduler` 之前由执行器初始化完成。
#### part9.初始化并行与模型状态
```python
self.use_pp = ...
self.current_step = 0
self.has_mamba_layers = ...
self.need_mamba_block_aligned_split = ...
```
这一部分保存流水线并行、异步调度以及 Mamba 模型需要的特殊状态。它们用于处理特殊的批次节奏、缓存对齐和前缀复用规则，普通 Transformer 模型通常不需要重点关注这些细节。
#### part10.初始化运行期辅助状态
```python
self.deferred_frees = deque()
self.perf_metrics = ...
self.routed_experts_mgr = ...
self._pause_state = PauseState.UNPAUSED
self._inflight_prefills = set()
```
最后一部分创建运行过程中需要持续更新的状态：
- `deferred_frees`：暂缓释放仍可能被异步任务使用的缓存块。
- `perf_metrics`：收集模型性能指标。
- `routed_experts_mgr`：记录 MoE 模型选择了哪些专家。
- `_pause_state`：记录调度器是否暂停。
- `_inflight_prefills`：记录尚未完成 Prefill 的请求。
### add request函数
```python
# scheduler.py:2547
def add_request(self, request: Request) -> None:
    request_id = request.request_id
    # ══════════════════════════════════════════════════════
    #  第一种情况：request_id 已存在
    #  说明这是同一个可恢复流式会话的后续输入
    #  L2548 - L2563
    # ══════════════════════════════════════════════════════
    existing = self.requests.get(request_id)
    if existing is not None:
        # ── 1. 从新 Request 中提取流式更新数据 ──
        # 包括新增 token、多模态特征、采样参数等
        # 如果 request.resumable=False，则 update=None，
        # 表示不再提供后续输入
        update = StreamingUpdate.from_request(request)
        # ── 2. 原请求仍在运行：先缓存后续输入 ──
        if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
            # 相同 ID 只允许用于已开启可恢复模式的请求
            assert existing.streaming_queue is not None, "duplicate request id"
            # update 也可能是 None，作为“输入流结束”的标记
            existing.streaming_queue.append(update)
        # ── 3. 原请求正在等待输入：立即恢复会话 ──
        elif update is not None:
            self._update_request_as_session(
                existing,
                update,
            )
            # 主要效果：
            #   追加新的输入 token / 多模态特征
            #   更新采样参数
            #   WAITING_FOR_STREAMING_REQ → WAITING
            #   重新进入等待队列
        # ── 4. 没有后续输入：结束流式会话 ──
        else:
            self.finish_requests(
                request_id,
                RequestStatus.FINISHED_ABORTED,
            )
    # ══════════════════════════════════════════════════════
    #  第二种情况：request_id 不存在
    #  这是第一次进入 Scheduler 的新请求
    #  L2564 - L2574
    # ══════════════════════════════════════════════════════
    else:
        # ── 1. 可恢复请求需要保存后续输入的队列 ──
        if request.resumable:
            request.streaming_queue = deque()
        # ── 2. ★ 放入合适的等待队列 ──
        self._enqueue_waiting_request(request)
        # 普通 WAITING 请求        → self.waiting
        # 等待 grammar / 远端 KV /
        # 流式输入的阻塞请求       → self.skipped_waiting

        # ── 3. ★ 注册到请求总表 ──
        self.requests[request_id] = request
        # 后续 schedule、abort、update_from_output
        # 都通过 request_id 查找该 Request

        # ── 4. 初始化推测解码统计 ──
        if self.spec_decode_metrics_level != "none":
            request.spec_decode_metrics = RequestSpecDecodeMetrics.new(
                self.num_spec_tokens
            )

        # ── 5. 通知 KV Connector 有新请求进入 ──
        if self.connector is not None:
            self.connector.on_new_request(request)

        # ── 6. 记录请求进入队列的事件 ──
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    # 无返回值；核心产物是 Scheduler 内部状态发生变化
```
1. **主要作用**：`Scheduler.add_request()` 把新的 `Request` 注册到调度器并放入合适的等待队列；如果 `request_id` 已存在，则把它当作可恢复流式会话的后续输入，而不是创建重复请求。

2. **输入（参数）**：输入一个 `Request` 对象，其中包含请求 ID、prompt token、采样参数、优先级、LoRA、多模态信息和当前状态等。

3. **输出（改变了什么值）**：函数没有返回值；它会修改 `Scheduler` 内部的 `requests`、`waiting` 或 `skipped_waiting`，并可能更新已有流式会话的输入队列、推测解码统计、KV Connector 状态和请求事件记录。

4. **术语解释**：
- `Request`：进入 `EngineCore` 后的内部请求对象。它已经完成分词，主要保存 token、生成参数和执行状态。
- `request_id`：请求的唯一标识。调度器通过它判断这是新请求，还是已有流式会话的后续输入。
- 可恢复流式会话：一个请求可以分多次提交输入。当前输入处理完后，请求进入 `WAITING_FOR_STREAMING_REQ`，后续使用相同 `request_id` 提交新的输入片段。
- `StreamingUpdate`：从新传入的 `Request` 中提取出的会话更新信息，例如新增 token、多模态特征和新的采样参数。
- `streaming_queue`：已有会话仍在运行时，后续输入先暂存在这里；它表示“流式输入队列”，不只是模型逐 token 返回结果的普通流式输出。
- `waiting`：可以正常参与后续调度的请求队列。
- `skipped_waiting`：暂时不能运行的请求队列，例如正在等待结构化输出 grammar、远端 KV Cache 或下一段流式输入。
- `spec_decode_metrics`：当前请求的推测解码统计信息，例如草稿 token 数量和接受数量。
- `connector.on_new_request()`：通知 KV Connector 有新请求进入，使远程 KV Cache 传输组件可以建立相应状态。
- `QUEUED` 事件：记录请求已经进入调度队列，主要用于性能统计和问题定位。

5. **核心流程**：
```
根据 request_id 查找已有请求
        │
        ├─ 已存在
        │    ├─ 提取 StreamingUpdate
        │    ├─ 已有请求仍在运行 → 后续输入放入 streaming_queue
        │    ├─ 正在等待输入 → 立即更新并恢复会话
        │    └─ 没有后续输入 → 结束该流式会话
        │
        └─ 不存在
             ├─ 可恢复请求则创建 streaming_queue
             ├─ 放入 waiting 或 skipped_waiting
             ├─ 写入 requests 字典
             ├─ 初始化推测解码统计
             └─ 通知 Connector 并记录 QUEUED 事件
```

6. **在整个项目中的定位**：
```
用户调用 LLM / AsyncLLM
          ↓
InputProcessor
把原始输入转换为 EngineCoreRequest
          ↓
EngineCore.preprocess_add_request()
转换为内部 Request
          ↓
EngineCore.add_request()
          ↓
Scheduler.add_request()             ← 当前函数
注册请求并放入等待队列
          ↓
Scheduler.schedule()
选择本轮请求并分配逻辑 KV Cache block
          ↓
ModelExecutor / ModelRunner
执行模型计算
          ↓
Scheduler.update_from_output()
更新 Request 状态并整理输出
          ↓
OutputProcessor
生成用户可见的结果
```
该函数位于“输入处理完成”和“正式调度执行”之间，是请求进入调度系统的入口；它只负责登记和排队，不会立即执行模型计算。
### schedule函数
```python
# scheduler.py:570
def schedule(self, throttle_prefills=False) -> SchedulerOutput:

    # ═══ 初始化预算 ═══════════════════════════════════════ L590
    token_budget = self.max_num_scheduled_tokens   # 本步总 token 预算，默认 = max_num_batched_tokens
    input_budget = self.max_num_batched_tokens     # 输入侧预算
    kv_cache_manager.new_step_starts()             # 通知 KV 管理器新的一步

    scheduled_new_reqs     = []   # 首次调度的请求
    scheduled_cached_reqs  = []   # 已调度过的请求（只记录增量）
    num_scheduled_tokens   = {}   # {req_id: 本步给它算多少 token} ★ 最终产物
    req_to_new_blocks      = {}   # {req_id: 新分配的 block}

    # ══════════════════════════════════════════════════════
    #  第一阶段：调度 RUNNING 队列（已经在跑的请求）
    #  L620 - L851
    # ══════════════════════════════════════════════════════
    for request in self.running:                    # L~640，按 FCFS 顺序

        # ── 1. 算这个请求这一步要算多少 token ──      L665-713
        num_new_tokens = (request.num_tokens_with_spec
                          - request.num_computed_tokens)
        # 被预算截断（关键！）                        L672
        num_new_tokens = min(num_new_tokens, token_budget)

        # ── 2. 如果算不了（预算用光 / 已算完），跳过 ── L715-733
        if num_new_tokens == 0:
            continue          # ★ 注意是 continue 不是 break
                              #   为了公平性，允许低优先级请求也被调度

        # ── 3. ★ 分配 KV Cache block（可能触发抢占）── L736-808
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)  # L738

            if new_blocks is not None:
                break                     # 分配成功

            # ★★ 分配失败 → 抢占一个受害者 ★★
            if self.policy == SchedulingPolicy.PRIORITY:
                victim = max(...)         # 优先级策略：挑 priority 最高的踢
            else:
                victim = self.running[-1] # 默认：踢最后一个（FCFS 里最晚来的）

            self._preempt_request(victim) # L796 ★ 见 8.7
            if victim is request:
                break                     # 连自己都被抢占了，放弃这个请求

        # ── 4. 记账 ──                               L811-818
        req_to_new_blocks[request.request_id] = new_blocks
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget     -= num_new_tokens         # ★ 扣预算
        input_budget     -= num_new_tokens

    # ══════════════════════════════════════════════════════
    #  第二阶段：从 WAITING 队列提拔新请求（连续批处理的关键！）
    #  L863 - L1348
    # ══════════════════════════════════════════════════════
    # 只有"第一步没发生抢占"时才提拔新请求              L864
    # 因为发生了抢占就说明显存已经紧张，不该再塞新请求
    while self.waiting and token_budget > 0:

        request = self.waiting[0]     # 取出队头请求（⚫ 阻塞态处理跳过）

        # ── 1. ★ 查前缀缓存能命中多少 ──              L927-1039
        #    "这个 prompt 的开头，之前是不是有别的请求算过？"
        #    命中 → 那些 token 直接复用，不用重算！
        num_computed_tokens = self._get_local_prefix_cache_hit(request, ...)  # L934

        # ── 2. 算新请求要算多少 token ──               L1084-1124
        num_new_tokens = request.num_tokens - num_computed_tokens

        # ★ 分块 prefill 的核心三行                    L1116-1124
        if not self.enable_chunked_prefill and num_new_tokens > token_budget:
            break                     # 不开分块 prefill：整块放不下就等下一步
        else:
            # ★ 开分块 prefill（V1 默认）：截断到预算大小
            num_new_tokens = min(num_new_tokens, token_budget)

        # ── 3. 分配 KV Cache block ──                  L1210-1222
        new_blocks = self.kv_cache_manager.allocate_slots(...)  # L1210
        if new_blocks is None:
            break                     # 分配失败 → 停止提拔

        # ── 4. ★ 提拔到 running 队列 ──                L1292-1339
        self.running.append(request)
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens

        # ★ 注意区分：新请求 vs 被抢占后恢复的请求
        if request.status == RequestStatus.WAITING:
            scheduled_new_reqs.append(NewRequestData.from_request(request, new_blocks))
        else:  # PREEMPTED
            scheduled_cached_reqs.append(CachedRequestData.from_request(...))

        token_budget -= num_new_tokens

    # ═══ 收尾：组装 SchedulerOutput ═══════════════════════ L1350-1507
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=scheduled_cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,     # ★ 最重要
        total_num_scheduled_tokens=...,
        finished_req_ids=...,
        ...
    )
```
1. **主要作用**：`schedule()` 为当前一轮模型计算选择请求，决定每个请求本轮需要处理多少个 token，并为它们分配逻辑 KV Cache block、编码器缓存等资源，最终生成供 `ModelExecutor` 使用的 `SchedulerOutput`。它只制定本轮执行计划，不进行真正的 GPU 模型计算。

2. **输入（参数）**：
- `self`：当前 `Scheduler`，内部保存等待请求、运行请求、缓存管理器和各类预算。
- `throttle_prefills=False`：是否暂缓 Prefill 请求，用于数据并行场景下平衡不同 DP rank 的 Prefill 负载；暂缓时通常仍允许 Decode 请求执行。

3. **输出（改变了什么值）**：返回一个 `SchedulerOutput`，其中记录本轮的新请求、已有请求、每个请求处理的 token 数、KV block、推测解码 token、多模态编码输入、远程缓存传输信息以及完成或抢占的请求 ID。函数还会修改调度器内部状态，例如移动请求队列、把请求设为 `RUNNING` 或 `PREEMPTED`、分配或释放逻辑缓存 block，并更新 `num_computed_tokens`、`num_in_flight_tokens` 和本轮预算等；但不会生成最终文本。

4. **术语解释**：
- 调度轮次：一次 `schedule()` 对应一次模型执行计划，通常随后触发一次模型 forward；一轮可以同时安排多个请求，也可以给不同请求安排不同数量的 token。
- 统一 token 调度：该函数没有完全分开的“Prefill 调度器”和“Decode 调度器”，而是计算：
    ```
    请求现有的 token 总数
    - 已经计算的 token 数
    = 本轮仍需要计算的 token 数
    ```
因此同一套逻辑可以处理 Prefill、Decode、分块 Prefill和推测解码。
- `token_budget`：本轮允许调度的 token 总量，所有请求共享，不能超过 `max_num_scheduled_tokens`。
- `input_budget`：本轮模型批次允许承载的输入 token 总量，通常来自 `max_num_batched_tokens`。
- `encoder_compute_budget`：图片、视频等多模态编码器本轮允许处理的输入量。
- `num_tokens_with_spec`：请求当前需要处理的 token 总数，包含 prompt token、已经生成的输出 token以及推测解码产生的候选 token。
- `num_computed_tokens`：该请求已经被模型计算过的 token 数量，也就是已经建立了有效计算进度的位置数。
- `num_new_tokens`：本轮计划交给模型继续计算的 token 数量。基础关系是：`num_tokens_with_spec`-`num_computed_tokens`
- `num_in_flight_tokens`：已经调度、但结果可能还没有完全返回的 token 数，用于异步调度。
- 前缀缓存：新请求进入时，调度器先查找本地或远端是否已经存在相同 prompt 前缀的 KV Cache；命中的 token 无需重新计算。
- `allocate_slots()`：从预先创建的 KV Cache 池中给请求分配逻辑 block。它不是重新创建 GPU KV Cache 张量，而是在已有缓存池里安排哪些 block 属于哪个请求。
- 抢占：如果 KV Cache 空间不足，调度器会选择较低优先级的运行请求，释放其缓存并将其设为 `PREEMPTED`，以后重新排队计算。
- `scheduled_new_reqs`：第一次进入模型执行端的请求，需要发送较完整的请求信息。
- `scheduled_cached_reqs`：模型执行端已经缓存过信息的请求，只需发送本轮变化，减少进程通信量。
- `scheduled_spec_decode_tokens`：推测解码器提前预测的候选 token，本轮由主模型验证。
- lookahead slots：为推测解码可能使用的未来 token 预留的 KV Cache 位置。
- `WAITING_FOR_REMOTE_KVS`：请求需要的 KV Cache 正在从远端异步加载，本轮暂时不执行模型。
- Connector metadata：描述本轮远程 KV Cache 或编码器缓存应该加载、保存或传输哪些内容。
- `num_common_prefix_blocks`：运行请求共同拥有的前缀 block 数，可以供 Cascade Attention 等优化使用。

4. 核心流程：
```
1. 初始化本轮状态
   ├─ current_step 加一
   ├─ 建立 token、输入和编码器预算
   └─ 检查暂停状态与 Prefill 节流
                  ↓
2. 优先处理 RUNNING 请求
   ├─ 计算每个请求还需处理多少 token
   ├─ 限制长度和本轮预算
   ├─ 安排推测 token、多模态编码输入
   ├─ 分配逻辑 KV Cache block
   └─ 空间不足时抢占低优先级请求
                  ↓
3. 处理 WAITING 请求
   ├─ 跳过仍在等待 grammar、远端 KV 或流式输入的请求
   ├─ 检查 LoRA 数量和活跃请求数限制
   ├─ 查询本地及远端前缀缓存
   ├─ 计算本轮需要处理的 token
   ├─ 分配 KV Cache 和编码器缓存
   └─ 将成功接纳的请求移入 running
                  ↓
4. 检查调度约束
   ├─ token 预算不能超限
   ├─ 请求数量不能超限
   └─ 各项预算不能为负
                  ↓
5. 构造 SchedulerOutput
   ├─ 新请求与已有请求数据
   ├─ 每个请求的 token 数和 KV block
   ├─ 推测解码与多模态输入
   ├─ 缓存清零、复制和 Connector 信息
   └─ 完成或抢占的请求 ID
                  ↓
6. 更新调度器内部状态
   ├─ 增加 num_computed_tokens
   ├─ 增加 num_in_flight_tokens
   ├─ 更新是否仍处于 Prefill
   └─ 清空已交付的 finished/preempted ID 集合
```

需要特别注意：只有在本轮没有发生抢占并且调度器未暂停时，才会继续接纳新的等待请求。这可以避免一边因缓存不足抢占旧请求，一边又把新请求加入运行队列。

**在整个项目中的定位**：
```
InputProcessor
把用户输入转换为 EngineCoreRequest
        ↓
EngineCore.add_request()
        ↓
Scheduler.add_request()
请求进入 waiting / skipped_waiting
        ↓
EngineCore.step()
        ↓
Scheduler.schedule()                     ← 当前函数
决定“哪些请求、本轮处理多少 token、使用哪些缓存 block”
        ↓
SchedulerOutput
        ↓
ModelExecutor.execute_model()
        ↓
Worker / ModelRunner
执行真正的 GPU 模型计算
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
更新 token、停止状态和缓存状态
        ↓
EngineCoreOutputs
        ↓
OutputProcessor
转换为用户最终看到的输出
        ↓
下一轮再次调用 Scheduler.schedule()
```
因此，`schedule()` 是 `EngineCore.step()` 中“模型执行之前”的核心决策函数：上接请求队列和缓存状态，下接 `ModelExecutor`，决定 GPU 下一轮具体计算什么。
### preempt request函数
```python
# scheduler.py:1528
def _preempt_request(
    self,
    request: Request,
    timestamp: float,
    drop_stale_output: bool = False,
) -> None:
    # ══════════════════════════════════════════════════════
    #  前置条件：只能抢占正在运行的请求
    #  L1541
    # ══════════════════════════════════════════════════════
    assert request.status == RequestStatus.RUNNING
    # 注意：调用者必须提前把 request 从 self.running 中移除，
    # 当前函数只负责释放资源、重置状态和重新排队。
    # ══════════════════════════════════════════════════════
    #  第一阶段：释放请求占用的缓存资源
    #  L1544 - L1546
    # ══════════════════════════════════════════════════════

    # ── 1. ★ 释放逻辑 KV Cache block ──
    self._free_request_blocks(request)
    # 如果仍有异步 GPU 操作在使用这些 block，
    # 可能先放入 deferred_frees，稍后再真正归还缓存池。

    # ── 2. 释放多模态/编码器缓存 ──
    self.encoder_cache_manager.free(request)

    # ── 3. 不再把它视为正在进行的 Prefill ──
    self._inflight_prefills.discard(request)

    # ══════════════════════════════════════════════════════
    #  第二阶段：重置请求的执行状态
    #  L1547 - L1552
    # ══════════════════════════════════════════════════════

    # ── 1. ★ 标记为“被抢占”，不是“已结束” ──
    request.status = RequestStatus.PREEMPTED

    # ── 2. KV Cache 已释放，旧计算进度不能继续信任 ──
    request.num_computed_tokens = 0

    # 原始 prompt 和已经保存的输出 token 不会删除；
    # 请求恢复时需要重新计算，也可能通过前缀缓存复用部分结果。

    # ── 3. 清除依赖旧模型状态的推测 token ──
    if request.spec_token_ids:
        request.spec_token_ids = []

    # ══════════════════════════════════════════════════════
    #  第三阶段：处理尚未返回的异步输出
    #  L1553 - L1562
    # ══════════════════════════════════════════════════════

    # 抢占发生时，之前提交给 GPU 的计算可能仍未返回。
    # 这些结果返回后属于 stale output（过期输出）。

    # ── 1. 决定旧输出返回后是交付还是丢弃 ──
    request.drop_stale_output = (
        drop_stale_output
        or (
            request.drop_stale_output
            and request.num_stale_output_tokens > 0
        )
    )
    # False：旧输出仍可交付，但不能修改已经重置的计算计数。
    # True ：旧输出返回后直接丢弃，相关位置以后重新计算。
    # 已进入丢弃模式且旧输出尚未排空时，继续保持丢弃模式。

    # ── 2. ★ 当前所有在途 token 都标记为 stale ──
    request.num_stale_output_tokens = request.num_in_flight_tokens
    # 注意这里是赋值，不是累加，因为 num_in_flight_tokens
    # 已经包含尚未排空的旧 token。

    # ── 3. 抢占后取消异步/推测输出占位 ──
    request.num_output_placeholders = 0

    # ══════════════════════════════════════════════════════
    #  第四阶段：更新抢占统计
    #  L1563 - L1566
    # ══════════════════════════════════════════════════════

    request.num_preemptions += 1

    if self.log_stats:
        request.record_event(
            EngineCoreEventType.PREEMPTED,
            timestamp,
        )

    # ══════════════════════════════════════════════════════
    #  第五阶段：重新排队并通知 ModelRunner
    #  L1568 - L1569
    # ══════════════════════════════════════════════════════

    # ── 1. ★ 放回等待队列，后续可以重新调度 ──
    self.waiting.prepend_request(request)

    # ── 2. 记录本轮被抢占的请求 ID ──
    self.reset_preempted_req_ids.add(request.request_id)
    # 随后会写入 SchedulerOutput.preempted_req_ids，
    # 通知 ModelRunner 清理该请求对应的旧运行状态。

    # 无返回值；核心结果是请求释放资源、状态重置并重新排队
```
1. **主要作用**：`_preempt_request()` 暂停一个正在运行的请求，释放它占用的 KV Cache 和编码器缓存，将状态改为 `PREEMPTED`，再放回等待队列，以便以后重新调度。抢占不是取消或结束请求，用户生成任务仍会继续。

2. **输入（参数）**：
- `request`：要被抢占的 `Request`，其状态必须为 `RUNNING`；调用者需要先将它从 `self.running` 中移除。
- `timestamp`：抢占发生的时间，用于性能统计和事件记录。
- `drop_stale_output=False`：是否丢弃抢占前已经发起、但尚未返回的异步模型输出；默认仍将这些输出交付，特殊情况下才丢弃。

3. **输出（改变了什么值）**：函数没有返回值。它会释放或延迟释放请求的逻辑 KV block，释放编码器缓存，将请求状态改为 `PREEMPTED`，重置已计算 token 数和推测 token，增加抢占次数，并把请求重新放到 `waiting` 队列；同时把请求 ID 加入 `reset_preempted_req_ids`，通知模型执行端清理该请求的旧状态。

4. **术语解释**：
- 抢占：当 KV Cache 空间不足时，调度器暂时停止某个运行请求，把资源让给其他请求。被抢占请求没有完成，之后还会重新运行。
- `PREEMPTED`：表示请求被暂停并等待恢复，不属于结束状态，也不会产生 `finish_reason`。
- `_free_request_blocks()`：解除该请求对 KV Cache block 的占用。如果异步 GPU 操作仍可能写入这些 block，会先放入延迟释放队列，等相关计算完成后再安全归还缓存池。
- 编码器缓存：保存图片、音频或编码器—解码器模型产生的中间特征；抢占时通过 `encoder_cache_manager.free()` 一并释放。
- `num_computed_tokens = 0`：调度器不再相信旧 KV Cache 对应的计算进度。请求恢复后需要重新计算，但原始 prompt token 和已经保留的输出 token 并没有被删除，并且可能通过前缀缓存减少实际重算量。
- `spec_token_ids`：推测解码模型给出的候选 token。它们依赖抢占前的模型状态，因此抢占后必须清空，不能继续使用。
- `num_output_placeholders`：异步或推测解码中，为尚未返回的输出预留的位置。抢占后原有计算进度已失效，因此重置为 `0`。
- in-flight output：已经提交给模型执行器、但尚未返回调度器的结果。异步调度下，请求被抢占时，这些 GPU 计算可能仍在进行，无法立即撤销。
- stale output：抢占前发起、抢占后才返回的旧结果。代码通过 `num_stale_output_tokens` 记录还有多少旧结果尚未返回。
- `drop_stale_output=False`：旧结果返回时通常仍会交付，但不能用它修改已经重置的计算计数；在旧结果排空前，请求也不会贸然恢复，以免同一位置被重复采样。
- `drop_stale_output=True`：旧结果返回后直接丢弃，适用于重置前缀缓存或 KV Connector 状态已失效等情况，因为这些位置将重新计算。
- `num_preemptions`：请求累计被抢占的次数，可用于统计和判断后续是否属于恢复执行。
- `waiting.prepend_request()`：把请求重新放回等待队列，使其以后可以由 `schedule()` 再次选择。
- `reset_preempted_req_ids`：记录本轮被抢占的请求 ID，随后写入 `SchedulerOutput.preempted_req_ids`，让 ModelRunner 重置对应的缓存状态。
- `_inflight_prefills.discard()`：如果该请求仍处于分块 Prefill，将它从正在进行的 Prefill 集合中移除，因为其当前缓存和计算进度已经被重置。
- 重要调用约束：该函数不会自行从 `self.running` 删除请求。`schedule()` 或 `reset_prefix_cache()` 必须先把请求弹出，再调用此函数。

5. 核心流程：
```
确认 request.status == RUNNING
              ↓
释放或延迟释放 KV Cache block
              ↓
释放多模态/编码器缓存
              ↓
从正在 Prefill 的集合中移除
              ↓
状态改为 PREEMPTED
              ↓
num_computed_tokens 清零
清空推测 token 和输出占位符
              ↓
标记尚未返回的异步输出为 stale
并决定交付还是丢弃
              ↓
num_preemptions 加一
记录 PREEMPTED 事件
              ↓
重新放入 waiting 队列
              ↓
记录到 reset_preempted_req_ids
```
6. **在整个项目中的定位**：
```
Scheduler.schedule()
尝试为运行请求分配 KV Cache block
              ↓
缓存空间不足
              ↓
选择低优先级或队尾请求作为被抢占者
              ↓
调用者先从 running 队列移除请求
              ↓
_preempt_request()                         ← 当前函数
    ├─ 释放缓存资源
    ├─ 重置计算进度
    ├─ 处理未返回的旧输出
    └─ 放回 waiting 队列
              ↓
SchedulerOutput.preempted_req_ids
通知 ModelRunner 清理请求状态
              ↓
后续 Scheduler.schedule()
重新选择该请求并分配 KV block
              ↓
重新计算或利用前缀缓存恢复
              ↓
ModelExecutor / ModelRunner 继续执行
```
除缓存不足外，`reset_prefix_cache(reset_running_requests=True)` 也会调用该函数，将所有运行请求抢占后再安全重置前缀缓存。
### update from output函数
```python
# scheduler.py:1922
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:

    # ══════════════════════════════════════════════════════
    #  第一阶段：提取本轮调度计划和模型计算结果
    #  L1927 - L1937
    # ══════════════════════════════════════════════════════

    sampled_token_ids = model_runner_output.sampled_token_ids
    logprobs = model_runner_output.logprobs
    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
    pooler_outputs = model_runner_output.pooler_output

    # ★ schedule() 决定的“每个请求本轮计算多少 token”
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens

    kv_connector_output = model_runner_output.kv_connector_output
    ec_connector_output = model_runner_output.ec_connector_output
    cudagraph_stats = model_runner_output.cudagraph_stats

    # 最终先按 client_index 收集每个客户端的请求输出
    outputs = defaultdict(list)

    # ══════════════════════════════════════════════════════
    #  第二阶段：处理批次级资源和缓存传输结果
    #  L1939 - L1986
    # ══════════════════════════════════════════════════════

    # ── 1. 释放已经确认不再被 GPU 使用的延期 KV block ──
    if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
        self.processed_step_seq += 1
        self._drain_deferred_frees()

    # ── 2. 收集本轮性能统计 ──
    perf_stats = None
    if self.perf_metrics and self.perf_metrics.is_enabled():
        perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(
            scheduler_output
        )

    # ── 3. 处理远端 KV 加载失败或无效 block ──
    failed_kv_load_req_ids = set()

    if kv_connector_output and kv_connector_output.failed_recving:
        failed_kv_load_req_ids.update(
            self._handle_failed_recving(
                kv_connector_output.failed_recving
            )
        )

    if kv_connector_output and kv_connector_output.invalid_block_ids:
        failed_kv_load_req_ids.update(
            self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )
        )
        # 根据配置：
        #   要么降低 num_computed_tokens，让这些 token 重新计算
        #   要么稍后把请求作为错误结束

    # ── 4. 保存本轮 MoE 专家路由信息 ──
    if model_runner_output.routed_experts is not None:
        routing_data = ...
        routing_offsets = ...
    else:
        routing_data = None
        routing_offsets = {}

    # ══════════════════════════════════════════════════════
    #  第三阶段：★ 逐个处理本轮执行过的请求
    #  L1988 - L2232
    # ══════════════════════════════════════════════════════

    stopped_running_reqs = set()
    stopped_preempted_reqs = set()
    spec_decoding_stats = None

    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():

        request = self.requests.get(req_id)

        # ── 1. 更新在途 token 和 stale output 计数 ──
        output_is_stale = False

        if request is not None:
            request.num_in_flight_tokens -= num_tokens_scheduled

            if request.num_stale_output_tokens > 0:
                output_is_stale = True
                request.num_stale_output_tokens -= num_tokens_scheduled

        # 远端 KV 加载失败：等待重新计算或错误处理
        if req_id in failed_kv_load_req_ids:
            continue

        # 请求可能在模型运行期间被用户取消
        if request is None or request.is_finished():
            continue

        # 抢占前的旧输出被配置为直接丢弃
        if output_is_stale and request.drop_stale_output:
            continue

        # ── 2. 找到该请求的模型输出 ──
        req_index = model_runner_output.req_id_to_index[req_id]

        generated_token_ids = (
            sampled_token_ids[req_index]
            if sampled_token_ids
            else []
        )

        # ── 3. ★ 修正推测解码结果 ──
        scheduled_spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(req_id)
        )

        if scheduled_spec_token_ids:
            num_draft_tokens = len(scheduled_spec_token_ids)
            num_accepted = max(
                len(generated_token_ids)
                - self.num_sampled_tokens_per_step,
                0,
            )
            num_rejected = num_draft_tokens - num_accepted

            # schedule() 曾乐观地把草稿 token 计入已计算数量，
            # 被主模型拒绝的部分必须回滚。
            if not output_is_stale:
                request.num_computed_tokens -= num_rejected

                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected

            # 记录推测 token 的提议、接受和拒绝情况
            spec_decoding_stats = self.make_spec_decoding_stats(...)
            if request.spec_decode_metrics is not None:
                request.spec_decode_metrics.observe(...)

        # ── 4. 模型已经真正使用完输入后，释放编码器输入 ──
        if request.has_encoder_inputs:
            self._free_encoder_inputs(request)

        # ══════════════════════════════════════════════════
        #  第四阶段：写入生成结果并判断请求是否停止
        #  L2079 - L2187
        # ══════════════════════════════════════════════════

        stopped = False
        new_token_ids = generated_token_ids
        pooler_output = (
            pooler_outputs[req_index]
            if pooler_outputs
            else None
        )

        status_before_stop = request.status

        # ── 1. 普通生成请求：追加 token 并检查停止条件 ──
        if new_token_ids:
            new_token_ids, stopped = self._update_request_with_output(
                request,
                new_token_ids,
                is_stale=output_is_stale,
            )
            # 检查：
            #   EOS / stop token
            #   max_tokens / max_model_len
            #   重复生成

        # ── 2. Pooling 请求：得到向量后立即完成 ──
        elif request.pooling_params and pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            stopped = True

        # ── 3. 纯多模态编码器：处理完整个输入后完成 ──
        elif (
            self.is_mm_encoder_only
            and request.num_computed_tokens >= request.num_prompt_tokens
        ):
            request.status = RequestStatus.FINISHED_STOPPED
            stopped = True

        # ── 4. 校验结构化输出 token ──
        if new_token_ids and not self.structured_output_manager.accept_tokens(
            request,
            new_token_ids,
        ):
            request.status = RequestStatus.FINISHED_ERROR
            request.resumable = False
            stopped = True

        # ── 5. 整理 MoE 路由和 Prefill 统计 ──
        routed_experts = ...
        should_emit_output = bool(
            new_token_ids
            or pooler_output is not None
            or stopped
        )

        prefill_stats = None
        if should_emit_output:
            prefill_stats = request.take_prefill_stats()
            if prefill_stats is not None:
                prefill_stats.finalize(
                    self.kv_cache_manager.estimate_cached_tokens(request)
                )

        # ── 6. ★ 请求停止：处理会话并释放资源 ──
        finish_reason = None
        kv_transfer_params = None
        ec_transfer_params = None

        if stopped:
            # 必须先保存结束原因；
            # 可恢复会话可能随后被重新设为 WAITING。
            finish_reason = request.get_finished_reason()

            finished = self._handle_stopped_request(request)

            if finished:
                (
                    kv_transfer_params,
                    ec_transfer_params,
                ) = self._free_request(request)
                # 释放编码器缓存和 KV block，
                # 或等待 Connector 传输结束后再释放。

            if status_before_stop == RequestStatus.RUNNING:
                stopped_running_reqs.add(request)
            else:
                stopped_preempted_reqs.add(request)

        # ══════════════════════════════════════════════════
        #  第五阶段：构造单个请求的 EngineCoreOutput
        #  L2189 - L2232
        # ══════════════════════════════════════════════════

        # 提取该请求对应的生成 logprob
        new_logprobs = None
        if request.sampling_params.num_logprobs is not None and logprobs:
            new_logprobs = logprobs.slice_request(
                req_index,
                len(new_token_ids),
            )

        # 可选：采样掩码、prompt logprob、NaN 统计等
        new_sampling_mask = ...
        prompt_logprobs = prompt_logprobs_dict.get(req_id)

        if should_emit_output:
            # ★ 按 client_index 分组，确保结果返回正确客户端
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=finish_reason,
                    new_logprobs=new_logprobs,
                    new_prompt_logprobs_tensors=prompt_logprobs,
                    pooling_output=pooler_output,
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    prefill_stats=prefill_stats,
                    spec_decode_metrics=...,
                    kv_transfer_params=kv_transfer_params,
                    ec_transfer_params=ec_transfer_params,
                    routed_experts=routed_experts,
                    ...
                )
            )
        else:
            # ★ 纯分块 Prefill 不向前端返回中间输出
            assert not prompt_logprobs

    # ══════════════════════════════════════════════════════
    #  第六阶段：从请求队列中清除已停止请求
    #  L2234 - L2241
    # ══════════════════════════════════════════════════════

    if stopped_running_reqs:
        self.running = remove_all(
            self.running,
            stopped_running_reqs,
        )

    if stopped_preempted_reqs:
        self.waiting.remove_requests(stopped_preempted_reqs)
        self.skipped_waiting.remove_requests(stopped_preempted_reqs)

    # ══════════════════════════════════════════════════════
    #  第七阶段：处理 grammar、KV 和 EC 错误
    #  L2243 - L2270
    # ══════════════════════════════════════════════════════

    error_req_ids = set(self.grammar_compile_error_reqs)
    self.grammar_compile_error_reqs.clear()

    if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
        error_req_ids.update(failed_kv_load_req_ids)

    if self.ec_connector is not None:
        error_req_ids.update(
            self.ec_connector.take_unavailable_requests()
        )

    if error_req_ids:
        error_reqs = self.finish_requests(
            error_req_ids,
            RequestStatus.FINISHED_ERROR,
        )

        for request in error_reqs:
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                )
            )

    # ══════════════════════════════════════════════════════
    #  第八阶段：更新缓存连接器状态和统计信息
    #  L2271 - L2334
    # ══════════════════════════════════════════════════════

    # 处理远端 KV 接收/发送完成事件
    if kv_connector_output:
        self._update_from_kv_xfer_finished(kv_connector_output)

    # 更新编码器缓存传输状态
    if self.ec_connector is not None and ec_connector_output:
        self.ec_connector.update_connector_output(
            ec_connector_output
        )

    # 合并 Worker 侧和 Scheduler 侧的 Connector 统计
    kv_connector_stats = ...
    ec_connector_stats = ...

    # 收集并发布 KV Cache 事件
    events = self.kv_cache_manager.take_events()

    if self.connector is not None:
        connector_events = self.connector.take_events()
        ...

    if events:
        self.kv_event_publisher.publish(
            KVEventBatch(
                ts=time.time(),
                events=events,
            )
        )

    # ══════════════════════════════════════════════════════
    #  第九阶段：★ 组装最终 EngineCoreOutputs
    #  L2335 - L2368
    # ══════════════════════════════════════════════════════

    # 每个 client_index 得到一份独立输出
    engine_core_outputs = {
        client_index: EngineCoreOutputs(outputs=client_outputs)
        for client_index, client_outputs in outputs.items()
    }

    # 多引擎模式：附加已完成的请求 ID
    if self.finished_req_ids_dict:
        for client_index, finished_set in self.finished_req_ids_dict.items():
            if client_index in engine_core_outputs:
                engine_core_outputs[client_index].finished_requests = finished_set
            else:
                engine_core_outputs[client_index] = EngineCoreOutputs(
                    finished_requests=finished_set
                )

        self.finished_req_ids_dict.clear()

    # 附加调度、推测解码、Connector 和性能统计
    stats = self.make_stats(
        spec_decoding_stats,
        kv_connector_stats,
        cudagraph_stats,
        perf_stats,
        ec_connector_stats=ec_connector_stats,
    )

    if stats is not None:
        # 统计只需要发送给一个前端
        if not engine_core_outputs:
            engine_core_outputs[0] = EngineCoreOutputs()

        first_output = next(iter(engine_core_outputs.values()))
        first_output.scheduler_stats = stats

    # ★ 最终产物：
    # {client_index: 该客户端本轮产生的所有 EngineCoreOutput}
    return engine_core_outputs
```
1. **主要作用**：`update_from_output()` 接收本轮模型执行结果，把新生成的 token 写回对应 `Request`，修正推测解码和异步调度状态，判断请求是否结束并释放资源，最后生成按客户端分组的 `EngineCoreOutputs`。它是“模型完成一轮计算”之后，调度器处理结果和更新内部状态的核心函数。

2. **输入（参数）**：
- `scheduler_output`：本轮执行前由 `schedule()` 生成的计划，记录哪些请求被执行、各自执行多少 token，以及推测解码、编码器输入和缓存信息。
- `model_runner_output`：`ModelRunner` 真正执行模型后返回的结果，可能包含生成的 token ID、logprob、Pooling 结果、KV/编码器缓存传输结果、MoE 专家路由信息和性能统计。
这两个参数必须属于同一轮执行，否则请求和输出将无法正确对应。

3. **输出（改变了什么值）**：返回 `dict[int, EngineCoreOutputs]`，键是 `client_index`，值是该客户端本轮收到的 token、结束原因、logprob、Pooling 结果、事件和统计信息。函数还会更新请求的输出 token、在途 token 数、停止状态和推测解码计数；移除并释放已结束请求；处理缓存传输结果、延迟释放的 KV block、缓存事件和调度统计。

4. **术语解释**：
- `SchedulerOutput`：模型执行前的“计算计划”。其中 `num_scheduled_tokens` 表示每个请求本轮被安排计算多少个 token。
- `ModelRunnerOutput`：模型执行后的“计算结果”。它保存 token ID 等底层结果，还不是用户最终看到的文本。
- `EngineCoreOutput`：单个请求本轮产生的引擎输出，包含新增 token、结束原因、logprob、Pooling 结果等。
- `EngineCoreOutputs`：一组请求输出的容器。同一个 `client_index` 的结果会放在一起，便于多前端或多客户端把结果送回正确用户。
- `num_in_flight_tokens`：已经提交执行、但结果尚未处理完的 token 数。本函数收到结果后会减去本轮完成的数量。
- stale output：请求被抢占后才返回的旧一轮异步输出。函数会减少对应的 stale 计数；如果设置了 `drop_stale_output`，则直接丢弃这些旧结果。
- 推测解码回滚：`schedule()` 会先假设草稿 token 都参与了计算，但主模型可能只接受其中一部分。本函数根据接受数量把被拒绝的部分从 `num_computed_tokens` 和输出占位符中扣除。
- `sampled_token_ids`：模型新生成的 token ID。函数通过 `_update_request_with_output()` 追加到请求中，并逐个检查停止条件。
- 停止条件：包括 EOS token、指定的 stop token、`max_tokens`、模型最大长度以及重复检测。命中后会设置相应的 `RequestStatus` 和 `finish_reason`。
- `finish_reason`：返回给上层的结束原因，例如 `stop`、`length`、`abort`、`error` 或 `repetition`。
- 结构化输出校验：生成 JSON、正则约束内容时，`StructuredOutputManager` 检查生成 token 是否符合 grammar；意外违反约束时，请求以错误状态结束。
- Pooling 输出：Embedding、分类等任务不生成普通文本 token，而是返回向量或分类结果；得到结果后即可结束。
- 可恢复流式会话：请求满足当前停止条件后，不一定永久结束。`_handle_stopped_request()` 可能把它转为等待下一段输入，而不是释放整个会话。
- logprob：模型为 token 给出的对数概率，可用于观察模型对生成结果的置信程度。
- prompt logprob：prompt 中各 token 的对数概率，与新生成 token 的 logprob 不同。
- 延迟释放：异步执行时，某些 GPU 操作可能仍在写 KV block。本函数确认对应执行轮次完成后，才通过 `_drain_deferred_frees()` 安全归还这些 block。
- KV/EC Connector：分别处理远程 KV Cache和编码器缓存传输。本函数接收传输成功、失败或 block 无效等结果，并决定重新计算还是让请求报错。
- 缓存加载失败：如果配置允许重新计算，会调整请求的计算进度，使无效 token 重新计算；否则以 `FINISHED_ERROR` 结束请求。
- 编码器缓存释放：图片等输入已经实际执行完后，函数才释放不再需要的编码器缓存，避免提前释放仍被 GPU 使用的数据。
- MoE 专家路由：如果启用相关返回功能，函数整理本轮 token 使用了哪些专家，并随请求输出返回。
- KV Cache event：缓存分配、释放或远程传输产生的事件，处理完成后统一发布。
- 无部分 Prefill 输出：如果本轮只处理了部分 prompt、没有生成 token、Pooling 结果或停止事件，通常不会创建 `EngineCoreOutput`。
- 该函数不负责 detokenize：它主要返回 token ID，后面的 `OutputProcessor` 才把 token ID 转换为文字，并处理字符串级 stop 条件。

25 核心流程：
```
1. 读取本轮计划和模型结果
   ├─ sampled token
   ├─ logprob / pooling output
   ├─ Connector 结果
   └─ 性能与 CUDA Graph 信息
                  ↓
2. 处理批次级状态
   ├─ 安全释放延期归还的 KV block
   ├─ 处理远程 KV 加载失败或无效 block
   └─ 保存 MoE 专家路由数据
                  ↓
3. 逐个处理本轮请求
   ├─ 减少 num_in_flight_tokens
   ├─ 忽略已结束、失效或应丢弃的旧结果
   ├─ 统计推测 token 的接受与拒绝
   ├─ 回滚被拒绝的推测计算进度
   ├─ 追加新生成 token
   ├─ 检查长度、EOS、stop token和重复条件
   ├─ 校验结构化输出
   └─ 提取 logprob、Pooling 和其他附加结果
                  ↓
4. 处理停止请求
   ├─ 可恢复会话 → 等待下一段输入
   └─ 真正结束 → 释放 KV/编码器缓存和请求状态
                  ↓
5. 构造 EngineCoreOutput
   └─ 按 request.client_index 分组
                  ↓
6. 清理请求队列
   ├─ 从 running 移除已停止请求
   └─ 从 waiting 移除结束的抢占请求
                  ↓
7. 处理额外错误与 Connector 状态
   ├─ grammar 编译错误
   ├─ KV/EC 传输错误
   └─ 更新远程缓存传输完成状态
                  ↓
8. 汇总事件与统计
   ├─ KV Cache 事件
   ├─ 推测解码统计
   ├─ Connector 统计
   └─ 性能与 CUDA Graph 统计
                  ↓
返回 dict[client_index, EngineCoreOutputs]
```
一个重要细节是：`schedule()` 在模型执行前已经先增加了 `num_computed_tokens`；本函数负责根据真实执行结果进行确认或修正。例如推测 token 被拒绝时，需要把多算的进度回滚。

6. **在整个项目中的定位**：
```
Scheduler.add_request()
请求进入等待队列
        ↓
EngineCore.step()
        ↓
Scheduler.schedule()
生成本轮 SchedulerOutput
        ↓
ModelExecutor.execute_model()
        ↓
Worker / ModelRunner
执行真正的 GPU 模型计算
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()              ← 当前函数
    ├─ 将结果写回 Request
    ├─ 修正推测解码和异步状态
    ├─ 判断请求是否结束
    ├─ 释放请求缓存资源
    └─ 生成 EngineCoreOutputs
        ↓
EngineCore.step() 返回
        ↓
EngineCoreClient / LLMEngine
        ↓
OutputProcessor
detokenize、处理字符串 stop 条件
并生成用户可见结果
        ↓
未结束请求进入下一轮 schedule()
```
因此，`schedule()` 负责决定“这一轮计算什么”，`ModelRunner` 负责“真正计算”，而 `update_from_output()` 负责“解释计算结果、更新请求并决定下一步”。  