本文基于的vllm commit hash为：`c16bb6068f70878fb8a2f7c4d6cda95cd03a778b`
# day7
InputProcessor把人类可读的 prompt（字符串 / 消息列表 / 多模态）变成引擎能处理的整数 token 序列，同时把采样参数补全、校验，最终打包成一个 `EngineCoreRequest`。
* 主要源码在：`vllm/v1/engine/input_processor.py`（约 600 行）
* 渲染（实际 tokenize 的地方）：`vllm/renderers/` 下的 `renderer.render_cmpl(...)`
## input_processor.py
**路径**：/home/dongmingzhe/vllm/vllm/v1/engine/input_processor.py
**简介**：它把用户提交的原始请求（文本、token id、多模态数据等加上 SamplingParams）翻译成引擎内部统一的 EngineCoreRequest——中间做参数校验（采样参数与模型支持的任务、LoRA、输入长度、多模态输入合法性），给请求分配唯一的内部 request_id，补全默认值（如 max_tokens），并登记多模态数据的缓存信息。除了 process_inputs 这个主流程，文件里还有几个校验辅助方法（`_validate_params、_validate_lora、_validate_prompt_len` 等），职责就是保证交给 EngineCore 的请求一定是合法、完整、可直接调度的。
**本小节阅读的内容**：
* 重点看`process_inputs`这个函数
* 略看InputProcessor类的构造函数
### InputProcessor类
**作用**：`InputProcessor` 是用户输入进入 vLLM 核心引擎前的“转换器和检查员”：它把文本、token、多模态输入及采样/池化参数整理成统一的 `EngineCoreRequest`。它会校验模型任务、参数、LoRA、输入长度和 token 范围，并补全默认生成配置、整理多模态特征，同时为请求生成唯一的内部 ID。它只负责准备合法、标准化的请求，不负责调度请求或执行 GPU 推理。
#### 构造函数
##### part1.参数列表
```python
class InputProcessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
```
这是 `InputProcessor` 的构造函数，创建对象时自动执行，主要接收三个参数：
- `vllm_config`：vLLM 的总配置，包含模型、缓存、调度器和 LoRA 等配置。
- `renderer`：输入渲染器，负责把文本或聊天输入整理成引擎需要的格式；没有传入时会自动创建。
- `mm_registry`：多模态能力注册表，用于判断当前模型是否支持图片、视频等输入。
其中 `*` 表示 `mm_registry` 必须使用参数名传入。
##### part2.保存各类引擎配置
```python
self.vllm_config = vllm_config
self.model_config = model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.scheduler_config = vllm_config.scheduler_config
self.speculative_config = vllm_config.speculative_config
self.structured_outputs_config = vllm_config.structured_outputs_config
self.observability_config = vllm_config.observability_config
```
这一部分从总配置中取出后续经常使用的子配置，并保存为成员变量。
这样后续处理请求时，不需要反复从 `vllm_config` 中查找。
##### part3.准备生成参数的校验与默认配置
```python
# Load the custom logits processor classes once; the returned callable
# runs their validate_params hooks per request at admission.
self.validate_logits_processors_params = (
	self._build_logits_processors_params_validator()
)

self.generation_config_fields = model_config.try_get_generation_config()
```
这一部分提前准备与文本生成有关的规则：
- `validate_logits_processors_params`：用于检查每个请求中的自定义 logits 处理参数是否合法。
- `generation_config_fields`：从模型配置中读取默认生成设置，例如默认的停止 token 等。
这些规则会在真正接收请求时使用，而不是在构造函数中立即校验某个请求。
##### part4.准备渲染器
```python
self.renderer = renderer or renderer_from_config(vllm_config)
```
`renderer` 负责把用户输入整理为 vLLM 引擎能够处理的内部格式。
这里有两种情况：
- 调用者传入了 `renderer`：直接使用。
- 没有传入：根据 `vllm_config` 自动创建。
##### part5.初始化多模态输入能力
```python
self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
self.mm_encoder_cache_size = 0
self.skip_prompt_length_check = False

if self.supports_mm_inputs:
    mm_budget = MultiModalBudget(vllm_config, mm_registry)
    self.mm_encoder_cache_size = mm_budget.encoder_cache_size
    self.skip_prompt_length_check = (
        mm_budget.processor.info.skip_prompt_length_check
    )
    mm_budget.reset_cache()
```
这一部分判断当前模型是否支持图片、视频等多模态输入。
如果支持，就进一步计算：
- 多模态编码结果可以使用多少缓存空间。
- 某些多模态输入是否需要跳过普通的长度检查。
如果不支持，就保留前面设置的默认值。
##### part6.创建异步输入处理接口
```python
self.process_inputs_async = make_async(
    self.process_inputs,
    executor=self.renderer._executor,
)
```
文本分词和图片处理可能比较耗时，并且会阻塞当前线程。因此这里把同步的 `process_inputs()` 包装成异步版本 `process_inputs_async()`。
异步调用者可以在线程池中执行输入处理，避免阻塞整个服务的事件循环。
#### process_inputs函数
```python
def process_inputs(
	self,
	request_id: str,
	prompt: PromptType | EngineInput,
	params: SamplingParams | PoolingParams,
	supported_tasks: tuple[SupportedTask, ...],
	arrival_time: float | None = None,
	lora_request: LoRARequest | None = None,
	tokenization_kwargs: dict[str, Any] | None = None,
	trace_headers: Mapping[str, str] | None = None,
	priority: int = 0,
	data_parallel_rank: int | None = None,
	resumable: bool = False,
	session_id: str | None = None,
) -> EngineCoreRequest:
	self._validate_params(params, supported_tasks)
	self._validate_lora(lora_request)

	parallel_config = self.vllm_config.parallel_config
	dp_size = parallel_config.data_parallel_size
	dp_local_size = parallel_config.data_parallel_size_local
	num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
	if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
		raise VLLMValidationError(
			f"data_parallel_rank {data_parallel_rank} "
			f"is out of range [0, {num_ranks})."
		)

	if isinstance(prompt, dict) and "type" in prompt:
		if arrival_time is None:
			arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

		engine_input: EngineInput = prompt  # type: ignore[assignment]
	else:
		logger.warning_once(
			"Passing raw prompts to InputProcessor is deprecated "
			"and will be removed in the future. You should instead pass "
			"the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
		)

		if arrival_time is None:
			arrival_time = time.time()

		renderer = self.renderer
		model_config = self.model_config

		parsed_prompt = parse_model_prompt(model_config, prompt)
		tok_params = renderer.default_cmpl_tok_params.with_kwargs(
			**(tokenization_kwargs or {})
		)

		(engine_input,) = renderer.render_cmpl(
			[parsed_prompt],
			tok_params,
		)

	current_platform.validate_request(engine_input, params)

	encoder_input, decoder_input = split_enc_dec_input(engine_input)
	self._validate_model_inputs(encoder_input, decoder_input)

	# Mypy can be conservative for TypedDict unions; normalize access.
	if decoder_input["type"] == "embeds":
		prompt_embeds = decoder_input["prompt_embeds"]
		prompt_token_ids = decoder_input.get("prompt_token_ids")
		prompt_is_token_ids = decoder_input.get("is_token_ids")
	else:
		prompt_token_ids = decoder_input["prompt_token_ids"]
		prompt_embeds = None
		prompt_is_token_ids = None

	sampling_params = None
	pooling_params = None
	if isinstance(params, SamplingParams):
		# TODO: can we avoid cloning here in multiproc case?
		sampling_params = params.clone()
		prompt_len = length_from_prompt_token_ids_or_embeds(
			prompt_token_ids, prompt_embeds
		)
		if not 0 <= sampling_params.routed_experts_prompt_start <= prompt_len:
			raise VLLMValidationError(
				f"routed_experts_prompt_start must be between 0 and "
				f"the prompt length ({prompt_len}), inclusive.",
				parameter="routed_experts_prompt_start",
				value=sampling_params.routed_experts_prompt_start,
			)
		# If unset max tokens, then generate up to the max_model_len.
		if sampling_params.max_tokens is None:
			sampling_params.max_tokens = (
				self.model_config.max_model_len - prompt_len
			)

		sampling_params.update_from_generation_config(
			self.generation_config_fields,
			self.renderer.get_eos_token_id(),
		)
		if self.tokenizer is not None:
			sampling_params.update_from_tokenizer(self.tokenizer)
		if sampling_params.trace_decode_token_ids:
			self._normalize_trace_replay_params(sampling_params, prompt_len)
	else:
		pooling_params = params.clone()

	# Multimodal related.
	mm_features: list[MultiModalFeatureSpec] | None = None

	if decoder_input["type"] == "multimodal":
		decoder_mm_inputs = decoder_input["mm_kwargs"]
		decoder_mm_positions = decoder_input["mm_placeholders"]
		decoder_mm_hashes = decoder_input["mm_hashes"]

		if not all(
			isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
		):
			raise ValueError(
				f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
				"This is likely due to an incorrect custom implementation of "
				"MultiModalProcessor.apply method."
			)

		# Merge and flatten multimodal placeholders, hashes and inputs
		# from dictionaries to lists, and sort them by each item's position
		# in the input sequence.
		sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

		mm_features = []
		for modality, idx in sorted_mm_idxs:
			base_mm_hash = decoder_mm_hashes[modality][idx]
			mm_features.append(
				MultiModalFeatureSpec(
					data=decoder_mm_inputs[modality][idx],
					modality=modality,
					identifier=self._get_mm_identifier(
						base_mm_hash,
						lora_request,
					),
					mm_position=decoder_mm_positions[modality][idx],
					mm_hash=base_mm_hash,
				)
			)

	return EngineCoreRequest(
		request_id=request_id,
		prompt_token_ids=prompt_token_ids,
		prompt_embeds=prompt_embeds,
		prompt_is_token_ids=prompt_is_token_ids,
		mm_features=mm_features,
		sampling_params=sampling_params,
		pooling_params=pooling_params,
		arrival_time=arrival_time,
		lora_request=lora_request,
		cache_salt=decoder_input.get("cache_salt"),
		priority=priority,
		data_parallel_rank=data_parallel_rank,
		trace_headers=trace_headers,
		resumable=resumable,
		session_id=session_id,
	)
```
1. **主要作用**
把用户输入、生成参数和请求元数据进行校验与标准化，最终组装成底层引擎统一使用的 `EngineCoreRequest`。它是请求进入调度器前的“输入检查与打包站”，不负责调度和 GPU 推理。
2. **输入（参数）**
    - `request_id`：请求编号。
    - `prompt`：原始 `PromptType` 或处理过的 `EngineInput`。
    - `params`：生成任务使用的 `SamplingParams`，或向量任务使用的 `PoolingParams`
    - `supported_tasks`：当前模型支持的任务类型。
    - `arrival_time`：请求到达时间。
    - `lora_request`：可选的 LoRA 配置。
    - `tokenization_kwargs`：额外的分词参数。
    - `trace_headers`：链路追踪信息。
    - `priority`：请求优先级。
    - `data_parallel_rank`：指定的数据并行引擎编号。
    - `resumable`、`session_id`：可恢复请求和会话相关信息。
3. **输出（改变了什么值）**
返回一个 `EngineCoreRequest`，其中包含标准化后的 token、prompt embedding、多模态特征、生成或池化参数以及请求元数据。函数不会把请求加入 Scheduler，也不会修改模型权重；它主要创建参数副本并补充默认值，但在 `PoolingParams.task` 为空时可能为原参数设置默认任务。
4. **术语解释**
    - `PromptType`：面向用户的输入形式，可以是文本、token ID 或多模态输入。
    - `EngineInput`：Renderer 生成的标准输入格式，其中已经整理好 token、多模态内容和相关位置。当前代码更推荐调用者先生成 `EngineInput`，直接传原始 prompt 的方式已经被标记为将来废弃。
    - Renderer：把原始文本或其他输入转换成 `EngineInput` 的组件。传入原始 prompt 时，`process_inputs()` 会临时调用它完成处理。
    - `PoolingParams`：embedding、分类等非生成任务使用的参数，通常控制如何从模型输出中提取向量或分类结果。
    - 默认生成参数：如果没有设置 `max_tokens`，函数会根据模型最大上下文长度和 prompt 长度计算可生成的最大 token 数，并补充模型 generation config 和 tokenizer 提供的默认设置。
    - encoder 输入：编码器—解码器模型中交给 encoder 处理的部分，例如翻译模型的原始句子。
    - decoder 输入：交给 decoder 处理的输入。对于普通自回归大模型，主要输入通常就在这一侧。
    - `prompt_token_ids`：prompt 分词后得到的整数 ID 序列，是模型实际读取的文本表示。
    - `prompt_embeds`：已经计算好的输入向量。使用这种输入时，可以不再通过普通 embedding 层查询 token 向量。
    - 多模态哈希：用于识别重复多模态内容，例如相同图片，以便后续复用缓存；使用特殊 LoRA 配置时，LoRA 信息也可能参与生成该标识。
    - LoRA：一种轻量模型适配技术。传入 `lora_request` 时，函数会检查引擎是否已经启用 LoRA。
    - `priority`：请求优先级。当前函数只负责把它写入请求，真正如何按优先级调度由 Scheduler 决定。
    - `trace_headers`：保存链路追踪数据，用于分析一个请求经过各组件时的耗时。
    - `session_id`：表示请求所属的会话，便于把相关请求关联起来。
    - `EngineCoreRequest`：前端与 EngineCore 之间使用的标准请求对象，将 prompt、参数、LoRA、多模态特征和调度元数据集中保存起来。
    - `request_id`：此时仍是调用者提供的外部 ID。内部随机 ID 并不是在 `process_inputs()` 中生成，而是随后由 `assign_request_id()` 添加。
    - Scheduler：vLLM 的调度器，负责决定每一轮运行哪些请求。`process_inputs()` 只生成请求对象，不会直接调用 Scheduler。
5. 核心流程
```
第一步：校验请求参数
    ├─ 检查 SamplingParams 或 PoolingParams
    ├─ 检查模型是否支持该任务
    └─ 检查 LoRA 和 data_parallel_rank

第二步：统一输入格式
    ├─ 已经是 EngineInput：直接使用
    └─ 原始 PromptType：通过 Renderer 转换

第三步：校验模型输入
    ├─ 执行当前硬件平台的请求检查
    ├─ 拆分 encoder 和 decoder 输入
    ├─ 检查 prompt 长度
    └─ 检查 token ID 是否越界

第四步：整理模型输入
    ├─ 提取 prompt_token_ids
    └─ 或提取 prompt_embeds

第五步：处理任务参数
    ├─ 复制 SamplingParams 或 PoolingParams
    ├─ 补充 max_tokens
    └─ 补充模型和 tokenizer 的默认生成配置

第六步：整理多模态特征
    ├─ 收集图片、视频等处理结果
    ├─ 生成缓存标识
    └─ 按输入位置排序

第七步：构造并返回 EngineCoreRequest
```
最终的EngineCoreRequest大概长什么样？
```text
EngineCoreRequest(
    request_id      = "0-a3f9b21c",
    prompt_token_ids= [104198, 100644, 100431, ...],   # ★ 这就是 tokenize 的结果
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64),
    mm_features     = [],          # 多模态特征，空
    arrival_time    = 1726900000.0,
    ...
)
```
6. **在整个项目中的定位**
`process_inputs()` 位于用户 API 与底层 EngineCore 之间，是请求正式进入调度系统前的标准化入口：
```
LLM.generate() / LLM.chat()
              ↓
OfflineInferenceMixin
整理 prompt 和调用参数
              ↓
LLMEngine.add_request()
              ↓
InputProcessor.process_inputs()       ← 当前函数
校验并生成 EngineCoreRequest
              ↓
InputProcessor.assign_request_id()
生成唯一的内部请求 ID
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
              ↓
LLMEngine.step()
调度 → 模型计算 → 输出处理
```
## 阶段性自检
**1.tokenize 具体发生在哪个函数的哪一行？用的是哪个库？**
`input_processor.py` 本身不直接调用 tokenizer；当传入原始文本时，`process_inputs()` 在 [input_processor.py (line 358)](/home/dongmingzhe/vllm/vllm/v1/engine/input_processor.py:358) 调用 `renderer.render_cmpl()`。
实际分词调用链是：
```
process_inputs()
→ BaseRenderer.render_cmpl()          base.py:1167
→ tokenize_prompts()                  base.py:720
→ _tokenize_singleton_prompt()        base.py:604
→ _tokenize_prompt()                  base.py:560
→ tokenizer(prompt["prompt"], ...)    base.py:570  ← 真正 tokenize
```
真正执行分词的是 [base.py (line 570)](/home/dongmingzhe/vllm/vllm/renderers/base.py:570)。默认通常使用 Hugging Face `transformers` 库加载的 tokenizer；快速 tokenizer 底层会使用 Hugging Face `tokenizers` 库。
注意：如果传入的已经是 `EngineInput` 或 `prompt_token_ids`，这里不会再次 tokenize；标准离线流程通常也会在进入 `process_inputs()` 之前由 Renderer 完成分词。

**2.`EngineCoreRequest` 里有没有原始文本？如果没有，那"原始 prompt"在什么时候还会被用到？（提示：想想 `RequestOutput.prompt`）**
`EngineCoreRequest` 中没有原始文本，只保留 `prompt_token_ids`、embedding、多模态特征和请求参数等引擎计算所需数据。
原始 prompt 会在 `LLMEngine.add_request()` 中单独提取，并传给 `OutputProcessor.add_request()` 保存到 `RequestState`；生成最终结果时，再用于填写 `RequestOutput.prompt`，方便用户看到输入原文。

**3.还是根据上述文件，回答问题： `params.max_tokens is None` 时补成 `max_model_len - prompt_len`，这个设计合理吗？如果 prompt 特别长会怎样？注意控制回答字数，直击核心但又不能遗漏重要内容。**
这个设计合理：当用户没有指定 `max_tokens` 时，就把模型上下文窗口中除 prompt 外的剩余空间全部作为最大生成长度，保证“输入 token + 输出 token”不超过 `max_model_len`。
如果 prompt 特别长：
- `prompt_len > max_model_len`：直接报错。
- `prompt_len == max_model_len`：生成任务会报错，因为连 1 个输出 token 的空间都没有。
- `prompt_len` 接近上限：只能生成很少的 token，例如只剩 10 个位置就最多生成 10 个。
但 prompt 较短时，这个默认值可能很大，带来较长耗时，因此实际使用通常建议明确设置 `max_tokens`。

**4.内部 `request_id` 加了随机后缀，那 `OutputProcessor` 怎么知道该把输出还给哪个用户请求？**
`assign_request_id()` 不会丢掉原始 ID，而是：

```
external_req_id = 原始用户 ID，例如 "0"
request_id      = 内部唯一 ID，例如 "0-a1b2c3d4"
```

`OutputProcessor` 创建 `RequestState` 时会同时保存两者，并维护“外部 ID → 内部 ID”的映射。EngineCore 返回内部 ID 后，`OutputProcessor` 用它找到对应状态；构造最终 `RequestOutput` 时，再把 `external_req_id` 填入 `RequestOutput.request_id` 返回给用户。

**5.如果你直接传 `prompt_token_ids` 而不是字符串，能跳过 tokenize 吗？这有什么用？**
可以。传入 `prompt_token_ids` 后，Renderer 会识别输入已经完成分词，跳过 tokenizer，`InputProcessor` 只检查 token 数量和 ID 是否越界。
用途主要是：
- 减少 CPU 分词开销，提高批量推理吞吐量。
- 使用外部或自定义 tokenizer。
- 精确控制特殊 token 和输入序列。
- 配合 `skip_tokenizer_init=True`，避免加载 tokenizer。
注意 token ID 必须与模型词表一致，特殊 token 也要自行处理；如果没有同时提供原始文本，最终 `RequestOutput.prompt` 可能是 `None`。
# day8-InprocClient与EngineCore
相关角色的核心职责：
- **`EngineCoreClient`**：抽象出"怎么把请求送进去、怎么把结果取出来"，屏蔽单进程/多进程差异（见core_client.py）
- **`EngineCore`**：★ **真正的引擎内循环**，持有 `Scheduler` 和 `ModelExecutor`，实现"调度 → 执行 → 回写"这三步（见core.py）
## core_client.py
**路径**：/home/dongmingzhe/vllm/vllm/v1/engine/core_client.py
**简介**：`core_client.py` 的核心作用是：在上层 `LLMEngine`/`AsyncLLM` 和底层 `EngineCore` 之间提供统一通信接口，负责发送请求、接收输出及执行缓存、LoRA、休眠等控制操作，并隐藏“同进程还是多进程、同步还是异步”的差异。示意图如下：
```text
LLMEngine / AsyncLLM
        ↓
EngineCoreClient
        ↓
EngineCore
调度 → 模型执行 → 输出回传
```
该文件主要包含了9个类：
* `EngineCoreClient`：抽象基类，定义添加请求、获取输出、取消请求等统一接口，并根据配置创建合适的客户端。**只规定接口，不直接通信**。
* `InprocClient`：在当前进程中直接创建并调用 `EngineCore`。**无进程通信，结构简单，适合关闭多进程模式的离线推理**。
* `BackgroundResources`：管理并清理后台进程、ZMQ socket 和异步任务。**辅助资源管理类，不处理推理请求**。
* `ElasticScalingCache`：暂存弹性扩缩容过程中的 EngineCore 信息和通知状态。**辅助状态类，主要用于弹性专家并行**。
* `MPClient`：多进程客户端的公共基类，负责启动或连接 EngineCore 进程、ZMQ 通信、序列化和生命周期管理。**不直接面向用户，由同步或异步子类继承**。
* `SyncMPClient`：同步多进程客户端，发送请求后以阻塞方式取得结果。**主要供同步 `LLM` 使用，后台线程接收输出**。
* `AsyncMPClient`：异步多进程客户端，通过 `asyncio` 队列和任务收发数据。**主要供在线异步 `AsyncLLM` 使用，不阻塞事件循环**。
* `DPAsyncMPClient`：支持多个数据并行 EngineCore 的异步客户端，并协调各引擎运行状态。**默认假设请求已由外部负载均衡器分配**。
* `DPLBAsyncMPClient`：在 `DPAsyncMPClient` 基础上增加内部负载均衡。**根据等待请求数、运行请求数和 KV Cache 使用率选择较空闲的 EngineCore**。
总结如下：
```text
InprocClient
└─ 同进程，直接调用 EngineCore

MPClient
├─ SyncMPClient
│  └─ 多进程 + 同步，供 LLM 使用
└─ AsyncMPClient
   ├─ 普通异步客户端
   └─ DPAsyncMPClient
      ├─ 外部负责数据并行负载均衡
      └─ DPLBAsyncMPClient
         └─ 客户端内部负责负载均衡
```
**本小节阅读的内容**：
* 看`EngineCoreClient`这个整体IDE抽象基类
* 重点看离线默认使用的`InprocClient`类以及相关函数
* 其他类暂时不看
### EngineCoreClient类
**作用**：`EngineCoreClient` 是上层 `LLMEngine` 与底层 `EngineCore` 之间的统一通信接口，负责提交和取消请求、获取推理输出，以及转发缓存管理、LoRA、休眠等控制命令；它通过不同子类屏蔽同进程/多进程、同步/异步等运行方式的差异。
#### make_client函数
```python
@staticmethod
def make_client(
	multiprocess_mode: bool,
	asyncio_mode: bool,
	vllm_config: VllmConfig,
	executor_class: type[Executor],
	log_stats: bool,
	renderer: BaseRenderer | None = None,
) -> "EngineCoreClient":
	# renderer is passed through to the multiprocess clients, which start
	# the frontend MM warmup (renderer.start_mm_warmup_in_background) once
	# the engine-core processes have been forked, so warmup overlaps
	# engine-core load without a live thread at fork() time. In-process
	# clients do not take a renderer: the engine core is built in this
	# process, so there is no other process to overlap with and the MM
	# warmup runs synchronously inside renderer.warmup() instead.
	# TODO: support this for debugging purposes.
	if asyncio_mode and not multiprocess_mode:
		raise NotImplementedError(
			"Running EngineCore in asyncio without multiprocessing "
			"is not currently supported."
		)

	if multiprocess_mode and asyncio_mode:
		return EngineCoreClient.make_async_mp_client(
			vllm_config,
			executor_class,
			log_stats,
			renderer=renderer,
		)

	if multiprocess_mode and not asyncio_mode:
		return SyncMPClient(
			vllm_config,
			executor_class,
			log_stats,
			renderer=renderer,
		)

	return InprocClient(vllm_config, executor_class, log_stats)
```
1. **主要作用**
`make_client()` 是一个“客户端工厂”：根据是否启用多进程和异步模式，自动创建合适的 `EngineCoreClient` 子类，让上层代码不必自己判断该使用 `InprocClient`、`SyncMPClient` 还是异步多进程客户端。

2. **输入（参数）**
    - `multiprocess_mode`：是否让 `EngineCore` 运行在独立进程中。
    - `asyncio_mode`：是否使用异步调用方式。
    - `vllm_config`：vLLM 的完整配置。
    - `executor_class`：执行模型计算的 Executor 类型。
    - `log_stats`：是否记录运行统计信息。
    - `renderer`：可选的输入渲染器，主要用于多进程模式下的多模态预热。

3. **输出（改变了什么值）**
返回一个具体的 `EngineCoreClient` 子类实例；函数不会修改传入参数，但创建客户端时可能同时创建 `EngineCore`，或者启动后台 EngineCore 进程并建立通信资源。不支持“异步但不开多进程”的组合，此时会抛出 `NotImplementedError`。

4. **术语解释**
    - 工厂方法：根据配置决定创建哪个具体类的方法。调用者只需调用 `make_client()`，不需要直接写复杂的分支判断。
    - `@staticmethod`：表示该方法属于类，但不依赖某个 `EngineCoreClient` 对象，因此没有 `self` 参数，可以直接通过 `EngineCoreClient.make_client()` 调用。
    - `EngineCoreClient`：上层引擎与底层 `EngineCore` 之间的统一通信接口，屏蔽同步、异步和多进程等运行方式的差异。
    - `multiprocess_mode`：决定 `EngineCore` 是否运行在独立进程中。开启后，客户端需要通过进程间通信发送请求和接收输出。
    - `asyncio_mode`：决定上层是否使用 Python `asyncio` 异步接口。异步调用可以在等待结果时处理其他请求，适合在线服务。
    - `InprocClient`：当多进程和异步模式都关闭时使用。它在当前进程中直接创建并调用 `EngineCore`，不需要进程间通信。
    - `SyncMPClient`：当开启多进程但关闭异步模式时使用。它通过 ZMQ 与后台 EngineCore 进程通信，并以同步方式向上层提供结果。
    - 异步多进程客户端：当多进程和异步模式同时开启时，通过 `make_async_mp_client()` 进一步选择。
    - `make_async_mp_client()`：异步客户端的二级工厂。单个数据并行引擎时返回 `AsyncMPClient`；多个数据并行引擎时，根据负载均衡配置返回 `DPAsyncMPClient` 或 `DPLBAsyncMPClient`。
    - `VllmConfig`：包含模型、缓存、调度、并行和多模态等配置，客户端及 EngineCore 初始化时都会使用。
    - `executor_class`：负责真正执行模型计算的 Executor 类型。它决定 EngineCore 通过什么执行器调用 worker 和模型。
    - `log_stats`：控制是否收集和记录吞吐量、延迟、KV Cache 使用率等运行指标。
    - `renderer`：把文本、聊天消息和多模态输入转换为引擎输入的组件。在多进程模式下，它还用于启动前端多模态预热。
    - 多模态预热：提前初始化图片、视频等输入的处理组件，减少第一次多模态请求的等待时间。
    - `fork`：创建子进程的一种方式。代码会等 EngineCore 子进程创建完成后再启动前端多模态预热线程，避免带着活跃线程执行 `fork` 所产生的问题。
    - 预热与加载重叠：多进程模式下，多模态预热可以和 EngineCore 加载模型同时进行，从而减少整体启动时间。
    - `NotImplementedError`：表示这种运行组合在当前版本中尚未实现。

5. 核心流程
```
开始
  ↓
asyncio=True 且 multiprocess=False？
  ├─ 是 → 抛出 NotImplementedError
  └─ 否
       ↓
multiprocess=True 且 asyncio=True？
  ├─ 是 → make_async_mp_client()
  │        ├─ 普通异步 → AsyncMPClient
  │        └─ 数据并行 → DPAsyncMPClient / DPLBAsyncMPClient
  └─ 否
       ↓
multiprocess=True 且 asyncio=False？
  ├─ 是 → SyncMPClient
  └─ 否 → InprocClient
```

6. **在整个项目中的定位**
它位于 `LLMEngine`/`AsyncLLM` 初始化和 `EngineCore` 创建之间，负责选择双方采用哪种连接方式：
```
LLM 或在线服务创建引擎
            ↓
LLMEngine / AsyncLLM 初始化
            ↓
EngineCoreClient.make_client()        ← 当前函数
            ↓
     根据运行模式选择
      ├─ InprocClient
      │      ↓ 直接调用
      │   EngineCore
      │
      ├─ SyncMPClient
      │      ↓ ZMQ 同步通信
      │   EngineCoreProc
      │
      └─ AsyncMPClient及其DP子类
             ↓ ZMQ 异步通信
          EngineCoreProc
            ↓
Scheduler → Executor → Worker → 模型
```

后续上层统一通过返回的客户端调用：
```
add_request()     提交请求
get_output()      获取结果
abort_requests()  取消请求
```
因此，这个函数本身不处理推理请求，而是在引擎启动阶段决定“上层以后通过哪种方式与 EngineCore 通信”。
### InprocClient类
**作用**：`InprocClient` 是 `LLMEngine` 与同一进程内 `EngineCore` 之间的轻量客户端：它不使用 ZMQ 或后台 EngineCore 进程，而是直接向 `EngineCore` 添加、取消请求，并通过调用 `step_fn()` 推动一轮调度和模型计算来获取输出；同时，它将缓存管理、休眠唤醒、LoRA、性能分析和权重版本等控制操作直接转发给 `EngineCore`，主要用于关闭多进程模式的同步离线推理。
#### 构造函数
```python
class InprocClient(EngineCoreClient):
    """InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # Takes no renderer: this class has no _start_mm_warmup (only MPClient
    # does), so a renderer would never be used here. EngineCore is built in
    # this process and MM warmup runs synchronously inside renderer.warmup().
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        *,
        executor_fail_callback: Callable | None = None,
    ):
        self.engine_core = EngineCore(
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback=executor_fail_callback,
        )
```
这个构造函数接收 vLLM 总配置 `vllm_config`、模型执行器类型 `executor_class`、是否记录统计信息 `log_stats`，以及可选的执行器失败回调 `executor_fail_callback`，然后直接在当前进程中创建一个 `EngineCore` 并保存到 `self.engine_core`。
```text
创建 InprocClient
      ↓
同进程创建 EngineCore
      ↓
初始化调度器和模型执行器等核心组件
      ↓
后续直接调用 self.engine_core
```

它不创建独立 EngineCore 进程，也不建立 ZMQ 通信；`renderer` 不作为参数传入，因为进程内模式的多模态预热由外部同步完成。构造函数本身没有返回值，但会让 `InprocClient` 获得可直接操作的底层推理核心。
#### get_out函数
```python
def get_output(self) -> EngineCoreOutputs:
	outputs, model_executed = self.engine_core.step_fn()
	self.engine_core.post_step(model_executed=model_executed)
	return outputs and outputs.get(0) or EngineCoreOutputs()
```
1. **主要作用**
`get_output()` 直接驱动同进程的 `EngineCore` 完成一轮“调度 → 模型计算 → 状态更新”，执行必要的轮后处理，然后取出属于当前客户端的结果。

2. **输入（参数）**
没有需要用户传入的参数；`self` 表示当前 `InprocClient`，通过其中的 `self.engine_core` 操作底层推理核心。

3. **输出（改变了什么值）**
返回一个 `EngineCoreOutputs`；如果本轮没有可返回结果，就返回空的 `EngineCoreOutputs()`。执行过程中会推进请求的计算进度、生成新 token，并可能改变调度器和 KV Cache 状态，但不会修改模型权重。

4. **术语解释**
- 一轮执行：表示一次引擎迭代，而不是把所有请求完整运行结束。一轮可以并行处理多个请求当前需要计算的一批 token。
- `step_fn`：`EngineCore` 初始化时选择的执行函数。普通模式下指向 `EngineCore.step()`；启用 batch queue 时指向 `step_with_batch_queue()`。
- `EngineCore.step()`：一次执行调度、模型计算和结果回写的核心方法，主要过程是：
```text
Scheduler.schedule()
        ↓
ModelExecutor.execute_model()
        ↓
Scheduler.update_from_output()
```
- `outputs`：`step_fn()` 返回的结果字典。字典按客户端编号保存输出；同进程模式通常只有编号`0`。
- `model_executed`：布尔值，表示本轮是否真的安排了 token 并执行了模型计算。没有请求或没有可调度 token 时可能为 `False`。
- `post_step()`：一轮模型计算后的补充处理。当前主要用于获取投机解码产生的草稿 token，并把它们更新到 Scheduler，供下一轮使用。
- 投机解码：先由较快的草稿模型预测多个候选 token，再让主模型验证，从而减少主模型的生成轮数。
- `outputs.get(0)`：从结果字典中取得客户端编号 `0` 的输出。`InprocClient` 只有一个本地客户端，所以读取这一项。
- `outputs and outputs.get(0) or EngineCoreOutputs()`：Python 的简写逻辑：
```text
outputs 非空且存在编号 0
    → 返回 outputs[0]
outputs 为空或没有编号 0
    → 返回空 EngineCoreOutputs()
```
- `EngineCoreOutputs`：一轮 EngineCore 的批量输出，可能包含多个请求的新 token、完成状态、调度统计和时间戳。
- 空的 `EngineCoreOutputs()`：表示这一轮没有用户侧输出，并不一定代表发生错误，也不一定代表所有请求都已完成。
- 请求计算进度：Scheduler 会更新每个请求已经计算的 token 数，并把未完成请求留给后续轮次继续处理。
- KV Cache 状态：调度过程中可能为请求分配、复用或释放 KV Cache block；这些变化由 Scheduler 和 KVCacheManager 完成。
- 阻塞行为：在同步的 `InprocClient` 中，`get_output()` 会等待本轮模型计算和结果处理完成后再返回，但不会等待所有请求全部完成。

5. 核心流程
```
调用 self.engine_core.step_fn()
        ↓
调度并执行本轮模型计算
        ↓
得到 outputs 和 model_executed
        ↓
调用 post_step(model_executed)
完成投机解码等轮后处理
        ↓
从 outputs 中取客户端 0 的结果
        ↓
有结果 → 返回对应 EngineCoreOutputs
无结果 → 返回空 EngineCoreOutputs
```

6. **在整个项目中的定位**
它是同步离线推理主循环与 `EngineCore` 之间的关键连接点：
```
OfflineInferenceMixin._run_engine()
        ↓ 循环调用
LLMEngine.step()
        ↓
InprocClient.get_output()             ← 当前函数
        ↓
EngineCore.step_fn()
        ├─ Scheduler.schedule()
        ├─ ModelExecutor.execute_model()
        └─ Scheduler.update_from_output()
        ↓
EngineCore.post_step()
        ↓
返回 EngineCoreOutputs
        ↓
LLMEngine.step()
        ↓
OutputProcessor.process_outputs()
反分词并组装 RequestOutput
        ↓
返回给 _run_engine()
```
`InprocClient.get_output()` 与多进程客户端的区别是：它直接调用当前进程中的 `EngineCore`；`SyncMPClient.get_output()` 则从后台进程的输出队列中读取结果。
## core.py
**路径**：/home/dongmingzhe/vllm/vllm/v1/engine/core.py
**简介**：`core.py` 是 vLLM V1 推理引擎的核心文件：它负责初始化模型执行器、创建 KV Cache 和调度器，并实现每一轮“调度请求 → 执行模型 → 回写结果”的内部循环；同时提供后台进程、ZMQ 通信、数据并行和 Ray 部署等封装。
该文件主要包含了7个类：
* `EngineCore`：最核心的引擎类，管理 Scheduler、ModelExecutor 和 KV Cache；负责添加/取消请求，并通过 `step()` 完成一轮调度、模型计算和结果更新
* `EngineShutdownState`：枚举后台引擎的生命周期状态：运行中、已请求关闭、正在关闭
* `EngineCoreProc`：`EngineCore` 的多进程封装，让 EngineCore 在后台进程的 busy loop 中运行，并通过 ZMQ 与前端收发请求和结果
* `DPEngineCoreProc`：数据并行 MoE 场景使用的后台 EngineCore，负责多个 DP rank 之间的同步、运行波次、暂停恢复和弹性专家并行扩缩容
* `EngineCoreActorMixin`：Ray Actor 的公共辅助类，负责 GPU 可见性、通信地址、初始化等待和 Actor 运行生命周期
* `DPMoEEngineCoreActor`：将 `EngineCoreActorMixin` 与 `DPEngineCoreProc` 组合起来，用于 Ray 部署下的数据并行 MoE 模型
* `EngineCoreActor`：将 `EngineCoreActorMixin` 与普通 `EngineCoreProc` 组合起来，用于 Ray 部署下的非 MoE 或非数据并行场景
类之间的关系可以简化为：
```text
EngineCore
├─ EngineCoreProc
│  ├─ DPEngineCoreProc
│  │  └─ DPMoEEngineCoreActor（再混入 EngineCoreActorMixin）
│  └─ EngineCoreActor（再混入 EngineCoreActorMixin）
│
├─ Scheduler
├─ ModelExecutor
└─ KV Cache
```
### EngineCore类
**作用**：`EngineCore` 是 vLLM V1 推理引擎的核心内循环：它在初始化时创建模型执行器、KV Cache 和调度器，接收并管理推理请求，并通过 `step()` 反复完成“调度请求 → 执行模型 → 更新请求状态和释放资源”的过程；它只处理 token、缓存和调度等核心逻辑，不直接负责用户输入预处理、文本反分词或网络通信。
#### 构造函数
##### part1.参数列表
```python
class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
```
这是 `EngineCore` 的构造函数，创建对象时自动执行。
主要参数：
- `vllm_config`：模型、缓存、调度和并行等总配置。
- `executor_class`：模型执行器类型，负责真正调用 worker 和模型。
- `log_stats`：是否记录性能统计。
- `executor_fail_callback`：模型执行器出错时的回调函数。
- `include_finished_set`：调度器输出中是否包含已完成请求集合。
##### part2.加载插件并保存基础状态
```python
# plugins need to be loaded at the engine/scheduler level too
from vllm.plugins import load_general_plugins
load_general_plugins()
self.vllm_config = vllm_config
if not vllm_config.parallel_config.data_parallel_rank_local:
	logger.info(
		"Initializing a V1 LLM engine (v%s) with config: %s",
		VLLM_VERSION,
		vllm_config,
	)
self.log_stats = log_stats
# Opaque weight version supplied by the caller.
self._weight_version = "default"
```
这一部分完成基础准备：
- 加载 vLLM 插件。
- 保存总配置和统计开关。
- 将初始模型权重版本标记为 `"default"`。
- 输出一次 V1 引擎初始化日志。
##### part3.创建模型执行器
```python
# Setup Model.
self.model_executor = executor_class(vllm_config)
self._pooler_config_logged = False
if executor_fail_callback is not None:
	self.model_executor.register_failure_callback(executor_fail_callback)
```
`model_executor` 负责模型加载和 GPU 计算，是 `EngineCore` 执行模型的入口。
如果提供了失败回调，执行器发生严重错误时会通知上层。
##### part4.初始化 KV Cache 和结构化输出
```python
self.available_gpu_memory_for_kv_cache = -1

if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
	self._eep_scale_up_before_kv_init()

# Setup KV Caches and update CacheConfig after profiling.
kv_cache_config = self._initialize_kv_caches(vllm_config)
self.structured_output_manager = StructuredOutputManager(vllm_config)
```
`_initialize_kv_caches()` 会：
- 分析模型需要什么样的 KV Cache。
- 测量可用 GPU 显存。
- 计算可以创建多少个 KV Cache block。
- 创建实际的 KV Cache。
`StructuredOutputManager` 用于支持 JSON、语法约束等结构化生成。
##### part5.创建调度器
```python
# Setup scheduler.
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
	# Encoder models without KV cache don't support
	# chunked prefill. But do SSM models?
	if vllm_config.scheduler_config.enable_chunked_prefill:
		logger.warning("Disabling chunked prefill for model without KVCache")
		vllm_config.scheduler_config.enable_chunked_prefill = False

scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
	kv_cache_config, vllm_config
)

self.scheduler: SchedulerInterface = Scheduler(
	vllm_config=vllm_config,
	kv_cache_config=kv_cache_config,
	structured_output_manager=self.structured_output_manager,
	include_finished_set=include_finished_set,
	log_stats=self.log_stats,
	block_size=scheduler_block_size,
	hash_block_size=hash_block_size,
)
```
这一部分根据配置选择 Scheduler 类型，并将 KV Cache 配置交给它。
Scheduler 后续负责：
```text
选择本轮请求
→ 分配 KV Cache block
→ 控制 token 预算
→ 检查请求结束
→ 回收资源
```
如果模型根本不使用 KV Cache，代码还会关闭不适用的分块预填充功能。
##### part6.初始化投机解码和数据连接器
```python
self.use_spec_decode = vllm_config.speculative_config is not None
self.check_for_draft_tokens = (
	self.use_spec_decode or vllm_config.model_config.is_diffusion
)
if self.scheduler.connector is not None:  # type: ignore
	self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore
if self.scheduler.ec_connector is not None:  # type: ignore
	self.model_executor.init_ec_output_aggregator()
```
这一部分初始化可选的高级功能：
- 判断是否使用投机解码。
- 决定是否需要读取草稿 token。
- 连接 Scheduler 与模型执行器之间的 KV/编码缓存传输组件。
入门阶段知道这些属于扩展功能即可。
##### part7.初始化多模态缓存和连接器握手
```python
self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
	vllm_config
)

# If a KV connector is initialized for scheduler, we want to collect
# handshake metadata from all workers so the connector in the scheduler
# will have the full context
kv_connector = self.scheduler.get_kv_connector()
if kv_connector is not None:
	# Collect and store KV connector xfer metadata from workers
	# (after KV cache registration)
	xfer_handshake_metadata = (
		self.model_executor.get_kv_connector_handshake_metadata()
	)

	if xfer_handshake_metadata:
		# xfer_handshake_metadata is list of dicts from workers
		# Each dict already has structure {(pp_rank, tp_rank): metadata}
		# Merge all worker dicts into a single dict
		content: dict[tuple[int, int], Any] = {}
		for worker_dict in xfer_handshake_metadata:
			if worker_dict is not None:
				content.update(worker_dict)
		kv_connector.set_xfer_handshake_metadata_pp_aware(content)
```
这里主要准备两类跨组件数据：
- 多模态缓存：保存图片等多模态输入的处理结果。
- KV connector 元数据：让不同 worker 或推理阶段知道如何传输 KV Cache。
普通单机文本推理通常不需要深入这一部分。
##### part8.配置批次队列和请求类型
```python
# Setup batch queue for pipeline parallelism.
# Batch queue for scheduled batches. This enables us to asynchronously
# schedule and execute batches, and is required by pipeline parallelism
# to eliminate pipeline bubbles.
self.batch_queue_size = vllm_config.max_concurrent_batches
self.batch_queue: (
	deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
) = None
if self.batch_queue_size > 1:
	logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
	self.batch_queue = deque(maxlen=self.batch_queue_size)

self.is_ec_consumer = (
	vllm_config.ec_transfer_config is None
	or vllm_config.ec_transfer_config.is_ec_consumer
)
self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"
```
如果允许同时处理多个 batch，就创建 `batch_queue`。它主要用于流水线并行，让调度和模型执行重叠，减少 GPU 等待。
同时记录当前引擎是否：
- 接收编码缓存数据。
- 运行 embedding、pooling 等非文本生成模型。
##### part9.配置前缀缓存、执行入口和运行状态
```python
self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
	caching_hash_fn = get_hash_fn_by_name(
		vllm_config.cache_config.prefix_caching_hash_algo
	)
	init_none_hash(caching_hash_fn)

	self.request_block_hasher = get_request_block_hasher(
		hash_block_size, caching_hash_fn
	)

self.step_fn = (
	self.step if self.batch_queue is None else self.step_with_batch_queue
)
self.async_scheduling = vllm_config.scheduler_config.async_scheduling

self.aborts_queue = queue.Queue[list[str]]()

self._idle_state_callbacks: list[Callable] = []
```
这一部分完成最后的运行配置：
- 为前缀缓存创建 block 哈希函数，用来识别相同的 prompt 前缀。
- 根据是否启用 batch queue，选择普通 `step()` 或队列版 `step_with_batch_queue()`
- 创建取消请求队列。
- 保存异步调度和引擎空闲回调状态。
最后的：
```
freeze_gc_heap()
maybe_attach_gc_debug_callback()
enable_envs_cache()
```
主要用于减少 Python 垃圾回收暂停和环境变量读取开销，属于运行性能优化。
整体初始化顺序是：
```
读取配置与插件
      ↓
创建 ModelExecutor
      ↓
测量显存并创建 KV Cache
      ↓
创建 Scheduler
      ↓
初始化投机解码、连接器和多模态缓存
      ↓
配置 batch queue 与前缀缓存
      ↓
选择 step 执行方式
      ↓
准备运行状态
```
其中最核心的三个对象是：
```
EngineCore
├─ model_executor：真正执行模型
├─ scheduler：决定本轮执行什么
└─ KV Cache：保存历史 token 的注意力状态
```
#### step函数
```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
	"""Schedule, execute, and make output.

	Returns tuple of outputs and a flag indicating whether the model
	was executed.
	"""
	# Check for any requests remaining in the scheduler - unfinished,
	# or finished and not yet removed from the batch.
	if not self.scheduler.has_requests():
		return {}, False
	scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
	future = self.model_executor.execute_model(scheduler_output, non_block=True)
	grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
	with (
		self.capture_iteration_details(scheduler_output) as iteration_details,
		self.log_error_detail(scheduler_output),
	):
		model_output = future.result()
		if model_output is None:
			model_output = self.model_executor.sample_tokens(grammar_output)

	# Before processing the model output, process any aborts that happened
	# during the model execution.
	self._process_aborts_queue()
	engine_core_outputs = self.scheduler.update_from_output(
		scheduler_output, model_output
	)
	self._attach_iteration_details(engine_core_outputs, iteration_details)

	return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```
1. **主要作用**
`EngineCore.step()` 完成 vLLM 核心引擎的一轮推理：由 Scheduler 选择本轮要处理的请求和 token，交给 ModelExecutor 执行模型计算，再把生成结果写回请求状态并生成 `EngineCoreOutputs`。一次 `step()` 只推进一轮，不会保证所有请求都完成。

2. **输入（参数）**
没有需要调用者传入的参数；函数使用 `self.scheduler`、`self.model_executor` 和内部取消队列完成工作。调度时会通过 `_should_throttle_prefills()` 判断是否暂缓新的 prefill 请求。

3. **输出（改变了什么值）**
返回一个二元组：
```
(
    dict[int, EngineCoreOutputs],
    bool,
)
```
- 字典保存本轮各客户端的输出。
- 布尔值表示本轮是否实际调度了 token 并执行模型。
执行过程中会更新请求的已计算 token、生成 token、完成状态和 KV Cache 占用，也可能处理取消请求；不会修改模型权重，也不会在这里把 token 转换成文本。

4. **术语解释**
- 一轮推理：一次“调度 → 模型计算 → 状态回写”。一轮可以并行处理多个请求，但未完成的请求还要参加后续轮次。
- `Scheduler`：调度器，决定本轮处理哪些请求、每个请求计算多少 token，以及为它们分配哪些 KV Cache block。
- `has_requests()`：检查调度器中是否还有请求，包括未完成请求，以及已经完成但尚未从执行 batch 中清理的请求。
    空返回：
```
if not self.scheduler.has_requests():
    return {}, False
```
表示没有任何请求需要继续处理，因此没有输出，也没有执行模型。
- `_should_throttle_prefills()`：判断是否暂缓新的 prefill。普通 `EngineCore` 通常返回 `False`；数据并行场景可能通过它平衡不同引擎的 prefill 负载。
- `scheduler_output`：Scheduler 对本轮工作的安排，包含本轮请求、每个请求的 token 数、KV Cache block table 等模型执行信息。
- `ModelExecutor`：负责把调度结果交给 worker 和模型执行器，最终运行 GPU 模型计算。
- `execute_model(..., non_block=True)`：以非阻塞方式启动模型执行并立即返回一个 `Future`，使 CPU 可以在等待 GPU 的同时准备其他信息。
- `Future`：代表一个尚未完成的计算任务。调用 `future.result()` 时，如果模型计算尚未结束，当前函数会等待它完成。
- 非阻塞与阻塞的区别：模型执行的启动是非阻塞的，但当前 `step()` 最终仍会在 `future.result()` 等待本轮结果，因此对调用者来说，它会在本轮完成后才返回。
- `grammar_output`：结构化输出的语法约束信息，用来限制本轮哪些 token 可以被采样。
- grammar bitmask：一个掩码，标记词表中哪些 token 当前合法。例如要求生成 JSON 时，可以屏蔽会破坏 JSON 结构的 token。
- `capture_iteration_details()`：记录本轮调度和执行的统计信息，例如本轮处理了多少 token。它主要用于监控和调试。
- `log_error_detail()`：当本轮执行发生异常时，提供更完整的请求和调度上下文，方便定位错误。
- `model_output`：模型执行器产生的结果，通常包含采样得到的新 token、logprob 或 pooling 输出。
- `model_output is None`：某些执行模式先完成模型前向计算，但尚未采样 token，此时需要再调用 `sample_tokens(grammar_output)` 完成采样。
- `sample_tokens()`：根据模型输出的 logits 和语法约束选择下一批 token。
- abort：取消一个尚未完成的请求，例如用户主动停止生成。
- `_process_aborts_queue()`：处理模型执行期间新到达的取消请求。它必须在回写模型结果前执行，避免继续更新已经取消的请求。
- `update_from_output()`：把模型产生的新 token 写回各个请求，并检查 EOS、`max_tokens` 等停止条件；请求结束时还会释放它占用的 KV Cache block。
- `EngineCoreOutputs`：EngineCore 返回给前端的结果，包含新 token ID、结束原因和调度统计等，但还不是最终用户看到的文本。
- `dict[int, EngineCoreOutputs]`：按客户端编号组织输出。普通进程内模式通常只使用客户端编号 `0`；多客户端模式可能产生多个条目。
- `iteration_details`：本轮的内部统计详情，通过 `_attach_iteration_details()` 附加到输出中。
- `total_num_scheduled_tokens`：Scheduler 本轮安排计算的 token 总数。大于 `0` 表示本轮确实执行了有效模型计算。
- 返回的布尔值：
```
scheduler_output.total_num_scheduled_tokens > 0
```
它会被 `post_step()` 使用，例如决定是否需要取得投机解码生成的草稿 token。
- KV Cache 状态：调度器可能在本轮分配、复用或释放 KV Cache block，但启动时预先申请的整个 KV Cache 显存池通常仍然保留。
- 不进行反分词：这里返回的是 token ID。将 token 转换成文本由后面的 `OutputProcessor` 和 Detokenizer 完成。

5. 核心流程
```
检查 Scheduler 中是否还有请求
        ↓
没有请求 → 返回 ({}, False)
        ↓
Scheduler.schedule()
选择本轮请求、token 和 KV Cache
        ↓
ModelExecutor.execute_model(non_block=True)
启动模型计算，得到 Future
        ↓
Scheduler.get_grammar_bitmask()
准备结构化输出约束
        ↓
future.result()
等待本轮模型计算完成
        ↓
必要时 sample_tokens()
完成 token 采样
        ↓
_process_aborts_queue()
处理计算期间收到的取消请求
        ↓
Scheduler.update_from_output()
写回 token、判断结束、回收资源
        ↓
附加本轮统计信息
        ↓
返回 EngineCoreOutputs 和模型执行标记
```

6. **在整个项目中的定位**
`EngineCore.step()` 是 vLLM V1 数据流中“调度、执行、回写”三步闭环的核心：
```
OfflineInferenceMixin._run_engine()
        ↓
LLMEngine.step()
        ↓
EngineCoreClient.get_output()
        ↓
EngineCore.step()                     ← 当前函数
        ├─ Scheduler.schedule()
        │      ↓
        ├─ ModelExecutor.execute_model()
        │      ↓
        │   Worker / GPUModelRunner
        │      ↓
        │   模型 forward + Sampler
        │
        └─ Scheduler.update_from_output()
               ↓
        返回 EngineCoreOutputs
               ↓
LLMEngine.step()
        ↓
OutputProcessor.process_outputs()
        ├─ Detokenizer：token → 文本
        └─ 生成 RequestOutput
        ↓
返回给用户
```
在进程内模式中，`InprocClient.get_output()` 直接调用它；在多进程模式中，则由后台 `EngineCoreProc` 的 busy loop 反复调用相同的核心执行逻辑。
#### initialize kv caches函数
```python
@instrument(span_name="Prepare model")
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
	start = time.time()

	# register all kvcache specs in enginecore process.
	register_all_kvcache_specs(vllm_config)

	# Get all kv cache needed by the model
	kv_cache_specs = self.model_executor.get_kv_cache_specs()

	# Some layers (e.g. Prefix LM attention) run non-causally and tag their
	# KV cache spec with ``non_causal=True``. The specs are collected here in
	# the engine-core process (the same process that builds the scheduler),
	# so this is the multiproc-safe place to translate that layer-level
	# signal into a scheduling policy: chunked prefill and prefix caching
	# both assume causal attention and would corrupt non-causal prefill.
	if any(
		getattr(spec, "non_causal", False)
		for worker_specs in kv_cache_specs
		for spec in worker_specs.values()
	):
		if vllm_config.scheduler_config.enable_chunked_prefill:
			logger.info(
				"Disabling chunked prefill: model has non-causal attention layers."
			)
			vllm_config.scheduler_config.enable_chunked_prefill = False
		if vllm_config.cache_config.enable_prefix_caching:
			logger.info(
				"Disabling prefix caching: model has non-causal attention layers."
			)
			vllm_config.cache_config.enable_prefix_caching = False

	# Resolve the KV cache layout before memory profiling: workers that
	# capture full cudagraphs initialize a minimal KV cache during it.
	# Attention-free models resolve the default so layout reads never precede
	# resolution.
	layout = resolve_kv_cache_layout(
		vllm_config,
		self.model_executor.get_supported_kv_cache_layouts(),
		[s for specs in kv_cache_specs for s in specs.values()],
	)
	self.model_executor.set_kv_cache_layout(layout.name)

	has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
	if has_kv_cache:
		if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
			# NOTE(yongji): should already be set
			# during _eep_scale_up_before_kv_init
			assert self.available_gpu_memory_for_kv_cache > 0
			available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
				kv_cache_specs
			)
		else:
			# Profiles the peak memory usage of the model to determine how
			# much memory can be allocated for kv cache.
			available_gpu_memory = self.model_executor.determine_available_memory()
			self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
	else:
		# Attention free models don't need memory for kv cache
		available_gpu_memory = [0] * len(kv_cache_specs)

	assert len(kv_cache_specs) == len(available_gpu_memory)

	# Track max_model_len before KV cache config to detect auto-fit changes
	max_model_len_before = vllm_config.model_config.max_model_len

	kv_cache_configs = get_kv_cache_configs(
		vllm_config, kv_cache_specs, available_gpu_memory
	)
	for kv_cache_config in kv_cache_configs:
		kv_cache_config.kv_cache_layout = vllm_config.cache_config.kv_cache_layout

	# If auto-fit reduced max_model_len, sync the new value to workers.
	# This is needed because workers were spawned before memory profiling
	# and have the original (larger) max_model_len cached.
	max_model_len_after = vllm_config.model_config.max_model_len
	if max_model_len_after != max_model_len_before:
		self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

	scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
	vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
	kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
	if kv_cache_groups:
		# Exclude groups that opt out of prefix caching (e.g. GLM-5.3-Flash
		# kpool tail, a 1-block/req scratch buffer with block_size=kpool):
		# their small block_size would otherwise drag the global block_size
		# below the real allocator block size and desync it from mamba.
		participating = [
			g.kv_cache_spec.block_size
			for g in kv_cache_groups
			if g.kv_cache_spec.prefix_cacheable
		]
		vllm_config.cache_config.block_size = min(
			participating
			if participating
			else [g.kv_cache_spec.block_size for g in kv_cache_groups]
		)
		update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config)

	vllm_config.validate_block_size()

	self.model_executor.initialize_from_config(kv_cache_configs)
	if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
		self.model_executor.compile_or_warm_up_model()

	elapsed = time.time() - start
	compile_time = vllm_config.compilation_config.compilation_time
	encoder_compile_time = vllm_config.compilation_config.encoder_compilation_time
	if encoder_compile_time > 0:
		logger.info_once(
			"init engine (profile, create kv cache, warmup model) took "
			"%.2f s (compilation: %.2f s — language_model: %.2f s, "
			"encoder: %.2f s)",
			elapsed,
			compile_time + encoder_compile_time,
			compile_time,
			encoder_compile_time,
		)
	elif compile_time > 0:
		logger.info_once(
			"init engine (profile, create kv cache, warmup model) took "
			"%.2f s (compilation: %.2f s)",
			elapsed,
			compile_time,
		)
	else:
		logger.info_once(
			"init engine (profile, create kv cache, warmup model) took %.2f s",
			elapsed,
		)
	return scheduler_kv_cache_config
```
1. **主要作用**
`_initialize_kv_caches()` 在引擎启动阶段确定模型需要怎样的 KV Cache，测量可用 GPU 显存，计算 block 数量并创建实际的 KV Cache；随后预热或编译模型，并返回 Scheduler 使用的 `KVCacheConfig`。

2. **输入（参数）**
- `self`：当前 `EngineCore`，通过其中的 `model_executor` 操作各个 worker。
- `vllm_config`：包含模型最大长度、显存使用比例、block 大小、前缀缓存和分块预填充等配置。

3. **输出（改变了什么值）**
返回一个供 Scheduler 使用的 `KVCacheConfig`。执行过程中还会：
- 记录可供 KV Cache 使用的 GPU 显存。
- 确定 KV Cache 布局、block 数量和 block 大小。
- 在 worker 的 GPU 上创建实际 KV Cache 张量。
- 必要时缩小 `max_model_len`。
- 对非因果注意力模型关闭不兼容的前缀缓存和分块预填充。
- 编译或预热模型。

4. **术语解释**
- KV Cache spec：描述某个模型层需要什么样的 KV Cache，包括数据类型、block 大小和注意力类型等。函数先从所有 worker 收集这些规格。
- `register_all_kvcache_specs()`：注册当前 vLLM 支持的 KV Cache 规格，使 EngineCore 能够识别不同模型和注意力层的缓存需求。
- `get_kv_cache_specs()`：向所有 worker 查询模型各层需要的 KV Cache。返回值通常按 worker 和模型层组织。
- 因果注意力：当前 token 只能看到自己和之前的 token，是普通自回归大模型最常见的注意力方式。
- 非因果注意力：token 可以看到后面的 token。前缀缓存和分块预填充通常假设注意力是因果的，因此遇到非因果层时必须关闭这两个功能，避免计算错误。
- 分块预填充：把很长的 prompt 拆成多个较小部分分轮处理，防止一个长 prompt 独占整轮 token 预算。
 - 前缀缓存：不同请求拥有相同 prompt 前缀时，可以复用已经计算好的 KV Cache block。  
- KV Cache layout：KV Cache 张量在显存中的排列方式。不同模型、注意力后端或硬件可能支持不同布局。
- `resolve_kv_cache_layout()`：综合配置、worker 支持能力和模型各层需求，选择一个可用的 KV Cache 布局。
- 显存 profiling：先运行或分析一次模型，测量模型权重、临时张量等内容占用的峰值显存，再确定剩余多少显存可以分配给 KV Cache。
- `determine_available_memory()`：让模型执行器计算每个 worker 可以用于 KV Cache 的显存大小。
- attention-free 模型：没有普通注意力 KV Cache 的模型。这类模型不需要为 KV Cache 预留显存，因此可用量直接记为 `0`。
- block：vLLM 分配 KV Cache 的基本单位，一个 block 可以保存固定数量 token 的 K/V 数据。
- block size：一个 block 能保存多少个 token。它不是显存字节数，而是以 token 数量表示的逻辑容量。
- `num_gpu_blocks`：GPU KV Cache 池中一共创建多少个 block。数量越多，通常能同时保存更多请求或更长的上下文。
- KV Cache group：缓存规格相同的一组模型层。调度器可以按组统一管理这些层的 KV Cache。
- `get_kv_cache_configs()`：根据 KV Cache 规格和可用显存，为每个 worker 生成具体的缓存分配配置。
- `generate_scheduler_kv_cache_config()`：把各 worker 的配置整理成 Scheduler 能够使用的统一配置。
- `max_model_len`：模型允许的最大上下文长度，包含输入 token 和输出 token。
- auto-fit：如果现有显存不足以支持原来的 `max_model_len`，配置生成过程可以自动把它缩小到可运行的长度。
- `collective_rpc("update_max_model_len", ...)`：如果 `max_model_len` 被缩小，通知所有 worker 更新它们保存的长度配置，保证 EngineCore 和 worker 一致。
- `update_kv_cache_capacity()`：根据最终 block 数量和 block 大小，更新引擎实际能够容纳的 KV Cache 容量。
- `validate_block_size()`：检查最终 block 大小是否满足模型和硬件的限制。
- `initialize_from_config()`：按照已经算好的配置在 worker 的 GPU 上创建真正的 KV Cache 张量。这是实际申请 KV Cache 显存的关键位置。
- `compile_or_warm_up_model()`：预先编译模型或运行几次模拟输入，减少第一次真实请求时的额外开销。
- Scheduler 使用的 `KVCacheConfig`：告诉调度器共有多少 block、每个 block 多大以及有哪些缓存组。运行时调度器根据它给具体请求分配 block。
- `@instrument(span_name="Prepare model")`：给整个初始化过程添加性能追踪区间，用于观察准备模型花费的时间，不改变功能。
        
5. **核心流程**
```
记录开始时间
    ↓
注册并收集各层的 KV Cache spec
    ↓
检查是否存在非因果注意力
    ├─ 有 → 关闭不兼容的前缀缓存和分块预填充
    └─ 无 → 保持原配置
    ↓
选择 KV Cache layout
    ↓
测量可用于 KV Cache 的 GPU 显存
    ↓
生成各 worker 的 KV Cache 配置
    ↓
必要时缩小 max_model_len 并通知 worker
    ↓
生成 Scheduler 使用的统一 KVCacheConfig
    ↓
更新 num_gpu_blocks、block_size 和缓存容量
    ↓
在 GPU 上创建 KV Cache 张量
    ↓
编译或预热模型
    ↓
记录耗时并返回 KVCacheConfig
```

这里最关键的三个阶段是：
```
模型需要多少缓存
        ↓
GPU 能提供多少显存
        ↓
最终创建多少 KV Cache block
```

6. **在整个项目中的定位**
该函数位于 `EngineCore` 启动阶段，连接模型执行器与调度器：
```
LLM / AsyncLLM 创建引擎
          ↓
EngineCore.__init__()
          ↓
创建 ModelExecutor
          ↓
_initialize_kv_caches()
    ├─ 获取模型各层的 KV Cache 规格
    ├─ 测量可用于 KV Cache 的 GPU 显存
    ├─ 计算 block 数量和布局
    ├─ 在 GPU 上创建空的物理 KV Cache 张量池
    └─ 返回 KVCacheConfig
          ↓
使用 KVCacheConfig 创建 Scheduler
          ↓
引擎初始化完成
          ↓
运行时 Scheduler.schedule()
          ↓
KVCacheManager.allocate_slots()
把缓存池中的 block 分配给具体请求
          ↓
模型 Attention forward
计算真实 K/V，并写入对应 block
          ↓
后续 decode 轮次
读取前面各轮保存的 K/V
```

它解决的是“启动时创建多大的公共 KV Cache 显存池”；运行时给具体请求分配和回收 block，则由 Scheduler 和 KVCacheManager 负责。

>容易出现的误区是：`_initialize_kv_caches()` 只在引擎启动时调用一次，不会在每个请求或每轮 decode 时重新创建 KV 张量池。正确流程如下
>![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261015190924.png)

#### add request函数
```python
def add_request(self, request: Request, request_wave: int = 0):
	"""Add request to the scheduler.

	`request_wave`: indicate which wave of requests this is expected to
	belong to in DP case
	"""
	# Validate the request_id type.
	if not isinstance(request.request_id, str):
		raise TypeError(
			f"request_id must be a string, got {type(request.request_id)}"
		)

	if pooling_params := request.pooling_params:
		supported_pooling_tasks = [
			task for task in self.get_supported_tasks() if task in POOLING_TASKS
		]

		if pooling_params.task not in supported_pooling_tasks:
			raise ValueError(
				f"Unsupported task: {pooling_params.task!r} "
				f"Supported tasks: {supported_pooling_tasks}"
			)

	if request.kv_transfer_params is not None and (
		not self.scheduler.get_kv_connector()
	):
		logger.warning(
			"Got kv_transfer_params, but no KVConnector found. "
			"Disabling KVTransfer for this request."
		)

	if (
		request.ec_transfer_params is not None
		and self.scheduler.get_ec_connector() is None
	):
		logger.warning(
			"Got ec_transfer_params, but no ECConnector found. "
			"Disabling ECTransfer for this request."
		)

	self.scheduler.add_request(request)
	if request.abort_immediately:
		# Immediately abort so the connector's request_finished hook runs
		# to free any pre-admission KV-transfer resources.
		self.abort_requests([request.request_id])
```
1. **主要作用**
`EngineCore.add_request()` 对内部 `Request` 做最后一轮基础校验，然后交给 `Scheduler.add_request()`，使新请求进入调度器的等待队列；它只负责接收请求，不会立即执行模型计算或分配 KV Cache block。

2. **输入（参数）**
- `request`：已经转换成 vLLM 内部格式的 `Request` 对象，包含请求 ID、token、采样或池化参数、优先级及缓存传输信息。
- `request_wave`：数据并行场景中的请求批次编号，默认是 `0`；普通 `EngineCore` 不直接使用它，`DPEngineCoreProc` 会使用。

3. **输出（改变了什么值）**
函数返回 `None`。正常情况下，它会把请求登记到 Scheduler 并加入 waiting 队列；如果 `abort_immediately=True`，则请求会在加入后立即被标记为取消，并释放相关的预接入资源。它不会修改模型权重，也不会在这里运行 GPU 模型。

4. **术语解释**
- `Request`：EngineCore 和 Scheduler 内部使用的请求对象，包含模型计算与调度需要的全部信息。
- `EngineCoreRequest`：前端传给 EngineCore 的标准请求对象。进入当前函数之前，通常已经由 `preprocess_add_request()` 转换成内部 `Request`。
- `request_id`：请求的唯一内部编号，必须是字符串。Scheduler 通过它保存、查找、更新和取消请求。
- `request_id` 类型检查：
```
if not isinstance(request.request_id, str):
    raise TypeError(...)
```
这是为了确保后续字典索引和请求状态管理使用统一的字符串 ID。
- `PoolingParams`：embedding、分类或 rerank 等非生成任务使用的参数。
- pooling task：具体的池化任务类型。代码会检查它是否存在于当前模型的 `supported_pooling_tasks` 中。
- `get_supported_tasks()`：查询当前模型实际支持哪些任务，防止把不兼容的请求送入 Scheduler。
- `:=`：Python 的赋值表达式。下面的代码会同时取出 `request.pooling_params` 并判断它是否存在：
```
if pooling_params := request.pooling_params:
```
- Scheduler：负责管理 waiting 和 running 请求，并在每一轮决定哪些请求可以进入 GPU 计算。
- waiting 队列：新请求通常首先进入的队列。只有后续调用 `Scheduler.schedule()` 时，请求才可能被选中运行。
- `Scheduler.add_request()`：登记请求并加入 waiting 队列；如果是可恢复的流式请求，也可能更新已有会话，而不是创建全新请求。
- 请求入队与请求执行：`add_request()` 只让 Scheduler 知道该请求存在；真正的 KV Cache block 分配和模型计算发生在后续的 `schedule()` 和 `execute_model()` 中。
- `kv_transfer_params`：描述请求如何从其他推理节点获取或传输 KV Cache 的信息，常用于 Prefill/Decode 分离等部署模式。
- `KVConnector`：负责跨引擎或跨节点传输 KV Cache 的组件。如果请求带有 `kv_transfer_params`，但引擎没有配置 KVConnector，相关传输功能会被忽略并输出警告。
- `ec_transfer_params`：描述编码缓存如何传输的信息，主要用于多模态 encoder cache 等场景。
- `ECConnector`：负责跨引擎传输 encoder cache 的组件。没有配置它时，`ec_transfer_params` 无法生效。
- `logger.warning()`：只记录配置不匹配的警告，不会因为缺少 Connector 直接拒绝整个请求。
- `request_wave`：数据并行场景下的一组请求批次标识，用于协调多个 EngineCore 同时开始或结束一轮工作。基础 `EngineCore.add_request()` 中没有直接读取它，但数据并行子类会处理它。
- `abort_immediately`：表示请求加入 Scheduler 后应立即取消。该设计看似多余，但可以确保 Connector 的“请求加入”和“请求结束”钩子都得到调用。
- 预接入资源：某些 KV 传输请求在正式进入 Scheduler 前就可能准备通信或缓存资源。立即取消时，需要通过正常结束流程释放它们。
- `abort_requests()`：通知 Scheduler 将指定请求标记为 `FINISHED_ABORTED`，并执行请求结束和资源释放逻辑。
- 不分配 KV Cache block：新请求入 waiting 队列时通常还没有专属 block。只有被 `Scheduler.schedule()` 选中后，才通过 KVCacheManager 分配。

5. 核心流程
```
接收内部 Request
        ↓
检查 request_id 是否为字符串
        ├─ 否 → 抛出 TypeError
        └─ 是
             ↓
如果是 pooling 请求
        ↓
检查模型是否支持对应 pooling task
        ├─ 不支持 → 抛出 ValueError
        └─ 支持
             ↓
检查 KV transfer 配置
        ├─ 有参数但无 KVConnector → 记录警告
        └─ 配置正常 → 继续
             ↓
检查 encoder cache transfer 配置
        ├─ 有参数但无 ECConnector → 记录警告
        └─ 配置正常 → 继续
             ↓
Scheduler.add_request(request)
请求进入调度系统
             ↓
abort_immediately？
        ├─ 是 → 立即取消并释放相关资源
        └─ 否 → 留在 waiting 队列等待调度
```

6. **在整个项目中的定位**
该函数位于“前端请求转换”和“Scheduler 正式调度”之间，是请求进入核心调度系统的最后入口：
```
LLM.generate() / 在线 API
              ↓
InputProcessor.process_inputs()
生成 EngineCoreRequest
              ↓
LLMEngine / EngineCoreClient
提交 EngineCoreRequest
              ↓
EngineCore.preprocess_add_request()
EngineCoreRequest → Request
              ↓
EngineCore.add_request()              ← 当前函数
校验请求并交给 Scheduler
              ↓
Scheduler.add_request()
请求进入 waiting 队列
              ↓
EngineCore.step()
              ↓
Scheduler.schedule()
选择本轮请求并分配 KV Cache block
              ↓
ModelExecutor.execute_model()
执行 GPU 模型计算
              ↓
Scheduler.update_from_output()
写回 token 和请求状态
```
因此，它解决的是“让一个已经准备好的内部请求安全进入 Scheduler”，而不是“立即运行这个请求”。
## 阶段性自检
**一、`EngineCore.step()` 的三步分别是什么？用一句话概括每一步的输入和输出。**
`EngineCore.step()` 的三步是：
1. 调度 `Scheduler.schedule()`：输入当前请求状态和可用 KV Cache/token 预算，输出本轮执行计划 `SchedulerOutput`，说明哪些请求计算多少 token、使用哪些缓存块
2. 执行 `ModelExecutor.execute_model()`：输入 `SchedulerOutput`，在 GPU 上完成模型前向与采样，输出包含新 token、logprob 或 pooling 结果的 `ModelRunnerOutput`
3. 写回 `Scheduler.update_from_output()`：输入调度计划和模型结果，更新请求进度、判断结束并释放资源，输出供前端处理的 `EngineCoreOutputs`

**二、为什么离线模式默认走 `InprocClient`？多进程模式（`MPClient`）存在的意义是什么？**
离线模式通常使用 `InprocClient`，因为调用是同步的，前端可以直接调用同一进程中的 `EngineCore.step()`，无需 ZMQ、序列化和后台通信，开销更小、调用链更清晰，也更方便调试。
`MPClient` 的意义是把前端和 `EngineCore` 分到不同进程：EngineCore 可在后台持续运行，前端不会被模型计算阻塞，从而支持异步在线服务、多客户端、数据并行、故障隔离以及通信与计算重叠。注意，`InprocClient` 只表示 EngineCore 与前端同进程，不代表模型一定只使用一个 GPU 或一个 worker。

**三、`EngineCore.__init__` 里为什么必须先 `_initialize_kv_caches` 再创建 `Scheduler`？如果反过来会怎样？**
因为 `Scheduler` 创建时必须知道 KV Cache 的最终配置：有多少 block、每个 block 多大、有哪些缓存组，以及实际支持的最大上下文长度。
`_initialize_kv_caches()` 会先：
```
分析模型的 KV Cache 需求
→ 测量可用 GPU 显存
→ 确定布局和 block 数量
→ 创建物理缓存池
→ 必要时调整 max_model_len
→ 返回 KVCacheConfig
```
然后 Scheduler 根据同一份 `KVCacheConfig` 创建 `KVCacheManager`，管理请求的 block 分配。
如果顺序反过来，Scheduler 只能使用未确定或过期的缓存配置，可能导致：
- Scheduler 认为可用 block 数与 GPU 实际分配不一致。
- 逻辑 block table 与物理 KV Cache 布局不匹配。
- 错误判断并发容量和最大上下文长度。
- 最终出现分配失败、显存越界或 OOM。
所以必须先创建并确定“物理缓存池”，再创建负责管理它的 Scheduler。

**四、`LLMEngine.step()` 和 `EngineCore.step()` 是两个不同的 `step`，它们各自做了什么？谁调谁？**
两者分工不同：
- `EngineCore.step()` 是底层计算循环：执行“调度请求 → GPU 模型计算与采样 → 写回 token 和请求状态”，返回 `EngineCoreOutputs` 和本轮是否执行模型的标记。
- `LLMEngine.step()` 是前端处理循环：通过 `EngineCoreClient` 获取底层输出，再进行反分词、停止字符串检查、统计记录，最后生成用户可见的 `RequestOutput`。
调用关系是：
```
OfflineInferenceMixin._run_engine()
        ↓
LLMEngine.step()
        ↓
EngineCoreClient.get_output()
        ↓
EngineCore.step()
        ↓
返回 token 级 EngineCoreOutputs
        ↓
LLMEngine 的 OutputProcessor
        ↓
返回文本级 RequestOutput
```
在 `InprocClient` 模式中会直接调用 `EngineCore.step()`；多进程模式下，`EngineCore.step()` 由后台 EngineCore 进程运行，`LLMEngine.step()` 通过客户端通信获取结果。

**五、`non_block=True` 让 `execute_model` 异步返回一个 `future`，这样做的收益是什么？**
`non_block=True` 的收益是：启动 GPU 模型计算后立即返回 `Future`，CPU 不必马上等待，可以同时准备语法约束 `grammar_bitmask` 等工作，让 CPU 处理与 GPU 计算发生重叠，减少空等时间。
```
GPU：执行模型 forward
CPU：同时计算 grammar bitmask
             ↓
future.result() 等待并汇合结果
```
在 batch queue 或流水线并行模式下，还可以进一步提前调度后续 batch，减少流水线空泡。需要注意，它并没有让整个 `EngineCore.step()` 变成异步；调用 `future.result()` 时仍会等待本轮模型执行完成。