本文基于的vllm commit hash为：`c16bb6068f70878fb8a2f7c4d6cda95cd03a778b`
# day5
`EngineArgs` 是一个巨大的 dataclass，把所有启动参数集中到一处，然后通过 `create_engine_config()` 加工成引擎真正使用的 `VllmConfig` 对象。我们不需要详细读它（文件有 2500+ 行，绝大部分是 CLI 参数定义），但需要知道几个对后续读源码影响最大的配置项。
- 主要源码在：`vllm/engine/arg_utils.py`（约 2500 行）
- 配置的实际定义分散在 `vllm/config/` 下：`model.py`、`cache.py`、`scheduler.py`、`parallel.py`、`compilation.py` 等
## arg_utils.py
**路径**：/home/dongmingzhe/vllm/vllm/engine/arg_utils.py
**简介**：定义了 vLLM 引擎的启动参数体系：`EngineArgs`/`AsyncEngineArgs` 两个 dataclass 汇总了模型、并行、缓存、调度等全部配置项，并通过 `add_cli_args` 自动把它们生成为命令行参数。它的核心作用是双向转换与默认值推导——通过 `from_cli_args` 把命令行解析成参数对象，再用 `create_engine_config` 等方法将其拆解校验成 `ModelConfig`、`LoadConfig`、`SchedulerConfig` 等各子配置，最终交给引擎初始化。。
**本小节阅读的内容**：
* 初识`block_size`参数
* 初识`max_num_seqs`参数
* 初识`max_num_batched_tokens`参数
* 初识`gpu_memory_utilization`参数
* 初识`enable_prefix_caching`参数
* 初识`enable_chunked_prefill`参数
* 初识`enforce_eager`参数
* 初识`max_model_len`参数
### block_size参数
**1.这是什么？**
vLLM 用 PagedAttention 管理 KV cache：显存被切成固定大小的「块」（block/page），`block_size` 就是每个块能存多少个 token 的 KV。
- 默认值 **16**（`CacheConfig.DEFAULT_BLOCK_SIZE`, [cache.py:70](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L70)）。
- 它决定：块表（block table）的寻址粒度、KV cache 张量形状、显存浪费（最后一个块存不满 → 内部碎片）、前缀缓存的命中粒度。
- block_size 越大 → 块表越小、调度开销低，但碎片多、前缀缓存命中粒度粗；越小则相反。

**2.在本文件中的体现**
三处，链路很清晰：
a) 定义配置字段（[arg_utils.py:551](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L551)）
```python
block_size: int | None = None
```
`None` 代表「用默认值/交给后端决定」，不是 0。
b) 暴露成 CLI 参数（[arg_utils.py:1283](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1283)）
```python
cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
```
参数定义（类型、help、校验）是直接从 `CacheConfig` 的 pydantic field 反射出来的（`get_kwargs(CacheConfig)`），所以这里只是转发，真正的语义在 `CacheConfig` 里。
c) 注入 CacheConfig（[arg_utils.py:2113](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2113)）
```python
cache_config = CacheConfig(block_size=self.block_size, ...)
```
`EngineArgs` 只是个中转站：CLI → EngineArgs 字段 → CacheConfig。
完整流程
```text
用户命令行：--block-size 32
↓
argparse 解析 → EngineArgs.block_size = 32
↓
构造 CacheConfig(block_size=32)
↓
vLLM 内核用 CacheConfig.block_size 分配KV Cache块

# 用户不写--block-size的情况
用户命令行：（不带--block-size）
↓
EngineArgs.block_size = None
↓
CacheConfig收到 None → 后端自动使用默认block_size
```

**3.在其它文件中的体现**
① 默认值与「用户是否显式指定」— [config/cache.py:73-77](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L73-L77)、[cache.py:315](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L315) 构造后把 `None` 解析为 16，并记录 `user_specified_block_size`。这个标志很关键：它让后端能在用户没指定时自动改 block_size。
② 后端适配 —— 最重要的一环— [platforms/interface.py:596](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/platforms/interface.py#L596) `update_block_size_for_backend`
- 若用户没指定：用 `backend_cls.get_preferred_block_size(16)` 取后端偏好的值；
- 混合模型（hybrid，如 Mamba/滑动窗口）再对齐 mamba page size；
- 多种 KV dtype 共享块池时可能覆盖用户设置。 即：`block_size` 最终值往往是后端约束说了算，而非简单等于命令行。
③ 后端声明支持哪些块大小 — [v1/attention/backend.py:116-155](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/attention/backend.py#L116-L155) `get_supported_kernel_block_sizes()` / `supports_block_size()`。多数后端要求 block_size 是某个「kernel block size」的整数倍（因为 kernel 内部还有自己的分页）。例如 Blackwell b12x 后端：见 [b12x.py:186](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/attention/backends/b12x.py#L186)。`flex_attention`、MLA 等各有约束。
④ KV cache 的物理形状与显存计算 — [v1/kv_cache_interface.py:160](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/kv_cache_interface.py#L160) 起 每个 `KVCacheSpec.block_size` → `get_page_size_bytes()` → `max_memory_usage_bytes()`（按 block 向上取整），决定 profiler 能挤出多少个块。
⑤ 运行时的块分配与调度 — [v1/core/kv_cache_manager.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/kv_cache_manager.py)、[kv_cache_coordinator.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/kv_cache_coordinator.py) `num_blocks = cdiv(num_tokens, block_size)`、块索引 = `token // block_size`。调度器也按 block_size 对齐地分配 token 数（这也是为什么 prompt 长度不是 block 整数倍时会有额外开销）。
⑥ 前缀缓存 — [config/cache.py:97](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L97) 前缀哈希按 `prefix_match_unit`（即 `hash_block_size`）计算；它必须能整除各 KV cache group 的 `block_size`。所以 block_size 直接影响「前缀缓存能命中到多细的边界」。

**4、总结**
`block_size` 在 arg_utils.py 里只承担「CLI → 配置」的转发；真正决定它最终值的顺序是：CLI(`--block-size`) → 用户指定则固定，否则由 attention 后端偏好决定 → 混合模型/多 dtype 再对齐 → config/vllm.py 校验；之后它贯穿 KV cache 的形状、显存分配、块表寻址和前缀缓存命中粒度。
### max_num_seqs参数
**1.这是什么？**
单次调度迭代（一个 batch）中最多同时处理多少个序列（请求），即调度器和模型 runner 的「并发上限」。
- 沿用 `SchedulerConfig.DEFAULT_MAX_NUM_SEQS = 128`（[scheduler.py:47](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L47)）。
- 它同时驱动三个东西：调度器的准入上限、model runner 的 per-request 静态缓冲区大小、CUDA graph 的捕获 size。所以它不只是个调度数字，还直接决定显存占用。

**2.在本文件中的体现**
a) 字段（[arg_utils.py:573](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L573)）
```python
max_num_seqs: int | None = None
```
`None` = 未指定，交由后面按 usage context / 硬件推断。
b) CLI 参数（[arg_utils.py:1608](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1608)）
```python
scheduler_group.add_argument("--max-num-seqs", default=None, ...)
```
注意这里强制覆盖 default 为 None：因为参数定义是从 `SchedulerConfig` 反射来的（默认 128），但上层需要区分「用户显式给了 128」和「没给」，所以把默认抹成 None 再在下面补。
c) 默认值推断 —— 本文件最核心的部分（[arg_utils.py:2920](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2920) `_set_default_max_num_seqs_and_batched_tokens_args`，在 [2428](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2428) 被调用） 逻辑链：
1. `get_batch_defaults()`（[arg_utils.py:2716](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2716)）按 usage_context × 硬件取默认：B200/B300(≥160GB) 和 H100/H200 → 1024；其他 GPU → 256；CPU → `256*world_size`；LLM_CLASS 与 OpenAI server 分开。
2. `performance_mode == "throughput"` 且用户没指定 → 翻倍。
3. 最后 `max_num_seqs = min(max_num_seqs, max_num_batched_tokens)` 夹一下（[arg_utils.py:3003](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L3003)）。
d) 注入 SchedulerConfig（[arg_utils.py:2448](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2448)） 和 `max_num_batched_tokens` 一起传进去，之后 [2437](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2437) 断言它已被解析为 int。
e) 交叉校验（[arg_utils.py:2499](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2499)） LoRA + 投机解码时要求 `max_num_batched_tokens >= max_num_seqs * (num_speculative_tokens + 1)`，否则报错。

**3.在其它文件中的体现**
① 定义与配置校验 — [config/scheduler.py:63](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L63)
- `ge=1`；
- `max_num_batched_tokens >= max_num_seqs`（每个序列至少 1 个 token，[scheduler.py:316](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L316)）；
- `max_num_batched_tokens <= max_num_seqs * max_model_len`（[scheduler.py:332](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L332)）——物理上每个序列不可能超过最大长度；
- `max_num_active_seqs <= max_num_seqs`。
② 调度器准入上限 — [v1/core/sched/scheduler.py:126](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/sched/scheduler.py#L126)
```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
```
这是它在运行时最直接的体现：RUNNING 队列装满就不再准入新请求。
③ 决定 CUDA graph / runner 形状 — [config/vllm.py:2286](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L2286) 捕获的 graph size 上界是 `max_num_seqs * decode_query_len * 2`（decode_query_len 含投机 token），再和 `max_cudagraph_capture_size` 取 min。这就是「调大 max_num_seqs 会吃显存」的原因。

**4.总结**
`max_num_seqs` 在 arg_utils.py 里的职责是：用 `None` 把「用户是否显式指定」编码下来 → 按 usage context + 硬件 + throughput 模式推断默认值 → 与 `max_num_batched_tokens` 互相夹取后注入 `SchedulerConfig`；在其它文件里它落成三件事：调度器的并发准入上限、runner 的 per-request 缓冲与 CUDA graph 捕获大小、以及编译缓存 key。
### max_num_batched_tokens参数
**1.这是什么？**
单次调度迭代（一个 batch）最多处理的 token 总数——prompt token（prefill）和生成 token（decode）合在一起算。
- 默认 2048（`SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS`，[scheduler.py:46](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L46)），但真实运行值几乎总由 arg_utils 按场景推断。
- 它同时是 chunked prefill 的 chunk 大小：长 prompt 会被切成不超过这个数的片段分多轮算。
- 与 `max_num_seqs` 的分工：前者管「一轮多少 token」，后者管「一轮多少序列」，两者互相约束（见下）。

**2.在本文件中的体现**
a) 字段（[arg_utils.py:570](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L570)）
```python
max_num_batched_tokens: int | None = None
```
`None` = 未指定，交给默认推断。
b) CLI 参数（[arg_utils.py:1594](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1594)）
```python
scheduler_group.add_argument("--max-num-batched-tokens", default=None, ...)
```
默认同样被抹成 `None` 以区分「用户显式指定」。它还被列入人类可读整数白名单（[arg_utils.py:387](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L387)），所以可以写 `--max-num-batched-tokens 16K`。
c) 默认值推断 —— 本文件核心（[arg_utils.py:2920](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2920)，调用点 [2428](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2428)） 顺序：
1. `get_batch_defaults()`（[arg_utils.py:2716](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2716)）按 usage context × 硬件取值：≥160GB(B200/B300) 和 H100/H200 → LLM_CLASS 16384 / server 8192~16384；其他 GPU → 8192/2048；CPU → `4096*world_size`；TPU 另有一套。
2. batched-DP MoE 特例 → `DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP = 256`（[arg_utils.py:2937](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2937)）。
3. `performance_mode == "throughput"` 且未指定 → 翻倍。
4. 未指定且关闭了 chunked prefill → 取 `max(max_model_len, 默认值)`（[arg_utils.py:2965](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2965)），保证一条完整序列能塞进一轮。
5. 多模态 prefix-LM（如 Gemma4）→ 用 `_get_min_mm_batched_tokens()` 抬高下限，让单个模态 item 能放下（[arg_utils.py:2972](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2972)）。
6. 最后 `min(max_num_seqs * max_model_len, ...)` 夹顶（[arg_utils.py:2990](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2990)）。
7. 反向影响 `max_num_seqs = min(max_num_seqs, max_num_batched_tokens)`（[arg_utils.py:3003](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L3003)）。
d) 注入 SchedulerConfig（[arg_utils.py:2446](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2446)），[2434](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2434) 断言已解析为 int。
e) 交叉校验（[arg_utils.py:2497](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2497)）：LoRA + 投机解码要求 ≥ `max_num_seqs * (num_speculative_tokens+1)`。

**3.在其它文件中的体现**
① 定义与校验 — [config/scheduler.py:49](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L49)、[scheduler.py:299](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L299)
- `ge=1`；
- 未开 chunked prefill 时必须 ≥ `max_model_len`，否则直接报错（否则等效于把可处理长度压到了这个值）；
- 必须 ≥ `max_num_seqs`；
- 必须 ≤ `max_num_seqs * max_model_len`；
- 被纳入编译缓存 hash（[scheduler.py:265](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L265)），因为 LoRA/Inductor 会按它分配静态缓冲、决定索引位宽。
② 生成本轮调度预算 — [v1/core/sched/scheduler.py:593](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/sched/scheduler.py#L593)
```python
input_budget = self.scheduler_config.max_num_batched_tokens
```
这就是它在运行时最直接的体现：调度器每轮按这个预算往 batch 里塞 token，塞满即停。[scheduler.py:137](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/sched/scheduler.py#L137) 里 `max_num_scheduled_tokens` 未设时也默认等于它。
③ 编码器/multimodal 容量— [scheduler.py:291](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L291) `max_num_encoder_input_tokens` 和 `encoder_cache_size` 都直接等于它——即单轮能缓存多少 MM encoder 输出。
④ CUDA graph / 编译边界— [compilation/piecewise_backend.py:144](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/compilation/piecewise_backend.py#L144)、[config/compilation.py:584](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/compilation.py#L584) 捕获的 graph size 上限和编译分段上界都引用它，所以调大它同样增加编译与显存开销。

**4.总结**
`max_num_batched_tokens` 在 arg_utils.py 里的职责是：用 `None` 编码「未指定」→ 按 usage context/硬件推断 → throughput 翻倍、非 chunked-prefill 时抬高到 max_model_len、多模态 prefix-LM 再抬高下限、与 max_num_seqs/max_model_len 互夹 → 注入 SchedulerConfig；在运行时它落成调度器每轮的token 预算（同时也是 chunked prefill 的 chunk 大小、encoder 缓存容量和 CUDA graph/编译的分段边界）
### gpu_memory_utilization参数
**1.这是什么？**
vLLM 实例允许占用的 GPU 显存比例（0~1），默认0.92。它是「每实例」的额度，不感知同卡上的其他进程——两个实例各设 0.5 是各自的 50%，不会自动协调。
它的作用机制是：先按 `总显存 × 该比例` 算出本轮请求额度，再用 profile 出的模型权重+激活开销去减，剩下的就是 KV cache 能用的显存。所以它本质上是「KV cache 显存预算」的旋钮。
 
**2.在本文件如何体现？**
本文件对它只做转发，没有任何特殊逻辑：
a) 字段（[arg_utils.py:568](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L568)）
```python
gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
```
直接继承 `CacheConfig` 的默认值 0.92（`EngineArgs` 不重新定义默认）。
b) CLI 参数（[arg_utils.py:1285](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1285)）
```python
cache_group.add_argument("--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"])
```
归到 CacheConfig 参数组（而非 Parallel/Model 组）——这本身就是个信号：它主要影响 KV cache。
c) 注入 CacheConfig（[arg_utils.py:2114](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2114)） 和 `kv_cache_memory_bytes`、`block_size` 一起传入，构建 `cache_config`。

**3.在其它文件中的体现**
① 定义（校验区间） — [config/cache.py:104](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L104) `Field(default=0.92, gt=0, le=1)`；文档明确说明它是 per-instance 限制、不感知同卡其他实例。
② 换算成显存额度 —— 最核心的一处 — [v1/worker/utils.py:537](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/utils.py#L537)
```python
requested_memory = ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)
```
启动时（[gpu_worker.py:456](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_worker.py#L456)）取一次显存快照并算出这个额度；若当前空闲显存 < 额度，直接报错退出（[utils.py:541](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/utils.py#L541)）——这是最常见的「启动失败」原因（同卡跑着别的进程时会撞上）。
③ 推出 KV cache 大小 — [gpu_worker.py:626](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_worker.py#L626) `determine_available_memory()`
```python
available_kv_cache_memory = requested_memory - non_kv_cache_memory - cudagraph_estimate
```
即：额度 − 权重/激活/profile 峰值 − CUDA graph 预留。这个结果再交给 [kv_cache_utils.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/kv_cache_utils.py) 除以每个 block 的字节数，得到 `num_gpu_blocks`。调大它 = 更多 KV block = 更高并发/更长上下文，代价是留给其他进程的空间变小。
④ 与 `kv_cache_memory_bytes` 互斥 — [gpu_worker.py:547](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_worker.py#L547) 一旦显式设了 `--kv-cache-memory-bytes`，就会跳过 profile 并完全无视 `gpu_memory_utilization`（只保留一次编译用 profile run）。这是手动控显存的高级用法。
⑤ CUDA graph 影响 — [gpu_worker.py:651](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_worker.py#L651) 自 v0.21.0 起默认把 CUDA graph 显存从额度里扣掉，于是同样的 `--gpu-memory-utilization` 值，有效 KV cache 比旧版本小；日志会给出「等效旧值」和「建议上调值」。
⑥ 遥测与报错文案 — [v1/utils.py:736](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/utils.py#L736)（usage 上报）、[kv_cache_utils.py:894](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/kv_cache_utils.py#L894)（KV cache 不够时的提示语）、[config/compilation.py:1530](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/compilation.py#L1530)（编译期显存不足提示）。

**4.总结**
`gpu_memory_utilization` 在 arg_utils.py 里只是从 CacheConfig 继承默认值并转发成 `--gpu-memory-utilization`；真正的语义在 worker 里：`总显存 × 比例 = 请求额度 → 扣除权重/激活/CUDA graph → 余量即 KV cache 显存`。它和 `max_num_seqs`/`block_size` 一起，决定最终能开多少个 KV block（即并发与上下文长度）。
### enable_prefix_caching参数
**1.这是什么？**
是否开启前缀缓存：多个请求如果开头有相同前缀（system prompt、few-shot 示例、多轮对话历史等），vLLM 会复用这些前缀已算好的 KV block，跳过重复 prefill。
- `CacheConfig` 里默认True（[cache.py:130](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L130)），但配置侧允许是 `None`，由「模型是否支持」来决定。
- 匹配是按 block 粒度的：对每个满块算哈希（`hash_block_size` / `prefix_match_unit`），哈希相同即命中，且要求 `block_size` 能被其整除。
- 收益：重复前缀越多，TTFT 越低；代价：每个 block 要额外维护哈希与引用计数，块回收逻辑更复杂。

**2.在本文件中的体现**
a) 字段（[arg_utils.py:552](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L552)）
```python
enable_prefix_caching: bool | None = None
```
类型是 `bool | None`，`None` 表示「未指定」，这是与其它布尔开关不同的关键点。
b) CLI 参数（[arg_utils.py:1296](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1296)）
```python
cache_group.add_argument("--enable-prefix-caching", default=None, ...)
```
默认值被强制改成 `None`（其它 cache 参数继承 pydantic 默认，它不继承）——因为要用 `None` 区分「用户没给」和「用户显式给了 True」，前者才允许被自动推断覆盖。
c) 默认推断（[arg_utils.py:2844](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2844)，在 [2093](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2093) 被调用）
```python
default_prefix_caching = model_config.is_prefix_caching_supported
if self.enable_prefix_caching is None:
    self.enable_prefix_caching = default_prefix_caching
```
即默认值由模型能力决定；用户显式开启但模型不支持时只告警不拦截（pooling 模型）。
d) 无条件关闭的平台特例（[arg_utils.py:2872](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2872)） RISC-V CPU 上直接强制置 False。
e) 注入 CacheConfig（[arg_utils.py:2120](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2120)），[2108](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2108) 断言此时必须已是 bool。

**3.总结**
`enable_prefix_caching` 在 arg_utils.py 里的特殊之处是用 `None` 作为「未指定」的哨兵，从而把默认值交给 `model_config.is_prefix_caching_supported` 推断；它在运行时落成两件事：engine core 是否建立 block hasher，以及 KV cache manager 的 `enable_caching` 开关（决定块命中复用）。注意它会被非因果注意力层、纯 MM encoder 和部分模型静默强制关闭

### enable_chunked_prefill参数
**1.这是什么？**
是否允许把 prefill「切块」分多轮执行
- 开启：一个长 prompt 不必在一轮里算完，按本轮剩余 token 预算（`max_num_batched_tokens`）切段，剩下的下轮继续。好处是长 prefill 不会独占整个 batch，decode 请求能插进来一起跑，显著降低 ITL 抖动，且 `max_num_batched_tokens` 可以设得远小于 `max_model_len`。
- 关闭：prefill 必须一轮算完 → 隐含要求 `max_num_batched_tokens >= max_model_len`，且超预算的长 prompt 会被拒绝调度。
- `SchedulerConfig` 默认 True（[scheduler.py:126](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L126)），配置侧允许 `None` 由模型能力推断。

**2.在本文件中的体现**
a) 字段（[arg_utils.py:662](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L662)）
```python
enable_chunked_prefill: bool | None = None
```
注意它属于 SchedulerConfig 组的字段（在 `EngineArgs` 里是普通属性，但注入的是 `SchedulerConfig`）。
b) CLI 参数（[arg_utils.py:1634](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1634)）
```python
scheduler_group.add_argument("--enable-chunked-prefill", default=None, ...)
```
默认同样被抹成 `None`，用于区分「未指定」。
c) 默认推断 —— 与 prefix caching 同一函数（[arg_utils.py:2816](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2816)，调用点 [2093](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2093)）
```python
default_chunked_prefill = model_config.is_chunked_prefill_supported
if self.enable_chunked_prefill is None:
    self.enable_chunked_prefill = default_chunked_prefill
```
用户与模型能力不一致时只告警：generate 模型被手动关闭、pooling 模型被手动开启都会 warning（可能崩溃或输出错误）。
d) 无条件关闭的平台特例（[arg_utils.py:2872](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2872)）：RISC-V CPU。
e) 与 `max_num_batched_tokens` 的联动（[arg_utils.py:2963](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2963)）
```python
if not self.enable_chunked_prefill:
    self.max_num_batched_tokens = max(model_config.max_model_len, self.max_num_batched_tokens)
```
这是两者在本文件里最实质的耦合：关闭 chunked prefill ⇒ 自动把 batched tokens 抬到至少 max_model_len，否则单个 prompt 永远放不下。
f) 注入 SchedulerConfig（[arg_utils.py:2453](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2453)），[2438](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2438) 断言已解析。

**3.总结**
`enable_chunked_prefill` 在 arg_utils.py 里同样是 `None` 哨兵 + 由 `is_chunked_prefill_supported` 推断，并与 `max_num_batched_tokens` 强耦合（关了就得把它抬到 `max_model_len`）；它在运行时落成调度器的一行分支——token 预算不足时是切块续跑还是停止调度。encoder-decoder、无 KV cache、非因果注意力三类模型会被自动强制关闭。
### enforce_eager参数
**1.这是什么？**
强制整个模型用 PyTorch eager 模式执行，即关闭 `torch.compile` 与 CUDA graph。
- 默认False（[model.py:242](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L242)）。此时是混合模式：小 batch 的 decode 走 CUDA graph（省 kernel launch 开销），prefill 走编译/eager。
- 设 True 则所有路径都是 eager。用途：调试定位问题、规避硬件/模型不支持 CUDA graph 或编译的情形、省掉编译等待时间和 graph 显存；代价是吞吐显著下降。
- 注意它只管「图」，不影响调度或算子实现。
### max_model_len参数
**1.这是什么？**
模型的上下文长度上限（prompt + 输出之和）。
- 不指定时会从模型 `config.json` 自动推导：取 `max_position_embeddings` 等字段，再乘 RoPE scaling 因子，并与 tokenizer 的 `model_max_length`、`sliding_window` 取 min；都找不到就 fallback 到 2048 并告警。
- 支持人类可读写法：`1k→1000`、`1K→1024`、`25.6k→25600`；`-1` 或 `auto` 表示自动选「能塞进当前 GPU 显存」的最大长度。
- 它同时是显存与调度的硬边界：KV cache 要按它预留，超过它的请求直接被拒。

**2.在本文件中的体现**
a) 字段（[arg_utils.py:475](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L475)）
```python
max_model_len: int = ModelConfig.max_model_len
```
普通转发，默认继承 `ModelConfig`（即 None → 后续自动推导），归 Model 参数组。
b) CLI 参数 —— 它是个特例（[arg_utils.py:914](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L914) + [arg_utils.py:393](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L393)）
```python
model_group.add_argument("--max-model-len", **model_kwargs["max_model_len"])
```
在 `get_kwargs` 的类型改写逻辑里，`max_model_len` 被单独挑出来：type 设为 `human_readable_int_or_auto`（而其它人类可读参数只有 `human_readable_int`）。它是唯一接受 `auto` 的整数参数。
c) 注入 ModelConfig（[arg_utils.py:1854](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1854)）
d) 作为其它默认值的上界——这是它在本文件里最有存在感的地方：
- 关闭 chunked prefill 时，`max_num_batched_tokens` 被抬到 `max(max_model_len, default)`（[arg_utils.py:2965](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2965)）；
- 反过来夹住 `max_num_batched_tokens ≤ max_num_seqs * max_model_len`（[arg_utils.py:2991](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2991)）；
- 多模态 prefix-LM 估算单 item token 数时也以它为 `seq_len`（[arg_utils.py:2910](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L2910)）。

**3.总结**
`max_model_len` 在 arg_utils.py 里是普通转发字段，但 CLI 层被特殊对待（唯一支持 `auto`/人类可读的整数参数），并在本文件内充当其它调度参数的上下界；真正的语义在 `ModelConfig.get_and_verify_max_len`（自动推导 + 校验）与 `kv_cache_interface`（按它预留 KV cache 显存）、`SchedulerConfig.verify_max_model_len`（与 batched tokens 互相约束）三处落地。
## 阶段型自检
**一、`block_size` 增大一倍，KV Cache 能装的 token 总数会变多还是变少？内存碎片呢？（提示：想想操作系统的页大小）**
- 理论总容量：基本不变（同一块显存重新分页而已）。
- 实际可用 token 数：变少，因为内部碎片翻倍。
- 外部碎片：不存在（这是 paged attention 的固有优势，与 block_size 无关）。

为什么可用 token 会变少？内部碎片
每个序列的最后一块通常装不满。设序列长度均匀分布，平均每序列浪费 ≈ block_size / 2。
- block_size=16 → 平均浪费 8 token/序列；
- block_size=32 → 平均浪费 16 token/序列。
序列越多，累计浪费越大。能同时容纳的序列数因此下降（KV cache 不够 → 调度器提前 preempt/排队）。
> 补充：vLLM 里实际还分 `block_size`（逻辑/Scheduler 粒度）和 `kernel_block_size`（kernel 分页粒度），前者必须是后者整数倍（[kv_cache_interface.py:293](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/kv_cache_interface.py#L293)）。上面讨论的是逻辑块——碎片是按逻辑块算的。

**二、`max_num_batched_tokens` 设得很小（比如 64）和很大（比如 8192），分别会发生什么？**
先记住它的角色：每轮调度迭代的 token 预算（`input_budget`，[scheduler.py:593](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/sched/scheduler.py#L593)）。它同时决定两件事——**prefill 被切成多大块**、**runner 的静态缓冲有多大**。
设得很小（如 64）
调度行为
- 长 prompt 被切成 `64` 一块，分很多轮算。prefill 总计算量不变，但被拉长到多轮 → TTFT 明显变高，且每轮都有调度/launch 固定开销。
- 好处：decode 请求可以和 prefill 交错执行，ITL 抖动小——这正是 chunked prefill 想要的低延迟流式效果。
- 太小（64）时一轮往往塞不满（几十个 decode 请求也就几十 token），GPU 利用率低、kernel launch 占比高 → 吞吐暴跌。
硬约束（很可能直接报错）
- 必须 `>= max_num_seqs`，否则 `ValueError`（[scheduler.py:316](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L316)）。
- 若关闭了 chunked prefill：必须 `>= max_model_len`，否则直接报错（[scheduler.py:299](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L299)）——64 面对任何正常模型都必然失败。所以「设 64」实际前提是 chunked prefill 开启。
- arg_utils 会把 `max_num_seqs = min(max_num_seqs, max_num_batched_tokens)`（[arg_utils.py:3003](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L3003)）→ 并发上限被压到 64。
- `encoder_cache_size` 也等于它（[scheduler.py:291](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L291)）→ 多模态大图可能缓存不下。

设得很大（如 8192）
调度行为
- 一轮能吞下整个长 prompt → TTFT 最低；prefill 的 GEMM 更大、GPU 利用率更高 → 吞吐更好。
- 代价：这一轮里 prefill 占满预算，decode 请求全被推迟 → ITL 出现尖刺（延迟抖动），对高并发在线服务不友好。
内存与编译
- runner 的 `max_num_tokens = max_num_batched_tokens`（[gpu_model_runner.py:548](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_model_runner.py#L548)）决定输入缓冲与 profile 峰值：越大 → 激活显存越高 → 留给 KV cache 的越少（与 `gpu_memory_utilization` 直接竞争）。
- CUDA graph 捕获尺寸与编译时间也随之增大（[compilation.py:584](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/compilation.py#L584)、编译缓存 hash 包含它）。
- 上界：不能超过 `max_num_seqs * max_model_len`，否则报错（[scheduler.py:332](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L332)）。
**三、`gpu_memory_utilization` 从 0.9 降到 0.5，能分配的 block 数量会怎么变？对你 demo 里观察 `cached_tok` 有什么影响？**
1.Block 数：不是减半，而是掉得远不止一半
关键公式（[gpu_worker.py:626](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_worker.py#L626)）：
```
requested      = 总显存 × gpu_memory_utilization
available_kv   = requested − 权重/激活(non_kv) − CUDAGraph预留
num_blocks     = available_kv // page_size_bytes
```
注意固定开销是先减掉的，它不随 utilization 缩放。举个 80GB 卡的例子（non_kv=20GB，cudagraph≈4GB）：

|util|requested|available_kv|相对 0.9 的 block 数|
|---|---|---|---|
|0.9|72 GB|72−20−4 = **48 GB**|1×|
|0.5|40 GB|40−20−4 = **16 GB**|**≈ 1/3**|

utilization 只降了 44%，block 数却掉到 1/3。模型越大、固定开销占比越高，跌幅越夸张。极端情况（如 non_kv=46GB）：0.5 时 `available_kv` 甚至为负 → 引擎直接启动失败，报 “Try increasing `gpu_memory_utilization`” 一类错误（[kv_cache_utils.py:2560](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/kv_cache_utils.py#L2560)）。

> 另注：代码里 `CacheConfig` 的默认其实是 **0.92**，指南里写的 0.9 是取整说法（[cache.py:104](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L104)）。

2.对 demo 里 `cached_tok` 的影响
你的 demo 是两条共享前缀的 prompt 顺序执行，打印 `output.num_cached_tokens`（第一个 0、第二个 3）。这里要分清两件事：
`cached_tok` 衡量的不是「显存有多少」，而是「第一轮用过的前缀块到第二轮时还在不在池子里」。前缀缓存按整块哈希命中，块被释放后只要没被淘汰就仍可复用。

于是分两种情况：
- 块池变小但仍装得下第一条 prompt 的块（demo 只有两个短 prompt，几十个 block 就够）→ 第二轮照样命中，`cached_tok` 仍然是 3，看不变化。
- 池子小到第一轮的块被挤掉（淘汰、或被别的请求覆盖） → 第二轮找不到对应哈希块，`cached_tok`掉回 0，而且如果连一条 prompt 都放不下，直接 OOM / 启动失败。
所以这个思考题的「陷阱」在于：降低 utilization 不一定让 demo 的 `cached_tok` 变差——demo 的显存压力太小，只要引擎还能起来，命中结果通常不变。真正会变差的是高并发/长前缀场景：块池越小 → 可用块越少 → 前缀块越快被淘汰 → 命中率下降。

总结：
0.9 → 0.5 时，`num_blocks` 因「固定开销先扣」而非线性骤降（可能只剩 1/3，甚至起不来）；而 `cached_tok` 取决于前缀块是否还被池子留住——你 demo 里两条短 prompt 顺序跑，0.5 通常仍打印 3，只有当块池小到保不住第一轮的块时才会掉到 0。要观察利用率对 `cached_tok` 的真实影响，需要构造并发 + 长共享前缀的实验。
# day6
## LLM_Engine
`LLMEngine` 是 V1 的前端门面。它站在"用户代码"和"GPU 引擎核心（EngineCore）"之间，负责：
- **进方向**：把用户的 prompt 变成引擎内部能理解的 `EngineCoreRequest`
- **出方向**：把引擎吐出来的 token ids 变成用户能看懂的 `RequestOutput`（带文本）
- **连接**：持有 `EngineCoreClient`，所有真正的推理都通过它下发
**一句话概括它的定位**：`LLMEngine` = `InputProcessor`（进） + `OutputProcessor`（出） + `EngineCoreClient`（连接）。
```text
        ┌───────────────── LLMEngine ─────────────────┐

用户 ──→│  InputProcessor  →  EngineCoreClient  → ... │──→ GPU

        │  OutputProcessor ←  EngineCoreClient  ← ... │←── GPU

        └─────────────────────────────────────────────┘
```
主要代码在：`vllm/v1/engine/llm_engine.py`
**别走错门**：`vllm/engine/llm_engine.py` 只是个 6 行的别名转发文件
## llm_engine.py
**路径**：/home/dongmingzhe/vllm/vllm/v1/engine/llm_engine.py
**简介**：这个文件是 V1 里**同步版的前端门面 `LLMEngine`**（类注释自称 "Legacy LLMEngine for backwards compatibility"）：它把真正跑推理、独立运行的 `EngineCoreClient`（异步引擎核心）包装成传统的「`add_request()` + `step()`」同步接口——请求进来时负责输入处理（分词、多模态预处理、采样参数校验），转成 `EngineCoreRequest` 丢给引擎核心；输出回来时负责 detokenize，把 `EngineCoreOutput` 组装成 `RequestOutput` / `PoolingRequestOutput` 返回给调用方。此外它还承载了一堆运维/控制入口：并发与排队状态查询（`has_unfinished_requests`）、LoRA 增删查、`sleep`/`wake_up`、profiling、prefix/encoder cache 重置、metrics 获取、tokenizer 访问和 `collective_rpc`。简单说，**它是给离线 `LLM` 与同步 serving 用的「用户代码 ↔ GPU 引擎核心」之间的同步适配层**，本身不做调度或模型计算。
**本小节阅读的内容**：LLMEngine类中的重要内容
### LLMEngine类
**作用**：见上
#### 构造函数
##### part1.参数列表
```python
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        multiprocess_mode: bool = False,
    ) -> None:
```
这个 `__init__` 的参数可以按「谁在用」分成三组：
**真正驱动初始化的**
- **`vllm_config: VllmConfig`** —— 唯一的必填配置，所有子配置（model/cache/scheduler/parallel/observability…）都从它取。它派生出了 `model_config`、`observability_config`、`parallel_config`，并用来构建 renderer、`InputProcessor`、`OutputProcessor` 和 `EngineCoreClient`。
- **`executor_class: type[Executor]`** —— 执行器类，决定 worker 怎么起（单机 / 多进程 / Ray 等），原样转交给 `EngineCoreClient.make_client()`。
- **`multiprocess_mode: bool = False`** —— 是否让 EngineCore 跑在**独立进程**。为 `False` 时用 InprocClient，前端与引擎核心同进程，并额外暴露 `self.model_executor`（v0 兼容）并注册显存清理的 finalizer；同时决定 DP group 的初始化路径。
**指标/日志组**
- **`log_stats: bool`** —— 是否采集并输出统计；它同时是 `OutputProcessor` 的开关和 `StatLoggerManager` 的创建条件。
- **`aggregate_engine_logging: bool = False`**、**`stat_loggers: list[StatLoggerFactory] | None = None`** —— 只在 `log_stats=True` 时生效：前者控制多引擎日志是否聚合，后者追加自定义的指标 logger（None 即只用默认）。
**已不再使用、仅为兼容保留的**
- **`usage_context`** 和 **`mm_registry`** —— 在当前 `__init__` 体里**没有任何引用**（`usage_context` 只在几个 classmethod 的签名里出现，`mm_registry` 完全没用到）。多模态能力现在由 `renderer_from_config(vllm_config)` 自己解析。这正呼应了类注释的 "Legacy … for backwards compatibility"——它们是为不破坏旧调用方签名而留的占位参数。
##### part2
```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config

tracing_endpoint = self.observability_config.otlp_traces_endpoint
if tracing_endpoint is not None:
	init_tracer("vllm.llm_engine", tracing_endpoint)

self.log_stats = log_stats
```
把总配置存到 `self` 供后续使用；如果配了 OTLP 追踪端点，就启动 tracer。属于「一次性设置」。
```python
self.external_launcher_dp = (
	parallel_config.data_parallel_size > 1
	and executor_backend == "external_launcher"
)
# important: init dp group before init the engine_core
# In the decoupled engine case this is handled in EngineCoreProc.
if (
	not multiprocess_mode
	and parallel_config.data_parallel_size > 1
	and not self.external_launcher_dp
):
	self.dp_group = parallel_config.stateless_init_dp_group()
else:
	self.dp_group = None
```
**只有多卡数据并行时才关心**：判断是不是「外部启动器」模式；如果不是外部启动器且非多进程模式，就自己新建一个 DP 通信组。注释强调「必须在启动引擎核心**之前**建组」。`should_execute_dummy_batch` 只是初始化一个占位标志。
>外部启动器模式是什么？
>GPU 不能凭空互相通信。vLLM 做多卡推理（TP 张量并行、DP 数据并行）时，每张 GPU 对应一个独立 Python 进程。 这些进程之间要通过 `torch.distributed` 库通信，互相传张量、同步信息。
>两个核心问题必须解决：谁来**启动**这些 GPU 对应的进程？（进程生命周期）谁来**初始化进程通信组队**（TP 组、DP 组）？（分布式通信组）
>vLLM 内置了 3 种不同方案，用来解决上面这两个问题，也就是executor_backend的可选值：mp / ray / external_launcher。 所有模式都要完成两件事：拉起进程 + 初始化分布式通信。差别只在于这件事是谁干的。
>1.**mp（multiprocess，本地多进程）** 执行者：vLLM 主进程自己。vLLM 主进程运行时，内部调用`multiprocessing`，`spawn`/fork，自动创建多个子进程（每个子进程绑定 1 张 GPU）。vLLM 内部代码，调用`torch.distributed`，创建 TP/DP 通信组。进程的创建、销毁、异常重启，全部由 vLLM 管控。限制：只能单机，不能跨机器多节点。
  2.**ray** 执行者：Ray 集群框架。vLLM 请求 Ray，Ray 在单机 / 多节点集群上，帮 vLLM 启动各个 worker 进程。vLLM 依然在引擎初始化阶段，创建 TP/DP 通信组。vLLM 只需要提交任务，进程调度交给 Ray；vLLM 依然管理 worker 生命周期。适合多节点集群部署。
  3.**external_launcher（外部启动器）** 执行者：不是 vLLM，外部工具。重点：`external_launcher`本质就是：把【拉起进程、初始化全局 torch 分布式环境】这件事，交给 vLLM 外面的工具来做。 vLLM 不再负责启动 worker 进程，也不再初始化全局 distributed环境
  什么是torch.distributed？
  `torch.distributed`是 PyTorch 分布式通信库。要使用它，必须在所有进程启动之后，统一执行一次`init_process_group()`。执行之后，所有进程属于同一个`WORLD`，每个进程有唯一`RANK`编号。之后，我们可以从这个大的全局 WORLD 里，切分出更小的子通信组：TP 组、DP 组。约束：`init_process_group()`全局只能调用一次，重复调用会报错。
  DP数据并行的通信组是什么？
  DP（数据并行）：多份输入样本，分给不同 DP 组，每个 DP 组内部可能还做 TP 张量并行。 DP 组，就是从全局大 WORLD 里面切出来的子通信集合，只有同一个 DP 组的进程互相通信。在`mp`和`ray`模式下：1.vLLM 或Ray框架先拉起所有 worker 进程；2.vLLM 调用`torch.distributed.init_process_group`建立全局 world；3.vLLM 再手动划分，新建 DP 子通信组（就是代码里`stateless_init_dp_group()`） ；而在`external_launcher`模式：1.外部工具（torchrun/deepspeed）提前启动全部进程；2.外部工具提前执行`init_process_group()`，全局 distributed 环境已经就绪；3.全局 WORLD、DP 分组，全部由外部工具提前划分完成；4.代码判断：如果是 external_launcher 模式，vLLM 不能再次调用创建 DP 组的代码，否则重复创建、冲突报错 → 所以`self.dp_group = None`
```python
self.renderer = renderer = renderer_from_config(self.vllm_config)

# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(
	renderer.tokenizer,
	log_stats=self.log_stats,
	stream_interval=self.vllm_config.scheduler_config.stream_interval,
	tracing_enabled=tracing_endpoint is not None,
)
```
- **renderer**：把文本/图片变成 token，是整个流水线的公共底座（`OutputProcessor` 也拿它的 tokenizer）。
- **input_processor**：用户输入 → 引擎核心能懂的内部请求。
- **output_processor**：引擎核心吐出的内部输出 → 用户能读的结果（含 detokenize，`tracing_enabled` 与①的 tracer 呼应）。
记住这条链：`EngineInput → EngineCoreRequest → [引擎核心] → EngineCoreOutput → RequestOutput`。
##### part3.创建引擎核心客户端
```python
# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
# Hand the renderer to the client. In multiprocess mode the client
# starts the MM warmup only after engine-core fork (the why is in
# BaseRenderer.start_mm_warmup_in_background); InprocClient takes no
# renderer, so MM warmup stays inside renderer.warmup() there.
self.engine_core = EngineCoreClient.make_client(
	multiprocess_mode=multiprocess_mode,
	asyncio_mode=False,
	vllm_config=vllm_config,
	executor_class=executor_class,
	log_stats=self.log_stats,
	renderer=renderer,
)
```
真正干活的部分（调度 + 跑模型）在 EngineCore 里；这里按 `multiprocess_mode` 决定它是同进程还是独立进程，并把 renderer 一起交给它。前面part2都是在为它做准备。
>同进程与独立进程的区别是什么？
>背景1：vLLM V0的痛点，为什么要做进程拆分？
>vLLM V0 是单进程大杂烩：HTTP 服务、分词、请求预处理、调度器、GPU 模型推理全部挤在同一个 Python 进程。Python 有GIL 全局解释器锁：只要 CPU 侧做 tokenizer、多模态图像预处理、日志等 CPU 密集工作，就会抢占 GIL，阻塞调度器，GPU 会空闲，吞吐下降。V1 的思路：把 CPU 业务和 GPU 推理调度拆成两个进程，互不抢 GIL；同时提供同进程模式，兼容老离线脚本、方便调试。
>背景2：CUDA fork的致命限制？
>Linux `fork()` 创建子进程时，如果父进程已经初始化 CUDA 上下文（调用过任何 cuda api、加载模型、显存管理器 MM 初始化），fork 出来的子进程无法正常使用 GPU。
>CUDA 驱动会给每个 cuda 上下文绑定进程 PID；fork 复制了句柄，但子进程没有所有权，后续任何 GPU 调用直接报错崩溃。👉 铁律：**必须先 fork 创建子进程，再在子进程内部初始化 CUDA、显存管理器 MM、多模态 Renderer**，这就是注释里写的：
>In multiprocess mode the client starts the MM warmup only after engine-core fork
  MM = Memory Manager，vLLM 显存管理器，管理 KV 缓存、GPU 显存分配
  MM warmup：预热显存管理器，跑 dummy 输入，提前编译 CUDA graph，避免第一个真实请求触发巨大 JIT 编译延迟；多模态模型还要预热 vision encoder。
  背景3:Renderer是什么？
  `Renderer`是 vLLM V1 多模态模块，负责图像 / 视频预处理、多模态输入编码。同进程：Renderer 和 EngineCore 在同一个进程，可以直接在初始化阶段完成 warmup；多进程：Renderer 传给子进程，fork 完成之后，子进程内部才执行 MM warmup，规避 cuda fork 问题
  背景 4：EngineCoreClient的作用？
  EngineCoreClient 是代理层，对外提供统一接口，上层 LLMEngine 只调用`EngineCoreRequest/EngineCoreOutput`，不用关心底层是函数调用还是 IPC 消息。
  InprocClient：直接函数调用代理；DPLBAsyncMPClient：基于 ZMQ 的 IPC 消息代理，还支持多 EngineCore 数据并行 DP 负载均衡
  一句话总结
`multiprocess_mode`就是 vLLM V1 提供的二选一运行模式开关：
 1.False（同进程）：EngineCore 只是普通对象，直接调用，简单好调试，但受 GIL 限制、无容错；True（独立进程）：EngineCore 跑在 fork 后的独立子进程，CPU/GIL 隔离，支持 DP、故障隔离；但是必须遵守 CUDA fork 规则，MM/Renderer 预热要放到 fork 之后执行，也就是代码注释解释的核心逻辑。
#### from engine args
```python
@classmethod
def from_engine_args(
	cls,
	engine_args: EngineArgs,
	usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
	stat_loggers: list[StatLoggerFactory] | None = None,
	enable_multiprocessing: bool = False,
) -> "LLMEngine":
	"""Creates an LLM engine from the engine arguments."""
	# Create the engine configs.
	vllm_config = engine_args.create_engine_config(usage_context)
	executor_class = Executor.get_class(vllm_config)

	if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
		logger.debug("Enabling multiprocessing for LLMEngine.")
		enable_multiprocessing = True

	# Create the LLMEngine.
	return cls(
		vllm_config=vllm_config,
		executor_class=executor_class,
		log_stats=not engine_args.disable_log_stats,
		usage_context=usage_context,
		stat_loggers=stat_loggers,
		multiprocess_mode=enable_multiprocessing,
	)
```
**主要作用**：它是一个构造工厂：把用户填好的参数包 EngineArgs 翻译成引擎初始化必需的运行配置 VllmConfig 和执行器类 executor_class，然后调用本类的构造器真正 new 出一个 LLMEngine 对象返回。它自己不做推理，只负责“把参数变成组件并装配起来”，是离线（同步）路线创建引擎的统一入口。
**输入（参数）** engine_args：EngineArgs 实例，用户或上层传进来的全部配置（模型名、并行度、显存占比、调度参数等都在这一个对象里）。 usage_context：UsageContext，说明这次启动来自哪里（离线 LLM 类、API server、benchmark 等），主要用于用量统计上报。 stat_loggers：StatLoggerFactory 列表或 None，可插拔的统计日志器，None 表示用默认那套。 enable_multiprocessing：布尔值，默认 False，表示是否把引擎核心放到独立子进程运行。
**输出（改变了什么值）** 不修改任何入参，只产生新值：
- vllm_config：由 engine_args.create_engine_config(usage_context) 校验并补全默认值后得到的 VllmConfig。
- executor_class：由 Executor.get_class(vllm_config) 依据配置查表得到的一个类（不是实例）。
- enable_multiprocessing：可能被函数内部改写——只要环境变量 VLLM_ENABLE_V1_MULTIPROCESSING 为真，就被强制置为 True，优先级高于调用方传进来的值。
- log_stats：把用户参数取反得到内部开关，即 not engine_args.disable_log_stats。
- 返回值：新构造的 LLMEngine 实例，其内部已经持有 InputProcessor、OutputProcessor、engine_core（EngineCoreClient）等成员。
**术语解释**
1. EngineArgs 与 VllmConfig（参数层 vs 配置层）。EngineArgs 是面向用户的扁平参数表，命令行里的 --model、--tensor-parallel-size 就对应它的字段；VllmConfig 是面向内部的分层配置对象，由 ModelConfig（模型结构、数据类型）、ParallelConfig（张量并行/数据并行）、SchedulerConfig（调度与批处理）、CacheConfig（KV cache 显存）等子配置组合而成。create_engine_config 就是两层之间的翻译官：它做合法性校验、互相推导默认值（例如根据模型权重大小和显存去推 max_model_len、gpu_memory_utilization 的可用空间），因此“创建引擎”的第一步几乎永远是先有 EngineArgs。
2. classmethod 与 from_xxx 命名。装饰器 @classmethod 表示这个函数挂在类上而不是某个已有实例上，专门用来造实例；第一个参数 cls 就是 LLMEngine 这个类本身。Python 里这类“从某物构造出实例”的构造器习惯命名为 from...，例如 from_pretrained、from_cli_args。真正创建对象的是最后那句 return cls(...)，也就是调用 init；所以这个函数本身只是“准备工作 + 一次构造调用”。
3. executor_class、Executor 与 get_class。Executor（执行器）是“把模型真正放到设备上跑、并驱动一次前向”的那层抽象，不同并行后端有不同实现：单卡是 UniProcExecutor，多进程是 MultiprocExecutor，Ray 集群是 RayDistributedExecutor。Executor.get_class 只是一个查表函数：读 parallel_config.distributed_executor_backend 的值，返回对应的类对象。注意这里传下去的是“类”而不是“实例”，真正的实例化被推迟到 engine core 内部——这是典型的延迟实例化写法，因为多进程/Ray 场景下执行器必须在正确的进程里创建。
4. VLLM_ENABLE_V1_MULTIPROCESSING 与 multiprocess_mode。环境变量集中在 vllm/envs.py 定义，默认值是 1（开启）。multiprocess_mode 决定 EngineCore 跑在哪里：False 时用 InprocClient 同进程运行，方便调试、单步跟踪；True 时 EngineCore 跑在独立子进程，通过进程间通信和共享内存交换数据，能绕开 Python GIL 对调度循环的拖累，也做到更好的故障隔离，所以是生产默认。代码里的判断是单向覆盖：环境变量为真就把参数强制改成 True，调用方传 False 也拦不住，只能通过环境变量来关。
5. LLMEngine 与 EngineCore 的分工（对应返回值里持有的那几个成员）。LLMEngine 是“前台”，把用户输入（文本 prompt、多模态数据、采样参数）转成引擎内部请求格式 EngineCoreRequest，再把引擎返回的 EngineCoreOutputs 还原成用户可见的 RequestOutput，这一步由 InputProcessor、OutputProcessor 完成；EngineCore 是“后台大脑”，只负责调度和批量执行，不知道文本长什么样；再往下 Executor 驱动 Worker，Worker 驱动 GPU。这种分层让“文本处理”和“GPU 计算”可以放在不同进程、甚至不同机器上。
**在整个项目中的定位**
```
   用户脚本 / benchmarks / tests
              │
   LLM.__init__(llm.py:343)        AsyncLLMEngine / AsyncLLM.from_engine_args
              │                     └ 异步路线也做同样两步，但直接构造 AsyncLLM，
              │                       不走本函数（与本函数是姊妹关系）
              ▼
   ★ LLMEngine.from_engine_args()      ← 本函数：参数 → 组件 → 装配
              │
   ┌──────────┼─────────────────────────────┐
   ▼          ▼                             ▼
create_engine_config()  Executor.get_class()  envs.VLLM_ENABLE_V1_MULTIPROCESSING
   │          │                             │
VllmConfig  executor_class           multiprocess_mode
   └──────────┴──────────────┬──────────────┘
                             ▼
              LLMEngine.__init__(...)      ← 真正的构造与资源分配
                             │  ├ InputProcessor / OutputProcessor（前台翻译）
                             │  └ StatLoggerManager（统计）
                             ▼
              EngineCoreClient.make_client(...)
                             │
                             ▼
        EngineCore（调度大脑）→ Executor 实例 → Worker → GPU
```
一句话概括它的位置：本函数是“同步离线引擎”的装配入口，向上被 LLM 类和各类离线示例/测试调用，向下负责把 EngineArgs 拆成 VllmConfig 与 executor_class 并交给 **init**，是用户参数与引擎内部世界之间的那道闸门。另外提示一点，[vllm/engine/llm_engine.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/llm_engine.py) 只是把这个类导出为 LLMEngine 别名，方便旧代码从老路径导入，两者是同一个类。
#### get num unfinished requests
```python
def get_num_unfinished_requests(self) -> int:
	return self.output_processor.get_num_unfinished_requests()
```
**主要作用**:一行转发，把“当前还有多少个请求没做完”这个问题原封不动地问给内部的 OutputProcessor，然后把它数的结果返回给调用者。它自己不数数，只做转发，属于对外暴露的查询接口。
**输入（参数）** 只有 self，没有任何真实参数——调用方式是 engine.get_num_unfinished_requests()。原因见下面术语部分。
**输出（改变了什么值）** 返回一个 int（未完成请求的个数），并且这个函数以及它调用的下层都不会修改任何字段，是纯读操作、没有副作用。这个数字在上层被用作离线推理进度条的总数、测试里判断“请求是否全部收尾”（内存泄漏检查断言等于 0），以及准入控制里交给 admission_stats 做并发限流统计。注意它统计的是前台自己的记账，和调度器内部那个同名计数不是同一个东西。
**术语解释**
1. def、self 与“无参数方法”。Python 里实例方法第一个参数 self 代表“调用这个方法的那个对象”，由解释器自动传入，所以写 def f(self) 的方法在调用时写成 obj.f()，看起来没有参数。这个函数确实不需要任何输入：要问的对象就是 self 自己，要问的内容固定是“未完成数”，所以参数表里只剩 self。类比的写法是 obj.output_processor.get_num_unfinished_requests()——把问题层层转包给真正持有数据的那个对象。
2. OutputProcessor 与 request_states 记账簿（这是 self.output_processor 指的东西）。LLMEngine 内部把职责分给了几个对象，其中 OutputProcessor 是“输出侧处理器”：引擎每跑完一步会吐出一批 EngineCoreOutputs（引擎内部格式的输出），OutputProcessor 负责把它们翻译成用户看得懂的 RequestOutput（多了一段新文本、是否结束、有没有报错）。同时它还维护一个字典 self.request_states，键是请求 id，值是 RequestState（这个请求的完整运行时状态：已生成多少 token、队列、采样参数等）。这个字典就是“未完成请求登记簿”：add_request() 时写入，请求正常结束 finishrequest() 或被中止 abort 时 pop 掉。所以 len(self.request_states) 天然等于“还活着的请求数”，不需要额外维护计数器，也就不可能算错。
3. unfinished 的准确含义。这里的“未完成”指：请求已经提交进来了，但还没有产生带 finished 标记的最终输出，也没有被 abort。它既包含正在等待调度的，也包含正在生成的——所有还挂在登记簿上的都算。一旦拿到最终输出或用户取消，就会被移出登记簿，数量随之减 1。因此对整批离线任务来说，这个数字从请求总数一路降到 0；如果降到 0 之后还有残留，就说明有请求“只进不出”，这正是测试用它来查内存泄漏/请求泄漏的原因。
4. 两个同名函数不要混淆（前台计数 vs 调度器计数）。项目里还有 Scheduler.get_num_unfinished_requests()（见 [vllm/v1/core/sched/scheduler.py:2725](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/sched/scheduler.py#L2725)），名字一模一样，但数的是调度器自己的 waiting / running / 等队列长度，而且会受暂停状态 PauseState 影响（比如 PAUSED_ALL 时直接返回 0）。本函数数的是前台 OutputProcessor 的登记簿。两者一个在“后端大脑”，一个在“前台”，因为中间隔着进程间通信和批处理，数值在某一瞬间可能不完全一致。看懂这一点，就不容易被同名函数误导。
5. 这个数字的下游用途。第一是进度条：离线批量推理在 [vllm/entrypoints/offline_utils.py:589](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/entrypoints/offline_utils.py#L589) 用它当作 tqdm 的 total，因为此时请求已经全部提交但还没处理完，用“未完成数”正好是总的待办量。第二是准入控制（admission control）：OutputProcessor 每次请求收尾时会调用 `_update_admission_stats()`，内部又调回这个同名方法，把当前在途请求数交给 SharedAdmissionStats，用来在请求过多时限流或排队，避免显存被撑爆。第三是测试与自检：断言它等于 0 是判断“这批请求确实干净地结束了”的常用手段。
**在整个项目中的定位**
```
   调用方（上层）
   ├─ offline_utils.py:589  LLM._run_engine  →  tqdm(total=未完成数)
   ├─ 测试 test_chat.py / test_memory_leak.py → 断言 == 0
   └─ 用户脚本 engine.get_num_unfinished_requests()
                 │
                 ▼
   LLMEngine.get_num_unfinished_requests()        ← 本函数：一行转发（门面）
                 │ self.output_processor.…
                 ▼
   OutputProcessor.get_num_unfinished_requests()  → len(self.request_states)
                 │
                 ▼
   request_states 登记簿  ← add_request() 写入 / _finish_request()、abort() 移除
                 ▲
                 │ 每步 process_outputs() 更新
   EngineCore ──► EngineCoreClient ──► LLMEngine（前台）

   姊妹接口：AsyncLLM.get_num_unfinished_requests()（异步）
   同名但不同物：Scheduler.get_num_unfinished_requests()（后端调度器自己的队列）
```
一句话概括它的位置：它是“前台引擎”对外查询在途请求数的小窗口，向上服务进度条、测试与准入控制，向下把问题转给 OutputProcessor 的登记簿，是分层设计里最薄的那一层适配代码。
#### abort request
```python
def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
	"""Remove request_ids from EngineCore and Detokenizer."""
	request_ids = self.output_processor.abort_requests(request_ids, internal)
	self.engine_core.abort_requests(request_ids)
```
**主要作用** 取消（中止）一批请求，让它们立刻停止生成并释放占用的资源。它做的是“两路广播”：先把请求从上层的记账簿里清掉并给等待者送去一个“已中止”的最终结果，再把真正要中止的内部 id 转告给后端引擎核心。典型使用场景有两个：上层主动取消，以及批量提交或推理中途抛异常时回滚已经提交的请求，避免请求泄漏。
**输入（参数）** `request_ids：list[str]`，要中止的请求 id 列表，可以是用户自己起的 id（外部 id），也可以是引擎内部随机化的 id，取决于 internal。 internal：bool，默认 False。False 表示传进来的是外部 id，需要先翻译成内部 id；True 表示传进来的已经是内部 id，跳过翻译直接使用。
**输出（改变了什么值）** 返回值为 None，但副作用不小：OutputProcessor 的 request_states 登记簿里对应的条目被 pop 掉，external_req_ids 映射被清理，若有父请求还会级联中止它的所有子请求，等待输出的队列里会被塞入一条 finish_reason 为 ABORT 的最终输出；同时后端 EngineCore 的调度器会把这些请求标记为 FINISHED_ABORTED，从而释放它们占用的 KV cache 等显存。另外注意第一行把 request_ids 这个局部变量重新赋值成了 OutputProcessor 返回的内部 id 列表——函数签名里的参数被覆盖了，这是本函数最关键也最容易看漏的一行。
**术语解释**
1. 外部 id 与内部 id，以及 internal 参数为什么存在。用户在调用时给请求起的 id（或 vLLM 自动生成的）叫外部 id external_req_id；引擎在处理前会在 [vllm/v1/engine/input_processor.py:285](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/input_processor.py#L285) 的 assign_request_id 里把它改造成“外部 id + 8 位随机串”的内部 id request_id，避免不同用户起了同名 id 时互相打架。外部 id 与内部 id 的对应关系记在 OutputProcessor 的 external_req_ids 字典里。于是取消请求时就有两种情形：上层一般只知道外部 id（比如 pooling 离线推理里用 `request_id.split("-", 1)[0]` 还原出外部 id，见 [vllm/entrypoints/pooling/offline.py:458](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/entrypoints/pooling/offline.py#L458)），此时传 internal=False 让它去查映射；而 [vllm/entrypoints/offline_utils.py:555](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/entrypoints/offline_utils.py#L555) 抛异常回滚时手里拿的已经是内部 id，就直接传 internal=True。一个 id 可能对应多个请求（一次提交多个 prompt 或并行采样），所以映射的值是列表，外部 id 一取消就把关联的请求全部带上。
2. 为什么要同时通知前后两端。vLLM 把工作拆成前台和后台：前台（OutputProcessor，和上面说的 request_states 登记簿）负责把引擎输出翻译成给用户看的文本、并决定“这个请求是否已结束”；后台（EngineCore 里的 Scheduler）负责排班和真正调用 GPU。只清前台，后台还会继续为这个请求算 token、白烧显存；只通知后台，前台的等待者就永远等不到结束信号。所以必须两边都做，缺一不可。
3. OutputProcessor.abort_requests 具体做了什么（对应输出里的那些变化）。第一步是按 1 中的规则把外部 id 解析成内部 id 列表；第二步逐个从 request_states 中 pop 出 RequestState，并做两件收尾的事：通知 lora_states 该请求已结束，以及往这个请求的输出队列里 put 一条 finish_reason=ABORT 的 RequestOutput——这非常关键，它保证了正在 await 结果的调用方会被立刻唤醒并收到明确的“被中止”信号，而不是无限阻塞。第三步是级联处理：如果 id 对应的是一个父请求（一次 prompt 配 n>1 的并行采样，或带 best_of 之类），会先递归中止它所有的子请求再删除父请求（见 [vllm/v1/engine/output_processor.py:566](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/output_processor.py#L566)）。最后它返回真正处理过的内部 id 列表，并顺带刷新一次准入控制的统计（`_update_admission_stats`）。
4. EngineCore 那一侧和“进程边界”。self.engine_core 是一个 EngineCoreClient（客户端代理），它背后可能是同进程的 EngineCore，也可能在另一个子进程里。进程内模式直接调用 EngineCore.abort_requests，最终落到 Scheduler 的 finish_requests(..., RequestStatus.FINISHED_ABORTED)（见 [vllm/v1/engine/core.py:536](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/core.py#L536)）；多进程模式则要把这次调用通过进程间通信发过去。这正是为什么这一行看起来简单、实际却是跨进程调用——也是为什么第 3 点里“先把前台处理完、拿到内部 id 列表再通知后端”的顺序不能颠倒，否则通知的内容就是错的。
5. 把 abort 当兜底手段的用法。批量推理的代码通常长这样：在 try 块里逐个提交请求并记录 id，一旦中途抛异常，就在 except 里用这些 id 调本函数回滚，然后重新抛出（见 [vllm/entrypoints/offline_utils.py:553-556](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/entrypoints/offline_utils.py#L553-L556)、[vllm/entrypoints/pooling/offline.py:468-471](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/entrypoints/pooling/offline.py#L468-L471)）。这在业界叫“取消传播”或“清理已提交资源”：报错时不能让已经进了引擎的请求继续占着 KV cache，否则下一次运行可能因为显存不足而失败。理解这一点，就明白本函数不只是“用户点取消”时才用的功能接口，也是错误处理路径上的基础设施。

**在整个项目中的定位**
```
   上层调用方
   ├─ offline_utils.py:555    批量提交中途异常 → abort(已提交id, internal=True)
   ├─ pooling/offline.py:470  推理循环异常     → abort(在途id, internal=False)
   └─ 用户 / AsyncLLM 异步取消（async_llm 里的 await 版本，结构相同）
                  │
                  ▼
   LLMEngine.abort_request(request_ids, internal)      ← 本函数：两路广播
                  │
      ┌───────────┴────────────────────────────┐
      │ ① 先问前台（并拿到内部 id）             │ ② 再用内部 id 通知后端
      ▼                                        ▼
  OutputProcessor.abort_requests()      EngineCoreClient.abort_requests()
   · 外部id → 内部id 映射查询             · 同进程：EngineCore.abort_requests()
   · pop request_states、父→子级联        · 多进程：走 IPC 发给子进程
   · 队列塞一条 finish_reason=ABORT 的      · 最终：Scheduler.finish_requests(
     最终输出（唤醒等待者）                     RequestStatus.FINISHED_ABORTED)
   · 返回真正的内部 id 列表 ────┐             · 释放 KV cache 等显存
                               └── 就是第 ② 步的输入
```
一句话概括它的位置：它是前台引擎统一的“取消请求”入口，向上服务异常回滚与主动取消两类调用者，向下把一次取消拆成“清前台记账 + 通知后端调度器”两个动作串联执行，是前后台之间一个典型的双写协调点。
#### add request
```python
def add_request(
	self,
	request_id: str,
	prompt: EngineCoreRequest | PromptType | EngineInput,
	params: SamplingParams | PoolingParams,
	arrival_time: float | None = None,
	lora_request: LoRARequest | None = None,
	tokenization_kwargs: dict[str, Any] | None = None,
	trace_headers: Mapping[str, str] | None = None,
	priority: int = 0,
	session_id: str | None = None,
	prompt_text: str | None = None,
) -> str:
	# Validate the request_id type.
	if not isinstance(request_id, str):
		raise TypeError(f"request_id must be a string, got {type(request_id)}")

	# Process raw inputs into the request.
	if isinstance(prompt, EngineCoreRequest):
		logger.warning_once(
			"Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
			"is deprecated and will be removed in the future. You should "
			"instead pass the outputs of Renderer.render_cmpl() or "
			"Renderer.render_chat()."
		)

		request = prompt
		if request_id != request.request_id:
			logger.warning_once(
				"LLMEngine.add_request() was passed a request_id parameter that "
				"does not match the EngineCoreRequest.request_id attribute. The "
				"latter will be used, and the former will be ignored."
			)
	else:
		request = self.input_processor.process_inputs(
			request_id,
			prompt,
			params,
			supported_tasks=self.get_supported_tasks(),
			arrival_time=arrival_time,
			lora_request=lora_request,
			tokenization_kwargs=tokenization_kwargs,
			trace_headers=trace_headers,
			priority=priority,
			session_id=session_id,
		)
		prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

	self.input_processor.assign_request_id(request)

	req_id = request.request_id

	# Use cloned params that may have been updated in process_inputs()
	params = request.params

	n = params.n if isinstance(params, SamplingParams) else 1

	if n == 1:
		# Make a new RequestState and queue.
		self.output_processor.add_request(request, prompt_text, None, 0)
		# Add the request to EngineCore.
		self.engine_core.add_request(request)
		return req_id

	# Fan out child requests (for n>1).
	parent_req = ParentRequest(request)
	for idx in range(n):
		request_id, child_params = parent_req.get_child_info(idx)
		child_request = request if idx == n - 1 else copy(request)
		child_request.request_id = request_id
		child_request.sampling_params = child_params

		# Make a new RequestState and queue.
		self.output_processor.add_request(
			child_request, prompt_text, parent_req, idx
		)
		# Add the request to EngineCore.
		self.engine_core.add_request(child_request)

	return req_id
```
**主要作用** 提交一个推理请求：把用户给的原始输入（文本、token id、多模态数据等）加工成引擎内部的请求对象 EngineCoreRequest，在前台的输出账本里登记一条，再把它交给后端 EngineCore 排队执行。如果采样参数里的 n 大于 1（一次要生成多个结果），它会自动把这个请求拆成 n 个子请求分别提交，最后统一返回父请求的内部 id。
**输入（参数）** 
* request_id：用户侧的请求标识，必须是字符串，否则直接抛 TypeError。 
* prompt：要推理的内容，可以是一条原始提示（文本、token id 列表等），也可以是已经渲染好的 EngineInput 字典，甚至是一个构造好的 EngineCoreRequest（属于旧用法，会打警告）。 
* params：采样参数 SamplingParams（做生成）或池化参数 PoolingParams（做 embedding/打分），决定温度、top_p、生成长度等。 
* 其余可选参数：arrival_time（到达时间，影响统计与调度指标）、lora_request（这次要挂哪个 LoRA 适配器）、tokenization_kwargs（分词时的额外参数）、trace_headers（链路追踪头）、priority（优先级）、session_id（会话 id）、prompt_text（原始提示文本）。
**输出（改变了什么值）** 返回值是一个字符串：内部 request id（在外部 id 后面追加了 8 位随机串）；注意 n>1 时返回的是父请求的 id，子请求的 id 不会返回给调用者。副作用方面：OutputProcessor 的 request_states 登记簿里多出条目（n>1 时是 n 个子请求 + 一条父请求记录），EngineCore 那边收到请求开始排队，同时函数内的局部变量 request、params、req_id 被重新赋值。函数不修改调用者传入的任何对象（除了 n>1 分支里最后一个子请求复用的那个 request）。
**术语解释**
1. 开头的类型检查与三分支结构（对应输入里的 prompt 和 request_id）。函数第一件事是 assert 式地检查 request_id 必须是 str，这是为了让后续 id 拼接（如 "0_" + id）不会因为传了整数而炸掉，把错误在使用方暴露而不是幕后暴露。接着按 prompt 的类型分两条路：如果传进来已经是 EngineCoreRequest（引擎的内部请求格式，包含 token id、采样参数、多模态数据等），就直接用它，并打两条一次性警告——一条说这种传法已废弃、以后应该传 Renderer.render_cmpl() 或 render_chat() 的结果，另一条说此时函数参数 request_id 会被忽略、以请求对象自带的 request_id 为准（这就是 [vllm/v1/engine/llm_engine.py:241-248](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/llm_engine.py#L241-L248) 那段 warning_once 的含义，once 表示只打一次不刷屏）。否则走正常路径，交给 InputProcessor.process_inputs 去加工。
2. process_inputs 到底做了什么（对应下面那句“Use cloned params that may have been updated”的注释）。这是本函数调用链里最重的一步，在 [vllm/v1/engine/input_processor.py:306](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/input_processor.py#L306)：先校验 params 与模型支持的任务是否匹配（supported_tasks 是模型能干什么的清单，例如生成、embedding、打分；注意本函数把 self.get_supported_tasks() 传了进去，这也是这段代码多一次调用的原因）、校验 LoRA；然后把原始 prompt 分词、渲染成 EngineInput，再把输入按编码器/解码器拆开；最后把 SamplingParams 克隆一份（clone），补上没填的字段——最典型的是 max_tokens 为空时按“模型最大长度减去提示长度”填满，并用模型的 generation_config 覆盖一部分默认值。所以在 process_inputs 之后，request.params 已经不是你传进来的那个对象了，函数里那句 params = request.params 就是“改用加工后的那份参数”，后面的 n 也是从这里读的。
3. assign_request_id 与内部 id（对应输出里的返回值）。[vllm/v1/engine/input_processor.py:288](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/input_processor.py#L288) 的 assign_request_id 把用户提供的 id 存进 external_req_id，再把 request_id 改写成“用户 id + 短横线 + 8 位随机串”，例如 "3-a1b2c3d4"。这么做是为了避免不同调用方起了重名 id 时在引擎内部互相覆盖。它与前面那节讲的 abort_request(request_ids, internal) 是配套的：上层手里通常只有外部 id，所以取消时要传 internal=False 让引擎去查这张映射表；本函数返回的就是加工之后的内部 id，谁拿到它就可以用 internal=True 直接取消。
4. n>1 的并行采样与 ParentRequest（对应主要作用里的“拆成 n 个子请求”）。SamplingParams.n 表示“这条提示要生成几个独立结果”，比如 n=3 就是一次问三个答案。vLLM 内部不会让一个请求带着 n=3 跑，而是拆成 3 个互不相干的子请求，各自独立采样，最后再合并成一条对用户可见的响应（用参数里的 best_of 时更明显，先多采样再挑最好的）。ParentRequest（[vllm/v1/engine/parallel_sampling.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/parallel_sampling.py)）就是这次拆分的记账人：它记录父请求 id、维护子请求 id 集合，并在所有子请求结束后把结果聚合回一条输出。子请求 id 的格式是 `index_父id`（见 `get_child_info` 里的` f"{index}_{self.request_id}"`），子请求的采样参数 n 被改成 1，如果设了随机种子 seed，则第 i 个子请求用 seed+i，保证每个子请求可复现又不重复。
5. 两次 add_request 的先后顺序为什么不能颠倒（对应输出里的两个副作用）。代码永远先调 self.output_processor.add_request(...) 再调 self.engine_core.add_request(...)。原因是引擎可能在极短时间内就把第一步的输出送回来，而处理输出时需要按 id 去 request_states 里查这条请求的状态；如果先交给后端，输出先到时就会查不到、直接报错或丢结果。这条“先登记再提交”的顺序在多进程模式下更明显——engine_core 那一侧可能只是把请求塞进 IPC 队列，返回得极快。
**在整个项目中的定位**
```
   用户代码 / LLM.generate / AsyncLLM
            │  offline_utils._add_request()（自增计数器生成 request_id）
            ▼
   LLMEngine.add_request()                    ← 本函数：请求的统一上岸点（同步版）
            │
     ┌──────┴───────────────────────────────────────┐
     │ prompt 是 EngineCoreRequest？                │ 否（正常路径）
     │  是 → 直接用 + 打废弃警告                     ▼
     │                        InputProcessor.process_inputs()
     │                        校验 params/LoRA、渲染、clone params、补 max_tokens
     │                                      │
     │                        InputProcessor.assign_request_id()
     │                        外部 id → 内部 id（追加 8 位随机串）
     │                                      │
     │                        params = request.params；n = params.n
     │                                      │
     │            ┌─────────────────────────┴──────────────────────────┐
     │         n == 1                                              n > 1
     │            │                                                   │
     │            │                          ParentRequest 拆分：循环 n 次，
     │            │                       子 id = "i_父id"，第 i 个 seed+1，
     │            │                              最后一个子请求复用原对象
     │            └─────────────────────────┬──────────────────────────┘
     ▼                                      ▼
   OutputProcessor.add_request()      登记 request_states + 输出队列（先）
   EngineCoreClient.add_request()     交给后端（后）
        ├ 同进程 InprocClient → EngineCore.add_request → Scheduler 排队
        └ 多进程 SyncMPClient → _send_input(ADD, ...) 经 IPC 进子进程
                                            │
            调度 → 执行 → 输出 ──► OutputProcessor.process_outputs ──► 用户
     （取消时回到上一节的 abort_request，用这里返回的内部 id 级联中止）
```
一句话概括它的位置：它是前台引擎的“请求入口”，向上被 LLM.generate 之类的离线接口和异步接口调用，向下把一次提交拆成“加工参数 → 换内部 id → 登记前台 → 交给后端”四步串联执行，是用户世界与引擎内部世界之间那道双向闸门的上行方向（下行方向就是前面讲的输出处理与 abort_request）。
#### step
```python
def step(self) -> list[RequestOutput | PoolingRequestOutput]:
	if self.should_execute_dummy_batch:
		self.should_execute_dummy_batch = False
		self.engine_core.execute_dummy_batch()
		return []

	# 1) Get EngineCoreOutput from the EngineCore.
	with record_function_or_nullcontext("llm_engine step: get_output"):
		outputs = self.engine_core.get_output()

	# 2) Process EngineCoreOutputs.
	with record_function_or_nullcontext("llm_engine step: process_outputs"):
		iteration_stats = (
			IterationStats() if self.log_stats and outputs.outputs else None
		)
		processed_outputs = self.output_processor.process_outputs(
			outputs.outputs,
			engine_core_timestamp=outputs.timestamp,
			iteration_stats=iteration_stats,
		)
		self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

	# 3) Abort any reqs that finished due to stop strings.
	with record_function_or_nullcontext("llm_engine step: abort_requests"):
		self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

	# 4) Record stats
	with record_function_or_nullcontext("llm_engine step: record_stats"):
		if self.logger_manager is not None and outputs.scheduler_stats is not None:
			# Record even when this step produced no request outputs.
			self.logger_manager.record(
				scheduler_stats=outputs.scheduler_stats,
				iteration_stats=iteration_stats,
				mm_cache_stats=self.renderer.stat_mm_cache(),
			)
			if outputs.outputs:
				self.do_log_stats_with_interval()

	return processed_outputs.request_outputs
```
**主要作用** 这是引擎的一步心跳：向后端要一轮已经算好的结果，把 token 级别的原始输出翻译成用户看得懂的文本输出，顺手把因为停用词而该停的请求通知后端中止，再记一笔统计，最后把这一轮产生的输出列表返回给调用者。上层只需要在一个 while 循环里反复调用它，直到没有未完成的请求为止。
**输入（参数）** 只有 self，没有显式参数——它是无参方法，输入来自两个地方：一是实例状态 self.should_execute_dummy_batch，二是后端通过 self.engine_core.get_output() 实时送回来的数据（一批 EngineCoreOutput、调度统计 scheduler_stats、时间戳 timestamp）。
**输出（改变了什么值）** 返回一个列表，元素是 RequestOutput（文本类任务）或 PoolingRequestOutput（池化类任务），可能为空。副作用链：should_execute_dummy_batch 被重置为 False（走 dummy 分支时）、OutputProcessor 的 request_states 账本里结束的请求被清理、后端调度器可能收到一批中止指令、统计管理器记录一轮指标、以及日志时间戳 `self._last_log_time` 被更新。另外要注意 dummy 分支会提前 return []，不走后面任何一步。
**术语解释**
1. 为什么叫 step，以及为什么要循环调用（对应主要作用）。step 这个词沿用了 vLLM 早期版本的命名：一次 step 对应引擎“跑一轮”——调度一批请求、做一次模型前向、把结果收上来。它天然是循环里的一个动作，调用方长这样（见 [vllm/entrypoints/offline_utils.py:603](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/entrypoints/offline_utils.py#L603)）：while self.llm_engine.has_unfinished_requests(): outputs = self.llm_engine.step()。理解这点就明白本函数为什么不接受参数：要处理哪些请求，是后端根据当前队列自己决定的，前端不需要告诉它。
2. 第 1 步 get_output 拿到了什么（对应输入里“后端送回来的数据”）。self.engine_core 是客户端代理：同进程时 InprocClient.get_output() 直接驱动一次引擎内部循环并把结果取出来（见 [vllm/v1/engine/core_client.py:370](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/core_client.py#L370)），多进程时则从进程间通信队列里取，取不到就阻塞等待。返回的 EngineCoreOutputs 是 msgspec 定义的结构体，三个关键字段是：outputs（一批 EngineCoreOutput，每个只含 request_id 和新生成的 token id，是“机器视角”的产物）、scheduler_stats（这一步调度器的统计：等待多少、运行多少、开了多少 KV cache 块）、timestamp（后端产生这批结果的时刻）。注意 outputs 可能为空列表，比如这一步只完成了调度、还没算出任何 token。
3. process_outputs 做的翻译工作（对应主要作用里的“翻译成文本”，也是输出的主要来源）。这是本函数里最重的一步，在 [vllm/v1/engine/output_processor.py:641](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/output_processor.py#L641)。EngineCoreOutput 里只有 token id，用户要看的是文字，所以要经过增量解码器（detokenizer，Attention 那类模型按 token 逐个生成，中文等多字节字符可能被切成半个，必须跨步拼起来才能正确还原）。这一步还负责：判断是否达到终止条件、计算 logprobs（只在实际请求时才算）、按 stream_interval 决定“每生成几个 token 才吐一个输出”（流式返回不用每个 token 都发一次）、以及把结束的请求从 request_states 里清掉。它返回的 OutputProcessorOutput 里有两个列表：request_outputs（给用户的）和 reqs_to_abort（要通知后端停下的）。
4. iteration_stats 与统计记录（对应输出里“记录一轮指标”）。IterationStats 是“迭代级”的统计（这一轮各请求的耗时、生成了多少 token 等），与 SchedulerStats 这种“系统级”统计互为补充，最终都会被 logger_manager.record(...) 打包输出到日志或 Prometheus。注意它只有在 self.log_stats 为真且这一轮确有输出时才创建，否则传的是 None，相当于关掉了细粒度统计以省开销。后面那句 if outputs.outputs: self.do_log_stats_with_interval() 也值得留意：统计记录每轮都做，但“打印日志”按时间间隔节流（默认由环境变量 VLLM_LOG_STATS_INTERVAL 控制，见 [vllm/v1/engine/llm_engine.py:410](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/llm_engine.py#L410)），并且只在真的有输出时才考虑打印，避免空转时刷屏。
**在整个项目中的定位**
```
   上层：离线推理主循环
   offline_utils._run_engine / pooling/offline.py
        while llm_engine.has_unfinished_requests():
              outputs = llm_engine.step()       ← 本函数：一次 step = 一轮心跳
                       │
     ┌─────────────────┴──────────────────────────────────────────┐
     │ 前置分支（DP 空转）：should_execute_dummy_batch == True      │
     │   → engine_core.execute_dummy_batch()，只为让各 DP rank     │
     │     的集合通信不卡死，然后直接 return []                     │
     └─────────────────┬──────────────────────────────────────────┘
                       ▼ 正常路径
   ① engine_core.get_output()          阻塞取回这一步的结果
        InprocClient：直接驱动内部循环；MPClient：从 IPC 队列取
        产出 EngineCoreOutputs{outputs, scheduler_stats, timestamp}
                       ▼
   ② output_processor.process_outputs()
        token id → 文本：增量解码 / 结束判断 / logprobs / 流式分段
        产出 request_outputs（给用户）+ reqs_to_abort（给后端）
                       ▼
   ③ engine_core.abort_requests(reqs_to_abort)
        前端检测到停用词，通知后端真正停下（补前后端分工的错位）
                       ▼
   ④ logger_manager.record(...)  记统计；按间隔节流地打日志
                       ▼
        return processed_outputs.request_outputs  → 交给上层收集
                       │
        （后端一侧的循环）EngineCore：Scheduler 调度 → Executor → Worker → GPU
                                        └─► 下一轮结果回到 ①
```
一句话概括它的位置：它是前台引擎的驱动齿轮——向上被离线推理的 while 循环和异步引擎的事件循环反复调用，向下通过 get_output 驱动后端跑一轮、通过 process_outputs 把机器输出变成人的输出、再通过 abort_requests 把前端的判断反馈给后端，是“请求提交（add_request）— 结果产出（step）”这条主干上的产出侧对应物。
## 阶段型自检
**1.`LLMEngine` 自己有没有直接碰 GPU？如果没有，那 GPU 是在哪个对象里被使用的？**
没有。LLMEngine 全程不导入 torch 做计算，也不持有模型权重，它手里只有 vllm_config、input_processor、output_processor 和 engine_core（EngineCoreClient），干的是文本/参数层面的活。
GPU 实际在后端这条链上被使用：EngineCore（调度出这一轮跑哪些请求）→ Executor（进程内/多进程/Ray 的调度器）→ Worker（GPUWorker）→ ModelRunner（真正调用模型前向、kv cache 也在这里）。所以 LLMEngine 与 GPU 之间隔着进程边界（多进程模式下甚至是另一个进程），它只能通过 engine_core 这个客户端代理间接驱动。

**2.`step()` 里 `engine_core.get_output()` 这一行，为什么说是"驱动"而不是"读取"？**
因为在同进程模式下，InprocClient.get_output() 内部直接调用了 engine_core.step_fn()——也就是真跑了一轮引擎循环（调度 + 模型前向 + 产出结果）。不是从某个缓冲区里取现成的数据，而是这次调用本身触发了计算，所以叫“驱动”。
补充一点：多进程模式下它退化成从 IPC 队列里阻塞取结果，此时后端有自己的循环在跑，语义更接近“等待读取”。所以“驱动”严格成立的是 inproc 那条路径。

**3.`add_request` 为什么必须先 `output_processor.add_request()` 再 `engine_core.add_request()`？调换顺序会出什么问题？**
因为后端可能立刻就把这个请求的结果产出来，而 process_outputs 处理输出时要按 id 去 output_processor.request_states 里查状态。
调换顺序的话，结果先到时查不到对应的 RequestState，[output_processor.py:672](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/output_processor.py#L672) 会走 `if req_state is None: continue`，这批输出就被静默丢弃了——不是报错，而是丢 token 或请求永远收不到结果，很难排查。所以这条顺序是“先建账本、再下单”。

**4.内部 `request_id` 为什么要加一个随机后缀？如果你用 UUID 直接做外部 id，还需要这个后缀吗？**
核心是为了保证唯一性：外部 id 由调用方自己起，不同客户端/不同代码路径完全可能撞名，而引擎内部到处拿 request_id 当字典键（request_states、调度队列、KV cache 块表），撞名会直接覆盖或串号。加 8 位随机后缀就把不同来源的同名请求彻底分开。
用 UUID 做外部 id 的话，实际碰撞概率已经可忽略，这个后缀在功能上就不再必要了；它只是默认的防御性做法。vLLM 也留了开关 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION，关掉后内部 id 就等于外部 id（代码里同时警告重名可能导致失败或隐蔽的正确性问题）。

**5.`has_unfinished_requests()` 是从哪里判断的？它看的是引擎的状态还是前端 `OutputProcessor` 的状态？**
主要是前端 OutputProcessor 的状态：底子是 `len(self.request_states) > 0`，即前台登记簿里还有没有活着的请求。
但它不是只看前台，还要或上后端的一个信号 `self.engine_core.dp_engines_running()`（[llm_engine.py:195-199](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/llm_engine.py#L195-L199)）：因为 DP 场景下请求可能被别的引擎实例接走了，本实例前台是空的，但全局还有活，此时不能返回 False 让循环退出。所以准确说是“前台账本为主，后端 DP 运行状态兜底”。
