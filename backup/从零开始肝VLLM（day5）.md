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
`EngineArgs` 只是个中转站：CLI → EngineArgs 字段 → CacheConfig。注意相邻的 `mamba_block_size` 是给 Mamba/线性注意力层的独立块大小。
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

⑦ 合法性校验 — [config/vllm.py:3199](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L3199) `validate_block_size` 在后端定稿后校验与 DCP（decode context parallel interleave size）、Mamba 的约束是否相容。

**4、总结**
`block_size` 在 arg_utils.py 里只承担「CLI → 配置」的转发；真正决定它最终值的顺序是：CLI(`--block-size`) → 用户指定则固定，否则由 attention 后端偏好决定 → 混合模型/多 dtype 再对齐 → config/vllm.py 校验；之后它贯穿 KV cache 的形状、显存分配、块表寻址和前缀缓存命中粒度。
### max_num_seqs参数
**1.这是什么？**
单次调度迭代（一个 batch）中最多同时处理多少个序列（请求），即调度器和模型 runner 的「并发上限」。
- 沿用 `SchedulerConfig.DEFAULT_MAX_NUM_SEQS = 128`（[scheduler.py:47](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L47)）。
- 它同时驱动三个东西：调度器的准入上限、model runner 的 per-request 静态缓冲区大小、CUDA graph 的捕获 size。所以它不只是个调度数字，还直接决定显存占用。
- 区别：`max_num_batched_tokens` 限制的是一轮的 token 数，`max_num_seqs` 限制的是一轮的序列数；`max_num_active_seqs` 是「只降准入、不缩 runner/graph」的变体。

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
完整流程：
```text
CLI：可选传入 --max-num-seqs
    ↓
argparse解析 → EngineArgs.max_num_seqs
    有用户值 → 赋值为用户数字
    无用户值 → 保持 None
    ↓
调用 _set_default_max_num_seqs_and_batched_tokens_args()
    ① get_batch_defaults() 按硬件拿基础值
    ② throughput模式、未手动指定 → 翻倍
    ③ min(max_num_seqs, max_num_batched_tokens)钳位
    ↓
断言：max_num_seqs 必须为int
    ↓
LoRA+投机解码 交叉校验token容量约束，不满足直接报错
    ↓
传给 SchedulerConfig(max_num_seqs=xxx, max_num_batched_tokens=xxx)
    ↓
Scheduler：调度准入，控制单batch最多序列数量
Runner：预分配per-request静态缓冲区
CUDA Graph：捕获的最大并发规模（graph捕获时会按这个上限预留张量尺寸）
```

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
④ 编译缓存 key 与 TP 校验
- `SchedulerConfig.compute_hash()` 把 `max_num_seqs` 纳入哈希（[scheduler.py:269](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L269)），因为 per-request buffer 形状被编进图里；
- `max_num_seqs` 小于 `tp_size` 时告警（[vllm.py:3009](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L3009)）。

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
>可能的问题：
>1.vLLM 的每个实例是什么意思？
>2.设置 kvcache 的预算有什么意义？当预算不够时是会限制decode生成token的个数还是直接会舍弃部分 token 的 kv？
>豆包解答：
>1.一个 `LLMEngine` 就是一个 vLLM 实例，对应一个独立的 vLLM 服务进程（`vllm serve` 启动一次 = 一个实例）。
>细节：
>- 一个实例：单独启动一次 `vllm serve`，有独立的：`Scheduler`、KV Cache 块池、模型权重、`EngineArgs`、参数（`gpu_memory_utilization` / `max_num_seqs`）。
>- 同一张 GPU 上，可以手动启动多个独立 vLLM 进程（多个实例）。
>- 重点：实例之间完全隔离，互不感知。 比如 A100 80G，同卡起 2 个 vLLM 实例，都设置`gpu_memory_utilization=0.5`：
    - 实例 1：按 80G × 0.5 = 40G 预算；
    - 实例 2：同样按 80G × 0.5 = 40G 预算；
    - 合计预算 80G，但系统真实显存只有 80G，两个实例不会互相协商。如果两个实例同时把预算打满，会直接触发 OOM，vLLM 不会自动跨实例做显存调度。
    - 区分：不是 worker。分布式 TP/PP 多 worker 的情况：同一个 LLMEngine 实例下的多个 GPU worker，共享同一份`gpu_memory_utilization`配置，属于同一个实例，不是多个实例。
>2. vLLM 启动阶段会用这个预算提前一次性分配 KV Cache 的块池（连续块内存），块池大小直接决定：最多能存多少 KV 块、能承载多少并发序列 + 上下文长度。当KV Cache预算不够的时候，vLLM不会主动丢弃已有的token kv，核心行为将会分为两层：
>**调度器侧**：拒绝新请求准入，不把新请求加入 batch。当 KV Cache 块池已经满了，调度器`Scheduler`不会再调度新的 pending 请求进 running 队列；新请求卡在 pending 排队，等待正在运行的序列生成结束、释放 KV 块。
>**正在运行的序列：** 如果序列继续生成，还需要申请新 KV 块，但块池已经耗尽 → 直接 OOM（CUDA out of memory），进程崩溃 
 
补充区分两个容易混淆的参数
- `max_model_len`：单条请求最大上下文上限。单条 prompt + 生成 token 超过这个长度，请求直接报错拒绝，这条请求本身就不会被调度。
- KV Cache 预算（由 gpu_memory_utilization 算出的总 KV 池）：全局所有 running 序列 KV 总和上限。单条序列没超过 max_model_len，但全局 KV 资源耗尽 → 不再放新请求；running 序列继续生成耗尽剩余块则 OOM。

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
>可能的问题：
>前缀缓存如何进行复用，复用的 kv 是不是那些还没被回收的？
>复用的对象：全局哈希表中存在、且还没有被物理回收（evict）的满 KV block。但块可以处于两种状态：①正在被请求使用（ref_cnt>0）；②已经没人在用（ref_cnt=0），但还留在块池里、没有被分配给别的新块。
>举一个例子：
>请求 A：`system prompt（32 token） + 用户问题A`
>- system prompt 刚好占 2 个 full block，prefill 完成后，两个块计算 hash 存入全局哈希表，ref_cnt=1。 请求 A 推理完成 → ref_cnt 变成 0，块放回空闲链表，哈希表还保留，KV 数据还在显存。
  请求 B 进来，同样 system prompt：
  - 计算前两个 block 哈希，命中哈希表，直接复用这两个物理块，ref_cnt 重新 + 1，跳过这 32token 的 prefill，TTFT 下降；后面用户问题部分正常 prefill。
> 只要这两个块还没有被 LRU 驱逐，哪怕 A 早就结束很久，B 依然可以复用。 如果在 A 结束后，大量新请求打满块池，空闲块用光，vLLM 会把这两个 ref_cnt=0 的块 evict，哈希删除，B 再来就无法命中。

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

**3.在其它文件中的体现**
① 定义与配套参数 — [config/cache.py:130](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/cache.py#L130) 默认 True；紧邻的 `prefix_caching_hash_algo`（sha256/xxhash…）和 `prefix_match_unit` 是它的直接配套。
② 决定是否构建 block hasher —— 运行时的开关点 — [v1/engine/core.py:230](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/core.py#L230)
```python
if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
    self.request_block_hasher = get_request_block_hasher(hash_block_size, ...)
```
关掉它，这张请求→块哈希的管线就完全不建立。
③ 传给 KV cache manager — [v1/core/sched/scheduler.py:312](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/sched/scheduler.py#L312)
```python
enable_caching=self.cache_config.enable_prefix_caching
```
这是它真正生效的地方：`KVCacheManager` 的 `enable_caching` 开关，决定分配块时是否走「先查哈希、命中则复用」的分支，以及块释放后是否保留哈希可被后续命中。
④ 多处自动强制关闭（比参数本身更容易踩坑）
- 非因果注意力层（如 Prefix-LM）→ 关闭，因为缓存复用假设因果注意力（[core.py:284](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/core.py#L284)）；
- 纯 MM encoder 实例 → 没有 KV cache 可复用（[vllm.py:2036](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L2036)）；
- 部分模型直接在其配置里置 False（[models/config.py:176](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/model_executor/models/config.py#L176)、[config.py:622](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/model_executor/models/config.py#L622)）。 所以「命令行开了但实际没生效」是常见现象，要查日志。
⑤ 与其它特性联动
- `mamba_block_size` 被设置时依赖它开启（[vllm.py:3257](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L3257)）；
- KV cache events 开启但前缀缓存没开 → 告警（[vllm.py:1843](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L1843)）；
- 块哈希计算本身在 [kv_cache_utils.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/kv_cache_utils.py)（Request→BlockHash 的构建）。

**4.总结**
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

**3.在其它文件中的体现**
① 校验规则 — [config/scheduler.py:126](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L126)、[scheduler.py:299](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L299) `verify_max_model_len()`：未开启且 `max_num_batched_tokens < max_model_len` 直接报错——这是它最硬的一条约束。
② 自动强制关闭
- encoder-decoder 模型：prefill/前缀缓存一并禁用（[scheduler.py:281](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L281)）；
- 没有任何 KV cache 的模型（如纯 encoder）→ 关闭（[core.py:158](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/core.py#L158)）；
- 含非因果注意力层的模型（Prefix-LM）→ 关闭，因为切块假设因果注意力（[core.py:279](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/engine/core.py#L279)）。
③ 运行时的核心分支 — [v1/core/sched/scheduler.py:1114](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/core/sched/scheduler.py#L1114)
```python
if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > request_token_budget:
    break   # 不能切块 → 放弃调度这个请求
```
这是它在调度循环里唯一、也是最本质的体现：开关直接决定了 token 预算不够时是「切一刀继续」还是「整个请求让路」。
④ 编译/显存侧的连带 — 开启时 `max_num_batched_tokens` 不必 ≥ `max_model_len`，从而 CUDA graph 捕获上限和 KV cache 峰值都更小（见 [config/vllm.py:1554](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L1554) 的 dtype 特例与 [vllm.py:2700](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L2700) 的日志）。

**4.总结**
`enable_chunked_prefill` 在 arg_utils.py 里同样是 `None` 哨兵 + 由 `is_chunked_prefill_supported` 推断，并与 `max_num_batched_tokens` 强耦合（关了就得把它抬到 `max_model_len`）；它在运行时落成调度器的一行分支——token 预算不足时是切块续跑还是停止调度。encoder-decoder、无 KV cache、非因果注意力三类模型会被自动强制关闭。
### enforce_eager参数
**1.这是什么？**
强制整个模型用 PyTorch eager 模式执行，即关闭 `torch.compile` 与 CUDA graph。
- 默认False（[model.py:242](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L242)）。此时是混合模式：小 batch 的 decode 走 CUDA graph（省 kernel launch 开销），prefill 走编译/eager。
- 设 True 则所有路径都是 eager。用途：调试定位问题、规避硬件/模型不支持 CUDA graph 或编译的情形、省掉编译等待时间和 graph 显存；代价是吞吐显著下降。
- 注意它只管「图」，不影响调度或算子实现。

**2.在本文件中的体现**
本文件对它纯粹是转发，没有任何特殊逻辑，这是它和前面几个参数最大的不同：
a) 字段（[arg_utils.py:599](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L599)）
```python
enforce_eager: bool = ModelConfig.enforce_eager
```
普通 `bool`，直接继承 `ModelConfig` 的默认 False——没有 `None` 哨兵，也没有「按模型能力推断」的逻辑，用户给了就是最终值。
b) CLI 参数（[arg_utils.py:923](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L923)）
```python
model_group.add_argument("--enforce-eager", **model_kwargs["enforce_eager"])
```
归到 Model 参数组（属于 `ModelConfig`），default 直接沿用 pydantic 的 False。
c) 注入 ModelConfig（[arg_utils.py:1858](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/engine/arg_utils.py#L1858)） 作为 `ModelConfig(...)` 的一个普通参数传入，不参与 CacheConfig/SchedulerConfig

**3.在其它文件中的体现**
① 定义 — [config/model.py:242](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L242) 文档直接点明语义：True → 永久 eager；False → graph 与 eager 混合以获得性能。
② 会被自动强制置 True 的特例 — [model.py:1364](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L1364) `_verify_cuda_graph()` ROCm 上的 encoder-decoder 模型不支持 CUDA graph → 自动 fallback 到 eager 并打 warning。所以「我没开 enforce_eager，但日志说它在 eager 模式」是正常现象。
③ 真正生效的地方 —— 关掉编译与图 — [config/vllm.py:1564](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L1564)
```python
if model_config.enforce_eager:
    compilation_config.mode = CompilationMode.NONE
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
```
并在 [vllm.py:1797](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L1797) 进一步把 `max_cudagraph_capture_size = 0`、`cudagraph_capture_sizes = []` 清空。日志会明确提示等价于 `-cc.mode=none -cc.cudagraph_mode=none`。这两处就是该参数在整个 vLLM 里的实际效果：它不是给模型用的，是给 `CompilationConfig` 用的开关。
④ 阻止 graph 尺寸计算 — [vllm.py:2268](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L2268) `enforce_eager` 为 True 时跳过 `max_cudagraph_capture_size` 的推导（自然也就不需要 `max_num_seqs` 去参与 graph 捕获预算）。
⑤ 执行侧与投机解码
- CUDA graph 的实际捕获/回放在 [v1/worker/gpu_model_runner.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_model_runner.py)（受上述 `cudagraph_mode` 支配）；
- draft 模型有独立覆盖项 `speculative_config.enforce_eager`（[speculative.py:378](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/speculative.py#L378)），默认继承 target 模型。

**4.总结**
`enforce_eager` 在 arg_utils.py 里是最"直白"的一个参数：从 ModelConfig 继承默认 False、作为普通 Model 组参数转发，没有哨兵、没有推断；它真正的落点是 `config/vllm.py` 把 `compilation_config` 的 `mode`/`cudagraph_mode` 一并置为 `NONE`（清空 capture sizes），从而关闭 torch.compile 与 CUDA graph。ROCm encoder-decoder 等场景会被自动强制打开。
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

**3.在其它文件中的体现**
① 定义与推导 —— 最核心的一处 — [config/model.py:215](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L215)、`get_and_verify_max_len` [model.py:2389](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L2389) 调用链在 [model.py:768](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L768) 的 `__post_init__`：先记录 `original_max_model_len`，再用 `get_and_verify_max_len` 把 None 解析成最终值。推导规则就是上面「①」描述的那套 min + RoPE scaling；若 config 里查不到任何 key 且用户也没指定，则取 2048 并 warning（[model.py:2419](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/model.py#L2419)）。`Field(ge=-1)` 保证只能是正整数或 -1(auto)。
② 调度层约束 — [config/scheduler.py:299](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/scheduler.py#L299) `verify_max_model_len()` 未开 chunked prefill 时必须 `max_num_batched_tokens >= max_model_len`，否则报错（否则相当于把可处理长度悄悄压到了 batched tokens）。这是它和上一个参数的联动点。
③ 决定 KV cache 显存上限 — [v1/kv_cache_interface.py:561](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/kv_cache_interface.py#L561)
```python
cdiv(max_model_len, block_size) * page_size_bytes
```
即「按最大长度预留 KV cache」，这是 `max_model_len` 对显存最直接的影响；调大它显存线性增长，常与 `gpu_memory_utilization` 一起决定能否启动。
④ 请求侧校验 — 如 [chat_completion/serving.py:298](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/entrypoints/openai/chat_completion/serving.py#L298) 收到请求时用它校验 prompt+max_tokens 是否超限，超限直接报错——用户最常见的「length exceeds」错误就来自这里。

**4.总结**
`max_model_len` 在 arg_utils.py 里是普通转发字段，但 CLI 层被特殊对待（唯一支持 `auto`/人类可读的整数参数），并在本文件内充当其它调度参数的上下界；真正的语义在 `ModelConfig.get_and_verify_max_len`（自动推导 + 校验）与 `kv_cache_interface`（按它预留 KV cache 显存）、`SchedulerConfig.verify_max_model_len`（与 batched tokens 互相约束）三处落地。
## 阶段型自检
**一、`block_size` 增大一倍，KV Cache 能装的 token 总数会变多还是变少？内存碎片呢？（提示：想想操作系统的页大小）**
- 理论总容量：基本不变（同一块显存重新分页而已）。
- 实际可用 token 数：变少，因为内部碎片翻倍。
- 外部碎片：不存在（这是 paged attention 的固有优势，与 block_size 无关）。

为什么总量不变？
KV cache 的显存是被 `gpu_memory_utilization` 扣掉权重/激活后固定下来的，跟 block_size 无关。而 page 大小正比于 block_size：
```python
page_size_bytes ∝ num_heads * state_size * block_size   # [kv_cache_interface.py:185](vllm/v1/kv_cache_interface.py#L185)
num_blocks = available_kv_bytes // page_size_bytes
```
block_size ×2 ⇒ page_size ×2 ⇒ num_blocks ÷2，两者相乘（= 理论 token 容量）几乎抵消。就像把 4KB 页换成 2MB 大页：物理内存没变，只是页表项变少了。

为什么可用 token 会变少？内部碎片
每个序列的最后一块通常装不满。设序列长度均匀分布，平均每序列浪费 ≈ block_size / 2。
- block_size=16 → 平均浪费 8 token/序列；
- block_size=32 → 平均浪费 16 token/序列。
序列越多，累计浪费越大。能同时容纳的序列数因此下降（KV cache 不够 → 调度器提前 preempt/排队）。
极端例子：每个请求只生成 1 个 token，block_size=16 时利用率 1/16，翻倍后掉到 1/32。

与 OS 页大小的类比
||小页 (小 block_size)|大页 (大 block_size)|
|块表 / 管理开销|大|小（寻址、元数据更省）|
|内部碎片|小|**大**|
|外部碎片|无|无（分页本身消除）|
|前缀缓存命中粒度|细，命中率高|粗，命中率低|

唯一「变多」的地方
如果序列长度恰好是 block_size 的整数倍，就没有浪费——此时总量确实不变。所以对超长、规整的请求，大 block_size 反而更划算（块表小、调度开销低）；对大量短/长度参差的请求，大 block_size 会实打实减少吞吐。
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
![4c9c5c9a8ae4a7f02622bf90a408a77c.png](https://raw.githubusercontent.com/waibibab123/blog_img/main/cdnimg/4c9c5c9a8ae4a7f02622bf90a408a77c.png)

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

**四、`enforce_eager=True` 关掉的 CUDA Graph 是什么？它为什么会让调试变困难？**
1.CUDA Graph 是什么
一句话：把一串 kernel launch「录一遍」，之后用一次调用整段回放。
- 背景：decode 阶段每个 kernel 只算几十微秒，但 CPU 逐个 launch kernel 的开销可能和计算本身一个量级——GPU 大部分时间在等 CPU。CUDA Graph 把整个 forward 的 kernel + 内存操作录成一张图，回放时 CPU 只发一次指令，消除 launch 开销，这是小 batch decode 提速的关键。
- vLLM 的做法：启动 profiling 时按一组 capture sizes（`[1, 2, 4, 8, …, 256, …]`，见 [vllm.py:2224](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L2224)）分别捕获图；运行时把实际 batch size 向上取整到最近的捕获尺寸再回放（[cudagraph_dispatcher.py:82-89](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/cudagraph_dispatcher.py#L82-L89)：`bs=3 → 回放 bs=4 的图`）。
- 两种模式：`FULL`（整段图）与 `PIECEWISE`（分段，中间可插入非图算子）。
2.为什么它会让调试变困难
这正是 `enforce_eager=True` 存在的原因——它是调试必开项：

| 问题                     | 原因                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| **断点/print 失效**        | 图一旦捕获，中间是纯 GPU 指令序列，插不进 CPU 侧操作；你在 forward 里加 `print`/`pdb` 会破坏捕获或根本不执行。                                |
| **张量地址被冻结**            | 输入输出通过固定地址的静态缓冲区交换，捕获时用的是占位 tensor。你在调试器里看到的地址和真实数据对不上。                                                 |
| **shape 不等于实际**        | batch 被 padding 到 capture size，`batch=3` 实际跑的是 `batch=4` 的图，缓冲里有填充的假 token——数值异常可能只是 padding 噪声。        |
| **报错栈指向 replay，不指向算子** | 失败常表现为非法内存访问或 capture 阶段的错误，堆栈停在回放处，看不到是哪个 attention/MoE 算子出的问题。                                        |
| **两条执行路径行为不一致**        | capture 下禁止动态 shape、`cudaMalloc`、CPU-GPU 同步，很多算子在图里走另一条分支（如 MoE all2all）。于是 bug 只在「图开」或「图关」其一边出现，非常难复现。 |
| **叠加 torch.compile**   | 编译后是 Inductor 生成的 kernel，Python 层已被抹掉，调试器无从下手。                                                          |
| **显存问题难以归因**           | 图有私有内存池，OOM 看起来莫名其妙。                                                                                    |

关掉之后（`enforce_eager=True`）每步都是普通的 eager kernel launch：shape 与真实一致、可以任意 print/断点、报错栈准确指向算子—但性能会明显下降，所以只是调试期临时手段。
3.代码落点
- `enforce_eager=True` → [config/vllm.py:1564](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L1564) 把 `compilation_config.mode = NONE`、`cudagraph_mode = NONE`，并清空 capture sizes（[vllm.py:1797](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/config/vllm.py#L1797)）；
- 图的捕获与回放：[v1/worker/gpu_model_runner.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/worker/gpu_model_runner.py) `capture_model()` / `_capture_cudagraphs()`；
- batch size → 图尺寸的映射与派发：[v1/cudagraph_dispatcher.py](vscode-webview://04q40l2rvarisapko3760nk8ftscavdv6a4bpnprq0nf358ms89e/vllm/v1/cudagraph_dispatcher.py)。
总结
CUDA Graph 是用「录制-回放」消除 kernel launch 开销的性能优化，代价是把执行变成静态的、地址固定的、shape 被 padding 的不可打断黑盒——所以但凡要断点、打印、看真实 shape 或复现只在某一模式下出现的 bug，就得 `enforce_eager=True` 退回 eager 执行。
