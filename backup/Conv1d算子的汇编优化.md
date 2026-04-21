# 题目描述
这是一道 CPU 汇编优化题。你将拿到一个使用 C++ 和 Make 构建的小项目，你的任务是在保持结果正确的前提下，对其中的 1D 卷积算子进行优化，使程序尽可能快。

一维卷积（1D Convolution）是信号处理、深度学习等领域中最基础的操作之一。它描述了一个输入信号与一个卷积核（滤波器）之间的运算关系。

其核心计算过程可表示为：

![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260420150259.png)
本题实现的是标准的 1D 卷积运算：
```text
out = in (*) weight
```
其中：
- `in` 的长度为 N
- `weight` 的长度为 K
- `out` 的长度为 N - K + 1
- 所有数组均按行优先存储
- 数据类型固定为 `float`
示例
```
输入 in:    [1, 2, 3, 4, 5]      (N=5)
卷积核 weight: [1, 0, -1]         (K=3)

输出 out:   [1*1+2*0+3*(-1), 2*1+3*0+4*(-1), 3*1+4*0+5*(-1)]
          = [-2, -2, -2]          (长度为 5-3+1=3)
```
## 项目结构
代码目录如下：
```text
.
├── data
│   └── cases.in *
├── include
│   └── conv1d.h *
├── Makefile *
├── run.sh *
├── README.md
└── src
    ├── conv1d_kernel.s
    └── main.cpp *
```
标有星号的文件/文件夹是**不可更改**的。你必须在不修改这些文件的前提下进行优化，保持它们的接口和行为不变。
其中：
- `include/conv1d.h` 声明了汇编函数接口（仅供参考，并没有被直接包含在 C++ 代码中）
- `src/conv1d_kernel.s` 是**你需要编辑的汇编文件**
- `src/main.cpp` 负责 reference 计算、正确性校验、测速和评分
- `data/cases.in` 是公开测试集
你要修改的文件在 `src/conv1d_kernel.s`，你可以完全重写这个文件的内容，但必须保持接口不变。
## 实现要求
你需要实现并优化如下接口：
```asm
.globl conv1d_asm
.type conv1d_asm, @function
```
函数签名（C语言视角）：
```c
void conv1d_asm(const float* in, const float* weight, float* out, int N, int K);
```
参数传递（System V AMD64 ABI）：
- `%rdi` = in (输入数组指针，float*)
- `%rsi` = weight (卷积核指针，float*)
- `%rdx` = out (输出数组指针，float*)
- `%ecx` = N (输入数组长度，int)
- `%r8d` = K (卷积核长度，int)
输出长度 = N - K + 1
要求：
- 保持接口不变
- 输出结果与参考实现足够接近
- 可以任意修改 `conv1d_kernel.s` 的实现方式
- 必须使用汇编语言实现（ATT 格式）
禁止事项：
- 不允许修改题目给定的输入规模文件来规避测试
- 不允许通过识别特定输入规模或特定数据内容做针对性特化
- 不允许提交错误结果或伪造性能数据
- 不允许使用已有高性能计算库
## 构建与运行
构建：
```bash
make
```
读取公开测试集直接运行：
```bash
./conv1d_bench data/cases.in
```
测试单个 case：
```bash
./conv1d_bench 65536 64
```
## 评分标准
总分为 `100` 分：
- 正确性：`20` 分
- 性能：`80` 分
评分方式如下：
1. 所有公开测试 case 都正确，才能拿到正确性分。
2. 每个 case 单独计算相对 reference 的加速比。
3. 不同规模的 case 使用不同的性能阈值。
4. 每个 case 先得到一个单独的性能分，再按权重汇总为最终的 `80` 分性能分。
程序运行时会输出：
- 每个 case 的 reference 时间
- 每个 case 的实现时间
- 每个 case 的加速比
- 每个 case 的性能得分
- 最终总分
## 提交方式
请直接提交你的汇编程序，我们会通过`run.sh`运行你的代码：
## 平台说明
测试平台为 Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz，支持 AVX-512 指令集。

我们有足够的时间限制和空间限制，请忽略页面上的时间和空间限制，当然，运行时间过长和内存占用过多会被杀死。
## 汇编编程资源

常用指令参考
**数据移动：**
- `movss` - 移动标量单精度浮点数
- `movaps` - 移动对齐的 128-bit 打包单精度浮点数
- `vmovaps` - AVX 版本的 movaps（256-bit/512-bit）
**算术运算：**
- `addss/mulss` - 标量加法/乘法
- `addps/mulps` - 打包（4个float）加法/乘法
- `vaddps/vmulps` - AVX 版本的打包加法/乘法（8个float）
**FMA（乘加融合）：**
- `vfmadd231ps` - 乘加融合: dst = src1 * src2 + dst
**寄存器：**
- SSE: `xmm0` - `xmm15` (128-bit)
- AVX: `ymm0` - `ymm15` (256-bit)
- AVX-512: `zmm0` - `zmm31` (512-bit)
> Tips: 利用反汇编工具你可以获得一份还不错的参考版本
# 前提知识学习
[csapp第三章笔记](https://waibibab-cs.github.io/post/CSAPP-di-san-zhang-bi-ji-%EF%BC%9A-cheng-xu-de-ji-qi-ji-biao-shi.html)
附16个寄存器（GPR）表格：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260420113948.png)
除了这些通用寄存器之外，还有一些是SIMD专用寄存器，例如XMM寄存器，他们的比较如下所示：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260420152718.png)
附寻址模式表：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260420114009.png)
# 基础写法（baseline）
将main.cpp使用-O1选项（若使用-O3可能会导致内联优化而找不到ref_conv1d这个函数）汇编得到main.s，然后参考其中ref_conv1d的汇编代码，可以修改得到baseline代码并加入conv1d_kernel.s中，具体的代码和注释如下所示：
```asm
	.text
	.globl conv1d_asm
	.type conv1d_asm, @function

# void conv1d_asm(const float* in, const float* weight, float* out, int N, int K)
#
# 版本：Baseline（根据O1 编译的 ref_conv1d 反汇编修改）
# 策略：标量双重循环，逐元素计算 out[i] = sum_{j=0}^{K-1} in[i+j] * weight[j]
#
# 寄存器分配（System V AMD64 ABI 入参）：
#   %rdi = in      (输入数组首地址，float*)
#   %rsi = weight  (卷积核首地址，float*)
#   %rdx = out     (输出数组首地址，float*)
#   %ecx = N       (输入数组长度，int)
#   %r8d = K       (卷积核长度，int)
#
# 内部寄存器分配：
#   %r10d = out_len = N - K + 1  (输出数组长度)
#   %r9d  = i                   (外层循环变量，输出索引)
#   %rax  = j / 索引偏移        (内层循环变量，同时是 in[j] / weight[j] 的偏移)
#   %rcx  = K                   (内层循环上界，每次重新加载)
#   %r10  = out_len             (外层循环上界)
#   %xmm0 = weight[j] * in[j]   (当前乘积，临时)
#   %xmm1 = sum                 (累加器)

conv1d_asm:
	endbr64                     # Intel CET 边界检查（兼容近代 CPU）

	# 计算 out_len = N - K，检查 N < K 的退化情况
	subl  %r8d, %ecx            # ecx = N - K（按 int 运算）
	js    .Lend                 # if (N < K) return

	leal  1(%rcx), %r10d        # r10d = (N-K) + 1 = out_len
	movl  $0, %r9d              # r9d = i = 0（外层循环变量）
	movl  %r8d, %ecx            # ecx = K（内层循环上界，每次内层循环不重置它）
	jmp   .Louter_setup

# =============================================================
# 内层循环：计算 out[i] = sum_{j=0}^{K-1} in[i+j] * weight[j]
# %rax = j（0..K-1），%xmm1 = sum
# =============================================================
.Linner:
	movss (%rsi,%rax,4), %xmm0  # xmm0 = weight[j]
	mulss (%rdi,%rax,4), %xmm0 # xmm0 *= in[i+j]
	addss %xmm0, %xmm1          # sum += xmm0
	addq  $1, %rax             # j++
	cmpq  %rcx, %rax
	jne   .Linner              # j != K 则继续

# -------------------------------------------------------------
# 存储结果并准备下一次外层迭代
# -------------------------------------------------------------
.Lstore_result:
	movss %xmm1, (%rdx,%r9,4)  # out[i] = sum
	addq  $1, %r9              # i++
	addq  $4, %rdi             # in += 1（指向下一个 in[i]）
	cmpq  %r10, %r9
	je    .Lend                # i == out_len 则结束

# =============================================================
# 外层循环初始化：重置 j=0 和 sum=0
# =============================================================
.Louter_setup:
	movl  $0, %eax             # rax = j = 0
	pxor  %xmm1, %xmm1         # xmm1 = sum = 0.0f（清零累加器）
	testl %r8d, %r8d           # if K == 0
	jg    .Linner              # K > 0 则进入内层循环
	jmp   .Lstore_result       # K == 0 直接存储（sum 仍为 0）

.Lend:
	ret

.section .note.GNU-stack,"",@progbits

```
由于这段代码优化等级不如main函数中的原始代码（-O3），经过测试（`./conv1d_bench 524288 64`），加速比只有0.6左右，这应该属于正常现象。
# 优化1：AVX2向量化
将最外层循环按 8 展开，使用 256-bit 的 `ymm` 寄存器。内层循环中，我们将单个权重 `weight[j]` 广播到全部 8 个通道，然后使用 FMA 指令 (`vfmadd231ps`) 进行融合乘加。这不仅减少了指令数，还消除了中间截断误差。为保证对齐安全，统一使用 `vmovups` 加载数据。代码和详细的注释如下：
```asm
	.text
	.globl conv1d_asm
	.type conv1d_asm, @function

# void conv1d_asm(const float* in, const float* weight, float* out, int N, int K)
#
# 版本：AVX2 基础向量化（1x8 展开）+ FMA 融合乘加
#
# 策略：
#   - 外层循环按 8 展开，使用 256-bit ymm 寄存器一次并行计算 8 个 out 元素
#   - 内层循环：对每个 weight[j]，将其广播（vbroadcastss）到全部 8 个通道，
#     再与对应的 8 个 in[i+j..i+j+7] 向量做 vfmadd231ps 融合乘加
#   - 收尾处理：out_len 不是 8 的倍数时，用标量 SSE 逐个处理剩余元素
#
# 寄存器分配（System V AMD64 ABI 入参）：
#   %rdi = in      (输入数组首地址，float*)
#   %rsi = weight  (卷积核首地址，float*)
#   %rdx = out     (输出数组首地址，float*)
#   %ecx = N       (输入数组长度，int)
#   %r8d = K       (卷积核长度，int)
#
# 内部寄存器分配：
#   %r10d = out_len = N - K + 1   (输出长度，32-bit)
#   %r10  = out_len (64位版)
#   %r8   = K       (64位版)
#   %r9   = i      (外层循环变量：当前处理的输出位置索引)
#   %rax  = j      (内层循环变量：卷积核偏移)
#   %r11  = vec_end (向量化循环的最大 i 值，= out_len & ~7，向下按 8 对齐)
#   %ymm0 = broadcast weight[j]   (8 通道广播后的权重向量)
#   %ymm1 = sum_vec               (8 元素 SIMD 累加向量)
#   %ymm2 = in_vec                (8 通道输入向量 in[i+j .. i+j+7])
#
# 算法流程：
#   out_len = N - K + 1
#   vec_end = out_len & ~7        # 向下按 8 对齐的边界
#
#   # --- 向量化路径：每次处理 8 个 out[i..i+7] 并行 ---
#   for i in [0, vec_end, 8):
#       sum_vec = {0,0,0,0,0,0,0,0}
#       for j in [0, K):
#           w_bc  = broadcast(weight[j])        # vbroadcastss → ymm0
#           in_v  = in[i+j .. i+j+7]            # vmovups → ymm2
#           sum_vec += w_bc * in_v            # vfmadd231ps
#       out[i..i+7] = sum_vec                  # vmovups → mem
#       i += 8
#
#   # --- 标量收尾路径：处理剩余 [vec_end, out_len) ---
#   for i in [vec_end, out_len):
#       sum = 0
#       for j in [0, K):
#           sum += weight[j] * in[i+j]
#       out[i] = sum

conv1d_asm:
	endbr64                        # Intel CET 边界检查（近代 CPU 兼容）

	# ------------------------------------------------------------
	# 边界检查，计算 out_len = N - K + 1
	# subl:  ecx = N - K（32 位减法）
	# js:    if (N < K) → 无输出，直接返回
	# leal:  r10d = (N - K) + 1 = out_len
	# movslq: 扩展到 64 位，用于后续的指针/索引运算
	# ------------------------------------------------------------
	subl   %r8d, %ecx             # ecx = N - K
	js     .Lend_v1                # if (N < K) return

	leal   1(%rcx), %r10d         # r10d = out_len = N - K + 1

	# 扩展到 64 位以进行安全的指针/索引运算
	movslq %r10d, %r10             # r10 = out_len (64-bit)
	movslq %r8d, %r8               # r8  = K (64-bit)

	movq   $0, %r9                 # r9 = i = 0（out 的写索引）

	# ------------------------------------------------------------
	# 计算 8 元素向量块的边界：vec_end = out_len & ~7
	# 向下按 8 对齐，剩余不足 8 个的元素走标量路径
	# 例如 out_len=10 时，vec_end=8，剩余 2 个走标量
	# ------------------------------------------------------------
	movq   %r10, %r11              # r11 = out_len
	andq   $-8, %r11                # r11 = vec_end = floor(out_len/8)*8
	cmpq   $0, %r11
	je     .Lscalar_setup_v1        # out_len < 8，直接走标量路径

# =============================================================
# 向量化外层循环：i = 0, 8, 16, 24, ... 直到 vec_end
# 每次处理 8 个连续的 out 元素
# =============================================================
.Lvec8_loop_v1:
	# ymm1 = sum_vec = {0,0,0,0,0,0,0,0}，8 通道累加器清零
	vxorps %ymm1, %ymm1, %ymm1

	movq   $0, %rax                # j = 0，内层循环偏移归零
	testq  %r8, %r8                 # if K == 0
	jle    .Lvec8_store_v1          # K == 0 跳过内层循环，直接写 0

# -------------------------------------------------------------
# 向量化内层循环：j = 0 .. K-1
#
# 对每个 weight[j] 执行：
#   1. vbroadcastss: 将 weight[j] 广播到 ymm0 的全部 8 个通道
#      ymm0 = {weight[j], weight[j], ..., weight[j]}
#   2. vmovups:      加载 in[i+j .. i+j+7] 到 ymm2
#   3. vfmadd231ps:  ymm1 += ymm0 * ymm2
#      即 sum_vec[k] += weight[j] * in[i+j+k]，对全部 k=0..7 同时生效
# -------------------------------------------------------------
.Lvec8_inner_v1:
	# vbroadcastss: 将 weight[j] 广播到 8 通道
	# 源: (%rsi+%rax*4)，即 weight[j] 的地址
	# ymm0 = {w, w, w, w, w, w, w, w}（8 个相同的 float）
	vbroadcastss (%rsi,%rax,4), %ymm0

	# vmovups: 加载 8 个连续的 float（内存不需要对齐）
	# 地址: (%rdi+%rax*4)，即 in[i+j] 的地址
	# ymm2 = {in[i+j+0], ..., in[i+j+7]}
	vmovups (%rdi,%rax,4), %ymm2

	# vfmadd231ps: 融合乘加（3 操作数 FMA）
	# 语义: ymm1 = ymm1 + ymm0 * ymm2
	# 等价: sum_vec[k] += broadcast(weight[j])[k] * in_vec[k]
	vfmadd231ps %ymm0, %ymm2, %ymm1

	addq   $1, %rax                # j++
	cmpq   %r8, %rax                # j < K ?
	jl     .Lvec8_inner_v1          # j < K，继续内层循环

# -------------------------------------------------------------
# 存储向量化结果：一次写入 8 个 float 到 out[i..i+7]
# -------------------------------------------------------------
.Lvec8_store_v1:
	# vmovups: 将 ymm1（8 个 float）写入内存
	# 地址: (%rdx+%r9*4)，即 out[i] 的地址
	vmovups %ymm1, (%rdx,%r9,4)    # out[i..i+7] = sum_vec

	addq   $8, %r9                 # i += 8
	addq   $32, %rdi               # ★ in 指针右移 8*4=32 字节
	                               # in 原本指向 in[i]，下一轮要指向 in[i+8]
	                               # 所以 in += 8，等价于 in = &in[i+8]

	cmpq   %r11, %r9               # i < vec_end ?
	jl     .Lvec8_loop_v1          # i < vec_end，继续向量化循环

# =============================================================
# 标量收尾循环：处理 out_len 不是 8 的倍数时剩余的 [vec_end, out_len)
# =============================================================
.Lscalar_setup_v1:
	cmpq   %r10, %r9               # i >= out_len ?
	jge    .Lend_v1                # 已全部处理完，退出

# -------------------------------------------------------------
# 标量外层循环：每次计算一个 out[i]
# -------------------------------------------------------------
.Lscalar_loop_v1:
	# xmm1 = sum = 0，标量累加器清零
	vxorps %xmm1, %xmm1, %xmm1

	movq   $0, %rax                # j = 0
	testq  %r8, %r8                 # if K == 0
	jle    .Lscalar_store_v1        # K == 0 直接写 0

# -------------------------------------------------------------
# 标量内层循环：j = 0 .. K-1，逐个累加
# 注意：rdi 在向量化阶段已被修改（in 指针右移了 vec_end*4 字节）
# 所以 in 的地址 = rdi + rax*4，即当前 in[i+j] 的位置
# -------------------------------------------------------------
.Lscalar_inner_v1:
	# 加载 weight[j] 到 xmm0
	vmovss (%rsi,%rax,4), %xmm0    # xmm0 = weight[j]

	# 加载 in[i+j] 到 xmm2
	vmovss (%rdi,%rax,4), %xmm2   # xmm2 = in[i+j]

	# 标量 FMA：xmm1 += xmm0 * xmm2
	vfmadd231ss %xmm0, %xmm2, %xmm1

	addq   $1, %rax
	cmpq   %r8, %rax
	jl     .Lscalar_inner_v1

# -------------------------------------------------------------
# 存储标量结果：写回 out[i]
# -------------------------------------------------------------
.Lscalar_store_v1:
	vmovss %xmm1, (%rdx,%r9,4)    # out[i] = sum

	addq   $1, %r9                # i++
	addq   $4, %rdi               # in 指针右移 1 个 float（1*4=4 字节）

	cmpq   %r10, %r9              # i < out_len ?
	jl     .Lscalar_loop_v1        # 继续处理下一个

.Lend_v1:
	vzeroupper                      # 清 ymm 高 128 位，避免调用者 AVX/SSE 切换惩罚
	ret

.section .note.GNU-stack,"",@progbits

```
测试了部分样例（全部测试会超时），结果如下：
```bash
./conv1d_bench 524288 64     #7.068x
./conv1d_bench 134217728 64  #6.039x
```
提交到网站加速比7.060，得分56
# 优化2：AVX-512向量化
利用目标平台 Xeon 6338 支持的 AVX-512 指令集，将向量寄存器升级为 512-bit 的 `zmm`，单次外层循环处理的输出元素从 8 个提升至 16 个，理论吞吐量翻倍。详细的代码和注释如下所示：
```asm
.text
	.globl conv1d_asm
	.type conv1d_asm, @function

# void conv1d_asm(const float* in, const float* weight, float* out, int N, int K)
#
# 版本：AVX-512 基础向量化（1x16 展开）+ FMA 融合乘加
#
# 策略：
#   - 升级到 512-bit zmm 寄存器，单次外层循环处理 16 个 out 元素（AVX2 的两倍）
#   - 内层循环：对每个 weight[j]，vbroadcastss 广播到全部 16 通道，
#     与 in[i+j..i+j+15] 向量做 vfmadd231ps 融合乘加
#   - 收尾处理：out_len 不是 16 的倍数时，用标量 SSE 逐个处理剩余元素
#
# 寄存器分配（System V AMD64 ABI 入参）：
#   %rdi = in      (输入数组首地址，float*)
#   %rsi = weight  (卷积核首地址，float*)
#   %rdx = out     (输出数组首地址，float*)
#   %ecx = N       (输入数组长度，int)
#   %r8d = K       (卷积核长度，int)
#
# 内部寄存器分配：
#   %r10d = out_len = N - K + 1   (输出长度，32-bit)
#   %r10  = out_len (64位版)
#   %r8   = K       (64位版)
#   %r9   = i      (外层循环变量：当前处理的输出位置索引)
#   %rax  = j      (内层循环变量：卷积核偏移)
#   %r11  = vec_end (向量化循环的最大 i 值，= out_len & ~15，向下按 16 对齐)
#   %zmm0 = broadcast weight[j]   (16 通道广播后的权重向量)
#   %zmm1 = sum_vec               (16 元素 SIMD 累加向量)
#   %zmm2 = in_vec                (16 通道输入向量 in[i+j .. i+j+15])
#
# 算法流程：
#   out_len = N - K + 1
#   vec_end = out_len & ~15       # 向下按 16 对齐的边界
#
#   # --- 向量化路径：每次处理 16 个 out[i..i+15] 并行 ---
#   for i in [0, vec_end, 16):
#       sum_vec = {0,0,...} (16个零)
#       for j in [0, K):
#           w_bc  = broadcast(weight[j])         # vbroadcastss → zmm0
#           in_v  = in[i+j .. i+j+15]             # vmovups → zmm2
#           sum_vec += w_bc * in_v                # vfmadd231ps
#       out[i..i+15] = sum_vec                    # vmovups → mem
#       i += 16
#
#   # --- 标量收尾路径：处理剩余 [vec_end, out_len) ---
#   for i in [vec_end, out_len):
#       sum = 0
#       for j in [0, K):
#           sum += weight[j] * in[i+j]
#       out[i] = sum

conv1d_asm:
	endbr64                        # Intel CET 边界检查（近代 CPU 兼容）

	# ------------------------------------------------------------
	# 边界检查，计算 out_len = N - K + 1
	# subl:  ecx = N - K（32 位减法）
	# js:    if (N < K) → 无输出，直接返回
	# leal:  r10d = (N - K) + 1 = out_len
	# movslq: 扩展到 64 位，用于后续的指针/索引运算
	# ------------------------------------------------------------
	subl   %r8d, %ecx             # ecx = N - K
	js     .Lend_v2                # if (N < K) return

	leal   1(%rcx), %r10d         # r10d = out_len = N - K + 1
	movslq %r10d, %r10             # r10 = out_len (64-bit)
	movslq %r8d, %r8               # r8  = K (64-bit)

	movq   $0, %r9                 # r9 = i = 0（out 的写索引）

	# ------------------------------------------------------------
	# 计算 16 元素向量块的边界：vec_end = out_len & ~15
	# 向下按 16 对齐，剩余不足 16 个的元素走标量路径
	# 例如 out_len=20 时，vec_end=16，剩余 4 个走标量
	# ------------------------------------------------------------
	movq   %r10, %r11              # r11 = out_len
	andq   $-16, %r11               # r11 = vec_end = floor(out_len/16)*16
	cmpq   $0, %r11
	je     .Lscalar_setup_v2        # out_len < 16，直接走标量路径

# =============================================================
# 向量化外层循环：i = 0, 16, 32, 48, ... 直到 vec_end
# 每次处理 16 个连续的 out 元素
# =============================================================
.Lvec16_loop_v2:
	# zmm1 = sum_vec = {0,0,...}（16 个零），16 通道累加器清零
	vxorps %zmm1, %zmm1, %zmm1

	movq   $0, %rax                # j = 0，内层循环偏移归零
	testq  %r8, %r8                 # if K == 0
	jle    .Lvec16_store_v2          # K == 0 跳过内层循环，直接写 0

# -------------------------------------------------------------
# 向量化内层循环：j = 0 .. K-1
#
# 对每个 weight[j] 执行：
#   1. vbroadcastss: 将 weight[j] 广播到 zmm0 的全部 16 个通道
#      zmm0 = {weight[j], weight[j], ..., weight[j]}（16 个相同 float）
#   2. vmovups:      加载 in[i+j .. i+j+15] 到 zmm2
#   3. vfmadd231ps:  zmm1 += zmm0 * zmm2
#      即 sum_vec[k] += weight[j] * in[i+j+k]，对全部 k=0..15 同时生效
# -------------------------------------------------------------
.Lvec16_inner_v2:
	# vbroadcastss: 将 weight[j] 广播到 16 通道
	# 源: (%rsi+%rax*4)，即 weight[j] 的地址
	# zmm0 = {w, w, ..., w}（16 个相同的 float）
	vbroadcastss (%rsi,%rax,4), %zmm0

	# vmovups: 加载 16 个连续的 float（内存不需要对齐）
	# 地址: (%rdi+%rax*4)，即 in[i+j] 的地址
	# zmm2 = {in[i+j+0], ..., in[i+j+15]}
	vmovups (%rdi,%rax,4), %zmm2

	# vfmadd231ps: 融合乘加（3 操作数 FMA）
	# 语义: zmm1 = zmm1 + zmm0 * zmm2
	# 等价: sum_vec[k] += broadcast(weight[j])[k] * in_vec[k]
	vfmadd231ps %zmm0, %zmm2, %zmm1

	addq   $1, %rax                # j++
	cmpq   %r8, %rax                # j < K ?
	jl     .Lvec16_inner_v2          # j < K，继续内层循环

# -------------------------------------------------------------
# 存储向量化结果：一次写入 16 个 float 到 out[i..i+15]
# -------------------------------------------------------------
.Lvec16_store_v2:
	# vmovups: 将 zmm1（16 个 float）写入内存
	# 地址: (%rdx+%r9*4)，即 out[i] 的地址
	vmovups %zmm1, (%rdx,%r9,4)    # out[i..i+15] = sum_vec

	addq   $16, %r9                # i += 16
	addq   $64, %rdi               # ★ in 指针右移 16*4=64 字节
	                               # in 原本指向 in[i]，下一轮要指向 in[i+16]
	                               # 所以 in += 16，等价于 in = &in[i+16]

	cmpq   %r11, %r9               # i < vec_end ?
	jl     .Lvec16_loop_v2         # i < vec_end，继续向量化循环

# =============================================================
# 标量收尾循环：处理 out_len 不是 16 的倍数时剩余的 [vec_end, out_len)
# =============================================================
.Lscalar_setup_v2:
	cmpq   %r10, %r9               # i >= out_len ?
	jge    .Lend_v2                # 已全部处理完，退出

# -------------------------------------------------------------
# 标量外层循环：每次计算一个 out[i]
# -------------------------------------------------------------
.Lscalar_loop_v2:
	# xmm1 = sum = 0，标量累加器清零
	vxorps %xmm1, %xmm1, %xmm1

	movq   $0, %rax                # j = 0
	testq  %r8, %r8                 # if K == 0
	jle    .Lscalar_store_v2        # K == 0 直接写 0

# -------------------------------------------------------------
# 标量内层循环：j = 0 .. K-1，逐个累加
# 注意：rdi 在向量化阶段已被修改（in 指针右移了 vec_end*4 字节）
# 所以 in 的地址 = rdi + rax*4，即当前 in[i+j] 的位置
# -------------------------------------------------------------
.Lscalar_inner_v2:
	# 加载 weight[j] 到 xmm0
	vmovss (%rsi,%rax,4), %xmm0    # xmm0 = weight[j]

	# 加载 in[i+j] 到 xmm2
	vmovss (%rdi,%rax,4), %xmm2   # xmm2 = in[i+j]

	# 标量 FMA：xmm1 += xmm0 * xmm2
	vfmadd231ss %xmm0, %xmm2, %xmm1

	addq   $1, %rax
	cmpq   %r8, %rax
	jl     .Lscalar_inner_v2

# -------------------------------------------------------------
# 存储标量结果：写回 out[i]
# -------------------------------------------------------------
.Lscalar_store_v2:
	vmovss %xmm1, (%rdx,%r9,4)    # out[i] = sum

	addq   $1, %r9                # i++
	addq   $4, %rdi               # in 指针右移 1 个 float（1*4=4 字节）

	cmpq   %r10, %r9              # i < out_len ?
	jl     .Lscalar_loop_v2         # 继续处理下一个

.Lend_v2:
	vzeroupper                      # 清 zmm/ymm 高位，避免调用者 AVX/SSE 切换惩罚
	ret

.section .note.GNU-stack,"",@progbits
```
测试了部分样例，结果如下：
```bash
./conv1d_bench 524288 64     #13.972
./conv1d_bench 134217728 64  #10.935
```
提交到网站加速比12.113x，得分80
# 优化3：外层循环4×16展开
由于 FMA 指令具有较长的延迟（通常为 4 个周期），单纯的 1x16 计算会导致 CPU 流水线处于等待状态。我们将外层循环进一步按 64 展开，同时使用 4 个独立累加器 (`zmm1`~`zmm4`)。这样一次 `vbroadcastss` 加载的权重被立即复用于 4 次并发的独立计算中，完美隐藏了延迟。代码与详细注释如下所示
```asm
.text
	.globl conv1d_asm
	.type conv1d_asm, @function

# void conv1d_asm(const float* in, const float* weight, float* out, int N, int K)
#
# 版本：AVX-512 寄存器分块（4x16 展开，64-element 大块）
#
# 策略：
#   FMA 指令延迟约 4 周期，单路累加会导致流水线停滞。
#   将外层按 64 展开，使用 4 个独立累加器（zmm1~zmm4）并行工作：
#     - 一次 vbroadcastss 加载的 weight[j] 同时参与 4 路独立的 FMA 计算
#     - 4 路累加器分别处理 4 组独立的 out 块（每块 16 个元素）
#     - 指令级并行（ILP）隐藏 FMA 延迟，使流水线持续满载
#
#   数据布局（每次内层循环迭代 j）：
#     weight[j] → vbroadcastss → zmm0
#     in[i+j+0..15]   → vmovups → zmm5 → zmm1 += zmm0*zmm5    (out[i+0..15])
#     in[i+j+16..31]  → vmovups → zmm6 → zmm2 += zmm0*zmm6    (out[i+16..31])
#     in[i+j+32..47]  → vmovups → zmm7 → zmm3 += zmm0*zmm7    (out[i+32..47])
#     in[i+j+48..63]  → vmovups → zmm8 → zmm4 += zmm0*zmm8    (out[i+48..63])
#
#   三级处理：
#     1. vec64_loop: 64 个 out 元素并行（4 路 x 16），out_len ≥ 64
#     2. vec16_loop: 16 个 out 元素并行（1 路 x 16），out_len % 64 ∈ [16, 63]
#     3. scalar_loop: 逐个处理剩余元素，out_len % 64 < 16
#
# 寄存器分配（System V AMD64 ABI 入参）：
#   %rdi = in      (输入数组首地址，float*)
#   %rsi = weight  (卷积核首地址，float*)
#   %rdx = out     (输出数组首地址，float*)
#   %ecx = N       (输入数组长度，int)
#   %r8d = K       (卷积核长度，int)
#
# 内部寄存器分配：
#   %r10d = out_len = N - K + 1   (输出长度，32-bit)
#   %r10  = out_len (64位版)
#   %r8   = K       (64位版)
#   %r9   = i      (外层循环变量：当前处理的输出位置索引)
#   %rax  = j      (内层循环变量：卷积核偏移)
#   %r11  = block_end (当前块边界，向量化循环的最大 i 值)
#   %zmm0 = broadcast weight[j]   (16 通道广播后的权重向量，4 路共享)
#   %zmm1 = sum_vec0              (第 0 路累加器，out[i+0..15])
#   %zmm2 = sum_vec1              (第 1 路累加器，out[i+16..31])
#   %zmm3 = sum_vec2              (第 2 路累加器，out[i+32..47])
#   %zmm4 = sum_vec3              (第 3 路累加器，out[i+48..63])
#   %zmm5-8 = in_vec0-3           (临时：各路输入向量)

conv1d_asm:
	endbr64                        # Intel CET 边界检查（近代 CPU 兼容）

	# ------------------------------------------------------------
	# 边界检查，计算 out_len = N - K + 1
	# subl:  ecx = N - K（32 位减法）
	# js:    if (N < K) → 无输出，直接返回
	# leal:  r10d = (N - K) + 1 = out_len
	# movslq: 扩展到 64 位，用于后续的指针/索引运算
	# ------------------------------------------------------------
	subl   %r8d, %ecx             # ecx = N - K
	js     .Lend_v3                # if (N < K) return

	leal   1(%rcx), %r10d         # r10d = out_len = N - K + 1
	movslq %r10d, %r10             # r10 = out_len (64-bit)
	movslq %r8d, %r8               # r8  = K (64-bit)

	movq   $0, %r9                 # r9 = i = 0（out 的写索引）

	# ------------------------------------------------------------
	# 计算 64 元素大块边界：block_end = out_len & ~63
	# 向下按 64 对齐，例如 out_len=100 时 block_end=64，剩余 36 个走 vec16
	# ------------------------------------------------------------
	movq   %r10, %r11              # r11 = out_len
	andq   $-64, %r11               # r11 = block_end = floor(out_len/64)*64
	cmpq   $0, %r11
	je     .Lvec16_setup_v3        # out_len < 64，跳到 vec16 路径

# =============================================================
# vec64_loop：每次处理 64 个 out 元素（4 路 × 16）
# i = 0, 64, 128, ... 直到 block_end
# =============================================================
.Lvec64_loop_v3:
	# 清零 4 个独立的累加器
	vxorps %zmm1, %zmm1, %zmm1     # sum_vec0 = 0
	vxorps %zmm2, %zmm2, %zmm2     # sum_vec1 = 0
	vxorps %zmm3, %zmm3, %zmm3     # sum_vec2 = 0
	vxorps %zmm4, %zmm4, %zmm4     # sum_vec3 = 0

	movq   $0, %rax                # j = 0
	testq  %r8, %r8                 # if K == 0
	jle    .Lvec64_store_v3        # K == 0 跳过内层循环，直接写 0

# -------------------------------------------------------------
# vec64 内层循环：j = 0 .. K-1
#
# 对每个 weight[j]，执行 4 路并行的 FMA：
#   1. vbroadcastss: weight[j] 广播到 zmm0（16 通道）
#   2. 依次加载 in[i+j+0..15], in[i+j+16..31], in[i+j+32..47], in[i+j+48..63]
#   3. 4 路并发 FMA：zmm1-4 分别累加 weight[j] × 对应 in 块
#
# 延迟隐藏原理：
#   4 条 vfmadd231ps 独立操作 zmm1-4，无数据依赖
#   CPU 发射单元可每周期发射 1 条，4 条流水线并行
#   延迟 ~4 周期，但吞吐量 = 每周期 1 条 → 流水线填满后无停顿
# -------------------------------------------------------------
.Lvec64_inner_v3:
	# 将 weight[j] 广播到 16 通道，4 路共享同一个 zmm0
	vbroadcastss (%rsi,%rax,4), %zmm0

	# 第 0 路：in[i+j+0..15]  → zmm5 → zmm1 += zmm0 * zmm5
	vmovups (%rdi,%rax,4), %zmm5
	vfmadd231ps %zmm0, %zmm5, %zmm1

	# 第 1 路：in[i+j+16..31] → zmm6 → zmm2 += zmm0 * zmm6
	vmovups 64(%rdi,%rax,4), %zmm6
	vfmadd231ps %zmm0, %zmm6, %zmm2

	# 第 2 路：in[i+j+32..47] → zmm7 → zmm3 += zmm0 * zmm7
	vmovups 128(%rdi,%rax,4), %zmm7
	vfmadd231ps %zmm0, %zmm7, %zmm3

	# 第 3 路：in[i+j+48..63] → zmm8 → zmm4 += zmm0 * zmm8
	vmovups 192(%rdi,%rax,4), %zmm8
	vfmadd231ps %zmm0, %zmm8, %zmm4

	addq   $1, %rax                # j++
	cmpq   %r8, %rax                # j < K ?
	jl     .Lvec64_inner_v3         # j < K，继续内层循环

# -------------------------------------------------------------
# 存储 64 元素大块的结果（4 次 16-element 存储）
# -------------------------------------------------------------
.Lvec64_store_v3:
	# 依次将 zmm1-4 写入 out 的连续 4 个 16-element 块
	vmovups %zmm1, (%rdx,%r9,4)     # out[i+0..15] = sum_vec0
	vmovups %zmm2, 64(%rdx,%r9,4)   # out[i+16..31] = sum_vec1
	vmovups %zmm3, 128(%rdx,%r9,4)  # out[i+32..47] = sum_vec2
	vmovups %zmm4, 192(%rdx,%r9,4)  # out[i+48..63] = sum_vec3

	addq   $64, %r9                # i += 64
	addq   $256, %rdi              # ★ in 指针右移 64*4=256 字节
	                               # 对应 4 路 × 16 个 float 的跨度

	cmpq   %r11, %r9               # i < block_end ?
	jl     .Lvec64_loop_v3         # i < block_end，继续 vec64 循环

# =============================================================
# vec16_setup + vec16_loop：处理剩余 [block_end, out_len) 中的 16-63 个元素
# 避免直接退化到慢速标量循环
# =============================================================
.Lvec16_setup_v3:
	# 计算 16 元素块边界：block_end = out_len & ~15
	movq   %r10, %r11              # r11 = out_len
	andq   $-16, %r11               # r11 = floor(out_len/16)*16
	cmpq   %r11, %r9               # i >= vec16_end ?
	jge    .Lscalar_setup_v3       # 无剩余或剩余 < 16，跳到标量路径

.Lvec16_loop_v3:
	# 清零单路累加器
	vxorps %zmm1, %zmm1, %zmm1

	movq   $0, %rax                # j = 0
	testq  %r8, %r8                 # if K == 0
	jle    .Lvec16_store_v3        # K == 0 跳过内层

# -------------------------------------------------------------
# vec16 内层循环：j = 0 .. K-1，单路 16 元素并行
# 与 vec64 的区别：只处理 1 路（zmm1），省去其他 3 路
# -------------------------------------------------------------
.Lvec16_inner_v3:
	vbroadcastss (%rsi,%rax,4), %zmm0
	vmovups (%rdi,%rax,4), %zmm2
	vfmadd231ps %zmm0, %zmm2, %zmm1

	addq   $1, %rax
	cmpq   %r8, %rax
	jl     .Lvec16_inner_v3

.Lvec16_store_v3:
	vmovups %zmm1, (%rdx,%r9,4)    # out[i..i+15] = sum_vec
	addq   $16, %r9                # i += 16
	addq   $64, %rdi               # in 指针右移 16*4=64 字节
	cmpq   %r11, %r9               # i < vec16_end ?
	jl     .Lvec16_loop_v3

# =============================================================
# 标量收尾循环：处理剩余 < 16 个元素
# =============================================================
.Lscalar_setup_v3:
	cmpq   %r10, %r9               # i >= out_len ?
	jge    .Lend_v3                # 已全部处理完

.Lscalar_loop_v3:
	vxorps %xmm1, %xmm1, %xmm1     # sum = 0
	movq   $0, %rax                # j = 0
	testq  %r8, %r8                 # if K == 0
	jle    .Lscalar_store_v3

.Lscalar_inner_v3:
	vmovss (%rsi,%rax,4), %xmm0    # weight[j]
	vmovss (%rdi,%rax,4), %xmm2   # in[i+j]
	vfmadd231ss %xmm0, %xmm2, %xmm1

	addq   $1, %rax
	cmpq   %r8, %rax
	jl     .Lscalar_inner_v3

.Lscalar_store_v3:
	vmovss %xmm1, (%rdx,%r9,4)    # out[i] = sum
	addq   $1, %r9
	addq   $4, %rdi               # in 指针右移 1 个 float
	cmpq   %r10, %r9
	jl     .Lscalar_loop_v3

.Lend_v3:
	vzeroupper                      # 清 zmm/ymm 高位，避免调用者 AVX/SSE 切换惩罚
	ret

.section .note.GNU-stack,"",@progbits

```
测试了部分样例，结果如下：
```bash
./conv1d_bench 524288 64     #25.679x
./conv1d_bench 134217728 64  #16.263x
```
提交到网站加速比15.530，得分97