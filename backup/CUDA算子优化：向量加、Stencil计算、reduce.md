这是一道偏入门的 GPU CUDA 性能优化题。你将拿到一个使用 CUDA 和 nvcc 构建的小项目，你的任务是在保持结果正确的前提下，对其中的三个基础算子进行优化，使程序尽可能快。
# 题目简介

向量加法、Stencil计算和归约求和是高性能计算中最经典、最重要的基础算子之一。在科学计算、机器学习、图像处理以及大量数值程序中，这些基础算子被广泛地使用。

本题的核心，就是使用 CUDA 优化三个基础算子：

1. **向量加法 (add)**：将两个输入向量逐元素相加得到输出向量
    
    ```
    C[i] = A[i] + B[i]
    ```
    
2. **一维 Stencil 计算 (stencil)**：对输入向量的每个元素，计算其邻域内元素的和
    
    ```
    C[i] = sum(A[i - radius + 1] ... A[i] ... A[i + radius - 1])
    ```
    
3. **归约求和 (reduce)**：将输入向量的所有元素求和，输出一个标量
    
    ```
    C[0] = sum(A[0] + A[1] + ... + A[N-1])
    ```
    

你不需要提前掌握很多数学知识，更重要的是理解 CUDA 编程模型、GPU 内存层次结构（共享内存、全局内存等）、线程组织方式，以及为什么某些写法会更快。

其中：

- 所有向量长度均为 `N`
- 数据类型固定为 `double`
- 数据存储在 GPU 显存中

## 项目结构

代码目录如下：

```text
.
├── data
│   └── cases.in
├── include
│   └── operators.h
├── Makefile
├── README.md
├── run.sh
└── src
    ├── main.cpp
    └── operators.cu
```

你**只能**修改 `src/operators.cu` 来实现和优化三个算子，请勿修改其他文件。

其中：

- `include/operators.h` 声明你需要实现的接口
- `src/operators.cu` 是你需要优化的文件
- `src/main.cpp` 负责 reference 计算、正确性校验、测速和评分
- `data/cases.in` 是公开测试集

## 实现要求

你需要实现并优化如下接口：

```cpp
void add(const double *A, const double *B, double *C, int N);
void stencil(const double *A, double *C, int N, int radius);
void reduce(const double *A, double *C, int N);
```

要求：

- 保持接口不变
- 输出结果与参考实现平均误差小于1e-3
- 可以任意修改 `operators.cu` 的实现方式
- 允许使用 CUDA 的各种优化技术（共享内存、寄存器优化、合并内存访问、warp 级原语等），但**不允许使用 cuBLAS、Thrust 等已有高性能计算库**

禁止事项：

- 不允许修改题目给定的输入规模文件来规避测试
- 不允许通过识别特定输入规模或特定数据内容做针对性特化
- 不允许提交错误结果或伪造性能数据

## 构建与运行

构建：

```bash
make
```

读取公开测试集直接运行：

```bash
./operators_bench data/cases.in
```

## 评分标准

总分为 `100` 分：

- 正确性：`30` 分
- 性能：`70` 分

评分方式如下：

1. 每个 case 每个算子正确，都可以拿到部分正确性分。
2. 每个 case 单独计算三个算子相对 reference 的加速比。
3. 不同规模的 case 使用不同的性能阈值。
4. 每个 case 先得到一个单独的性能分（add 占 30 分，stencil 占 30 分，reduce 占 40 分），再按权重为最终的 `100` 分性能分。

程序运行时会输出：

- 每个 case 的 reference 时间
- 每个 case 的实现时间
- 每个 case 的加速比
- 每个 case 的性能得分
- 最终总分

## 提交方式

请将你`operators.cu`直接提交，我们会通过`run.sh`运行你的代码。

## 平台说明

我们的测试平台是NVIDIA L40 48G。

你可以充分利用 GPU 的并行计算能力进行优化。但是运行时间超过 baseline 的提交会被杀死，显存占用过多的提交也会被杀死。

# 向量加法
## 基础写法
首先是 CUDA 编程最直觉的写法：**让 GPU 上的每一个线程负责计算数组中的一个元素**
```cpp
__global__
void addKernel(const double *A, const double *B, double *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void add(const double *A, const double *B, double *C, int N) {
    addKernel<<<(N + 255) / 256, 256>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
此处设定每个线程块有256个线程
## 优化1：网格跨步循环
当数据量 $N$ 极其巨大（比如十亿级别）时，基础写法会启动成百上千万个 Block。虽然 GPU 支持庞大的 Grid，但过多的 Block 会增加调度开销。更专业的写法是**网格跨步循环**。
- **解耦网格大小与数据大小**：我们不再为每个元素分配一个线程，而是启动一个**固定大小的 Grid**（通常正好能填满 GPU 的所有流多处理器 SM 即可）。
- **跨步处理**：每个线程处理完一个元素后，它会向前跳跃一个“整个 Grid 的跨度（Stride）”，去处理下一个元素，直到处理完整个数组。
```cpp
__global__
void addKernel(const double *A, const double *B, double *C, int N) {
    // 获取当前线程的起始索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 获取整个网格的总线程数作为跨度 (Stride)
    int stride = blockDim.x * gridDim.x;

    // 线程通过循环处理多个元素，每次跳跃 stride 个位置
    for (; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Host 端接口
void add(const double *A, const double *B, double *C, int N) {
    int blockSize = 256;
    // NVIDIA L40 有 142 个 SM，每个 SM 驻留几个 Block 就能跑满
    // 这里我们限制最大 Block 数量为 1024 或 2048，不再随 N 无限增长
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 1024) {
        numBlocks = 1024;
    }

    addKernel<<<numBlocks, blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
## 优化2：向量化访存
可以通过向量化访存进一步减少CPU/GPU内部发出的访存指令数量
在循环中，每次循环读取一个 `double`（8 字节）。为了读取 32 字节，需要发射 4 条 Load 指令。
CUDA 提供了内置的向量类型。我们可以强制把普通的 `double` 指针转换成 `double4` 指针。
不过因为数组总长度 $N$ 未必是 4 的整数倍，所以我们在前面用 `double4` 跑完 $N/4$ 的主体部分后，剩下的几个零头（$N\%4$）再用普通的标量方式处理。
具体见代码
```cpp
// 向量化访存 + 跨步循环 Kernel
__global__ void add_kernel_vectorized(const double4 *A_vec, const double4 *B_vec, double4 *C_vec, int N_vec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 主体部分：每次处理 4 个 double
    for (; i < N_vec; i += stride)
    {
        double4 a = A_vec[i];
        double4 b = B_vec[i];
        double4 c;
        // 逐分量相加
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C_vec[i] = c;
    }
}

// 处理剩余元素的 Kernel
__global__ void add_kernel_tail(const double *A, const double *B, double *C, int N, int offset)
{
    int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// Host 端接口
void add(const double *A, const double *B, double *C, int N)
{
    // 1. 计算有多少个完整的 double4 (即 N 的 1/4)
    int N_vec = N / 4;

    // 注意：cudaMalloc 分配的显存默认是 256 字节对齐的，所以安全地转换指针是没问题的
    const double4 *A_vec = reinterpret_cast<const double4 *>(A);
    const double4 *B_vec = reinterpret_cast<const double4 *>(B);
    double4 *C_vec = reinterpret_cast<double4 *>(C);

    int blockSize = 256;
    // 适当设置 Grid 大小以跑满 L40
    int numBlocks = 1024;

    // 2. 启动向量化 Kernel 处理绝大部分数据
    if (N_vec > 0)
    {
        // 根据 N_vec 的大小动态调整 block 数量，防止 N_vec 很小的情况
        int blocks = (N_vec + blockSize - 1) / blockSize;
        if (blocks > numBlocks)
            blocks = numBlocks;
        add_kernel_vectorized<<<blocks, blockSize>>>(A_vec, B_vec, C_vec, N_vec);
    }

    // 3. 处理不能被 4 整除的尾巴 (最多只有 3 个元素)
    int tail_elements = N % 4;
    if (tail_elements > 0)
    {
        int offset = N_vec * 4; // 计算尾巴在原数组中的起始偏移量
        add_kernel_tail<<<1, tail_elements>>>(A, B, C, N, offset);
    }

    cudaDeviceSynchronize();
}
```
# Stencil计算
## 基础写法
每个线程负责计算一个 $C[i]$，通过一个 `for` 循环，直接去全局内存中把邻域内的元素读出来累加。
缺陷：假设 $radius = 16$，窗口大小为 31。这意味着全局内存中的每个元素会被 **重复读取 31 次**
```cpp
__global__ void stencil_basic(const double *A, double *C, int N, int radius) {
    // 1. 获取全局线程 ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 确保不越界
    if (i < N) {
        double sum = 0.0;
        int r = radius - 1; 

        // 3. 直接从全局内存读取并累加
        for (int j = -r; j <= r; ++j) {
            int idx = i + j;
            // 越界保护，处理矩阵边缘
            double val = (idx >= 0 && idx < N) ? A[idx] : 0.0;
            sum += val;
        }
        C[i] = sum;
    }
}

void stencil(const double *A, double *C, int N, int radius) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    stencil_basic<<<numBlocks, blockSize>>>(A, C, N, radius);
    cudaDeviceSynchronize();
}
```
## 优化1：引入共享内存
为了解决重复读取的问题，我们引入 GPU 内部速度极快（接近寄存器）的**共享内存**
我们让一个 Block 内的 256 个线程，**齐心协力**先从全局内存把它们需要的所有数据一次性搬到共享内存中。然后，所有的累加计算都在共享内存里完成。
由于最边缘的线程计算时需要用到 Block 外部的数据，所以共享内存的实际大小必须比 Block 的大小还要宽。宽度 = `BlockSize + 2 * (radius - 1)`。左右多出来的部分就叫 Halo。
**分工搬运**：
1. 所有线程搬运中间的 Main Data。
2. 最左边的几个线程负责搬运左边的 Halo。
3. 最右边的几个线程负责搬运右边的 Halo。
必须调用 `__syncthreads()`，确保此线程块内所有线程都把数据搬完后，才能开始计算。

缺陷：
1. 搬运 Halo 的代码用了 `if (tid < r)`。这会导致**Warp 分支发散**
2. 读取边界数据的操作在内存上是不连续的，打破了**合并访存 (Memory Coalescing)** 的规则：相邻的线程（例如 thread 0 和 thread 1）访问相邻的内存地址（`A[0]` 和 `A[1]`）时，GPU 硬件会将这些请求合并成一次宽泛的内存事务，提高性能
```cpp
__global__ void stencil_shared_naive(const double *A, double *C, int N, int radius)
{
    // 动态分配共享内存
    extern __shared__ double s_A[];

    int tid = threadIdx.x;                         // Block 内部的本地 ID
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 全局 ID
    int r = radius - 1;

    // 1. 协作搬运：主数据区
    if (i < N)
    {
        s_A[r + tid] = A[i];
    }
    else
    {
        s_A[r + tid] = 0.0;
    }

    // 2. 协作搬运：左侧 Halo
    if (tid < r)
    {
        int left_idx = i - r;
        s_A[tid] = (left_idx >= 0) ? A[left_idx] : 0.0;
    }

    // 3. 协作搬运：右侧 Halo
    if (tid >= blockDim.x - r)
    {
        int right_idx = i + r;
        s_A[r + tid + r] = (right_idx < N) ? A[right_idx] : 0.0;
    }

    // 4. 等待所有线程（同一block内线程）搬运完毕
    __syncthreads();

    // 5. 在高速的共享内存中完成 Stencil 计算
    if (i < N)
    {
        double sum = 0.0;
        for (int j = -r; j <= r; ++j)
        {
            sum += s_A[r + tid + j];
        }
        C[i] = sum;
    }
}

void stencil(const double *A, double *C, int N, int radius)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    // 共享内存大小 = blockSize + 2 * (radius - 1)，用于存放主数据区 + 左右 halo
    int sharedMemSize = (blockSize + 2 * (radius - 1)) * sizeof(double);
    stencil_shared_naive<<<numBlocks, blockSize, sharedMemSize>>>(A, C, N, radius);
    cudaDeviceSynchronize();
}
```
## 优化2：跨步循环与循环展开
为消除优化1中的Warp分支发散问题，我们不再让线程“按职责”去搬数据。我们把共享内存看作一个长度为 `shared_size` 的一维数组。让 Block 里的所有线程组成一个**跨步循环 (Stride Loop)**，像扫地机一样从头扫到尾。这样写，所有的内存读取指令天然 100% 连续对齐（Coalesced），完全消除了 `if-else` 带来的线程发散。
此外，我们使用`#pragma unroll`强制编译器把累加的 `for` 循环展开成一长串 `+` 号，榨干指令级并行度，这种优化在小循环场景下能起到不错的效果。
```cpp
__global__ void stencil_shared_coalesced(const double *A, double *C, int N, int radius)
{
    extern __shared__ double s_A[];

    int tid = threadIdx.x;
    int r = radius - 1;
    // 共享内存的总大小
    int shared_size = blockDim.x + 2 * r;
    // 当前 Block 负责的全局起始位置
    int block_start = blockIdx.x * blockDim.x;

    // 1. 无发散、完全合并的共享内存加载 (绝杀技巧)
    // 无论 shared_size 多大，所有线程共同参与，循环填充
    for (int j = tid; j < shared_size; j += blockDim.x)
    {
        // 计算当前要加载的全局索引 (向左偏移 r)
        int global_idx = block_start - r + j;

        // 统一处理边界
        if (global_idx >= 0 && global_idx < N)
        {
            s_A[j] = A[global_idx];
        }
        else
        {
            s_A[j] = 0.0;
        }
    }

    // 必须同步，等待缓冲区填满
    __syncthreads();

    // 2. 高效计算
    int i = block_start + tid;
    if (i < N)
    {
        double sum = 0.0;
// 使用 #pragma unroll 提示编译器展开循环，减少跳转开销
#pragma unroll
        for (int j = -r; j <= r; ++j)
        {
            sum += s_A[tid + j + r];
        }
        C[i] = sum;
    }
}

void stencil(const double *A, double *C, int N, int radius)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // 动态计算所需的共享内存大小（字节数）
    int r = radius - 1;
    size_t shared_bytes = (blockSize + 2 * r) * sizeof(double);

    stencil_shared_coalesced<<<numBlocks, blockSize, shared_bytes>>>(A, C, N, radius);
    cudaDeviceSynchronize();
}
```
# reduce
## 基础写法
每个 Block 将自己负责的数据搬入共享内存，然后在共享内存中两两相加，直到只剩下一个总和，最后再将所有block的这个总和全部加在一起。
例如：
- 第 1 轮：线程 0 计算 `sdata[0] += sdata[1]`；线程 2 计算 `sdata[2] += sdata[3]`...（跨度 stride = 1）
- 第 2 轮：线程 0 计算 `sdata[0] += sdata[2]`；线程 4 计算 `sdata[4] += sdata[6]`...（跨度 stride = 2）
- 以此类推
缺陷：GPU 调度是以 Warp（32 个线程）为单位的。在第一轮中，线程 0, 2, 4... 在干活，而线程 1, 3, 5... 在闲置。同一个 Warp 内一半干活一半发呆，导致计算效率直接减半。
**Host 端注意点**：因为 Kernel 内部最后使用的是 `atomicAdd(C, sdata[0])`，如果 `C` 的初始值不是 0，结果就会错。所以必须在 Host 端调用 `cudaMemset` 将 `C` 清零。
```cpp
// --- Kernel 函数 ---
__global__ void reduce_kernel_basic(const double *A, double *C, int N)
{
    // 动态共享内存，由 Host 端在 <<< >>> 中指定大小
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 将数据从全局内存加载到共享内存
    sdata[tid] = (i < N) ? A[i] : 0.0;
    __syncthreads(); // 等待所有线程加载完毕

    // 交错寻址的树状归约
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2 * stride) == 0)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads(); // 等待这一层级所有人加完
    }

    // 每个 Block 的 0 号线程将局部和累加到全局 C[0]
    if (tid == 0)
    {
        atomicAdd(C, sdata[0]);
    }
}

// --- Host 端函数 ---
void reduce(const double *A, double *C, int N)
{
    // 关键：规约的全局输出必须初始化为 0
    cudaMemset(C, 0, sizeof(double));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    // 计算动态共享内存的字节数
    size_t sharedSize = blockSize * sizeof(double);

    reduce_kernel_basic<<<numBlocks, blockSize, sharedSize>>>(A, C, N);
    cudaDeviceSynchronize();
}
```
## 优化1：连续寻址
为了解决交错寻址导致的 Warp 发散，我们改变“配对”的逻辑，让跨度从大到小递减。
假设 Block 有 256 个线程：
- 第 1 轮：线程 0~127 活跃。线程 0 计算 `sdata[0] += sdata[128]`，线程 1 计算 `sdata[1] += sdata[129]`...（跨度 stride = 128）
- 第 2 轮：线程 0~63 活跃。跨度 stride = 64。
- 以此类推
这样以来的好处是可以在前几轮实现前几个warp的满载运行
```cpp
// --- Kernel 函数 ---
__global__ void reduce_kernel_opt1(const double *A, double *C, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? A[i] : 0.0;
    __syncthreads();

    // 连续寻址的树状归约：stride 从块大小的一半开始，每次减半
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(C, sdata[0]);
    }
}

// --- Host 端函数 ---
void reduce(const double *A, double *C, int N)
{
    cudaMemset(C, 0, sizeof(double));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    size_t sharedSize = blockSize * sizeof(double);

    reduce_kernel_opt1<<<numBlocks, blockSize, sharedSize>>>(A, C, N);
    cudaDeviceSynchronize();
}
```
## 优化2：网格跨步循环+引入寄存器
目前的写法是每个线程只处理 1 个元素。如果 $N$ 很大，启动太多 Block 会增加 `atomicAdd` 的竞争和系统调度开销，因此，我们做出下面的优化：
- **提升计算密度**：我们引入上一题向量加法中讲过的 **网格跨步循环 (Grid-Stride Loop)**。
- **寄存器缓存**：在数据还没放进共享内存之前，就先在线程的**私有寄存器**里用 `while` 循环把多个全局内存元素加起来。
- **Host 端调整**：我们将最大 Block 数量限制为 1024。这意味着无论 $N$ 是一百万还是一百亿，都只会有 1024 个 Block 去竞争最后的 `atomicAdd`，大大缓解了写入冲突。
```cpp
// --- Kernel 函数 ---
__global__ void reduce_kernel_opt2(const double *A, double *C, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // 1. 加载时归约：用寄存器变量 sum 收集跨步元素
    double sum = 0.0;
    for (; i < N; i += grid_stride)
    {
        sum += A[i];
    }

    // 2. 将寄存器里的局部和存入共享内存
    sdata[tid] = sum;
    __syncthreads();

    // 3. 块内共享内存归约 (连续寻址)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(C, sdata[0]);
    }
}

// --- Host 端函数 ---
void reduce(const double *A, double *C, int N)
{
    cudaMemset(C, 0, sizeof(double));

    int blockSize = 256;
    // 限制最大 Block 数量，强迫每个线程在跨步循环中多干点活
    int numBlocks = std::min((N + blockSize - 1) / blockSize, 1024);
    size_t sharedSize = blockSize * sizeof(double);

    reduce_kernel_opt2<<<numBlocks, blockSize, sharedSize>>>(A, C, N);
    cudaDeviceSynchronize();
}

```
## 优化3：Warp Shuffle 原语
`__syncthreads()` 依然存在极大的性能开销。
现代 NVIDIA 显卡支持 **Warp Shuffle (洗牌)** 指令。它允许同一个 Warp（32个线程）里的线程**直接读取互相的寄存器**，完全不需要走共享内存，也不需要 `__syncthreads()` 同步
`__shfl_down_sync(0xffffffff, val, offset)`：这条指令让当前线程直接向比它 ID 大 `offset` 的线程索要 `val` 的值。
**终极流程**：
1. **网格跨步**求和到寄存器 `sum`。
2. **Warp 内部**通过 Shuffle 指令折叠 `sum`。
3. 每个 Warp 只需要把这 1 个最终结果写入共享内存（不再需要 256 大小的共享内存，只需要 32 大小）。
4. 最后一个 Warp 读取这部分共享内存，再做一次 Shuffle 归约，得到整个 Block 的总和。
```cpp
#define WARP_SIZE 32

// Warp 级洗牌归约内联函数 (完全在寄存器中发生，速度极快)
__inline__ __device__ double warpReduceSum(double val)
{
    // 每次 offset 减半：16, 8, 4, 2, 1
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// --- Kernel 函数 ---
__global__ void reduce_kernel_final(const double *A, double *C, int N)
{
    // 静态分配共享内存：一个 Block 最多 1024 个线程，即最多 32 个 Warp
    __shared__ double shared_warp_sums[32];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;       // Warp 内的局部 ID (0~31)
    int wid = tid / WARP_SIZE;        // Warp 的 ID (0~31)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // 1. 加载时归约 (Grid-Stride Loop)
    double sum = 0.0;
    for (; i < N; i += grid_stride)
    {
        sum += A[i];
    }

    // 2. Warp 内洗牌归约 (无共享内存读取，无 syncthreads)
    sum = warpReduceSum(sum);

    // 3. 每个 Warp 的 0 号线程将部分和存入共享内存
    if (lane == 0)
    {
        shared_warp_sums[wid] = sum;
    }

    // 必须同步一次，等待所有 Warp 把自己的总和写进共享内存
    __syncthreads();

    // 4. 让第一个 Warp 负责最后的规约
    // 共有 blockDim.x / WARP_SIZE 个有效 Warp 数据需要规约
    int num_warps = blockDim.x / WARP_SIZE;
    if (wid == 0)
    {
        // 读取共享内存，如果超出了活跃 warp 数量，则补 0
        sum = (lane < num_warps) ? shared_warp_sums[lane] : 0.0;
        // 最后一次 Warp 洗牌归约
        sum = warpReduceSum(sum);
    }

    // 5. Block 0 号线程使用原子操作汇聚到全局结果
    if (tid == 0)
    {
        atomicAdd(C, sum);
    }
}

// --- Host 端函数 ---
void reduce(const double *A, double *C, int N)
{
    // 初始化输出为 0
    cudaMemset(C, 0, sizeof(double));

    int blockSize = 256;
    // 限制 block 数量以利用 Grid-Stride
    int numBlocks = std::min((N + blockSize - 1) / blockSize, 1024);

    // 注意：最终版代码内部使用了静态共享内存，不需要在调用时传递 sharedSize
    reduce_kernel_final<<<numBlocks, blockSize>>>(A, C, N);
    cudaDeviceSynchronize();
}

```
# 最终得分
