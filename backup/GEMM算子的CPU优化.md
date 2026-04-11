# 题目描述
这是一道偏入门的 CPU 性能优化题。你将拿到一个使用 C++ 和 Make 构建的小项目，你的任务是在保持结果正确的前提下，对其中的 GEMM 算子进行优化，使程序尽可能快。

矩阵乘法（GEMM, General Matrix-Matrix Multiplication）是高性能计算中最经典、最重要的基础算子之一。在科学计算、机器学习、图形学以及大量数值程序中，很多更复杂的计算最后都可以转化为矩阵乘法或与矩阵乘法类似的计算。

其核心计算过程即将输入矩阵的行向量与列向量求内积得到输出矩阵的每一个结果

本题的核心，就是优化一个标准的 GEMM 算子。你不需要提前掌握很多数学知识，更重要的是理解程序是如何访问内存、如何组织循环，以及为什么某些写法会更快。

本题实现的是最经典的矩阵乘法加法：
```text
C = A * B + C
```
其中：
- `A` 的形状为 `M x K`
- `B` 的形状为 `K x N`
- `C` 的形状为 `M x N`
- 所有矩阵均按 row-major 存储
- 数据类型固定为 `float`
## 项目结构
代码目录如下：
```text
.
├── data
│   └── cases.in *
├── include
│   └── gemm.h *
├── Makefile
├── README.md
├── run.sh
└── src
    ├── gemm.cpp
    └── main.cpp *
```
标有星号的文件/文件夹是**不可更改**的。你必须在不修改这些文件的前提下进行优化，保持它们的接口和行为不变。  
其中：
- `include/gemm.h` 声明你需要实现的接口
- `src/gemm.cpp` 是你主要需要优化的文件
- `src/main.cpp` 负责 reference 计算、正确性校验、测速和评分
- `data/cases.in` 是公开测试集
你可以修改Makefile来改变编译逻辑，但是必须要包含main.cpp和gemm.h以保证逻辑正确
你也可以修改run.sh来改变环境，但是必须保留原有的运行指令以保证测试正确
## 实现要求
你需要实现并优化如下接口：
```cpp
void gemm(const float* A, const float* B, float* C, int M, int N, int K);
```
要求：
- 保持接口不变
- 输出结果与参考实现足够接近
- 可以任意修改 `gemm.cpp` 的实现方式
- 允许使用多线程、SIMD或你认为合理的其他优化方式, 但**不允许使用已有高性能计算库**
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
./gemm_bench data/cases.in
```
测试单个 case：
```bash
./gemm_bench 256 512 256
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
请将你的代码放在名为cpu-gemm的文件夹中并打包为.tar.gz格式提交，我们会通过以下方式运行你的代码：
```bash
tar -xzf submission.tar.gz
cd cpu-gemm
./run.sh
```
## 平台说明
测试平台为Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz,你可以使用最多16核的资源。
我们有足够的时间限制和空间限制，请忽略页面上的时间和空间限制，当然，运行时间过长和内存占用过多会被杀死。

# 原始得分
本文采用的测试平台均为： 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
```bash
==> Summary
cases             : 6
total ref time    : 31207.638 ms
total impl time   : 29565.342 ms
overall speedup   : 1.056x
correctness score : 20.000 / 20.000
performance score : 0.964 / 80.000
total score       : 20.964 / 100.000
```
# 优化1：循环重排
在原始代码中，最内层循环是 $k$。计算 `B[k * N + j]` 时，$k$ 每增加 1，内存地址就会跳跃 $N$ 个位置。CPU 从内存读取数据时，实际上会一次性读取一整块（Cache Line，通常是 64 字节）到高速缓存中，跳跃访问会导致刚读进缓存的数据还没用完就被丢弃，缓存命中率极低（Cache Thrashing）。
下图展示了缓存结构，参考下面的命令查看cache相关信息
```bash
lscpu
ls /sys/devices/system/cpu/cpu0/cache/
cat /sys/devices/system/cpu/cpu0/cache/index0/level 
cat /sys/devices/system/cpu/cpu0/cache/index0/type 
cat /sys/devices/system/cpu/cpu0/cache/index0/size 
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size 
```
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260411154654.png)
我们将循环顺序改为 $i \rightarrow k \rightarrow j$。这样在最内层 $j$ 循环时，`C[i * N + j]` 和 `B[k * N + j]` 的内存地址都是**连续**加 1 的。CPU 的预取（Prefetcher）机制会完美生效，大幅减少内存等待时间。
代码：
```cpp
#include "gemm.h"

void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a = A[i * K + k];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}
```
得分：
```bash
==> Summary
cases             : 6
total ref time    : 30781.374 ms
total impl time   : 15831.562 ms
overall speedup   : 1.944x
correctness score : 20.000 / 20.000
performance score : 0.136 / 80.000
total score       : 20.136 / 100.000
```
# 优化2：分块
如果矩阵非常大（比如 $4096 \times 4096$），当最内层 $j$ 跑完一行时，产生的数据量早已超出了 L1/L2 缓存的容量。等到下一轮 $k$ 循环再去访问 `C` 的同一行时，它已经被挤出缓存，只能重新去慢速主存（RAM）里捞。

**分块**的思想是：我们不一次性算完一整行，而是把大矩阵切成（比如 $128 \times 256$）的“小瓷砖”（Tile）。我们保证每次计算的一块子矩阵大小，刚好能完全塞进 L1/L2 缓存里，算完这一块再算下一块。至于分块大小，需要根据CPU对应的缓存大小进行调优。

先给出实验平台（左）和本地平台（右）的cpu规格

| 参数          | **Xeon Gold 6338 **        | **i5-1135G7 **             |
| :---------- | :------------------------- | :------------------------- |
| **架构**      | Ice Lake (10nm)            | Tiger Lake (10nm SuperFin) |
| **核心 / 线程** | 32 核 / 64 线程               | 4 核 / 8 线程                 |
| **基频 / 睿频** | 2.0 GHz / **3.2 GHz**Intel | 2.4 GHz / 4.2 GHzIntel     |
| **L1d / 核** | 48 KB                      | 48 KB                      |
| **L2 / 核**  | 1.25 MB（32 核共 40MB）        | 1.25 MB（4 核共 5MB）          |
| **L3**      | 48 MB（共享）Intel             | 8 MB（共享）Intel              |

初始我们的分块方案采用BM=64；BN=128；BK=64，代码如下：
```cpp
#include "gemm.h"
#include <algorithm>

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  // 定义分块大小（需根据 CPU 的 L1/L2 Cache 大小调优）
  const int BM = 64;
  const int BN = 128;
  const int BK = 64;

  // 外层：遍历块 (Blocks)
  for (int i0 = 0; i0 < M; i0 += BM)
  {
    for (int j0 = 0; j0 < N; j0 += BN)
    {
      for (int k0 = 0; k0 < K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = std::min(i0 + BM, M);
        int j_end = std::min(j0 + BN, N);
        int k_end = std::min(k0 + BK, K);

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i < i_end; ++i)
        {
          for (int k = k0; k < k_end; ++k)
          {
            float a_ik = A[i * K + k];
            for (int j = j0; j < j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
```
得分：
```bash
==> Summary
cases             : 6
total ref time    : 37113.107 ms
total impl time   : 12385.678 ms
overall speedup   : 2.996x
correctness score : 20.000 / 20.000
performance score : 0.672 / 80.000
total score       : 20.672 / 100.000
```
分块参数调优参考：
计算一块参与 GEMM 的子矩阵总大小的公式为（单位：字节）：
$Size = (BM \cdot BN + BM \cdot BK + BK \cdot BN) \times 4$
# 优化3：SIMD
传统的标量计算，CPU 每个时钟周期只能执行一次 `float` 的乘加运算。现代 CPU 内部有很宽的向量寄存器（比如 AVX-512 是 512 位宽），可以**一条指令同时计算 16 个 float 的乘加**。通过 `#pragma omp simd` 强制告诉编译器同时处理多个 FLOP
```cpp
#include "gemm.h"
#include <algorithm>

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  // 定义分块大小（需根据 CPU 的 L1/L2 Cache 大小调优）
  const int BM = M;
  const int BN = N;
  const int BK = K;

  // 外层：遍历块 (Blocks)
  for (int i0 = 0; i0 < M; i0 += BM)
  {
    for (int j0 = 0; j0 < N; j0 += BN)
    {
      for (int k0 = 0; k0 < K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = std::min(i0 + BM, M);
        int j_end = std::min(j0 + BN, N);
        int k_end = std::min(k0 + BK, K);

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i < i_end; ++i)
        {
          for (int k = k0; k < k_end; ++k)
          {
            float a_ik = A[i * K + k];
#pragma omp simd
            for (int j = j0; j < j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
```
此外，在Makefile中需要修改前几行为如下
```bash
CXX := g++
COMMON_CXXFLAGS ?= -std=c++17 -g -Wall -Wextra -fopenmp
MAIN_CXXFLAGS ?= -O0
GEMM_CXXFLAGS ?= -O3 -march=native
CPPFLAGS ?=

TARGET := gemm_bench
OBJECTS := src/main.o src/gemm.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
    $(CXX) $(OBJECTS) -o $(TARGET) $(COMMON_CXXFLAGS)

src/main.o: src/main.cpp include/gemm.h
    $(CXX) $(CPPFLAGS) $(COMMON_CXXFLAGS) $(MAIN_CXXFLAGS) -I include -c $< -o $@

src/gemm.o: src/gemm.cpp include/gemm.h
    $(CXX) $(CPPFLAGS) $(COMMON_CXXFLAGS) $(GEMM_CXXFLAGS) -I include -c $< -o $@

run: $(TARGET)
    ./$(TARGET)

clean:
    rm -f $(TARGET) $(OBJECTS)

.PHONY: all run clean
```
得分：
```bash
==> Summary
cases             : 6
total ref time    : 36410.445 ms
total impl time   : 808.016 ms
overall speedup   : 45.062x
correctness score : 20.000 / 20.000
performance score : 17.288 / 80.000
total score       : 37.288 / 100.000
```
# 优化4：多核并行
在第二步中，我们已经做了矩阵分块，对于C=A×B+C，最外面的两层循环其实就对应着C矩阵的每一个分块处理，而这些分块之间的计算恰好是独立的，所以可以把每个分块的计算用不同核心并行化。但注意第三层循环的k0不能够加入并行，可以理解成每个分块的计算是第三层循环累加到C的结果，若并行，可能发生数据竞争，如下所示：
- **线程 A** 读取了 $C_{i,j}$ 的当前值（假设是 10）。
- **线程 B** 也读取了 $C_{i,j}$ 的当前值（也是 10）。
- **线程 A** 加上了自己的结果（+5），准备写回 15。
- **线程 B** 加上了自己的结果（+2），准备写回 12。
- **最终结果**：取决于谁最后写回。如果是 B 最后写，结果就是 12；如果是 A 最后写，就是 15。
- **正确结果**：应该是 $10 + 5 + 2 = 17$。

此处我们使用openmp做并行化，除了使用常规的parallel for之外，我们还加入了collapse子句，它的作用是将多个的循环“压扁（展平）”成一个巨大的一维循环，然后再把这些任务分给各个线程，若不使用，parallel for只会并行化最外面的一层循环。
```cpp
#include "gemm.h"

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  // 定义分块大小（需根据 CPU 的 L1/L2 Cache 大小调优）
  const int BM = 64;
  const int BN = 128;
  const int BK = 64;

  // 外层：遍历块 (Blocks)
  // 开启多线程并行
  // collapse(2) 表示将 i0 和 j0 这两层外循环合并为一个大的任务池进行分配
  // schedule(dynamic) 让计算快的线程主动领新任务，防止各个核心工作量不均
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i0 = 0; i0 < M; i0 += BM)
  {
    for (int j0 = 0; j0 < N; j0 += BN)
    {
      for (int k0 = 0; k0 < K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = (i0 + BM < M) ? i0 + BM : M;
        int j_end = (j0 + BN < N) ? j0 + BN : N;
        int k_end = (k0 + BK < K) ? k0 + BK : K;

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i < i_end; ++i)
        {
          for (int k = k0; k < k_end; ++k)
          {
            float a_ik = A[i * K + k];
#pragma omp simd
            for (int j = j0; j < j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
```
得分：
```bash
==> Summary
cases             : 6
total ref time    : 37793.549 ms
total impl time   : 263.726 ms
overall speedup   : 143.306x
correctness score : 20.000 / 20.000
performance score : 36.292 / 80.000
total score       : 56.292 / 100.000
```
# 优化5：条件判断
跑分时发现小规模矩阵(case1、case2）的性能损失很大，这是因为加入多线程的开销超过了其收益，因此我们在原来基础上加一个阈值判断，若规模很小，无需开启上述优化。
```cpp
#include "gemm.h"

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  if (M <= 128 && N <= 128 && K <= 128)
  {
    for (int i = 0; i < M; ++i)
    {
      for (int k = 0; k < K; ++k)
      {
        float a_ik = A[i * K + k];
#pragma omp simd
        for (int j = 0; j < N; ++j)
        {
          C[i * N + j] += a_ik * B[k * N + j];
        }
      }
    }
    return;
  }

  // 定义分块大小（需根据 CPU 的 L1/L2 Cache 大小调优）
  const int BM = 64;
  const int BN = 128;
  const int BK = 64;

  // 外层：遍历块 (Blocks)
  // 开启多线程并行
  // collapse(2) 表示将 i0 和 j0 这两层外循环合并为一个大的任务池进行分配
  // schedule(dynamic) 让计算快的线程主动领新任务，防止各个核心工作量不均
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i0 = 0; i0 < M; i0 += BM)
  {
    for (int j0 = 0; j0 < N; j0 += BN)
    {
      for (int k0 = 0; k0 < K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = (i0 + BM < M) ? i0 + BM : M;
        int j_end = (j0 + BN < N) ? j0 + BN : N;
        int k_end = (k0 + BK < K) ? k0 + BK : K;

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i < i_end; ++i)
        {
          for (int k = k0; k < k_end; ++k)
          {
            float a_ik = A[i * K + k];
#pragma omp simd
            for (int j = j0; j < j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
```
得分：
```bash
==> Summary
cases             : 6
total ref time    : 44226.380 ms
total impl time   : 238.423 ms
overall speedup   : 185.495x
correctness score : 20.000 / 20.000
performance score : 43.767 / 80.000
total score       : 63.767 / 100.000
```
# 最终得分
上述得分测试均是在本地测试环境（11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz）下测的，把代码上传至实验平台（Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz）上得分如下：
