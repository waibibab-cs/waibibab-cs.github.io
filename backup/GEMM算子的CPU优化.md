<html>
<body>
<!--StartFragment--><!-- obsidian --><h1 data-heading="题目描述">题目描述</h1>
<p>这是一道偏入门的 CPU 性能优化题。你将拿到一个使用 C++ 和 Make 构建的小项目，你的任务是在保持结果正确的前提下，对其中的 GEMM 算子进行优化，使程序尽可能快。</p>
<p>矩阵乘法（GEMM, General Matrix-Matrix Multiplication）是高性能计算中最经典、最重要的基础算子之一。在科学计算、机器学习、图形学以及大量数值程序中，很多更复杂的计算最后都可以转化为矩阵乘法或与矩阵乘法类似的计算。</p>
<p>其核心计算过程即将输入矩阵的行向量与列向量求内积得到输出矩阵的每一个结果</p>
<p>本题的核心，就是优化一个标准的 GEMM 算子。你不需要提前掌握很多数学知识，更重要的是理解程序是如何访问内存、如何组织循环，以及为什么某些写法会更快。</p>
<p>本题实现的是最经典的矩阵乘法加法：</p>
<pre><code class="language-text">C = A * B + C
</code></pre>
<p>其中：</p>
<ul>
<li><code>A</code> 的形状为 <code>M x K</code></li>
<li><code>B</code> 的形状为 <code>K x N</code></li>
<li><code>C</code> 的形状为 <code>M x N</code></li>
<li>所有矩阵均按 row-major 存储</li>
<li>数据类型固定为 <code>float</code></li>
</ul>
<h2 data-heading="项目结构">项目结构</h2>
<p>代码目录如下：</p>
<pre><code class="language-text">.
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
</code></pre>
<p>标有星号的文件/文件夹是<strong>不可更改</strong>的。你必须在不修改这些文件的前提下进行优化，保持它们的接口和行为不变。<br>
其中：</p>
<ul>
<li><code>include/gemm.h</code> 声明你需要实现的接口</li>
<li><code>src/gemm.cpp</code> 是你主要需要优化的文件</li>
<li><code>src/main.cpp</code> 负责 reference 计算、正确性校验、测速和评分</li>
<li><code>data/cases.in</code> 是公开测试集<br>
你可以修改Makefile来改变编译逻辑，但是必须要包含main.cpp和gemm.h以保证逻辑正确<br>
你也可以修改run.sh来改变环境，但是必须保留原有的运行指令以保证测试正确</li>
</ul>
<h2 data-heading="实现要求">实现要求</h2>
<p>你需要实现并优化如下接口：</p>
<pre><code class="language-cpp">void gemm(const float* A, const float* B, float* C, int M, int N, int K);
</code></pre>
<p>要求：</p>
<ul>
<li>保持接口不变</li>
<li>输出结果与参考实现足够接近</li>
<li>可以任意修改 <code>gemm.cpp</code> 的实现方式</li>
<li>允许使用多线程、SIMD或你认为合理的其他优化方式, 但<strong>不允许使用已有高性能计算库</strong><br>
禁止事项：</li>
<li>不允许修改题目给定的输入规模文件来规避测试</li>
<li>不允许通过识别特定输入规模或特定数据内容做针对性特化</li>
<li>不允许提交错误结果或伪造性能数据</li>
</ul>
<h2 data-heading="构建与运行">构建与运行</h2>
<p>构建：</p>
<pre><code class="language-bash">make
</code></pre>
<p>读取公开测试集直接运行：</p>
<pre><code class="language-bash">./gemm_bench data/cases.in
</code></pre>
<p>测试单个 case：</p>
<pre><code class="language-bash">./gemm_bench 256 512 256
</code></pre>
<h2 data-heading="评分标准">评分标准</h2>
<p>总分为 <code>100</code> 分：</p>
<ul>
<li>正确性：<code>20</code> 分</li>
<li>性能：<code>80</code> 分<br>
评分方式如下：</li>
</ul>
<ol>
<li>所有公开测试 case 都正确，才能拿到正确性分。</li>
<li>每个 case 单独计算相对 reference 的加速比。</li>
<li>不同规模的 case 使用不同的性能阈值。</li>
<li>每个 case 先得到一个单独的性能分，再按权重汇总为最终的 <code>80</code> 分性能分。<br>
程序运行时会输出：</li>
</ol>
<ul>
<li>每个 case 的 reference 时间</li>
<li>每个 case 的实现时间</li>
<li>每个 case 的加速比</li>
<li>每个 case 的性能得分</li>
<li>最终总分</li>
</ul>
<h2 data-heading="提交方式">提交方式</h2>
<p>请将你的代码放在名为cpu-gemm的文件夹中并打包为.tar.gz格式提交，我们会通过以下方式运行你的代码：</p>
<pre><code class="language-bash">tar -xzf submission.tar.gz
cd cpu-gemm
./run.sh
</code></pre>
<h2 data-heading="平台说明">平台说明</h2>
<p>测试平台为Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz,你可以使用最多16核的资源。<br>
我们有足够的时间限制和空间限制，请忽略页面上的时间和空间限制，当然，运行时间过长和内存占用过多会被杀死。</p>
<h1 data-heading="原始得分">原始得分</h1>
<p>本文采用的测试平台均为： 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz</p>
<pre><code class="language-bash">==> Summary
cases             : 6
total ref time    : 31207.638 ms
total impl time   : 29565.342 ms
overall speedup   : 1.056x
correctness score : 20.000 / 20.000
performance score : 0.964 / 80.000
total score       : 20.964 / 100.000
</code></pre>
<h1 data-heading="优化1：循环重排">优化1：循环重排</h1>
<p>在原始代码中，最内层循环是 <span class="math math-inline">k</span>。计算 <code>B[k * N + j]</code> 时，<span class="math math-inline">k</span> 每增加 1，内存地址就会跳跃 <span class="math math-inline">N</span> 个位置。CPU 从内存读取数据时，实际上会一次性读取一整块（Cache Line，通常是 64 字节）到高速缓存中，跳跃访问会导致刚读进缓存的数据还没用完就被丢弃，缓存命中率极低（Cache Thrashing）。<br>
下图展示了缓存结构，参考下面的命令查看cache相关信息</p>
<pre><code class="language-bash">lscpu
ls /sys/devices/system/cpu/cpu0/cache/
cat /sys/devices/system/cpu/cpu0/cache/index0/level 
cat /sys/devices/system/cpu/cpu0/cache/index0/type 
cat /sys/devices/system/cpu/cpu0/cache/index0/size 
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size 
</code></pre>
<p><img src="https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260411154654.png" alt="image.png"><br>
我们将循环顺序改为 <span class="math math-inline">i \rightarrow k \rightarrow j</span>。这样在最内层 <span class="math math-inline">j</span> 循环时，<code>C[i * N + j]</code> 和 <code>B[k * N + j]</code> 的内存地址都是<strong>连续</strong>加 1 的。CPU 的预取（Prefetcher）机制会完美生效，大幅减少内存等待时间。<br>
代码：</p>
<pre><code class="language-cpp">#include "gemm.h"

void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i &#x3C; M; ++i) {
    for (int k = 0; k &#x3C; K; ++k) {
      float a = A[i * K + k];
      for (int j = 0; j &#x3C; N; ++j) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}
</code></pre>
<p>得分：</p>
<pre><code class="language-bash">==> Summary
cases             : 6
total ref time    : 30781.374 ms
total impl time   : 15831.562 ms
overall speedup   : 1.944x
correctness score : 20.000 / 20.000
performance score : 0.136 / 80.000
total score       : 20.136 / 100.000
</code></pre>
<h1 data-heading="优化2：分块">优化2：分块</h1>
<p>如果矩阵非常大（比如 <span class="math math-inline">4096 \times 4096</span>），当最内层 <span class="math math-inline">j</span> 跑完一行时，产生的数据量早已超出了 L1/L2 缓存的容量。等到下一轮 <span class="math math-inline">k</span> 循环再去访问 <code>C</code> 的同一行时，它已经被挤出缓存，只能重新去慢速主存（RAM）里捞。</p>
<p><strong>分块</strong>的思想是：我们不一次性算完一整行，而是把大矩阵切成（比如 <span class="math math-inline">128 \times 256</span>）的“小瓷砖”（Tile）。我们保证每次计算的一块子矩阵大小，刚好能完全塞进 L1/L2 缓存里，算完这一块再算下一块。至于分块大小，需要根据CPU对应的缓存大小进行调优。</p>
<p>先给出实验平台（左）和本地平台（右）的cpu规格</p>

参数 | Xeon Gold 6338 | i5-1135G7
-- | -- | --
架构 | Ice Lake (10nm) | Tiger Lake (10nm SuperFin)
核心 / 线程 | 32 核 / 64 线程 | 4 核 / 8 线程
基频 / 睿频 | 2.0 GHz / 3.2 GHzIntel | 2.4 GHz / 4.2 GHzIntel
L1d / 核 | 48 KB | 48 KB
L2 / 核 | 1.25 MB（32 核共 40MB） | 1.25 MB（4 核共 5MB）
L3 | 48 MB（共享）Intel | 8 MB（共享）Intel


<p>初始我们的分块方案采用BM=64；BN=128；BK=64，代码如下：</p>
<pre><code class="language-cpp">#include "gemm.h"
#include &#x3C;algorithm>

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  // 定义分块大小（需根据 CPU 的 L1/L2 Cache 大小调优）
  const int BM = 64;
  const int BN = 128;
  const int BK = 64;

  // 外层：遍历块 (Blocks)
  for (int i0 = 0; i0 &#x3C; M; i0 += BM)
  {
    for (int j0 = 0; j0 &#x3C; N; j0 += BN)
    {
      for (int k0 = 0; k0 &#x3C; K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = std::min(i0 + BM, M);
        int j_end = std::min(j0 + BN, N);
        int k_end = std::min(k0 + BK, K);

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i &#x3C; i_end; ++i)
        {
          for (int k = k0; k &#x3C; k_end; ++k)
          {
            float a_ik = A[i * K + k];
            for (int j = j0; j &#x3C; j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
</code></pre>
<p>得分：</p>
<pre><code class="language-bash">==> Summary
cases             : 6
total ref time    : 37113.107 ms
total impl time   : 12385.678 ms
overall speedup   : 2.996x
correctness score : 20.000 / 20.000
performance score : 0.672 / 80.000
total score       : 20.672 / 100.000
</code></pre>
<p>分块参数调优参考：<br>
计算一块参与 GEMM 的子矩阵总大小的公式为（单位：字节）：<br>
<span class="math math-inline">Size = (BM \cdot BN + BM \cdot BK + BK \cdot BN) \times 4</span></p>
<h1 data-heading="优化3：SIMD">优化3：SIMD</h1>
<p>传统的标量计算，CPU 每个时钟周期只能执行一次 <code>float</code> 的乘加运算。现代 CPU 内部有很宽的向量寄存器（比如 AVX-512 是 512 位宽），可以<strong>一条指令同时计算 16 个 float 的乘加</strong>。通过 <code>#pragma omp simd</code> 强制告诉编译器同时处理多个 FLOP</p>
<pre><code class="language-cpp">#include "gemm.h"
#include &#x3C;algorithm>

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  // 定义分块大小（需根据 CPU 的 L1/L2 Cache 大小调优）
  const int BM = M;
  const int BN = N;
  const int BK = K;

  // 外层：遍历块 (Blocks)
  for (int i0 = 0; i0 &#x3C; M; i0 += BM)
  {
    for (int j0 = 0; j0 &#x3C; N; j0 += BN)
    {
      for (int k0 = 0; k0 &#x3C; K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = std::min(i0 + BM, M);
        int j_end = std::min(j0 + BN, N);
        int k_end = std::min(k0 + BK, K);

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i &#x3C; i_end; ++i)
        {
          for (int k = k0; k &#x3C; k_end; ++k)
          {
            float a_ik = A[i * K + k];
#pragma omp simd
            for (int j = j0; j &#x3C; j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
</code></pre>
<p>此外，在Makefile中需要修改前几行为如下</p>
<pre><code class="language-bash">CXX := g++
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
    $(CXX) $(CPPFLAGS) $(COMMON_CXXFLAGS) $(MAIN_CXXFLAGS) -I include -c $&#x3C; -o $@

src/gemm.o: src/gemm.cpp include/gemm.h
    $(CXX) $(CPPFLAGS) $(COMMON_CXXFLAGS) $(GEMM_CXXFLAGS) -I include -c $&#x3C; -o $@

run: $(TARGET)
    ./$(TARGET)

clean:
    rm -f $(TARGET) $(OBJECTS)

.PHONY: all run clean
</code></pre>
<p>得分：</p>
<pre><code class="language-bash">==> Summary
cases             : 6
total ref time    : 36410.445 ms
total impl time   : 808.016 ms
overall speedup   : 45.062x
correctness score : 20.000 / 20.000
performance score : 17.288 / 80.000
total score       : 37.288 / 100.000
</code></pre>
<h1 data-heading="优化4：多核并行">优化4：多核并行</h1>
<p>在第二步中，我们已经做了矩阵分块，对于C=A×B+C，最外面的两层循环其实就对应着C矩阵的每一个分块处理，而这些分块之间的计算恰好是独立的，所以可以把每个分块的计算用不同核心并行化。但注意第三层循环的k0不能够加入并行，可以理解成每个分块的计算是第三层循环累加到C的结果，若并行，可能发生数据竞争，如下所示：</p>
<ul>
<li><strong>线程 A</strong> 读取了 <span class="math math-inline">C_{i,j}</span> 的当前值（假设是 10）。</li>
<li><strong>线程 B</strong> 也读取了 <span class="math math-inline">C_{i,j}</span> 的当前值（也是 10）。</li>
<li><strong>线程 A</strong> 加上了自己的结果（+5），准备写回 15。</li>
<li><strong>线程 B</strong> 加上了自己的结果（+2），准备写回 12。</li>
<li><strong>最终结果</strong>：取决于谁最后写回。如果是 B 最后写，结果就是 12；如果是 A 最后写，就是 15。</li>
<li><strong>正确结果</strong>：应该是 <span class="math math-inline">10 + 5 + 2 = 17</span>。</li>
</ul>
<p>此处我们使用openmp做并行化，除了使用常规的parallel for之外，我们还加入了collapse子句，它的作用是将多个的循环“压扁（展平）”成一个巨大的一维循环，然后再把这些任务分给各个线程，若不使用，parallel for只会并行化最外面的一层循环。</p>
<pre><code class="language-cpp">#include "gemm.h"

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
  for (int i0 = 0; i0 &#x3C; M; i0 += BM)
  {
    for (int j0 = 0; j0 &#x3C; N; j0 += BN)
    {
      for (int k0 = 0; k0 &#x3C; K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = (i0 + BM &#x3C; M) ? i0 + BM : M;
        int j_end = (j0 + BN &#x3C; N) ? j0 + BN : N;
        int k_end = (k0 + BK &#x3C; K) ? k0 + BK : K;

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i &#x3C; i_end; ++i)
        {
          for (int k = k0; k &#x3C; k_end; ++k)
          {
            float a_ik = A[i * K + k];
#pragma omp simd
            for (int j = j0; j &#x3C; j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
</code></pre>
<p>得分：</p>
<pre><code class="language-bash">==> Summary
cases             : 6
total ref time    : 37793.549 ms
total impl time   : 263.726 ms
overall speedup   : 143.306x
correctness score : 20.000 / 20.000
performance score : 36.292 / 80.000
total score       : 56.292 / 100.000
</code></pre>
<h1 data-heading="优化5：条件判断">优化5：条件判断</h1>
<p>跑分时发现小规模矩阵(case1、case2）的性能损失很大，这是因为加入多线程的开销超过了其收益，因此我们在原来基础上加一个阈值判断，若规模很小，无需开启上述优化。</p>
<pre><code class="language-cpp">#include "gemm.h"

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
  if (M &#x3C;= 128 &#x26;&#x26; N &#x3C;= 128 &#x26;&#x26; K &#x3C;= 128)
  {
    for (int i = 0; i &#x3C; M; ++i)
    {
      for (int k = 0; k &#x3C; K; ++k)
      {
        float a_ik = A[i * K + k];
#pragma omp simd
        for (int j = 0; j &#x3C; N; ++j)
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
  for (int i0 = 0; i0 &#x3C; M; i0 += BM)
  {
    for (int j0 = 0; j0 &#x3C; N; j0 += BN)
    {
      for (int k0 = 0; k0 &#x3C; K; k0 += BK)
      {

        // 处理边界情况，防止矩阵维度不是分块大小的整数倍导致越界
        int i_end = (i0 + BM &#x3C; M) ? i0 + BM : M;
        int j_end = (j0 + BN &#x3C; N) ? j0 + BN : N;
        int k_end = (k0 + BK &#x3C; K) ? k0 + BK : K;

        // 这里依然保持 i -> k -> j 的连续访存顺序
        for (int i = i0; i &#x3C; i_end; ++i)
        {
          for (int k = k0; k &#x3C; k_end; ++k)
          {
            float a_ik = A[i * K + k];
#pragma omp simd
            for (int j = j0; j &#x3C; j_end; ++j)
            {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
</code></pre>
<p>得分：</p>
<pre><code class="language-bash">==> Summary
cases             : 6
total ref time    : 44226.380 ms
total impl time   : 238.423 ms
overall speedup   : 185.495x
correctness score : 20.000 / 20.000
performance score : 43.767 / 80.000
total score       : 63.767 / 100.000
</code></pre>
<h1 data-heading="最终得分">最终得分</h1>
<p>上述得分测试均是在本地测试环境（11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz）下测的，把代码上传至实验平台（Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz）上得分如下：</p><!--EndFragment-->
</body>
</html>