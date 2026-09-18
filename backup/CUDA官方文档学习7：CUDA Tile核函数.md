原文链接：
[2.4.Writing Tile Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html)
# 前言
CUDA Tile 提供了一种不同于前面章节介绍的单指令多线程（SIMT）模型的 GPU 内核代码编写方式。Tile 编程允许程序员以另一种形式表达并行逻辑，而将最底层的并行处理交由编译器与内置算子完成。借助该机制，Tile 能够让开发者更简便地调用 NVIDIA GPU 的各类新型高性能硬件单元，例如**TMA 多维单元（Tensor Memory Access，张量内存访问）** 与张量核心（Tensor Core），
* Python 环境下可通过 cuTile 软件包 `cuda.tile` 使用 CUDA Tile 编程。
* C++ 版本的 CUDA Tile 自 CUDA Toolkit 13.3 版本起正式纳入工具集。

围绕 Tile 核函数的应用代码，包括设备内存分配、主机与设备间的数据传输、核函数启动调度等任务，与前面章节介绍的 SIMT 核函数完全一致。Tile 核函数操作由标准 CUDA API 分配的全局内存，计算结果同样以相同方式回传至主机。唯一的区别在于**核函数内部**的代码编写逻辑。

在 SIMT 核函数中，开发者以单个线程为思考单位：计算全局线程索引，加载该线程对应的数据元素，对元素执行运算，并保存结果。而在 Tile 核函数中，开发者以整个线程块为思考单位：加载包含大量元素的一个分片（tile），对整个分片执行运算，再存储结果。由编译器负责将分片运算映射到线程块内的硬件线程；这部分工作在 SIMT 编程中需要开发者手动显式处理。

本章专门聚焦这一差异：如何编写核函数入口，以及在入口内实现分片运算。所有代码范式均同时提供 CuTile Python（`cuda.tile`）与 CUDA Tile C++（`cuda::tiles`）两种实现。二者共用同一套编译器后端（CUDA Tile IR，CUDA 分片中间表示），因此拥有完全相同的执行语义。

按照约定，两种语言中均将 Tile API 别名定义为`ct`：
- Python：`import cuda.tile as ct`
- C++：`namespace ct = cuda::tiles;`
Python 中，Tile API 位于`cuda.tile`模块，导入方式如上所示。
C++ 中，Tile API 位于`cuda::tiles`命名空间，由头文件`cuda_tile.h`暴露：
```cpp
# include "cuda_tile.h"
namespace ct = cuda::tiles;
```
下文代码片段中的`ct.`（Python）与`ct::`（C++）前缀，代表对应语言下的 Tile API。
# 1.内核和函数声明
Tile 核函数是 GPU 入口函数，在启动网格中的每个线程块执行一次。Tile 函数可由 Tile 核函数或其他 Tile 函数调用，但本身不能由主机函数调用。与 SIMT 核函数一样，Tile 核函数无法由主机代码直接调用，**必须通过启动（launch）的方式执行**。

在 CUDA Tile C++ 中：
- `__tile_global__` 对应 SIMT 中的`__global__`，用于标记 Tile 核函数入口。
- `__tile__` 对应 SIMT 中的`__device__`，表示该函数编译至 GPU 端，可被其他`__tile__`或`__tile_global__`函数调用。

数组与标量参数的传递方式和 SIMT 核函数完全相同。Tile 代码与 SIMT 代码可以共存：单个`.cu`源文件中可以同时定义`__tile_global__`核函数与`__global__`核函数，同一个主机程序也可以启动这两类核函数。

>**注意** 当前，`__tile__` 函数**不能**被 `__global__` 或 `__device__` 函数调用。 同理，`__device__` 函数**不能**被 `__tile_global__` 或 `__tile__` 函数调用。该限制在未来版本的 CUDA 中可能会被解除。

在 cuTile Python 中：
- 装饰器 `@ct.kernel`：将函数标记为 Tile 核函数入口
- 装饰器 `@ct.function`：标记该函数可被 Tile 核函数或其他 Tile 函数调用
实际使用时，凡是在核函数内部调用的函数都会自动编译为 Tile 代码（是创造了一个Tile函数副本，而不是说这个函数就变成了Tile函数，所以还是可以被主机函数调用的）因此 `@ct.function` 装饰器是可选的。
数组参数支持任意驻留在设备显存（条件1）、并且对外暴露 DLPack 或者 CUDA Array Interface 的数组类型（条件2），标量参数直接传参，主机会直接把数值拷贝给kernel参数，不需要显存也不需要DLPack
>哪些库的GPU数组/张量支持上述两个接口？（举5个例子）
>1.pytorch；2.CuPy；3.Numba CUDA Device Array；4.JAX；5.CUDA Python原生数组

C++：
```cpp
#include "cuda_tile.h"
// Tile kernel entry point. Cannot be called directly; must be launched.
__tile_global__ void my_kernel(float* a, float* b, float* c) {
    ...
}
// Tile function. Callable from tile kernels and tile functions.
__tile__ float helper(float x, float y) {
    return x + y;
}
```
Python：
```python
import cuda.tile as ct

# Tile kernel entry point. Cannot be called directly; must be launched.
@ct.kernel
def my_kernel(a, b, c):
    ...

# Tile function. Callable from tile kernels and tile functions.
# @ct.function is optional, any function called from tile code
# is automatically compiled as tile code.
@ct.function
def helper(x, y):
    return x + y
```
# 2.启动内核
Tile 核函数在**Tile 块网格**上启动，就像 SIMT 核函数在线程块网格上启动一样。由程序员指定网格形状，最多支持三维。在程序员视角下，**每个 Tile 块由单个逻辑线程执行**；块内部的并行交由编译器管理。

在 C++ 中，Tile 核复用 SIMT 大家熟悉的三尖括号`<<<>>>`启动语法： 第一个尖括号参数是网格形状（Tile 块的数量）；第二个是 SIMT 中使用的**单块线程数**。对于 Tile 核，线程数量由编译器在内部决定，**第二个参数必须写为 1**。

Tile 核本质也是普通 CUDA 核函数，因此可以通过运行时已有 API `cudaLaunchKernel`、`cudaLaunchKernelEx` 启动，同样使用`grid, 1`的配置。当需要把 Tile 核集成到已使用上述 API 完成核启动的代码库时，该特性十分有用。
```cpp
// second arg must be 1
my_kernel<<<dim3(num_blocks_x, num_blocks_y), 1>>>(a, b, c);  
```
在 Python 中，`ct.launch`接收 4 个位置参数：CUDA 流、指定各维度 Tile 块数量的网格元组、核函数对象，以及核函数参数组成的元组。
```python
import torch
stream = torch.cuda.current_stream()     # CUDA stream object
grid = (num_blocks_x, num_blocks_y, 1)   # tile-block grid (x, y, z)
ct.launch(stream, grid, my_kernel, (a, b, c))
```
## 2.1 Grid-Sizing 模式
一种常用范式是启动足够多的块，使其能够覆盖整个数组；其中最后一个块在一个或多个维度上，其覆盖范围可能会超出数组本身的边界。
```cpp
int num_blocks = (N + tile_size - 1) / tile_size;   // ceil division -> covers partial tail
kernel<<<num_blocks, 1>>>(in, out, N);
```
```python
import math
grid = (math.ceil(N / TILE),)   # ceil division -> covers partial tail
ct.launch(stream, grid, my_kernel, (arr_in, arr_out, TILE))
```
后续第六节的子小节会介绍：当数组长度无法被分片大小整除时该如何处理。
# 三、查询块的位置
每个块都需要知道自身在网格中的位置，从而确定要处理哪一部分数据。 在 SIMT 编程模型中，程序员需要结合`blockIdx`与`threadIdx`来计算全局线程索引。 而在 Tile 代码中，**只需要块索引**；块内部所有线程级索引的计算都由编译器自动处理。

在 C++ 中： `ct::bid()` 返回一个`uint3`类型值，包含三个维度下的块索引。 `ct::num_blocks()` 返回一个`dim3`类型值，存放各个维度的总块数（由核函数启动参数决定），可通过`.x`、`.y`、`.z`访问各个维度分量。

在 Python 中： `ct.bid(axis)` 返回当前块在指定维度（0、1 或 2）上的索引，类型为 int32 标量。 `ct.num_blocks(axis)` 返回该维度上的总块数，常用于边界检查与循环计数。
```cpp
#include "cuda_tile.h"
__tile_global__ void my_kernel(float* a, float* b, float* c) {
    namespace ct = cuda::tiles;
    int bid_x = ct::bid().x;          // block index along .x
    int bid_y = ct::bid().y;          // block index along .y
    int num_x = ct::num_blocks().x;   // total blocks along .x
}
```
```python
@ct.kernel
def my_kernel(a, b, c):
    bid_x = ct.bid(0)          # block index along axis 0
    bid_y = ct.bid(1)          # block index along axis 1
    num_x = ct.num_blocks(0)   # total blocks along axis 0
```
# 四、创建Tiles
确定了块的标识之后，下一个问题就是 Tile 核实际操作的对象是什么 —— 也就是**Tile**：它是固定大小的多维标量元素数组，其形状与元素类型在编译期就已知。Tile 的每个维度长度必须是 2 的幂。Tile 具备**值语义**，这意味着拷贝 Tile 时会复制它的所有元素，两份拷贝完全独立。尽管如此，Tile 拷贝的开销很低，因为编译器控制 Tile 在硬件内部的表示形式。程序员无需为 Tile 分配或释放内存。

实际使用中，Tile 的创建方式有两种：从数组加载数据（Tile 空间加载与存储），或是使用工厂函数生成按指定模式填充的 Tile。

在 C++ 中，Tile 类型是显式声明的：`ct::tile<T, ct::shape<dims...>>`。其中`T`是元素类型，`ct::shape<dims...>`通过模板参数编码各个维度（整数值代表每个轴上编译期确定的尺寸）。例如，`ct::tile<float, ct::shape<8>>`是包含 8 个 float 的一维 Tile；`ct::tile<float, ct::shape<4, 4>>`是 4×4 的 float 二维 Tile。由于形状是类型的一部分，它在编译期始终可知。

工厂函数将完整 Tile 类型（下文记作`Tile`）作为模板参数：
- `ct::zeros<Tile>()` 与 `ct::ones<Tile>()`：生成全部元素为 0 或 1 的 Tile
- `ct::full<Tile>(val)`：每个元素都取值为`val`的 Tile
- `ct::iota<Tile>()`：元素依次为`(0, 1, ..., N−1)`的 Tile，N 是 Tile 总元素数量

本章所有 C++ 示例都会使用`using`类型别名（例如`using f32x4x4 = ct::tile<float, ct::shape<4, 4>>`），让调用处的 Tile 类型更易阅读。

在 Python 中，传给 Tile 工厂函数的 shape 元组与 dtype 参数都必须是**编译期常量**。Python 字面量（如`(64, 64)`）和`ct.float32`天然满足该要求；也可以使用带`Constant`注解的核参数来提供，详见后文`Python Constant[T]`。生成的 Tile 会暴露`.shape`、`.dtype`、`.ndim`属性，反映其编译期属性。

以下是这些工厂函数：
- `ct.zeros(shape, dtype)` 与 `ct.ones(shape, dtype)` —— 生成全部元素为 0 或 1 的 Tile。
- `ct.full(shape, fill_value, dtype)` —— 所有元素填充为任意指定常量值的 Tile。
- `ct.arange(size, dtype=...)` —— 一维 Tile，元素内容为 `[0, 1, …, size−1]`。
```cpp
#include "cuda_tile.h"
__tile__ void factories() {
    namespace ct = cuda::tiles;
    using i32x8   = ct::tile<int,   ct::shape<8>>;      // 1-D: 8 ints
    using f32x4x4 = ct::tile<float, ct::shape<4, 4>>;   // 2-D: 4x4 floats
    auto z      = ct::zeros<f32x4x4>();       // all zeros
    auto o      = ct::ones<f32x4x4>();        // all ones
    auto filled = ct::full<f32x4x4>(3.14f);   // all 3.14
    auto seq    = ct::iota<i32x8>();          // {0, 1, 2, 3, 4, 5, 6, 7}
}
```
```python
import cuda.tile as ct
@ct.function
def factories():
    zeros  = ct.zeros((64, 64), dtype=ct.float32)            # 64x64 tile of 0.0
    ones   = ct.ones((128,), dtype=ct.float16)               # 128-element tile of 1.0
    filled = ct.full((32, 32), 3.14, dtype=ct.float32)       # 32x32 tile of 3.14
    seq    = ct.arange(8, dtype=ct.int32)                    # [0, 1, 2, 3, 4, 5, 6, 7]
```
# 五、编译期常量
Tile 编译器会为每一组 Tile 形状、数据类型以及其他结构参数的组合，生成专用机器码。 因此，所有会影响生成代码的参数值，都必须在**编译期就确定**。也就是说：Tile 的形状和数据类型必须编译期已知。
在【创建 Tiles】章节，使用字面量来指定 Tile 的形状与数据类型，例如： `ct.zeros((64, 64), dtype=ct.float32)` 以及 `ct::tile<int, ct::shape<8>>`。
形状也可以作为编译期已知值，通过核函数接口传入，后续章节会展示相关用法。
## 5.1 Python Constant[T]
在核函数参数上添加 `ct.Constant[T]` 类型注解，会将该参数标记为**常量嵌入**。 这意味着，在内核中每次使用该参数时，效果等价于直接在对应位置写入字面量。
类型参数是可选的；不带类型参数的 `ct.Constant` 可以嵌入任意类型的常量。
`ct.Constant` 最常用于整型（`ct.Constant[int]`），一般用来作为控制 Tile 形状与循环边界的参数。
```python
import cuda.tile as ct
@ct.kernel
def my_kernel(TILE: ct.Constant[int]):
    # TILE is constant-embedded: wherever TILE appears, the compiler sees its
    # literal value (e.g., 128) and generates specialized code. Here TILE drives
    # the shape of a factory-built tile.
    zeros = ct.zeros((TILE,), dtype=ct.float32)
```
## 5.2 C++ integral constant 与 ic 字面量
在 CUDA Tile C++ 中，编译期数值通过 `ct::integral_constant` 表达，这是一种**数值编码在类型本身之中**的类型。 `ct::literals` 命名空间下的 `_ic` 字面量提供了简洁简写形式：`0_ic` 会生成 `ct::integral_constant<0>`。

接收编译期数值的 API，既支持非类型模板参数（NTTP）形式，也支持 `_ic` 字面量形式。 例如：`ct::cat` 用于沿着指定维度拼接两个 Tile，该维度必须在编译期确定。下面两行代码调用 `ct::cat` 时使用同一个编译期轴，二者的区别仅在于编译期数值的书写位置。
```cpp
#include "cuda_tile.h"
__tile__ void concat_demo() {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using T = ct::tile<int, ct::shape<4, 8>>;
    T lhs = ct::full<T>(0);
    T rhs = ct::full<T>(1);
    auto a = ct::cat<0>(lhs, rhs);     // NTTP form
    auto b = ct::cat(lhs, rhs, 0_ic);  // _ic form
}
```
`_ic`字面量还有另一个常用场景。`ct::extents` 和 `ct::shape` 都支持两种形式：非类型模板参数（NTTP）形式（例如 `ct::extents<std::uint32_t, 4, 8>`）与大括号初始化形式。

和 NTTP 形式不同，大括号形式**支持运行时值**。当存在一个或多个维度要到核启动阶段才能确定时，就使用这种写法：编译期维度用`_ic`字面量，运行时维度使用普通变量。 像`ct::tensor_span`、`ct::partition_view`这类 Tile 空间 API，就用该形式封装这类数组。
```cpp
auto shape2d = ct::extents{8_ic, length};  // 8 is compile-time; length is runtime
```
在任何需要传入编译期值的 API 参数位置，`_ic`字面量都是统一简写方式，例如`ct::cat`的拼接维度、extents 或 shape 的分量。
# 六、加载与存储Tiles
CUDA Tile 编程模型中有两类核心内存对象：**tile（片）** 与**array（数组）**。
array 是位于全局显存中的多维元素容器，tile 核的所有线程块都可以访问它。 
tile 同样是多维元素容器，但它仅属于单个 CUDA Tile 代码线程块。tile 通常是 array 中元素的一个子集。本节介绍如何将 array 的数据加载到 tile 中以供 tile 核使用，以及将 tile 的数据写回 array。

后续小节会介绍两种 tile 加载与存储方式：
- **Tile-Space Loads and Stores（Tile 空间加载与存储）**：使用 tile 空间索引，借助视图对象规定 array 元素映射到 tile 的规整可预测访问模式。
- **Gather and Scatter（聚集与分散）**：加载 / 存储时，依靠存放索引或指针的 tile，分别指定 tile 元素读写对应的 array 源元素或目标元素。

**性能提示**：在支持的硬件上，编译器可将 Tile 空间加载指令下沉到**张量内存加速器（TMA）** 执行，性能远高于逐元素 gather 操作。（C++ 相关内容可参见后续小节《C++ 性能优化建议》）

程序员需要自行决定加载时越界元素取何值。 Python 版本默认会静默丢弃越界写操作；C++ 中使用带掩码版本 API 时，越界写同样会被静默丢弃。
## 6.1 Tile-Space Loads and Stores
使用 Tile 空间加载时，会创建一个**视图对象**，用来规定如何将数组切分为由多个 Tile 大小区域构成的网格。这种映射关系就称为**Tile 空间**；Tile 核可以借助 Tile 空间索引，每次加载或存储其中一块区域。

Tile 空间加载的核心概念是数组的**分块视图（tiled view）**，它定义数组元素如何映射到指定尺寸的各个 Tile。图 19 展示的分块视图是一种**partition view（分区视图）**，它属于 Tile 空间，其中所有 Tile 大小固定、互不重叠，且 Tile 之间没有空隙。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260917114225.png)
当数组各维度无法被 Tile 尺寸整除时，在一个或多个维度上跨越数组边界的 Tile 只会被部分填充。程序员可以指定加载这类 Tile 时的处理逻辑，相关内容将在后面的小节介绍。

分区视图（partition view）是可在数组上定义的多种 Tile 空间之一。两种编程语言中还支持另外两类 Tile 空间：
- 固定形状的数据块，块之间由步长隔开；块与块之间可以存在空隙，也可以相互重叠。 C++ 中对应的是`ct::strided_view`；Python 则向`Array.tiled_view`传入`traversal_steps`参数。
- 通过一个存放元素索引的 Tile，在单个维度上选取数据块，用于聚集与分散小节介绍的不规则访存模式。 C++ 中为`ct::gather_scatter_view`；Python 使用`ct.load_advanced_indexing`与`ct.store_advanced_indexing`。
本章仅演示分区视图，它也是`Array.tiled_view`默认生成的 Tile 空间。其余视图类型请查阅对应语言的 API 参考文档（CUDA Tile C++ API 参考、cuTile Python 分块视图与高级索引）。
### 6.1.1 分区视图的加载与存储
结构化 Tile 空间加载是全局显存与 Tile 之间迁移数据的首选方式。核函数必须先构建一个视图对象用于定义 Tile 空间，之后再凭借 Tile 空间索引，逐块加载或存储 Tile。

在 C++ 中，分区视图分为两步构建：
- `ct::tensor_span`：将原生指针与`ct::extents`绑定，为指针赋予多维结构。
- `ct::partition_view`：将这个 span 切分为固定大小 Tile 组成的网格，并提供基于 Tile 空间坐标操作的`.load(idx...)`与`.store(tile, idx...)`接口。
```cpp
__tile_global__ void vec_add(float* __restrict__ a, float* __restrict__ b, float* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
	// 向cuTile承诺：指针地址是16字节对齐，因为TMA硬件对内存对齐有要求
    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);

    // Step 1: attach a shape to each raw pointer. 128_ic marks 128 as a compile-time constant.
    auto aSpan = ct::tensor_span{a,   ct::extents{128_ic}};
    auto bSpan = ct::tensor_span{b,   ct::extents{128_ic}};
    auto oSpan = ct::tensor_span{out, ct::extents{128_ic}};

    // Step 2: partition each span into a tile space of fixed 8-element tiles.
    auto aView = ct::partition_view{aSpan, ct::shape{8_ic}};
    auto bView = ct::partition_view{bSpan, ct::shape{8_ic}};
    auto oView = ct::partition_view{oSpan, ct::shape{8_ic}};

    int  bx    = ct::bid().x;             // this block's tile-space index along .x
    auto aTile = aView.load(bx);          // pick the bx-th tile of a
    auto bTile = bView.load(bx);
    oView.store(aTile + bTile, bx);       // write the tile back at the bx-th position of out
}
```
在 Python 中，`Array.tiled_view(tile_shape)`会返回一个`TiledView`，把数组划分成指定形状的 Tile。该视图提供接收 Tile 空间索引的`.load(index)` / `.store(index, tile)`方法，与 C++ 的`partition_view`一一对应。
```cpp
@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    a_view = a.tiled_view((TILE,))
    b_view = b.tiled_view((TILE,))
    c_view = c.tiled_view((TILE,))

    bid = ct.bid(0)
    a_tile = a_view.load((bid,))
    b_tile = b_view.load((bid,))
    c_view.store((bid,), a_tile + b_tile)

```
> **注** 本章 C++ 示例代码会给指针参数加上`__restrict__`注解，并在核函数体开头调用`ct::assume_aligned(ptr, 16_ic)`。这些是重要的性能注解，将在后续小节《C++ 性能优化建议》详述。数值字面量后的`_ic`后缀（例如`128_ic`、`8_ic`）用于标记其为编译期常量，相关内容在《编译期常量》小节已介绍。

### 6.1.2 Python 一次性加载与存储接口
Python 额外提供了**单调用形式**：可以在每次 load /store 调用时直接内联指定 Tile 形状，无需显式创建 view 视图对象。
`ct.load(array, index, shape)`：在给定的 Tile 空间索引位置，读取一块指定形状的 Tile。
`ct.store(array, index, tile)`：对应的写回操作。
`ct.load` / `ct.store` 与 `Array.tiled_view` 表达**完全相同的 Tile 空间访问模式**，二者的区别仅在于 Tile 形状定义的位置：
- 使用`Array.tiled_view`：Tile 形状一次性绑定到 view 视图对象上。
- 使用`ct.load` / `ct.store`：每次调用都要内联传入 Tile 形状。
当多次加载、存储复用同一种分块规则时，推荐使用`tiled_view`；
如果只是一次性单次加载，追求代码简洁，则可以使用`ct.load` / `ct.store`。
```python
@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)                                    # this block's tile-space index along axis 0
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))   # (index, shape) = pick the bid-th TILE-sized region of a
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))
    ct.store(c, index=(bid,), tile=a_tile + b_tile)    # write the tile back to the bid-th region of c
```
### 6.1.3 Tile空间边界处理
在 C++ 中，`partition_view` 提供**无掩码**与**带掩码**两套接口：
- `.load(idx...)` / `.store(tile, idx...)`：假定整个 Tile 完全落在数组合法范围内。访问部分越界的 Tile 属于**未定义行为**。
- `.load_masked(idx...)` / `.store_masked(tile, idx...)`：安全处理边缘的部分填充 Tile。
    - `.load_masked()`：默认将越界位置填充 0；也可选择其他填充模式（例如浮点 Tile 填充 NaN）。
    - `.store_masked()`：静默丢弃越界位置的写操作，不会写入越界内存。

当数组长度可以被 Tile 尺寸整除时，推荐使用无掩码版本的加载 / 存储。当必须处理边界情况时，即便 Tile 是完全填充的完整 Tile，也可以使用带掩码版本。

本节也是本指南中**第一个数组维度为运行时值**的 C++ 示例。 `ct::extents{N}` 支持运行时维度；`ct::extents` 可以混合编译期常量（`_ic`）与运行时值。因此`span`和`partition_view`可以封装那些**仅在内核启动时才确定大小**的数组。
```cpp
__tile_global__ void edge_safe(float* __restrict__ in, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    in  = ct::assume_aligned(in,  16_ic);
    out = ct::assume_aligned(out, 16_ic);

    // ct::extents{N} uses a runtime dimension; 128_ic stays compile-time.
    auto inView  = ct::partition_view{ct::tensor_span{in,  ct::extents{N}}, ct::shape{128_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{128_ic}};

    int  bx   = ct::bid().x;
    auto tile = inView.load_masked(bx);    // masked load: OOB lanes default to 0
    outView.store_masked(tile, bx);        // masked store: OOB writes silently discarded
}
```
在 Python 中，`ct.load`支持`padding_mode`参数，用于指定越界元素填充的值。两种常用模式：
- `PaddingMode.UNDETERMINED`（默认值）：越界元素的值由底层实现决定。加载操作仍然会做边界检查，**不会读取数组以外的内存**，只是越界通道的值是未指定的。适合永远不会读取这些越界通道的场景。
- `PaddingMode.ZERO`：越界元素填充 0。

对于写操作`ct.store`，不需要`padding_mode`参数：默认会对照数组做边界检查，**静默丢弃对越界位置的写入**。`tiled_view`遵循相同规则，它会在创建视图时就固定`padding_mode`。

`ct.load`和`ct.store`还支持`check_bounds`参数，默认值为`True`。 `padding_mode`决定越界通道里填什么值；而`check_bounds`决定**是否开启边界检测**： 若设置`check_bounds=False`，访问超出数组范围属于未定义行为，可能读取或覆盖相邻内存。 这就等价上面 C++ 无掩码版本的`.load()` / `.store()`，适用原则也完全一致。 `TiledView.load`与`TiledView.store`也支持该关键字参数。
```python
@ct.kernel
def edge_safe(arr_in, arr_out, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tile = ct.load(arr_in, index=(bid,), shape=(TILE,),
                   padding_mode=ct.PaddingMode.ZERO)   # OOB lanes of a partial edge tile become 0
    ct.store(arr_out, index=(bid,), tile=tile)         # OOB writes are silently discarded
```
在 C++ 核函数中，`.load_masked()` 和 `.store_masked()` 用于处理边缘的**部分 Tile**。 在 Python 核函数中，加载时指定`PaddingMode.ZERO`可以保证边缘部分 Tile 用 0 填充；而`ct.store`在默认开启边界检查的情况下，会静默丢弃超出数组边界的写入操作。

完整的填充模式、掩码选项与填充值说明，请查阅对应语言的 API 参考文档（CUDA Tile C++ 视图填充、cuTile Python 填充模式）。
## 6.2 Gather和Scatter
在上一小节里的 Tile 空间加载操作使用了分区视图，它对数组做**规则、块对齐**的划分。 当访问模式是**不规则、数据依赖型**时（例如查找表、置换重排场景），**聚集（gather）与分散（scatter）操作**支持通过任意索引或地址，从数组里非均匀、非连续的位置读取 Tile，或将 Tile 写入这类位置。

Gather 与 scatter 操作在 C++ 和 Python 中的写法略有差异：

- Python：向`ct.gather()` / `ct.scatter()`传入**整型索引 Tile**，内置边界检查。
- C++：传入**指针 Tile**给`ct::load()` / `ct::store()`；配套掩码版本`ct::load_masked()`、`ct::store_masked()`，可接收布尔掩码 Tile 来处理数组边界处的 Tile。

在 C++ 中，gather/scatter 的实现方式是构造一个指针 Tile：每个元素对应一个指针，再将该指针 Tile 传给`ct::load()`或`ct::store()`。标量指针与整型 Tile 之间按元素做算术运算，生成指针 Tile。这是 C++ 里构造 gather/scatter 索引 Tile 的标准写法。
```cpp
__tile_global__ void vec_add_gather(int* __restrict__ a, int* __restrict__ b, int* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using i32x8 = ct::tile<int, ct::shape<8>>;
    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);
    int bx       = ct::bid().x;
    auto offsets = 8 * bx + ct::iota<i32x8>();   // element-level offsets, one per lane
    // scalar pointer + int tile = tile of pointers (one pointer per offset).
    auto aPtrs = a + offsets;
    auto bPtrs = b + offsets;
    auto aTile = ct::load(aPtrs);                // gather: one load per pointer
    auto bTile = ct::load(bPtrs);
    ct::store(out + offsets, aTile + bTile);     // scatter: one store per pointer
}
```
在 Python 中，`ct.gather`读取索引 Tile 中每个下标对应的元素。默认开启边界检查：越界索引会返回填充值（默认为 0，可通过`padding_value=`配置）；也可以设置`check_bounds=False`关闭边界检查。 `ct.scatter`按每个索引写入一个值；越界写操作会被静默丢弃，同样支持`check_bounds=False`。
### 6.2.1 Gather与Scatter的边界处理
上一节介绍的 gather/scatter 操作，其边界处理遵循一套独立的规则。

在 Python 中，`ct.gather` 和 `ct.scatter` **默认自带边界安全保护**：越界读会返回填充值（默认是 0）；越界写会被静默丢弃。 如果能保证所有索引都在合法范围内，可以关闭边界检查；一旦关闭，发生越界访问就属于未定义行为。 可选的掩码、填充值参数，请查阅 API 参考文档（CUDA Tile C++ 加载操作、cuTile Python 加载 / 存储操作）。

在 C++ 中，**不会自动做边界检查**。需要程序员自己构造布尔掩码（例如：把偏移量和数组长度做比较），再将掩码传入 `ct::load_masked` 或者 `ct::store_masked`。
```cpp
__tile_global__ void gather_safe(int* __restrict__ arr, int* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using i32x8 = ct::tile<int, ct::shape<8>>;

    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    int bx       = ct::bid().x;
    auto offsets = 8 * bx + ct::iota<i32x8>();   // element-level offsets, one per lane
    auto mask    = offsets < N;                  // boolean tile: true where the offset is in-bounds

    auto ptrs = arr + offsets;                   // tile of pointers, one per offset
    auto tile = ct::load_masked(ptrs, mask, 0);  // masked lanes get the pad value 0
    ct::store_masked(out + offsets, tile, mask); // masked lanes are skipped on the store
}
```
# 七、控制流
从程序员视角来看，一个 Tile 核函数**每个 block 只走一条控制流路径**。条件判断、循环边界中的标量值负责驱动控制流；而函数体内的 Tile 运算，则由编译器分发到硬件线程上执行。
并非所有控制流语法都支持。例如，Tile 代码**不允许在循环内部 return 返回**。完整限制列表请查阅各语言的 API 参考文档。
## 7.1 循环
一种常见写法：遍历数组里的多个 Tile，依次处理每一块。
C++ 中，`ct::irange` 是一个正向区间，表示从下界开始、**递增到上界（不含上界）** 的整数序列，可指定可选步长。 使用`ct::irange`能够向编译器提供迭代边界的结构化信息，便于编译器更好地优化生成代码。想要启用该优化，循环变量必须通过基于`ct::irange`的范围 for 语法绑定。
Python 中，Tile 代码支持内置`range()`、`for`、`while`以及嵌套循环。
步长参数**必须严格为正数**，不支持负步长区间。
下面的单 Block 核函数示例，用于对一维数组的全部 Tile 求和：
```cpp
__tile_global__ void tile_sum(float* __restrict__ arr, float* __restrict__ out, int num_tiles) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float, ct::shape<8>>;

    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto inView  = ct::partition_view{ct::tensor_span{arr, ct::extents{8 * num_tiles}},ct::shape{8_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{8_ic}},ct::shape{8_ic}};

    auto acc = ct::full<f32x8>(0.0f);
    // range-for over ct::irange gives the compiler structured iteration bounds.
    for (auto k : ct::irange(0, num_tiles)) {
        auto tile = inView.load(k);
        acc = acc + tile;                               // accumulate the k-th tile into acc
    }
    outView.store(acc, 0);                              // write the final result as the 0-th tile of out
}
```
## 7.2 条件
标准的`if/else`条件分支可以正常使用。由于每个 block 只会走**单条控制流路径**，因此传统 CUDA 中需要考虑的 warp 内分支发散问题，**不适用于 Tile 核函数**。
```cpp
__tile_global__ void conditional_load(float* __restrict__ arr, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float, ct::shape<8>>;
    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto inView  = ct::partition_view{ct::tensor_span{arr, ct::extents{N}}, ct::shape{8_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};
    int bx   = ct::bid().x;
    int nb_x = ct::num_blocks().x;
    auto tile = ct::full<f32x8>(0.0f);    // default for the last-block branch
    // Scalar condition -> one control-flow path per block; no divergence to reason about.
    if (bx < nb_x - 1) {
        tile = inView.load(bx);           // all blocks except the last
    }
    outView.store_masked(tile, bx);       // masked to handle a potentially partial final tile
}
```
```python
@ct.kernel
def conditional_load(arr, out, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    # Scalar condition -> one control-flow path per block; no divergence to reason about.
    if bid < ct.num_blocks(0) - 1:
        tile = ct.load(arr, index=(bid,), shape=(TILE,))    # all blocks except the last
    else:
        tile = ct.zeros((TILE,), dtype=ct.float32)          # last block: emit zeros
    ct.store(out, index=(bid,), tile=tile)
```
# 八、元素级算术与广播
## 8.1 broadcasting
广播机制遵循 NumPy 语义：标量会在整个 Tile 上复制；**单例维度（长度为 1）** 会被扩展，去匹配另一个操作数对应的维度；低秩操作数会把缺失的前置维度视作单例维度，对齐到高秩操作数的尾部维度。如果两个对应的维度都不是单例且大小不相等，那么该操作属于非法构造。

下面示例在一次加法运算中，同时演示单例维度扩展与秩提升： 一个形状为`8×2`的 2 阶 Tile，先做秩提升变为`1×8×2`，再和形状为`4×1×2`的 3 阶 Tile 广播，最终得到公共形状`4×8×2`。
```cpp
auto x = ct::iota<ct::tile<int, ct::shape<8, 2>>>();      // 8x2   (rank 2)
auto y = ct::iota<ct::tile<int, ct::shape<4, 1, 2>>>();   // 4x1x2 (rank 3)
auto z = x + y;    
```
```python
x = ct.full((8, 2),    3, dtype=ct.int32)   # 8x2   (rank 2)
y = ct.full((4, 1, 2), 5, dtype=ct.int32)   # 4x1x2 (rank 3)
z = x + y                                    # x promoted to 1x8x2, then broadcasts to 4x8x2
```
## 8.2 Arithmetic Operators
所有受支持的算术运算符都会对 Tile 按元素执行运算，并生成具有广播后形状的新 Tile。标量与 Tile 运算时，该标量会广播至 Tile 的每一个元素。当操作数类型不同时，会优先选用信息保留能力更强的类型：
- **Tile 与 Tile 运算**：结果 Tile 的类型为精度或表示范围更大的类型。例如：
    - `int + float` 结果为 `float`
    - `int16 + int32` 结果为 `int32`
- **标量与 Tile 运算**：若该标量的类型可以在 Tile 的元素类型中被精确表示（比如整型字面量`2`与`int`类型 Tile 运算，或是`2.0f`与`float`类型 Tile 运算），运算将直接以 Tile 的元素类型执行。 如果标量需要截断缩小才能适配 Tile 的元素类型（例如字面量`2.5`与`int`类型 Tile 运算），Python 与 C++ 两种语言的处理逻辑存在差异：
    - Python：将结果提升为可容纳两者取值的类型
    - C++：判定该表达式格式非法，直接拒绝编译
下文代码片段展示了这种标量 - Tile 运算的差异化行为。
```cpp
using i32x8 = ct::tile<int, ct::shape<8>>;
i32x8 x = ct::full<i32x8>(3);
x + 2;       // OK - int literal matches int tile element type
x + 2.5;     // ill-formed - 2.5 would narrow to int
```
```cpp
x = ct.full((8,), 3, dtype=ct.int32)
x + 2          # int32 - int literal matches int32 tile dtype
x + 2.5        # float32 - result promoted to hold both
```
在实际开发中，只要条件允许，应当使用 Tile 元素对应的类型来书写标量字面量，并进行显式类型转换。当操作数精度不同时，上述规则在内核函数内同样生效。
```cpp
__tile_global__ void elementwise(float* __restrict__ a, float* __restrict__ b, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto aView = ct::partition_view{ct::tensor_span{a,   ct::extents{N}}, ct::shape{8_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b,   ct::extents{N}}, ct::shape{8_ic}};
    auto cView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};

    int  bx = ct::bid().x;
    auto x  = aView.load(bx);
    auto y  = bView.load(bx);
    // 2.0f matches the float tiles' element type, so no narrowing conversion is required.
    // The scalar is broadcast across every element; + then runs elementwise.
    auto z  = 2.0f * x + y;
    cView.store(z, bx);
}
```
```python
@ct.kernel
def elementwise(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    x = ct.load(a, index=(bid,), shape=(TILE,))
    y = ct.load(b, index=(bid,), shape=(TILE,))
    # 2.0 is a loosely typed float constant; with float tiles, the result stays float.
    # Scalars are broadcast across every element of the tile, then + runs elementwise.
    z = 2.0 * x + y
    ct.store(c, index=(bid,), tile=z)
```
当你需要**显式控制舍入模式**或**次正规数处理逻辑**时，CUDA Tile API 提供了可暴露这些选项的数学函数，既支持逐元素算术运算（例如 `ct.add`、`ct::add`），也支持超越函数（例如 `exp` 和 `tanh`）。
# 九、Tile原语
工厂函数（用于创建 Tile）、加载与存储（Tile 空间读写）以及逐元素算术运算（逐元素运算与广播机制）都属于**Tile 原语**，即该编程语言内置的基础操作。开发者以 Tile 粒度编写这些操作，编译器会将其映射到硬件执行，支持时会调用张量核心。本节介绍 CUDA Tile 中其他可用的原语。
## 9.1 矩阵乘法
两个 Tile 之间的矩阵乘法，是实现两个数组矩阵乘的基础操作。CUDA Tile 提供两种 Tile 矩阵乘形式：纯矩阵乘法（`matmul`，写法 `a @ b`），以及**乘累加运算（mma）**，写法 `a @ b + acc`。 在 MMA 运算中，累加器会将部分乘积从一个 K 维度 Tile 传递到下一个 K 维度 Tile，这对分块矩阵乘法的内层循环十分有用。`matmul`与`mma`均支持二维矩阵乘法、三维批量矩阵乘，同时支持操作数与累加器使用不同数据类型（精度）。

下面内核示例采用一种常见范式：**无论输入精度如何，都使用 FP32 做累加，在存储阶段再转换为输出元素类型**。 Python 中写法：`ct.mma(a, b, acc)`，其中`acc`为 FP32 类型； C++ 中写法：`ct::mma(a, b, acc)`，需要显式指定 FP32 累加器类型。

K 循环迭代次数为`ceil(K/tk)`，以此覆盖矩阵 A 的右边界与矩阵 B 的下边界；加载时对不完整的 K 维度 Tile 做零填充（Python 使用`PaddingMode.ZERO`，C++ 使用`.load_masked()`）；C 矩阵侧不完整的 M/N 边界 Tile，通过存储侧的越界丢弃机制处理（Python 使用`ct.store`，C++ 使用`.store_masked()`）。
```cpp
__tile_global__ void gemm(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C,std::size_t M, std::size_t K, std::size_t N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32_acc = ct::tile<float, ct::shape<32, 32>>;
    A = ct::assume_aligned(A, 16_ic);
    B = ct::assume_aligned(B, 16_ic);
    C = ct::assume_aligned(C, 16_ic);
    constexpr auto tm = 32_ic;
    constexpr auto tn = 32_ic;
    constexpr auto tk = 16_ic;
    auto aView = ct::partition_view{ct::tensor_span{A, ct::extents{M, K}}, ct::shape{tm, tk}};
    auto bView = ct::partition_view{ct::tensor_span{B, ct::extents{K, N}}, ct::shape{tk, tn}};
    auto cView = ct::partition_view{ct::tensor_span{C, ct::extents{M, N}}, ct::shape{tm, tn}};
    auto [bx, by, bz] = ct::bid();
    auto acc = ct::full<f32_acc>(0.0f);                 // FP32 accumulator
    std::size_t num_k = (K + tk - 1) / tk;
    for (auto k : ct::irange(std::size_t{0}, num_k)) {
        acc = ct::mma(aView.load_masked(bx, k),  // zero-pad partial K-tile
                      bView.load_masked(k, by),
                      acc);                             // acc += a @ b
    }
    cView.store_masked(acc, bx, by);              // drop OOB edge lanes
}
```
图解：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260918111426.png)
## 9.2 规约与扫描
归约（Reduction）用于将一个 Tile 压缩为标量，或是一行标量。计算 Softmax 的分母、层归一化的均值与方差、注意力打分中的最大值，这些场景都会用到归约操作。

有一点需要提前理解：**归约结果的形状**。Python 版本默认会删掉被归约的维度（传入`keepdims=True`可以保留长度为 1 的维度）；C++ 版本则始终保留该维度，维持 Tile 的维度数不变。下面两段代码都是沿着第 1 轴对 2×4 的 Tile 做归约，二者最直观的差异就体现在输出形状上。
```cpp
using namespace ct::literals;
using i32x2x4 = ct::tile<int, ct::shape<2, 4>>;
// [[0,1,2,3],[4,5,6,7]]
auto x = ct::iota<i32x2x4>();                         
auto row_sums = ct::sum(x, 1_ic);                     // shape (2, 1) - axis kept
// row_sums == [[6], [22]]
```
```python
x   = ct.arange(8, dtype=ct.int32).reshape((2, 4))    # [[0,1,2,3],[4,5,6,7]]
s   = ct.sum(x, axis=1)                               # shape (2,)    - axis dropped
s_k = ct.sum(x, axis=1, keepdims=True)                # shape (2, 1)  - axis kept
# s == [6, 22];  s_k == [[6], [22]]
```
扫描（Scan）是归约的 “累积版本”，沿着某条轴生成累积结果。例如前缀和（`cumsum`）输出张量与输入维度完全相同，指定轴上每个位置的值，等于该位置之前（含自身）所有元素的累加和。完整 API 列表请查阅各语言的参考文档.
## 9.3 转置与全排序
有两个相关的原语可以**重排 Tile 的维度顺序，而不改动其内部数据**： `transpose`用于交换前两个维度；`permute`支持任意维度重排。 当需要在运算之间修改 Tile 的逻辑布局时就会用到它们，例如：生成矩阵乘法操作数的转置、注意力块内行列互换，或是在广播运算前对齐各维度。

在 Python 中： 对 2 维 Tile 调用`ct.transpose(x)`会直接交换它的两个维度；更高维的 Tile 需要显式指定`axis0`/`axis1`参数。 `ct.permute(x, axes)`接收一组维度下标构成的元组。
```python
tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
ty = ct.transpose(tx)                                            # shape (4, 2)

tz = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
tw = ct.permute(tz, (2, 0, 1))                                   # axes (0,1,2) -> (2,0,1)
```
在 C++ 中： `ct::transpose(x)`交换前两个维度，其余靠后的维度保持不变； `ct::permute(x, map)`接收一个`ct::dimension_map`类型对象，用来描述新的维度顺序。
```cpp
using namespace ct::literals;
using t2d = ct::tile<int, ct::shape<2, 4>>;
using t3d = ct::tile<int, ct::shape<2, 2, 2>>;

auto tx = ct::iota<t2d>();
auto ty = ct::transpose(tx);                             // shape (4, 2)

auto tz = ct::iota<t3d>();
// axes (0,1,2) -> (2,0,1)
auto tw = ct::permute(tz, ct::dimension_map{2_ic, 0_ic, 1_ic});  
```
## 9.4 元素级选择
按元素选择是 Tile 形式的条件分支操作：给定一个布尔类型 Tile 与两个操作数 Tile，输出的每个元素会根据对应位置的布尔值，从两个操作数之一选取。条件张量会广播至操作数的形状；两个操作数的类型必须兼容 Python 中该操作为 `ct.where(cond, x, y)`；C++ 中为 `ct::select(cond, lhs, rhs)`。
```cpp
auto cond = ct::iota<ct::tile<int, ct::shape<4>>>() < 2;   // {T, T, F, F}
auto t    = ct::full<ct::tile<float, ct::shape<4>>>( 1.0f);
auto f    = ct::full<ct::tile<float, ct::shape<4>>>(-1.0f);
auto r    = ct::select(cond, t, f);                        // {1, 1, -1, -1}
```
```python
cond    = ct.arange(4, dtype=ct.int32) < 2                 # [T, T, F, F]
x_true  = ct.full((4,),  1.0, dtype=ct.float32)
x_false = ct.full((4,), -1.0, dtype=ct.float32)
result  = ct.where(cond, x_true, x_false)                  # [1, 1, -1, -1]
```
## 9.5 算术函数
Tile 代码中可在`ct`命名空间下调用常用按元素数学运算函数：
- add（加法）、sub（减法）、mul（乘法）
- truediv（真除法）、floordiv（向下取整除法）、cdiv（复数除法）
- mod（取模）
- pow（幂运算）
- exp（自然指数）、exp2（以 2 为底指数）、log（自然对数）、log2（以 2 为底对数）
- sqrt（平方根）、rsqrt（平方根倒数）
- sin、cos、tan（正弦、余弦、正切）
- sinh、cosh、tanh（双曲正弦、双曲余弦、双曲正切）
- minimum（取较小值）、maximum（取较大值）
- negative（取负数）
- floor（向下取整）、ceil（向上取整）

每个函数对输入 Tile 执行按元素运算，并返回**相同形状**的 Tile。 这些运算也可以在 Tile 代码中作用于标量。
如需详细说明以及完整的支持按元素运算列表，请查阅 API 参考文档：
- [cuTile Python 数学运算](https://docs.nvidia.com/cuda/cutile-python/operations.html#math)
- [CUDA Tile C++ 数学运算](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/math_operations.html)
# 十、原子级内存操作
Tile 代码中有两种场景需要使用内存原子操作：
- **块间竞争**：每个线程块计算出部分结果，使用原子操作将该部分结果与其他线程块的部分结果合并，写入全局内存的同一位置。
- **块内竞争**：同一个 Tile 内的多个元素写入内存的同一个地址。
对 Tile 执行原子操作时，会为 Tile 中的**每一个元素各执行一次原子更新**。单个元素上的操作具备原子性，但**整个原子调用整体不具备原子性**；各个逐元素原子操作的执行顺序无明确规定。

在 Python 中，原子操作通过数组下标寻址目标，寻址规则与`ct.gather`、`ct.scatter`保持一致。可选参数用于控制边界检查、内存序与线程作用域。默认配置（开启边界检查、ACQ_REL、设备作用域）下，普通调用仅需传入数组、索引、待更新值。 `TiledView`还提供归约风格的原子成员方法，方法名以`atomic_store_`作为前缀（例如 `TiledView.atomic_store_add(index, update)`）；这类接口基于 Tile 空间索引寻址目标，**无返回值**，底层会编译为 PTX 中的原子归约。如果不需要读取内存旧值，推荐使用`TiledView`版本，性能更优。

在 C++ 中，原子操作接收指针与对应数值：单个内存地址使用原生指针 + 标量；也支持指针 Tile + 数值 Tile。内存序在调用处以编译期类型标签指定，例如`ct::memory_order_relaxed_t{}`。线程作用域同样使用同形式类型标签；若省略，则默认为系统全局可见。

`ct::partition_view`与`ct::strided_view`额外提供归约原子操作（`atomic_add`、`atomic_sub`、`atomic_and`、`atomic_or`、`atomic_xor`、`atomic_max`、`atomic_min`）作为成员函数。 这类接口基于 Tile 空间索引寻址，而非裸指针；参数顺序和`.store()`一致：**先传数值 Tile，最后传索引**。 独立全局函数会返回内存中修改前的旧值，而成员函数**无返回值**。和 Python 一样：不需要旧值时优先选用成员接口。成员接口底层会映射为硬件归约；而指针 Tile 形式会对每个元素单独发起一次原子操作。`ct::gather_scatter_view`不提供该类原子成员方法。
## 10.1 块间冲突
下方代码示例中会出现块间竞争：多个不同线程块向同一个内存位置`out`写入数据。如果不使用原子操作，并行执行的各个线程块会导致计算结果错误。 本例使用**设备线程作用域**（C++ 为`ct::thread_scope_device_t{}`；Python 默认就是设备全局作用域），因为该内存操作的结果必须对 GPU 上所有正在运行的线程块可见。 Python 核函数选用`TiledView.atomic_store_add`，原因是每个块的部分和只需要累加进`out[0]`，不需要保留旧值。
```cpp
__tile_global__ void block_sum(int* __restrict__ arr, int* __restrict__ out, std::size_t N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    constexpr auto TILE = 16_ic;
    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto aView = ct::partition_view{ct::tensor_span{arr, ct::extents{N}},
                                    ct::shape{TILE}};
    int bid = ct::bid().x;
    // partial final tile -> OOB lanes default to 0
    auto tile    = aView.load_masked(bid);      
    // reduce to a 1-element tile  
    auto partial = ct::sum(tile, 0_ic);           
	// accumulate the scalar into out[0]
    ct::atomic_add(out, (int)partial,             
                   ct::memory_order_relaxed_t{},  // single-location accumulator -> relaxed suffices
                   ct::thread_scope_device_t{});  // visible across the device
}
```
## 10.2 块内冲突
下面代码片段出现**块内竞争**：一个 Tile 内的所有数值，以原子方式累加至内存的同一个位置。

本例中，`ptrs`这个 Tile 里的每一个元素都指向同一个内存地址`slot`。`ct::iota<i32x16>()`生成的 Tile 中的每个元素，都会原子地加到该内存地址保存的值上。 同一个 Tile 针对单一内存地址发起的多次原子操作，**执行顺序未做规定**。 这里使用块级线程作用域`ct::thread_scope_block_t{}`，表示该原子操作的结果**仅需要在当前这个线程块内部可见**。
```cpp
using i32x16 = ct::tile<int, ct::shape<16>>;

int* slot = /* pointer to the contended location */;

// 16 lanes all aim at the same address. Add is commutative, so the
// unspecified ordering doesn't affect this sum; block scope suffices
// since contention stays within one block.
auto ptrs = ct::full<ct::tile<int*, ct::shape<16>>>(slot);
ct::atomic_add(ptrs, ct::iota<i32x16>(),
            ct::memory_order_relaxed_t{},
            ct::thread_scope_block_t{});
```
## 10.3 支持的原子操作
Tile 代码支持多种内存原子操作，各类操作的区别在于：待写入的值与内存中原有值的合并方式不同：
- `atomic_and`：将传入值与内存中保存的值执行**逐元素原子按位与**运算
- `atomic_or`：将传入值与内存中保存的值执行**逐元素原子按位或**运算
- `atomic_xor`：将传入值与内存中保存的值执行**逐元素原子按位异或**运算
- `atomic_max`：将传入值与内存中保存的值逐元素比较，把较大值存入内存
- `atomic_min`：将传入值与内存中保存的值逐元素比较，把较小值存入内存
- `atomic_add`：将传入值加到内存原有值上，并将结果存入内存
- `atomic_sub`（仅 C++ 支持）：用内存原有值减去传入值，并将结果存入内存
- `atomic_xchg`：将传入值写入内存，**返回写入前内存里的旧值**
- `atomic_cas`（C++ 中名为`atomic_compare_exchange`）：将内存中的值与传入的期望值逐元素比较；若二者相等，则把内存值替换为目标值。
# 十一、优化提示
**优化提示（optimization hint）** 是附加在源码构造体（Tile 核函数、load/store 调用点等）上的元数据，用于指导编译器生成目标代码。 优化提示**不会改变程序语义**：无论是否添加该提示，核函数的编译与运行行为完全一致。因此开发者可以自由增删、调优提示，**不会影响程序正确性**。编译器也可以选择忽略任意优化提示。
优化提示具备两项通用特性：
- **提示作用于单独构造体**：提示仅作用于它所附着的特定核函数或特定调用表达式，不会影响周边其他代码。
- **提示可按硬件架构单独指定**：针对不同 GPU 架构，同一条提示可以设置不同取值；也可以设置单一值，作用于所有目标硬件。
C++ 与 Python 两种语言暴露优化提示的方式不同：
- C++：采用 C++ 属性，写在对应的声明或语句上。
- Python：在核函数装饰器、以及各个内存操作调用点上，通过关键字参数传入。
提示的种类集合、以及每一类提示实际控制的内容，在两种语言中保持一致，详细文档参见后续「提示类型（Hint Kinds）」小节。
## 11.1 C++ the cutile::hint Attribute
在 C++ 中，优化提示通过 C++ 属性 `cutile::hint` 来表达：
```cpp
[[ cutile::hint(arch, kind1=value1, kind2=value2, ...) ]]
```
第一个参数是目标硬件架构，使用和宏 `__CUDA_ARCH__` 相同编码规则的整数（例如 `900` 代表 sm_90，`1000` 代表 sm_100）。特殊值 **0** 表示**架构无关提示**，适用于所有目标架构。其余每一个参数都是 `kind=value` 键值对，用于指定提示类型及其取值。

`cutile::hint` 属性作用于紧跟在它后面的源码构造体：
- 对于 Tile 核函数：将该属性写在函数声明处。
- 对于内存操作（如`ct::load`、`ct::store`以及`ct::partition_view`对应的加载 / 存储操作）：将该属性写在包含此调用的表达式语句上。

其他放置位置存在使用限制；完整规则参见[《CUDA Tile C++ 优化提示规范》](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/optimization_hints.html#hint-specification)。

下面这个核函数展示了两种放置方式：**核函数级提示**（为 sm_90 与 sm_100 设置不同的`num_cta_in_cga`值），以及**表达式语句级提示**（标记某次特定加载为带宽密集型访问）。
```cpp
[[ cutile::hint(900,  num_cta_in_cga=4),    // sm_90:  prefer 4 CTAs per cluster
   cutile::hint(1000, num_cta_in_cga=8) ]]  // sm_100: prefer 8 CTAs per cluster
__tile_global__ void optimization_hints(float* __restrict__ in,
                                        float* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    in  = ct::assume_aligned(in,  16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto inSpan  = ct::tensor_span{in,  ct::extents{128_ic}};
    auto outSpan = ct::tensor_span{out, ct::extents{128_ic}};
    auto inView  = ct::partition_view{inSpan,  ct::shape{8_ic}};
    auto outView = ct::partition_view{outSpan, ct::shape{8_ic}};

    int bx = ct::bid().x;

    // Expression-statement hint: tag this particular load as bandwidth-heavy.
    ct::tile<float, ct::shape<8>> tile;
    [[ cutile::hint(0, latency=8) ]]
    tile = inView.load(bx);

    outView.store(tile, bx);
}
```
当多条同类型优化提示作用于同一个源码构造体时，**特定架构的提示优先级高于架构无关提示**。
## 11.2 Python:装饰器参数与调用点关键字参数
Python 提供两种方式设置优化提示：

- **核函数级提示**：作为`@ct.kernel(...)`装饰器的关键字参数。编译后的核对象还提供`.replace_hints(**hints)`方法，该方法会返回一个覆盖了新提示的全新核函数；新核拥有独立的 JIT 缓存，因此`replace_hints`非常适合作为自动调优循环的基础组件。
- **单次调用提示**：在内存操作的调用位置通过关键字参数指定，包括：`ct.load` / `ct.store`、`TiledView.load` / `TiledView.store`，以及`ct.gather` / `ct.scatter`。

如需按不同架构设置不同取值，用`cuda.tile.ByTarget(*, default=..., sm_XXX=..., sm_YYY=...)`封装数值。架构键必须采用`"sm_<主版本><次版本>"`格式的字符串（例如`"sm_100"`、`"sm_120"`）。不使用`ByTarget`的普通值会作用于所有目标架构，等价于 C++ 中`arch=0`的架构无关提示。

下方核函数是上面 C++ 示例的 Python 直接对等实现：`ByTarget`承载核函数级提示，`latency=8`关键字代表单次调用提示，而`replace_hints`可以在**不修改源码**的前提下生成重新调参后的核函数。
```python
@ct.kernel(num_ctas=ByTarget(sm_90=4, sm_100=8))
def optimization_hints(in_, out, TILE: ct.Constant[int]):
    bid = ct.bid(0)

    # Per-call hint: this particular load is bandwidth-heavy.
    tile = ct.load(in_, index=(bid,), shape=(TILE,), latency=8)

    ct.store(out, index=(bid,), tile=tile)


# Autotuning: produce a new kernel with overridden hints without editing the
# source. The new kernel has its own JIT cache.
tuned_kernel = optimization_hints.replace_hints(num_ctas=8)
```
## 11.3 Hint Kinds
### 11.3.1 每个CGA中CTA的数量
- CTA：Cooperative Thread Array，协作线程阵列，等价于 CUDA 的线程块 block
- CGA：Cooperative Group Array，协作组阵列（GPU 硬件集群调度单元）
- C++ 名称：`num_cta_in_cga`（核函数属性）
- Python 名称：`num_ctas`（`@ct.kernel`装饰器参数）
- 允许取值：1、2、4、8、16。在 sm_80 架构下，仅支持取值 1。
- 含义：启动核函数时，编译器倾向于在每个协作组阵列（CGA）内放置的协作线程阵列（CTA）数量。
### 11.3.2 占用率
- C++ 名称：`occupancy`（核函数属性）
- Python 名称：`occupancy`（`@ct.kernel`装饰器参数）
- 允许取值：闭区间`[1, 32]`内的任意整数
- 含义：每个流式多处理器（SM）上目标活跃 CTA 数量。编译器将该值视为一条建议，并会在代码生成阶段尽量遵循该设置。
### 11.3.3 内存访问延迟
- C++ 名称：`latency`（附加在包含调用的表达式语句上的属性）
- Python 名称：`latency`（调用点处的关键字参数）
- 适用范围：Tile 空间加载与存储（C++ 中`ct::partition_view`；Python 中`Array.tiled_view`以及`ct.load` / `ct.store`），同时适用于聚集 / 分散访存（C++ 中使用指针 Tile 的`ct::load` / `ct::store`；Python 中`ct.gather` / `ct.scatter`）。
- 允许取值：闭区间`[1, 10]`内任意整数。**1**代表 DRAM 访存压力小，**10**代表访存压力大。更大的取值通常会让编译器调度更深的预取深度。
### 11.3.4 允许使用TMA
- C++ 名称：`allow_tma`（附加在包含调用的表达式语句上的属性）
- Python 名称：`allow_tma`（调用点处的关键字参数）
- 适用范围：**仅 Tile 空间加载与存储**（C++ 中`ct::partition_view`；Python 中`Array.tiled_view`和`ct.load` / `ct.store`）。gather、scatter 操作不支持该提示。
- 允许取值：C++ 用`true`/`false`；Python 用`True`/`False`。默认允许启用 TMA；若将该提示设为`false`/`False`，则指示编译器：即使硬件支持 TMA，本次特定的加载 / 存储也**不要编译降级为 TMA 指令**。
# 十二、C++性能提示
本指南中所有 C++ 核函数都用到了少数相同的注解与惯用写法。本节将解释它们的作用以及重要意义。
## 12.1 对内存数组使用__restrict__指针
`__restrict__` 关键字向编译器做出承诺：在该指针的生命周期内，**只能通过这一个指针**访问它所指向的内存区域。参见 5.4.1.4 节。

在 Tile C++ 中，如果内存上的数组满足上述条件，**必须**为其指针标注 `__restrict__` 关键字，才能获得良好的内存操作性能。

我们可以通过例子理解背后的原因：考虑一个逐元素拷贝场景，若数组指针**未添加 `__restrict__`**：
```cpp
__tile_global__ void tile_elementwise_copy(float* out, float const* in) {
    namespace ct = cuda::tiles;

    using f32x64 = ct::tile<float, ct::shape<64>>;
    using i32x64 = ct::tile<int, ct::shape<64>>;

    auto inPtrs  = in  + 64 * ct::bid().x + ct::iota<i32x64>();
    auto outPtrs = out + 64 * ct::bid().x + ct::iota<i32x64>();

    auto data = ct::load(inPtrs);   // (1)
    ct::store(outPtrs, data);       // (2)
}
```
在 CUDA Tile 程序中，开发者通常无需关心编译器对 Tile 操作的并行化实现细节。但本节会对其进行分析，以此理解：**使用无重叠数组为何能让编译器生成性能更优的代码**。

我们来看编译器如何并行化`load`与`store`这两个 Tile 操作。如果输入数组和输出数组不存在内存重叠，`load`操作可以被并行拆分为一组相互独立的内存读操作。同理，`store`操作可以并行拆分为多个内存写操作，每一个写操作仅依赖它所要写入数据元素对应的加载操作。

但倘若输入、输出数组**可能发生内存重叠**，编译器就必须保证：**当前整个 Tile 的所有内存加载操作全部完成之后，才能发起任意存储写操作**，以此保障程序语义正确。否则，写操作有可能提前执行，在元素被读取之前就覆盖它，造成程序运行结果错误。这会限制编译器交错调度读写操作的能力 —— 所有读操作必须全部结束，才能发起写操作。

简言之：当编译器无法保证数组内存互不重叠时，只能生成更保守的代码。这就是为什么使用无重叠数组，并通过在指针上标注`__restrict__`关键字把该信息告知编译器，有助于拿到最优性能。

如果某块内存区域还可以被其他指针访问，却给该指针加上`__restrict__`标注，会引发**未定义行为**。
## 11.2 将数组指针标记为 16 字节对齐
使用 `ct::assume_aligned` 将数组指针标记为 16 字节对齐。
```cpp
__tile_global__ void foo(float* __restrict__ in) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    in = ct::assume_aligned(in, 16_ic);

    ct::tensor_span t{in, ct::extents{256_ic, 256_ic}};
    ct::partition_view{t, ct::shape{4_ic, 4_ic}};

    // ...
}
```
该对齐保证是 `ct::partition_view` 能够使用张量内存加速器（TMA）的必要条件。采用该方式时，运行时传入的指针**必须满足 16 字节对齐**，否则会产生未定义行为。

`cudaMalloc` 这类 CUDA 内存分配器返回的指针，天然保证至少为 16 字节对齐。
## 11.3 内存访问优先选用 `ct::partition_view`
对于结构化内存访问，优先使用`ct::partition_view`，而不是 gather/scatter 形式的`ct::load`与`ct::store`。基于 view 的写法在支持的硬件上可以编译为张量内存加速器（TMA）指令，性能远优于逐元素 gather 访存。关于 gather/scatter 的相关内容参见《聚集与分散访存》章节。
## 11.4 有界循环使用 `ct::irange`
在固定范围迭代时，请使用`ct::irange`，而非普通 for 循环。这种结构化形式能够让编译器启用流水线、向量化等优化；若循环边界与步长是不透明整型表达式，则无法使用这类优化
```cpp
for (auto idx : ct::irange(lowerBound, upperBound, step)) {
    // ...
}
```
