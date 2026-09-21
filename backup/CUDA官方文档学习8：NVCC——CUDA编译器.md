原文链接：
[2.7.NVCC: The NVIDIA CUDA Compiler](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
# 前言
NVIDIA CUDA 编译器（nvcc）是 NVIDIA 提供的工具链，用于编译 CUDA C/C++ 代码以及 PTX 代码。该工具链属于 CUDA 工具包的一部分，包含多种工具：编译器、链接器，以及 PTX 与 Cubin 汇编器。顶层工具 nvcc 负责统筹编译流程，在编译的各个阶段调用对应的工具。

nvcc 负责 CUDA 代码的离线编译；与之相对，CUDA 运行时编译器 nvrtc 实现**即时编译（JIT，在线编译）**。

本章介绍开发应用程序所需的 nvcc 最常用用法与细节。nvcc 的完整说明请参阅 nvcc 官方文档。
# 一、CUDA 源文件与头文件
由 nvcc 编译的源文件可以同时包含**主机代码（host code）** 与**设备代码（device code）**：主机代码在 CPU 上执行，设备代码在 GPU 上执行。
nvcc 支持常见 C/C++ 源文件后缀：`.c`、`.cpp`、`.cc`、`.cxx`，用于仅包含主机代码的文件；`.cu`后缀用于包含设备代码，或是主机、设备混合代码的文件。
包含设备代码的头文件一般使用`.cuh`后缀，以此区分仅主机代码的头文件（`.h`、`.hpp`、`.hh`、`.hxx`等）。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260921153121.png)
# 二、NVCC编译工作流
在初始阶段，nvcc将设备代码与主机代码分离，并分别交由 GPU 编译器与主机编译器进行编译。

编译主机代码时，CUDA编译器nvcc要求系统中存在兼容的主机编译器。CUDA Toolkit规定了Linux与Windows平台下的主机编译器支持策略。仅含主机代码的文件，既可以用 nvcc 构建，也可以直接使用主机编译器构建。生成的目标文件可在链接阶段与 nvcc 输出的、包含 GPU 代码的目标文件合并。

GPU 编译器将 C/C++ 设备代码编译为 PTX 汇编代码。对于编译命令行中指定的每一种虚拟机指令集架构（例如 `compute_90`），GPU 编译器都会执行一次编译。

随后，独立的 PTX 代码会交给 `ptxas` 工具，由它针对目标硬件指令集（ISA）生成 Cubin。硬件指令集由其 SM 版本标识。

可以将多个 PTX 和 Cubin 目标文件嵌入到应用或库内的单个二进制 Fatbin 容器中，这样一份二进制程序就能够支持多种虚拟指令集与目标硬件指令集。

nvcc 会自动调用并协调上述所有工具。`-v` 参数可展示完整编译流程与工具调用过程；`-keep` 参数可将编译过程产生的中间文件保存至当前目录，或是由 `--keep-dir` 指定的目录。

下面的示例展示 CUDA 源文件 `example.cu` 的编译工作流：
```cpp
// ----- example.cu -----
#include <stdio.h>
__global__ void kernel() {
    printf("Hello from kernel\n");
}

void kernel_launcher() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main() {
    kernel_launcher();
    return 0;
}
```
`nvcc`基本的编译工作流如下：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260921153702.png)
支持多 PTX 与 Cubin 架构的 nvcc 编译工作流
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260921154439.png)
# 三、NVCC基本使用
使用nvcc编译一个CUDA源文件的基本命令是：
```bash
nvcc <source_file>.cu -o <output_file>
```
nvcc 支持常用的编译器选项，用于指定头文件搜索目录 `-I <path>`、库文件搜索路径 `-L <path>`、链接其他库 `-l<library>`，以及定义宏 `-D<macro>=<value>`。
```bash
nvcc example.cu -I path_to_include/ -L path_to_library/ -lcublas -o <output_file>
```
## 3.1 NVCC PTX与Cubin的生成
默认情况下，nvcc 会为 CUDA Toolkit 所支持的**最早一代 GPU 架构**（最低版本的`compute_XY`与`sm_XY`）生成 PTX 和 Cubin，以此最大化兼容性。
- `-arch`选项：可用于为**单个指定 GPU 架构**生成 PTX 与 Cubin。
- `-gencode`选项：可用于为**多个 GPU 架构**同时生成 PTX 与 Cubin。
支持的虚拟 GPU 架构与真实 GPU 架构完整列表，可分别通过`--list-gpu-code`和`--list-gpu-arch`参数查看
```bash
nvcc --list-gpu-code # list all supported real GPU architectures
nvcc --list-gpu-arch # list all supported virtual GPU architectures
```
```bash
# e.g. -arch=compute_80 for NVIDIA Ampere GPUs and later
# PTX-only, GPU forward compatible
nvcc example.cu -arch=compute_<XY> 
# e.g. -arch=sm_80 for NVIDIA Ampere GPUs and later
# PTX and Cubin, GPU forward compatible
nvcc example.cu -arch=sm_<XY>      
# automatically detects and generates Cubin for the current GPU
# no PTX, no GPU forward compatibility
nvcc example.cu -arch=native       
# generate Cubin for all supported GPU architectures
# also includes the latest PTX for GPU forward compatibility
nvcc example.cu -arch=all          
# generate Cubin for all major supported GPU architectures, e.g. sm_80, sm_90,also includes the latest PTX for GPU forward compatibility
nvcc example.cu -arch=all-major    
```
更高级的用法支持单独指定 PTX 与 Cubin 目标：
```bash
# generate PTX for virtual architecture compute_80 and compile it to Cubin for real architecture sm_86, keep compute_80 PTX
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,compute_80 # (PTX and Cubin)

# generate PTX for virtual architecture compute_80 and compile it to Cubin for real architecture sm_86, sm_89
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,sm_89    # (no PTX)
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 # same as above

# (1) generate PTX for virtual architecture compute_80 and compile it to Cubin for real architecture sm_86, sm_89
# (2) generate PTX for virtual architecture compute_90 and compile it to Cubin for real architecture sm_90
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 -gencode=arch=compute_90,code=sm_90
```
## 3.2 主机代码编译说明
编译单元（即源文件及其头文件）如果**不含设备代码或设备符号**，可以直接使用主机编译器进行编译。 如果任意编译单元调用了 CUDA 运行时 API 函数，则应用程序必须链接 CUDA 运行时库。CUDA 运行时同时提供静态库与动态库，分别为`libcudart_static`和`libcudart`。 默认情况下，nvcc 会链接静态版本的 CUDA 运行时库。若要使用 CUDA 运行时的动态库版本，需要在编译或链接命令中给 nvcc 传入`--cudart=shared`参数。

nvcc 可通过`-ccbin <compiler>`参数指定用于编译主机函数的主机编译器。 也可以定义环境变量`NVCC_CCBIN`，以此指定 nvcc 所使用的主机编译器。 nvcc 的`-Xcompiler`参数可以将参数透传给主机编译器。例如，下面示例中，`-O3`参数会由 nvcc 传递给主机编译器。
```bash
nvcc example.cu -ccbin=clang++

export NVCC_CCBIN='gcc'
nvcc example.cu -Xcompiler=-O3
```
## 3.3 GPU代码的分离编译
nvcc 默认采用**全程序编译（whole-program compilation）**，该模式要求所有 GPU 代码与符号都必须位于使用它们的同一个编译单元内。CUDA 设备函数可以调用其他编译单元中定义的设备函数，或访问其他编译单元里的设备变量；但必须在 nvcc 命令行指定`-rdc=true`（或别名`-dc`），才能启用跨编译单元的设备代码链接。这种支持链接不同编译单元内设备代码与符号的能力，就称为**分离编译（separate compilation）**。

分离编译可以实现更灵活的代码组织，能够缩短编译耗时，还可生成体积更小的二进制程序。但相比全程序编译，分离编译会增加构建阶段的复杂度。设备代码链接可能会对性能产生影响，因此默认不开启。**链接时优化（LTO，Link-Time Optimization）** 有助于降低分离编译带来的性能开销。

启用分离编译需要满足以下条件：
- 在某个编译单元中定义的**非 const 设备变量**，在其他编译单元中引用时必须使用`extern`关键字。
- 所有**const 设备变量**，其定义与引用都必须使用`extern`关键字。
- 所有 CUDA 源文件（`.cu`）编译时都必须带上`-dc`或`-rdc=true`参数。

主机函数与设备函数默认拥有外部链接属性，**不需要 extern 关键字**。注意：从 CUDA 13 开始，`__global__`函数以及`__managed__`/`__device__`/`__constant__`变量默认为内部链接属性。

下面示例中，`definition.cu`定义了一个变量和一个函数，`example.cu`引用它们。两个文件分开编译，再链接合并为最终二进制程序。
```cpp
// ----- definition.cu -----
extern __device__ int device_variable = 5;
__device__        int device_function() { return 10; }
```
```cpp
// ----- example.cu -----
extern __device__ int  device_variable;
__device__        int device_function();

__global__ void kernel(int* ptr) {
    device_variable = 0;
    *ptr            = device_function();
}
```
```bash
nvcc -dc definition.cu -o definition.o
nvcc -dc example.cu    -o example.o
nvcc definition.o example.o -o program
```
# 四、常用编译器选项
本节介绍 nvcc 最常用的编译器选项，涵盖语言特性、优化、调试、性能剖析以及构建相关方面。全部选项的完整说明可查阅 nvcc 官方文档。
## 4.1 语言特性
nvcc 支持 C++ 核心语言特性，从 C++03 到 C++23。可使用`-std`选项指定要启用的 C++ 语言标准：
- `--std={c++03|c++11|c++14|c++17|c++20|c++23}`
除此之外，nvcc 还支持以下语言扩展：
- `-restrict`：声明所有核函数指针参数均为 restrict 指针。
- `-extended-lambda`：允许在 lambda 表达式声明中使用`__host__`、`__device__`修饰符。
- `-expt-relaxed-constexpr`：（实验性选项）允许主机代码调用`__device__ constexpr`函数，同时允许设备代码调用`__host__ constexpr`函数。
关于这些特性的更多细节，可参见扩展 Lambda 与 constexpr 相关章节。
## 4.2 调试选项
nvcc 支持以下选项用于生成调试信息：
- `-g`：为主机代码生成调试信息。gdb/lldb 等工具依靠该信息对主机代码调试。
- `-G`：为设备代码生成调试信息。cuda-gdb 依靠该信息调试设备端代码；该选项同时定义宏`__CUDACC_DEBUG__`。
- `-lineinfo`：为设备代码生成行号信息。该选项**不影响运行性能**，常配合 compute-sanitizer 工具追踪核函数执行。
nvcc 默认对 GPU 代码使用最高优化等级`-O3`。调试选项`-G`会禁用一部分编译器优化，因此调试版本性能会低于发布版本。可定义`-DNDEBUG`关闭运行时断言（断言同样会拖慢执行速度）。
## 4.3 优化选项
nvcc 提供大量性能优化选项。本节简要介绍开发者常用的部分选项，附带相关参考链接；完整内容请查阅 nvcc 文档。
- `-Xptxas`：将参数传递给 PTX 汇编器 ptxs。nvcc 文档列出了 ptxs 的常用参数。例如`-Xptxas=-maxrregcount=N`用于指定每个线程最多可使用的寄存器数量。
- `-extra-device-vectorization`：开启更激进的设备代码向量化优化。
- `--apply-controls=/path/to/file`：向 nvcc 与 ptxas 传入高级控制文件（ACF）。该文件会修改默认编译行为，使编译更适配特定负载。使用高级控制文件可能引发编译失败或运行结果错误，请自行承担风险。可访问 CompileIQ GitHub 主页了解高级控制文件的生成方法。
其余用于精细控制浮点数行为的选项
下面这些选项用于输出编译器报告，对高级代码优化很有用：
- `-res-usage`：编译结束后打印资源使用报告，包含每个核函数分配的寄存器、共享内存、常量内存与本地内存大小。
- `-opt-info=inline`：打印函数内联相关信息。
- `-Xptxas=-warn-lmem-usage`：使用本地内存时输出警告。
- `-Xptxas=-warn-spills`：寄存器溢出到本地内存时输出警告。
## 4.4 链接时优化（LTO）
分离编译由于跨文件优化的能力受限，性能可能低于全程序编译。**链接时优化（LTO）** 可以解决该问题：它在链接阶段，对多个独立编译得到的文件执行跨文件优化；代价是编译耗时增加。LTO 能够在保留分离编译灵活性的同时，恢复大部分全程序编译的性能。

nvcc 需要使用 **`-dlto`** 选项，或是 `lto_<SM版本>` 这类链接时优化目标，来开启 LTO：
```bash
nvcc -dc -dlto -arch=sm_100 definition.cu -o definition.o
nvcc -dc -dlto -arch=sm_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program
```
```bash
nvcc -dc -arch=lto_100 definition.cu -o definition.o
nvcc -dc -arch=lto_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program
```
## 4.5 性能分析选项
可直接使用 Nsight Compute 和 Nsight Systems 工具对 CUDA 程序做性能分析，**编译阶段无需额外传参数**。不过，nvcc 可以生成附加信息，将源码与生成的机器代码关联起来，辅助性能分析：
- `-lineinfo`：为设备代码生成行号信息；可在性能分析工具里查看源代码。性能分析工具要求**原始源码必须放在编译时的相同路径下**。
- `-src-in-ptx`：将原始源代码保留在 PTX 中，规避上面 `-lineinfo` 的路径限制。**该选项必须搭配 `-lineinfo` 使用**。
## 4.6 Fatbin压缩
nvcc 默认会对程序或库二进制文件内存储的 fatbin 进行压缩。可通过下面选项控制 fatbin 压缩行为：
- `-no-compress`：关闭 fatbin 压缩。
- `--compress-mode={default|size|speed|balance|none}`：设置压缩模式。
    - `speed`：优先保证快速解压；
    - `size`：目标是减小 fatbin 文件体积；
    - `balance`：在解压速度与文件大小之间做权衡；
    - 默认模式为 `speed`；
    - `none`：关闭压缩。
## 4.7 编译器性能控制选项
nvcc 提供了若干选项，用于分析并**加速编译过程本身**：
- `-t <N>`：针对单个编译单元，为多 GPU 架构并行编译所使用的 CPU 线程数。
- `-split-compile <N>`：优化阶段并行处理所用的 CPU 线程数。
- `-split-compile-extended <N>`：更激进的拆分编译模式，**需要开启链接时优化（LTO）**。
- `-Ofc <N>`：设备代码编译速度等级。
- `-time <filename>`：生成 CSV（逗号分隔值）表格，记录每个编译阶段耗时。
- `-fdevice-time-trace`：生成设备代码编译的时间追踪日志。