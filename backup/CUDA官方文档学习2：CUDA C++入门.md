原文章链接：https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
本章通过展示如何在C++中体现这些概念，介绍了CUDA编程模型的一些基本概念。本编程指南重点介绍CUDA运行时API。CUDA运行时API是在C++中使用CUDA最常用的方式，它构建在底层的CUDA驱动程序API之上。
# 1.核函数
在 GPU 上执行且可以从主机端（Host）调用的函数被称为 **核函数（Kernels）**。核函数的编写逻辑旨在由许多并行线程同时执行。
## 1.1 指定核函数
核函数的代码是通过 `__global__` 声明限定符来指定的。这会告知编译器，该函数将以一种允许通过“核函数启动（Kernel Launch）”来调用的方式被编译为 GPU 代码。核函数启动是指启动核函数运行的操作，通常由 CPU 发起。核函数是返回类型为 `void` 的函数。
```cpp
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```
## 1.2 启动核函数
执行核函数的并行线程数量是在核函数启动时指定的，这被称为**执行配置（Execution Configuration）**。同一个核函数的不同调用可以使用不同的执行配置，例如使用不同数量的线程或线程块。从 CPU 代码启动核函数有两种方式：**三括号记法（Triple Chevron Notation）** 和 `cudaLaunchKernelEx`。这里介绍最常用的启动方式——三括号记法。

三括号记法： `<<<gridDim, blockDim>>>` 指定**执行配置**。其中 `gridDim` 定义网格中线程块的数量，`blockDim` 定义每个线程块中的线程数。
```cpp
__global__ void vecAdd(float* A, float* B, float* C)
 {

 }

int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```
内核会被配置在 GPU 上执行，但主机代码在继续执行之前不会等待内核在 GPU 上完成（甚至开始）执行。必须通过某种形式的 GPU 与 CPU 同步机制来确定内核是否已完成执行。

对于多维结构，需使用 `dim3` 类型进行定义。
```cpp
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```
在核函数内部，CUDA 提供了四个核心内建变量（Intrinsics），用于定位线程在并行结构中的位置。每个变量都是包含 `.x`, `.y`, `.z` 三个分量的向量：
- **`threadIdx`**：当前线程在所属线程块（Block）内的索引。
- **`blockIdx`**：当前线程块在整个网格（Grid）中的索引。
- **`blockDim`**：执行配置中指定的线程块维度（即每个块包含多少线程）。
- **`gridDim`**：执行配置中指定的网格维度（即整个网格包含多少块）。
```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x
   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}
int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```
上述示例假设向量的长度是线程块大小（在本例中为 256 个线程）的整数倍。为了使核函数能够处理任意长度的向量，我们可以添加检查机制，以确保内存访问不会超出数组的边界（如下所示），然后启动一个包含部分非活动线程（Inactive Threads）的线程块。
```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```
在线程块中启动不工作的额外线程并不会产生巨大的开销成本，但应当尽量避免启动那些“没有任何线程在工作”的线程块。

所需的线程块数量可以通过“所需线程数（本例中为向量长度）除以每个块的线程数”并**向上取整（Ceiling）** 来计算。
```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```
**CUDA 核心计算库 (CCCL)** 提供了一个便捷的实用程序 `cuda::ceil_div`，用于执行这种向上取整除法，以计算核函数启动所需的线程块数量。该实用程序可以通过包含头文件 `<cuda/cmath>` 来使用。注意：经过查询cuda c++ core libraries文档，ceil_div 需要cuda12.8以上版本。
```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```
# 2.内存访问
为了使用上面所示的 vecAdd 内核，数组 A、B 和 C 必须位于 GPU 可访问的内存中。实现这一点有几种不同的方法，这里将介绍其中两种
## 2.1 统一内存
**统一内存（Unified Memory）** 是 CUDA 运行时的一项核心特性，它建立了一个在 CPU（主机）和 GPU（设备）之间共享的单一逻辑地址空间。通过使用 `cudaMallocManaged` API 或 `__managed__` 限定符，NVIDIA 驱动程序会自动监控数据访问需求，并在硬件底层透明地按需迁移数据。
```cpp
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}
```
## 2.2 显式内存管理
通过手动控制数据迁移实现更高的性能。其核心流程包括：使用 `cudaMalloc` 在 GPU 上开辟空间，并通过同步的 `cudaMemcpy` API 在主机与设备间传输数据（其中 `cudaMemcpyDefault` 可根据指针地址自动判定传输方向）。为了优化传输效率，最佳实践是使用 `cudaMallocHost` 分配**页锁定内存（Page-locked/Pinned Memory）**，这种内存可以防止操作系统将其交换至磁盘，从而显著提升带宽表现，并为后续的异步流传输奠定基础；但需注意，过度锁定主机内存可能会导致系统整体性能下降，应仅针对需频繁与 GPU 交换数据的缓冲区使用。
```cpp
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```
# 3.完整实例
以“统一内存”为例：
```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
//unified-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);		
    return 0;
}
```
核函数的启动相对于调用它们的 CPU 线程是异步的。这意味着 CPU 线程的控制流将在核函数完成执行之前、甚至可能在核函数正式启动之前就继续向下执行。为了确保在执行后续主机端代码前核函数已完成运行，必须使用某种同步机制。同步 GPU 与主机线程最简单的方法是使用 cudaDeviceSynchronize，它会阻塞主机线程，直到此前发布到 GPU 上的所有工作全部完成。在上述示例中，这种方法是足够的，因为 GPU 上每次只执行单个操作。但在更大规模的应用中，可能会有多个 **流（Streams）** 在 GPU 上并行执行任务，此时 cudaDeviceSynchronize 会等待所有流中的工作全部完成，此时建议使用“流同步（Stream Synchronization）”API 来仅与特定流进行同步
# 4.设备函数与主机函数
`__global__` 限定符用于指明核函数（Kernel）的入口点。也就是说，该函数将被调用并在 GPU 上进行并行执行。核函数通常从主机端（Host）启动，但也可以利用**动态并行（Dynamic Parallelism）** 机制从一个核函数内部启动另一个核函数。

`__device__` 限定符表示该函数应被编译为 GPU 代码，并能被其他 `__device__` 或 `__global__` 函数调用。一个函数（包括类成员函数、仿函数和 Lambda 表达式）可以同时指定为 `__device__` 和 `__host__`

当一个函数被指定为 `__host__ __device__` 时，编译器会被指示为该函数同时生成 GPU 和 CPU 两套代码。在这种函数中，可能需要使用预处理器（宏）来分别为函数的 GPU 副本或 CPU 副本指定特定的代码。
# 5.静态变量声明
CUDA 限定符（Specifiers）可用于静态变量声明，以控制变量的物理存储位置，具体见下一节对GPU设备内存空间的介绍
- **`__device__`**：指定变量存储在**全局内存（Global Memory）** 中。
- **`__constant__`**：指定变量存储在**常量内存（Constant Memory）** 中。
- **`__managed__`**：指定变量存储为**统一内存（Unified Memory）**。
- **`__shared__`**：指定变量存储在**共享内存（Shared Memory）** 中。

当在 `__device__` 或 `__global__` 函数内部声明一个没有任何限定符的变量时，如果可能，它会被分配到**寄存器（Registers）** 中；而在必要时（如寄存器溢出），则会被分配到**局部内存（Local Memory）** 中。任何在 `__device__` 或 `__global__` 函数外部声明的、且没有任何限定符的变量，都将被分配在**系统内存（System Memory，即主机端内存）** 中
# 6.线程块集群
CUDA 在线程块（Thread Block）与网格（Grid）之间引入了可选的“集群”层级。类似于线程块内的线程被保证调度到同一个流式多处理器（SM），集群内的所有线程块被保证共同调度到同一个 GPU 处理集群（GPC）中。这种硬件级的协同调度确保了块与块之间能够进行更紧密的通信与同步。

集群的核心优势在于**分布式共享内存（Distributed Shared Memory）**。集群内的线程块不仅可以访问自身的共享内存，还可以对集群内其他线程块的共享内存进行读、写和原子操作。这打破了传统模型中“共享内存仅限块内私有”的限制，极大提升了数据交换效率。

通过 Cooperative Groups API，集群内的线程块可以执行硬件支持的同步操作（如 `cluster.sync()`）。此外，API 还提供了 `num_threads()` 和 `num_blocks()` 等函数来查询集群规模，以及 `dim_threads()` 和 `dim_blocks()` 来获取线程或块在集群内的秩（Rank/坐标）

集群可以通过两种方式启用：
- **编译时固定**：使用 `__cluster_dims__(X, Y, Z)` 属性，随后通过传统的三括号 `<<< >>>` 启动，大小不可更改。
- **运行时动态**：使用 `cudaLaunchKernelEx` API 进行配置。 需要注意的是，为了保持兼容性，即使启用了集群，`gridDim` 变量在内核中依然表示**线程块的总数**，而非集群的总数。
```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}
int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```