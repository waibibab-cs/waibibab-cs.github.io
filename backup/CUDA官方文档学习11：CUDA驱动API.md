本指南前面章节已经介绍过CUDA运行时。正如《CUDA运行时API与CUDA驱动API》章节所述，CUDA运行时构建于底层CUDA驱动API之上。本节介绍CUDA运行时API与驱动API之间的部分差异，以及二者混合使用的方法。大多数应用程序完全可以只使用CUDA运行时，并且发挥全部性能，无需调用CUDA驱动API。不过，部分新接口会率先在驱动API中推出，之后才会加入运行时API；还有一些高级接口，例如虚拟内存管理（Virtual Memory Management），仅在驱动API中对外暴露。
驱动API实现在CUDA动态库（Windows平台为cuda.dll，Linux平台为cuda.so）中，安装设备驱动时该库会被复制到系统内。驱动API所有入口函数均以cu作为前缀。
驱动API是基于句柄的命令式API：绝大多数对象通过不透明句柄引用，可将句柄传入函数以操作对应的对象。
> 句柄是什么？
> 在CUDA驱动API中，句柄是一种不透明的凭证标识，由CUDA驱动内部维护，用来代表设备、上下文、流、模块等底层对象；开发者只需要拿到句柄并将其传入cu系列API函数，驱动内部通过句柄查表找到对应的底层对象执行操作，开发者无法直接访问、修改对象内部结构，这样驱动可以随意调整内部实现而不改变上层接口，同时对资源和权限进行管控。
> 但是为什么不能说句柄一定就是结构体指针？因为句柄只是抽象概念：它底层也可以实现成单纯的整数ID（就像Linux的fd），内核维护一张映射表，根据ID查找对象；只是CUDA驱动在实现的时候，选择了用未定义结构体的指针来实现；也就是说：结构体指针只是句柄的其中一种实现方案，不是句柄的本质。

驱动API支持的对象汇总如表6所示。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261010102301.png)
在调用驱动 API 的任意函数之前，必须先通过`cuInit()`完成驱动 API 的初始化。 随后需要创建一个 CUDA 上下文，该上下文绑定到指定 GPU 设备，并且设置为当前宿主线程的上下文，详细说明见本文的「上下文（Context）」章节。
在 CUDA 上下文内部，由主机代码将核函数以 PTX 或者二进制目标文件的形式显式加载，详见本文「模块（Module）」章节。 因此，用 C++ 编写的核函数，必须单独编译为 PTX 或二进制目标文件。 核函数通过本文「核执行（Kernel Execution）」章节介绍的 API 入口来启动。
任何希望能够在未来新硬件架构上运行的应用程序，必须加载 PTX 代码，而非二进制代码。 原因是：二进制代码与硬件架构强绑定，无法兼容未来新架构；而 PTX 代码会在加载阶段，由设备驱动实时编译为二进制机器码。
下面是「核函数」章节中，使用驱动 API 编写的示例对应的主机端代码：
```cpp
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Initialize
    cuInit(0);

    // Get number of devices supporting CUDA
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit (0);
    }

    // Get handle for device 0
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    // Create context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Create module from binary file
    CUmodule cuModule;
    cuModuleLoad(&cuModule, "VecAdd.ptx");

    // Allocate vectors in device memory
    CUdeviceptr d_A;
    cuMemAlloc(&d_A, size);
    CUdeviceptr d_B;
    cuMemAlloc(&d_B, size);
    CUdeviceptr d_C;
    cuMemAlloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    // Get function handle from module
    CUfunction vecAdd;
    cuModuleGetFunction(&vecAdd, cuModule, "VecAdd");

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &d_A, &d_B, &d_C, &N };
    cuLaunchKernel(vecAdd,
                   blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
                   0, 0, args, 0);

    ...
}
```
# 一、上下文（Context）
CUDA 上下文可以类比为 CPU 进程。驱动 API 中所有的资源与操作都被封装在 CUDA 上下文内部；当上下文销毁时，系统会自动清理这些资源。除模块、纹理或表面引用这类对象之外，每个上下文拥有独立的地址空间。因此，来自不同上下文的`CUdeviceptr`会指向不同的内存位置。
>CUDA Context、GPU、CUDA驱动、CPU完整关系梳理：
>**核心结论**：CUDA Context是CPU线程操作某一块GPU时，在GPU上建立的一套独立运行环境
>**CPU**：跑宿主线程；线程可以调用CUDA Driver API（cuXXX）、每个CPU线程单独维护自己的上下文栈，记录哪个 Context 是本线程的`current`当前上下文；注意：上下文对象不属于线程，只是线程可以把它 “激活” 变成自己的当前上下文。一个 Context 可以被多个 CPU 线程轮流激活使用，但同一个 CPU 线程同一时刻只能激活 1 个 Context。
>**GPU**：设备硬件，可以同时存在多个独立CUDA Context；每个 Context 拥有独立 GPU 虚拟地址空间（GPU VA）；每个 Context 持有自己的资源：显存分配记录、加载的 PTX 模块、kernel 状态、纹理 / 表面对象、事件、流（stream）；GPU 硬件的 SM、L2、显存硬件被多个 Context分时复用；
>**CUDA驱动**：介于 CPU 用户态程序和 GPU 硬件之间，负责：1.响应`cuInit`/`cuCtxCreate`这类用户态 API；2.在 GPU 上创建、销毁、切换 Context；3.管理 GPU 虚拟地址映射、显存管理、kernel JIT 编译、任务调度；4.处理上下文切换：当 CPU 线程切换 current context，驱动完成 GPU 侧环境切换。
>**CUDA上下文**：可以类比GPU上的进程，属于GPU侧的运行环境，封装了这一套环境下所有 GPU 资源；销毁 Context 时，驱动自动回收该 Context 持有的全部显存、模块等资源。
>上下文切换：
>- **CPU 上下文切换**：操作系统切换 CPU 进程 / 线程（OS 内核，CPU 侧）
>- **CUDA 上下文切换**：CUDA 驱动切换 GPU 上的 CUDA Context

一个宿主线程（CPU端线程）同一时刻**只能有一个设备上下文处于当前（current）状态**。通过`cuCtxCreate()`创建上下文时，该上下文会自动设为调用该函数的宿主线程的当前上下文。如果线程没有绑定有效的当前上下文，那么所有依赖上下文执行的 CUDA 函数（绝大多数不涉及设备枚举、上下文管理的函数）都会返回`CUDA_ERROR_INVALID_CONTEXT`错误。

每个宿主线程维护一个**当前上下文栈**。`cuCtxCreate()`会将新建上下文压入栈顶。可调用`cuCtxPopCurrent()`将上下文从宿主线程剥离；此时该上下文变为 “浮动状态（floating）”，可以被压入任意宿主线程，作为它的当前上下文。`cuCtxPopCurrent()`同时会恢复栈中之前的当前上下文（如果存在）。

每个上下文还维护一个**引用计数（usage count）**。`cuCtxCreate()`创建上下文时，引用计数初始化为 1。`cuCtxAttach()`增加引用计数，`cuCtxDetach()`减少引用计数。当调用`cuCtxDetach()`或`cuCtxDestroy()`后，引用计数降为 0 时，上下文才会被销毁。
>cuCtxPopCurrent()将上下文从宿主线程剥离之后不会降低引用次数吗，那此时上下文是被谁引用？
>`cuCtxPopCurrent()`**不会降低引用计数**，它只把上下文从当前线程的上下文栈剥离，变成浮动上下文。 这个 context 的引用，**保存在你代码持有的 CUcontext 句柄上**；只要你保留这个句柄，引用计数保持不变，上下文继续存活，等待被其他线程`cuCtxPushCurrent`重新激活。只有`cuCtxDetach`或者`cuCtxDestroy`才会减少引用计数。

驱动 API 可与 Runtime API 互操作。可以通过`cuDevicePrimaryCtxRetain()`，在驱动 API 侧访问由 Runtime 管理的**主上下文（primary context）**（参考文档链接）。
>主上下文是什么，驱动与Runtime管理的上下文区别是什么？
>主上下文（Primary Context）是 CUDA Runtime API 为进程内每一块 GPU 自动懒创建的标准 CUDA 上下文，当进程首次调用`cudaXXX`类 Runtime 接口时自动生成，同一进程内单块 GPU 仅有一个主上下文，Runtime 自动维护它的上下文栈与生命周期，开发者几乎感知不到 current 上下文、引用计数等底层细节；主上下文本质仍是普通 CUDA Context，具备独立 GPU 虚拟地址空间与引用计数，可通过 Driver API 的`cuDevicePrimaryCtxRetain`获取其句柄（调用会使引用计数 + 1，用完需调用`cuDevicePrimaryCtxRelease`释放引用），实现 Runtime 与 Driver API 互操作；而 Driver API 通过`cuCtxCreate`显式创建的自定义上下文，需要开发者手动管理上下文栈（`cuCtxPushCurrent`/`cuCtxPopCurrent`仅切换线程激活状态，不增减引用计数）、引用计数与生命周期，句柄持有引用，引用计数归 CUcontext 对象本身而非宿主线程；主上下文与自定义上下文在 GPU 硬件层面无差异，地址空间互相隔离，同一进程内可同时存在主上下文和自定义上下文，来回切换会触发 CUDA 上下文切换并带来性能开销；多进程共享同一块 GPU 时，每个进程会拥有各自独立的主上下文，驱动对多个上下文做时分复用，仅在不同上下文任务交替执行时触发上下文切换。

引用计数机制用于支持多方第三方代码在同一个上下文内协同工作。举例：如果加载了三个库，共用同一个上下文，每个库都会调用`cuCtxAttach()`增加引用计数，使用完毕后调用`cuCtxDetach()`减少引用计数。对于大多数库，要求应用程序在加载、初始化库之前就已经创建好上下文；这样应用程序可以基于自身策略创建上下文，库直接在传入的上下文上执行操作。 如果库想要**自行创建上下文**（调用方客户端可能已经创建、也可能未创建上下文，库在后台独立管理），则使用`cuCtxPushCurrent()`与`cuCtxPopCurrent()`，如下图所示。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20261010114205.png)
# 二、模块（Module）
**模块（Module）** 是可动态加载的设备代码与数据包，类似于 Windows 中的 dll，由 nvcc 编译输出（参考文档：[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#compilation-with-nvcc](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#compilation-with-nvcc)）。所有符号（包括函数、全局变量、纹理引用或表面引用）的名称均在**模块作用域**内维护，由此，由不同独立第三方编写的模块可以在**同一个 CUDA 上下文中**实现互操作。 下面这段代码示例用于加载一个模块，并获取某个核函数（kernel）的句柄：
```cpp

// `CUmodule` 是CUDA 模块的句柄类型，`cuModule` 变量只是一个句柄，代表加载到当前 CUDA 上下文里的整套设备代码包；它会绑定到当前宿主线程对应的当前 CUDA 上下文！ 这个模块只能在加载它的这个 Context 中使用；切换到别的 CUDA 上下文，这个 CUmodule 句柄直接无效。
CUmodule cuModule;

// 从磁盘读取ptx文件，加载设备代码到当前激活的CUDA上下文。
//`"myModule.ptx"`：nvcc 编译输出的 PTX 中间汇编代码（虚拟 ISA，JIT 会在 GPU 硬件上即时编译成硬件可执行指令）；也可以传入`.cubin`（已编译好的二进制）
// 底层行为：1.读取 ptx 文件到内存；2.在当前线程的 current CUDA Context内做JIT编译（PTX → GPU 硬件指令）；3.在该 Context 内创建 Module 对象；4.将模块句柄写入变量`cuModule`；
cuModuleLoad(&cuModule, "myModule.ptx");

// `CUfunction`：核函数句柄类型，用来代表模块内部的一个GPU kernel函数。
CUfunction myKernel;
// 在已经加载好的 Module 里，按符号名字查找 kernel，拿到 kernel 句柄
cuModuleGetFunction(&myKernel, cuModule, "MyKernel");
```
>**当前CUDA上下文**
>1. 对单条宿主线程：同一时刻，线程的current当前CUDA上下文最多只有 1 个。线程有一个上下文栈，栈顶就是当前`current`上下文；你可以 push 多个压进栈里，但只有栈顶那一个是激活的 current。push 新的，栈顶切换；pop 弹出栈顶，回到上一层栈顶。
>2. 对整张物理 GPU 硬件：同一时刻，GPU 上可以存在多个 CUDA Context 对象（已经创建、引用计数 > 0），但GPU 硬件当前正在激活执行任务的 Context 最多只能有 1 个（原生 CUDA 非抢占式 kernel 调度）。

以下代码示例从 PTX 代码编译并加载一个新模块，同时解析编译报错信息。
```cpp
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[3];
void* values[3];
char* PTXCode = "some PTX code";
char error_log[BUFFER_SIZE];
int err;
options[0] = CU_JIT_ERROR_LOG_BUFFER;
values[0]  = (void*)error_log;
options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[1]  = (void*)BUFFER_SIZE;
options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
values[2]  = 0;
err = cuModuleLoadDataEx(&cuModule, PTXCode, 3, options, values);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
```
以下代码示例基于多段 PTX 代码完成编译、链接并加载新模块，同时解析链接错误与编译错误。
```cpp
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[6];
void* values[6];
float walltime;
char error_log[BUFFER_SIZE], info_log[BUFFER_SIZE];
char* PTXCode0 = "some PTX code";
char* PTXCode1 = "some other PTX code";
CUlinkState linkState;
int err;
void* cubin;
size_t cubinSize;
options[0] = CU_JIT_WALL_TIME;
values[0] = (void*)&walltime;
options[1] = CU_JIT_INFO_LOG_BUFFER;
values[1] = (void*)info_log;
options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
values[2] = (void*)BUFFER_SIZE;
options[3] = CU_JIT_ERROR_LOG_BUFFER;
values[3] = (void*)error_log;
options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[4] = (void*)BUFFER_SIZE;
options[5] = CU_JIT_LOG_VERBOSE;
values[5] = (void*)1;
cuLinkCreate(6, options, values, &linkState);
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                    (void*)PTXCode0, strlen(PTXCode0) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                    (void*)PTXCode1, strlen(PTXCode1) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
cuLinkComplete(linkState, &cubin, &cubinSize);
printf("Link completed in %fms. Linker Output:\n%s\n", walltime, info_log);
cuModuleLoadData(cuModule, cubin);
cuLinkDestroy(linkState);
```
可利用多线程对模块链接 / 加载流程的部分环节进行加速，加载 cubin 文件时同样适用。以下代码示例使用`CU_JIT_BINARY_LOADER_THREAD_COUNT`来加快模块加载速度。
```cpp
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[3];
void* values[3];
char* cubinCode = "some cubin code";
char error_log[BUFFER_SIZE];
int err;
options[0] = CU_JIT_ERROR_LOG_BUFFER;
values[0]  = (void*)error_log;
options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[1]  = (void*)BUFFER_SIZE;
options[2] = CU_JIT_BINARY_LOADER_THREAD_COUNT;
values[2]  = 0; // Use as many threads as CPUs on the machine
err = cuModuleLoadDataEx(&cuModule, cubinCode, 3, options, values);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
```
# 三、内核执行（Kernel Execution）
`cuLaunchKernel()` 根据给定的执行配置启动一个核函数。

参数有两种传递方式： 一种是**指针数组**（`cuLaunchKernel()` 的倒数第二个入参）：数组中第`n`个指针对应第`n`个参数，指向一块内存区域，参数将从该内存拷贝； 另一种是通过**扩展选项**（`cuLaunchKernel()` 的最后一个入参）传入。

若采用扩展选项传参（即`CU_LAUNCH_PARAM_BUFFER_POINTER`选项），需要传入指向**单个缓冲区**的指针；要求缓冲区内部各个参数之间的偏移量必须符合设备代码中各参数类型的对齐要求。

设备代码中内置向量类型的对齐要求，可查阅文档：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#vector-types-alignment-requirements-in-device-code](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#vector-types-alignment-requirements-in-device-code)。其余所有基础类型，其设备端对齐要求与主机端一致，因此可通过`__alignof()`获取。 唯一例外场景：主机编译器将`double`、`long long`（64 位系统下的`long`）按单字边界对齐，而非双字边界（例如使用 gcc 编译标识`-mno-align-double`）；**设备代码中这些类型永远按双字边界对齐**。

`CUdeviceptr`本质是整数，但它代表指针，因此对齐要求等于`__alignof(void*)`。

下面的代码示例使用宏`ALIGN_UP()`调整每个参数的偏移量以满足对齐要求，并用另一个宏`ADD_TO_PARAM_BUFFER()`把各个参数添加到参数缓冲区，该缓冲区最终传给`CU_LAUNCH_PARAM_BUFFER_POINTER`选项。
```cpp
#define ALIGN_UP(offset, alignment) \
      (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

char paramBuffer[1024];
size_t paramBufferSize = 0;

#define ADD_TO_PARAM_BUFFER(value, alignment)                   \
    do {                                                        \
        paramBufferSize = ALIGN_UP(paramBufferSize, alignment); \
        memcpy(paramBuffer + paramBufferSize,                   \
               &(value), sizeof(value));                        \
        paramBufferSize += sizeof(value);                       \
    } while (0)

int i;
ADD_TO_PARAM_BUFFER(i, __alignof(i));
float4 f4;
ADD_TO_PARAM_BUFFER(f4, 16); // float4's alignment is 16
char c;
ADD_TO_PARAM_BUFFER(c, __alignof(c));
float f;
ADD_TO_PARAM_BUFFER(f, __alignof(f));
CUdeviceptr devPtr;
ADD_TO_PARAM_BUFFER(devPtr, __alignof(devPtr));
float2 f2;
ADD_TO_PARAM_BUFFER(f2, 8); // float2's alignment is 8

void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, paramBuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &paramBufferSize,
    CU_LAUNCH_PARAM_END
};
cuLaunchKernel(cuFunction,
               blockWidth, blockHeight, blockDepth,
               gridWidth, gridHeight, gridDepth,
               0, 0, 0, extra);
```
结构体的对齐要求等于其各个成员对齐要求中的最大值。因此，如果结构体包含内置向量类型、`CUdeviceptr`，或是对齐规则存在差异的`double`与`long long`，该结构体在设备代码和主机代码中的对齐要求就可能不一致；这类结构体的填充（padding）布局也会有所区别。

例如下面这个结构体：在主机代码中完全没有填充字节，但在设备代码里，会在成员`f`之后插入 12 字节的填充。原因是成员`f4`的对齐要求为 16 字节。
```cpp
typedef struct {
    float  f;
    float4 f4;
} myStruct;
```
# 四、Runtime API 与 Driver API 的互操作
应用程序可以混用运行时 API 代码与驱动 API 代码。

如果通过驱动 API 创建一个上下文并将其设为当前上下文，后续的运行时 API 调用将使用该上下文，而不会新建上下文。

若运行时已完成初始化，可使用`cuCtxGetCurrent()`获取初始化阶段创建的上下文；该上下文可用于后续的驱动 API 调用。

由运行时隐式创建的上下文称为**主上下文**（参见文档：[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-runtime-initialization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-runtime-initialization)）。可通过这组接口从驱动 API 侧管理主上下文：[https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)。

设备内存可使用两套 API 中的任意一套完成分配与释放。`CUdeviceptr`可转换为普通指针，反之亦然。以下代码展示 CUDA Runtime API 和 Driver API 设备地址互转：两套 API 分配出来的GPU 显存地址本身是同一个数值，只是用了不同类型包装，在同一个 CUDA 上下文内可以互相强制类型转换。
```cpp
CUdeviceptr devPtr;
float* d_data;

// Allocation using driver API
cuMemAlloc(&devPtr, size);
d_data = (float*)devPtr;

// Allocation using runtime API
cudaMalloc(&d_data, size);
devPtr = (CUdeviceptr)d_data;
```
具体而言，这意味着使用驱动 API 编写的应用程序可以调用基于运行时 API 开发的库（例如 cuFFT、cuBLAS 等）。