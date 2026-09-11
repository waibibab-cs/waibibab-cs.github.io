# 1.PCIe BDF
**BDF = Bus‑Device‑Function**，PCI/PCIe 设备的**全局唯一标识符**，Linux 内核、lspci、sysfs、VFIO、GDS/SPDK 全部依赖 BDF 定位 PCIe 设备。 
PCIe的四个坐标是：
```text
Domain : Bus : Device . Function
0000   : 21  : 00     . 0
```
Domain：一个 PCIe 配置空间的编号，用于区分不同 PCIe 域
Bus：某一级 Root Complex 或 PCIe Bridge（Root Port）下游的逻辑总线编号，例如：
```text
Root Complex 20:00.0
  └─ 下游 Bus 21
       └─ GPU 21:00.0
Root Complex 00:00.0
  └─ 下游 Bus 01、02、03、04……
       └─ 多块 NVMe
```
Device：同一 bus 上的设备槽位，范围通常 0–31
Function：一个 PCIe 设备中的功能单元，范围通常 0–7
使用命令`lspci -Dtv`可以查看本机的树形拓扑
# 2.PCIe Domain与segment
·PCIe Segment：是硬件概念。它代表一个完全独立的PCIe总线树，由独立的Root Complex（根复合体）管理。不同Segment之间的地址空间和配置空间是互相隔离的。多Segment常见于大型服务器，用于突破单个总线域只能有256个总线的限制。
·PCIe Domain：是Linux内核中的软件表示。内核使用域号来区分不同的硬件Segment。因此，可以在Linux的lspci -t命令的输出中看到“Domain: Bus: Device. Function”的格式，这个Domain号实际上对应的就是硬件Segment。

老的 Intel 平台，不同 Socket 的 RC 会作为独立 Segment，所以会看到 domain:0000、domain:0001；
新型平台如Genoa 单颗 CPU IOD 内的多 RC，共享同一个 PCI Segment（domain=0000），靠不同 root bus 区分，因此同一个domain中可以有不同的RC
# 3.Host Bridge/Root Complex/Root Port
**Host Bridge**：源自传统 PCI 的术语，指连接 CPU/内存子系统与 PCI 总线层级的主机侧桥接逻辑；在 Linux/`lspci` 的实际语境中，常被用来描述或近似指代Root Complex

**Root Complex**：在 PCI‑Express（PCIe）系统中，**根复合体（Root Complex）** 设备将 CPU 与内存子系统，连接到由一个或多个 PCIe/PCI 设备构成的 PCIe 交换架构。根复合体有时也被称作**PCI 根桥（PCI root bridge）**。一个根复合体可以具备多个 PCIe 端口；多个交换设备可以连接到根复合体的端口上，也可以进行级联。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260910100923.png)

**Root Port（根端口）**：RC 内的一个 PCIe 下行端口，本质上是 RC 内实现的 PCIe Bridge。它所在的上游 Bus 是 Root Bus；它向下游分配 **secondary bus**，并由此启动一棵独立的 PCIe 子树。层级关系可理解为：
```text
CPU / 内存
  └─ Root Complex
       └─ Root Bus 00
            ├─ Root Port A（位于 Bus 00）
            │    └─ Secondary Bus 01 → NVMe
            └─ Root Port B（位于 Bus 00）
                 └─ Secondary Bus 02 → GPU
```
下图当中Host Bridge可以近似理解成Root Complex，对应的上游总线为Bus0，PCI-PCI Bridge可以近似理解成Root Port，分别开出一条下游Secondary Bus
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260910154906.png)
# 4.PCIe Switch
PCIe Switch是一种硬件设备，提供扩展或聚合能力，并允许更多的设备连接到一个PCle端口，它包含以下功能：
1. 连接多个设备：PCIe switch允许多个设备通过单个PCIe总线连接到主机系统，从而扩展系统的连接性。
2. 数据交换：PCIe switch可以在多个设备之间传输数据，允许设备之间直接通信而无需通过主机处理器。
3. 动态分配：支持动态分配带宽和资源，根据需要调整设备之间的通信速率和优先级。
4. NTB（Non-Transparent Bridge）：支持NTB技术，允许两个或多个系统之间直接通信，提高数据传输效率。
5. Peer to Peer：支持点对点通信，设备之间可以直接进行数据交换而无需通过主机。
6. MRIOV（Multi-Root I/O Virtualization）：一种PCIe技术，支持多根系统共享同一个PCIe设备。普通 PCIe 设备只能被单根系统独占，而 MR-IOV 可为每个根端系统暴露设备的多个虚拟功能，分配给虚拟机或容器使用，实现多系统共享协作，提升硬件利用率、降低成本，简化云环境设备管理。如下图所示
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260910160326.png)
# 5.PCIe Hierarchy
这个概念实际上是对上面的总结。
PCIe 层级采用**树形点对点网络拓扑**，将计算机中央处理器与系统内存连接到外围硬件。
核心层级组件
- **Root Complex (RC，根复合体)**：这棵树的锚点，位于拓扑顶端。它把 CPU、系统内存接入 PCIe 子系统，并负责配置、路由与电源管理。
- **Switches（PCIe 交换芯片）**：透明桥接器件，用于扩展网络。它拥有 1 个上游端口，并分出多个下游端口，可接入更多设备，且不会改变软件的设备可见性。
- **Bridges（桥）**：转换器，用来把 PCIe 树形结构连接至传统或其他总线架构（例如老式 PCI、PCI-X 插槽）。
- **Endpoints (EP，端点设备)**：树形结构边缘的叶子节点，即实际的外设硬件，如 GPU 显卡、NVMe 固态硬盘、网卡。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260910160503.png)
# 6.PCIe TLP路由
下图是一个PCIe总线系统示意图。此时RC发出一个TLP，经过Switch访问EP，TLP的路径为红色箭头所示。首先TLP从RC的下游OUT端口发出，Switch的上游IN端口接收到该TLP后，根据其路由信息，将其转发到Switch的下游OUT端口，随后TLP达到EP的IN端口，最后TLP到达EP设备。TLP从RC到EP的转发过程被称为TLP的路由过程。PCIe总线总共定义了三种路由方式，分别是基于地址（Address）路由、基于ID（BDF）路由和隐式（Implicit）路由。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260910161526.png)
TLP使用的路由方式和TLP的类型有关，具体如下表所示：

|TLP 类型|使用的路由方式|
|---|---|
|内存读 [锁定]、内存写、原子操作|地址路由（Address Routing）|
|IO 读写|地址路由（Address Routing）|
|配置读写|ID 路由（ID Routing）|
|消息、带数据消息|地址路由、ID 路由 或 隐式路由（Implicit routing）|
|完成包、带数据完成包|ID 路由（ID Routing）|

# 7.PCIe BAR0
BAR空间是指PCI设备中的基地址寄存器（Base Address Register）所映射的地址空间。每个PCI设备都有多个BAR，用于指示设备在系统地址空间中的位置。
PCIe BAR0 是 PCIe 设备配置空间中的第一个基地址寄存器，用来决定设备在系统内存或I/O空间中的映射起始地址。
- **位置**：位于设备中的PCIe 配置空间（Configuration Space）的头部。Type 0 设备（如普通Endpoint端点设备；Type1设备一般指PCIe桥，Root Port等设备）一共有 6 个 BAR（BAR0 到 BAR5）。
- **作用**：CPU通过访问BAR0指向的物理地址，来直接读写PCIe设备的内部寄存器或内存。
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260910170702.png)
- Bit0: 用于指示设备寄存器是映射到内存空间0还是IO空间（1）。
- Bit1: 保留位，值为0。
- Bit2: 在Memory BAR中，0表示32位地址空间，1表示64位地址空间。
- Bit3: 在Memory BAR中用于表示设备是否允许预取数据，1表示可以预取，0表示不可以预取。
- Bit4~31: 用来表示设备需要占用的地址空间大小。其中，某些位为只读，且0表示需要的地址空间大小。

PCIe 设备上电复位后，BIOS / 操作系统 PCI 子系统对 BAR0 的完整处理流程：
1. **BAR 探测（扫盲）**：向 BAR 寄存器写入全 1（`0xFFFFFFFF`），再回读寄存器值；硬件通过 Bit4~31 中的只读 0 比特，上报该 BAR 所需地址空间大小与地址对齐约束，Bit0~Bit3 固定为硬件预定义的空间类型、寻址位数、预取等属性。若读到 BAR 全部为 0，则代表该 BAR 未启用。
2. **地址分配**：操作系统在全局 PCIe 内存 / I/O 地址空间中，寻找一块大小、对齐要求均匹配的空闲地址区间，取出这段区间的**全局起始基地址**。
3. **写入 BAR 寄存器**：将分配好的全局基地址写入 BAR 寄存器的 Bit4~31 高位区域，**保留 Bit0~Bit3 不变**，维持设备的地址空间属性标记。若是 64 位 Memory BAR，则占用 BARn 与 BARn+1 两个连续寄存器，分别存放基地址低 32 位与高 32 位。
4. **正常访问与路由**：CPU/GPU 发起访问请求，**全局 PCIe 地址 = BAR 内保存的基地址 + 设备内部偏移 offset**。PCIe 根复合体与交换机根据全局地址区间路由 TLP 报文；端点设备收到报文后，用全局地址减去 BAR 基地址，得到设备内部偏移，读写内部寄存器或片上资源。
# 8.PCIe Gen
**PCIe各代带宽**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260911090902.png)
**NVLink各代带宽**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260911090938.png)
**主流GPU互连与显存规格**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260911091002.png)
**NVMe SSD典型速度**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260911091036.png)
**典型场景带宽**
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260911091055.png)
# 9.ACS
PCIe ACS（Access Control Services，访问控制服务）是 PCIe 提供的一项扩展功能，用于增强 PCIe 总线拓扑中的访问控制和安全性。它主要用来管理和控制设备之间的点对点（P2P, Peer-to-Peer）通信
基础功能：默认情况下，PCIe 设备间的 P2P 流量可以直接在 Switch 之间转发，不经过 CPU 或 Root Complex。开启 ACS 后，可以将 P2P 请求重定向到 Root Complex，由上游的 IOMMU/SMMU 验证逻辑决定是否允许该访问。是P2P通信最常见的“拦路虎”
# 参考资料
1.https://blog.csdn.net/qq_37344125/article/details/136903500
2.https://zhuanlan.zhihu.com/p/693852582
3.https://blog.csdn.net/u011037593/article/details/139398579
4.https://zhuanlan.zhihu.com/p/683933089
5.https://github.com/ForceInjection/AI-fundamentals/blob/main/01_hardware_architecture/performance/05_pcie_nvlink_speed_reference.md