# CPU-SSD ANNS系统
## 一、DiskANN（2019；开源；已读）
DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node

摘要：目前最先进的**近似最近邻搜索（ANNS）** 算法所生成的索引必须存储于**主内存**中，才能实现高召回率的快速搜索——这使得此类算法成本高昂，同时限制了数据集的规模。我们提出了一款名为**DiskANN**的全新基于图索引的搜索系统，它仅需**64GB主存**与一台普通**固态硬盘（SSD）** ，即可单机完成十亿级数据量的索引构建、存储与检索。  
我们证实，DiskANN所构建的SSD基索引可同时满足大规模ANNS的三大核心需求：**高召回率、低查询延迟、高密度（每个节点可索引的数据量）** 。在十亿级SIFT1B数据集上，DiskANN在一台16核机器上的**单秒查询处理量超5000次**，均值延迟不足3毫秒，且**1-recall@1高达95%以上**。而当前类似内存占用下的十亿级ANNS算法（如FAISS与IVFOADC+P ），其1-recall@1效率仅止步于50%左右。此外，与HNSW、NSG等当前主流的图基方法相比，DiskANN在高召回率场景下，每个节点可索引与检索的数据量是它们的5-10倍。  
最终，作为DiskANN完整架构的核心组件，我们提出了一款名为**Vamana**的全新图基ANNS索引，它的灵活性远超现有主流图索引算法——即使与全内存基的图索引相比亦是如此。

对比工作：NSG、HNSW

`Jayaram Subramanya S, Devvrit F, Simhadri H V, et al. Diskann: Fast accurate billion-point nearest neighbor search on a single node[J]. Advances in neural information processing Systems, 2019, 32.`

论文阅读：
https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-xi-du-%3BNeurIPS%E2%80%9919%E3%80%91DiskANN%EF%BC%9A-dan-jie-dian-shang-kuai-su-%E3%80%81-jing-zhun-de-shi-yi-ji-shu-ju-dian-zui-jin-lin-sou-suo.html
## 二、FreshDiskANN（2021；开源；已读）
FreshDiskANN: A Fast and Accurate Graph-Based  ANN Index for Streaming Similarity Search

摘要：近似最近邻搜索（ANNS）是信息检索领域的基础核心技术，其中**图索引**是当前的业界最优方案，已在工业界得到广泛应用。图索引领域的最新研究进展，现已能够在单台配备固态硬盘的普通商用机器上，对十亿级向量数据集完成索引构建与检索，同时保证高召回率和毫秒级查询延迟。
然而，现有的近似最近邻搜索图算法仅支持**静态索引**，无法适配众多实际业务场景所需的语料实时变更需求（例如文档句子索引、邮件索引、新闻索引等场景）。为解决这一缺陷，目前工业界的通用做法是通过**周期性重建索引**来同步数据更新，但这种方式的成本极高、开销难以承受。
本文提出首个可**实时同步语料更新**、且不损失检索性能的图结构近似最近邻搜索索引。基于该索引的更新规则，我们设计实现了 FreshDiskANN 系统：该系统可在搭载固态硬盘、内存资源有限的工作站上完成十亿级向量的索引构建，每秒可支持数千次并发的实时插入、删除与检索请求，同时保持**5-recall@5 召回率高于 95%**。与现有方案相比，该系统将索引时效性维护成本降低了 5 至 10 倍。

挑战-解决措施：
- **挑战**：现有图基 ANNS 索引仅支持静态结构，无法适配文档、邮件、新闻等真实场景的语料实时增删需求。**解决方案**：提出首个可实时同步语料更新、且不牺牲检索性能的图基 ANNS 索引，解决动态更新适配问题。
- **挑战**：工业界通过周期性全量重建索引实现数据更新，维护索引时效性的成本极高、开销巨大。**解决方案**：设计 FreshDiskANN 系统，支持单机 SSD 环境下十亿级向量索引构建，每秒处理数千次并发增删查请求，将索引维护成本降低 5-10 倍。
- **挑战**：传统静态图索引（如 HNSW、NSG）直接增删会导致图结构稀疏、检索召回率持续下降，无稳定更新策略。**解决方案**：研发 FreshVamana 算法，采用 α-RNG 松弛剪枝规则与懒删除 + 批量合并策略，保障多轮更新后索引召回率稳定。
- **挑战**：内存无法承载十亿级动态索引，SSD 直接更新又会产生大量随机读写，拖慢检索和更新效率。**解决方案**：构建 “内存临时索引 + SSD 长期索引” 双结构，设计 StreamingMerge 算法，后台高效合并更新，内存占用仅与更新量相关，不影响检索性能。

对比工作：-

`Singh A, Subramanya S J, Krishnaswamy R, et al. Freshdiskann: A fast and accurate graph-based ann index for streaming similarity search[J]. arXiv preprint arXiv:2105.09613, 2021.`

论文阅读：https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-yue-du-%3B21%E3%80%91FreshDiskANN%EF%BC%9A-kuai-su-qie-zhun-que-de-yong-yu-liu-shi-ANNS-de-tu-suo-yin.html
## 三、ParlayANN（2024；开源）
ParlayANN: Scalable and Deterministic Parallel Graph-Based Approximate Nearest Neighbor Search Algorithms

摘要：近似最近邻搜索（ANNS）算法是现代深度学习技术体系中的重要组成部分，它能够针对数据的高维向量表征（即嵌入向量）实现高效的相似度检索。在各类近似最近邻搜索算法中，**基于图结构的算法**在吞吐量与召回率的综合表现上优势最为突出。
如今近似最近邻搜索的数据集规模已达海量级别，但现有的并行化图结构实现方案存在明显短板：这类方案大量使用锁机制，还存在各类串行性能瓶颈。这两大问题一方面导致算法无法充分利用多核处理器实现高效扩展，另一方面还会带来结果不确定性，而这一特性在部分应用场景中是无法接受的。
本文提出了**ParlayANN**库，这是一套具备确定性、支持并行运算的图结构近似最近邻搜索算法库，同时配套了一系列辅助开发工具。该库针对四种当前主流的图结构近似最近邻搜索算法设计了全新并行实现方案，可支持十亿级规模数据集。本文所提算法结果确定，且在各类复杂数据集下均具备出色的可扩展性。
除创新算法设计外，本文还对新算法以及两种现有非图结构检索方法开展了详尽的实验分析。实验结果不仅验证了本文技术方案的有效性，还基于大规模数据集对各类近似最近邻搜索算法进行了全面对比，并总结出多项具有参考价值的研究结论。

挑战-解决措施：
- **挑战**：现有并行图结构 ANNS 实现大量使用锁与串行瓶颈，无法高效扩展至多处理器，且结果具有非确定性，不满足持久化、崩溃恢复等场景需求。**解决方案**：提出 ParlayANN 库，采用无锁、确定性的 fork-join 并行设计，实现可扩展至百亿级数据集与上百线程的并行图结构 ANNS 算法。
- **挑战**：增量式 ANNS 算法（DiskANN、HNSW）并行插入依赖逐点锁，引发竞争与非确定性。**解决方案**：创新采用**前缀加倍（prefix doubling）** 与批量插入 / 剪枝技术，以指数级增长批次插入点，用并行半排序合并边，全程无锁且保持确定性。
- **挑战**：基于聚类树的 ANNS 算法（HCNNG、PyNNDescent）并行度受限、合并边时用锁导致竞争，且局部图构建占用大量缓存。**解决方案**：对聚类树做并行分治构建，用半排序无锁合并边；针对 HCNNG 提出**边受限最小生成树**优化，降低缓存占用，提升并行扩展性。
- **挑战**：PyNNDescent 的两跳邻域计算呈度平方复杂度，内存占用过高，无法扩展至十亿级数据。**解决方案**：限制顶点最大度并随机采样边，批量计算两跳邻域以控制中间内存，使其可扩展至亿级规模。
- **挑战**：通用贪心搜索的已访问集合判断慢、图布局存在间接访问开销，影响查询性能。**解决方案**：使用单侧错误近似哈希表加速成员判断，优化图存储布局，并引入 (1+ϵ) 剪枝，平衡查询精度与速度。
- **挑战**：现有 ANNS 评测集中在百万级小数据、串行性能，缺乏十亿级大规模并行对比基准。**解决方案**：在三类十亿级真实数据集上做全面实验，首次给出确定性并行图算法与非图方法的公平对比，揭示高召回与分布外查询性能规律。

对比工作：四种当前最优图 ANNS 算法的官方实现（DiskANN、HNSW、HCNNG、PyNNDescent）；两种主流非图 ANNS 方法（FAISS-IVF、FALCONN）

`Manohar M D, Shen Z, Blelloch G, et al. Parlayann: Scalable and deterministic parallel graph-based approximate nearest neighbor search algorithms[C]//Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming. 2024: 270-285.`
# GPU-CPU-SSD ANNS系统
## 一、FusionANNS（2025；已读）
Towards High-throughput and Low-latency Billionscale Vector Search via CPU/GPU Collaborative Filtering and Re-ranking

摘要
近似最近邻搜索（ANNS）已成为数据库与人工智能基础设施的关键组成部分。向量数据集规模持续增长，给近似最近邻搜索服务在**性能、成本与精度**方面带来了严峻挑战。**当前主流的近似最近邻搜索系统均无法同时兼顾这些问题**。
本文提出**FusionANNS**系统：面向**十亿级向量数据集**，仅借助固态硬盘（SSD）与**单块入门级图形处理器（GPU）**，即可实现**高吞吐、低延迟、低成本、高精度**的近似最近邻搜索。
FusionANNS 的核心思想是**CPU/GPU 协同过滤与重排序机制**，该机制大幅减少 CPU、GPU 与 SSD 之间的 I/O 操作，进而突破 I/O 性能瓶颈。具体来说，本文提出三项创新设计：
1. **多层索引**，避免 CPU 与 GPU 之间的数据交换；
2. **启发式重排序**，在保证高精度的同时消除不必要的 I/O 与计算开销；
3. **冗余感知 I/O 去重**，进一步提升 I/O 效率。
我们实现了 FusionANNS 系统，并将其与当前最先进的**基于 SSD 的 ANNS 系统 SPANN**、**GPU 加速的内存式 ANNS 系统 RUMMY**进行对比。实验结果表明：
相较于 SPANN，FusionANNS 的每秒查询数（QPS）提升**9.4–13.1 倍**，成本效率提升**5.7–8.8 倍**；
相较于 RUMMY，FusionANNS 的 QPS 提升**2–4.9 倍**，成本效率提升**2.3–6.8 倍**，同时保持低延迟与高精度。

挑战-解决措施：
- 面对 GPU 显存无法容纳大规模压缩索引、CPU 与 GPU 间数据交换频繁的问题——FusionANNS 采用多层索引架构，将原始向量存在 SSD、PQ 压缩向量常驻 GPU 显存、向量 ID 与导航图放在 CPU 内存，只传输轻量 ID 而非向量内容，大幅减少设备间数据搬运。
- 针对 PQ 压缩带来精度损失、固定重排数量导致无效 I/O 与计算过多的问题——系统提出启发式分批次重排策略，通过轻量反馈判断 Top-k 结果是否稳定，满足条件立即终止重排，在保证精度的同时最小化 I/O 与距离计算。
- 由于向量远小于 SSD 最小读取单元引发严重读放大——FusionANNS 优化 SSD 存储布局，将相似向量紧凑存放，并支持批次内与批次间的 I/O 去重，合并同页请求、复用内存缓存，显著降低读放大并提升 I/O 效率。

对比工作：diskann、spann、rummy

`Tian B, Liu H, Tang Y, et al. Towards High-throughput and Low-latency Billion-scale Vector Search via {CPU/GPU} Collaborative Filtering and Re-ranking[C]//23rd USENIX Conference on File and Storage Technologies (FAST 25). 2025: 171-185.`
## 二、FlashANNS（2026；开源；已读）
FlashANNS: GPU-Driven Asynchronous I/O Pipelining for Eliminating Storage-Compute Bottlenecks in Billion-Scale Similarity Search

摘要：近似最近邻搜索（ANNS）能够在高维向量空间中实现高效的相似度检索，已成为推荐系统、检索增强生成（RAG）等上层任务的核心基础组件。现代 ANNS 系统通过集成固态硬盘（SSD）来支撑 TB 级规模的向量数据集，主要采用**聚类索引**与**图索引**两种架构。然而，聚类索引类 ANNS 系统因向量索引粒度粗糙，存在查询吞吐性能不佳的问题；图索引系统则受两项固有缺陷制约，难以达到最优性能：
1. 无法将 SSD 数据读取与距离计算过程**并行执行**；
2. I/O 长尾延迟导致存储访问性能低下。
为解决上述难题，本文提出**FlashANNS**—— 一款基于 GPU 加速、核外（out-of-core）图索引的 ANNS 系统，通过 I/O 与计算重叠执行突破性能瓶颈。本研究的核心思路是对 I/O 操作与计算过程进行精细调度，主要包含三项创新设计：
3. **带严格理论收敛保证的松依赖异步流水线**：FlashANNS 解除 I/O 与计算间的依赖约束，让 GPU 距离计算与 SSD 数据传输实现**完全重叠**；
4. **查询粒度并发 SSD 访问**：FlashANNS 构建无锁 I/O 栈并采用查询级并发控制，避免 I/O 长尾延迟引发的性能下降；
5. **计算 - I/O 平衡的图度选择策略**：针对不同硬件配置、数据集与查询需求，在计算负载与存储访问延迟间实现最优平衡。
我们实现了 FlashANNS 系统，并与三款当前顶尖的核外 ANNS 系统（SPANN、DiskANN、FusionANNS）进行对比实验。结果表明：在**Recall@10≥95%** 的相同精度条件下，单 SSD 配置中本方法的查询吞吐比现有最优方案**提升 2.7–5.9 倍**；多 SSD 配置下，查询吞吐提升幅度可达**3.9–12.2 倍**。

挑战-解决措施：
- 针对计算与 I/O 串行执行、GPU 和 SSD 利用率低的问题——FlashANNS 提出松依赖异步 I/O 流水线，放宽步骤间数据依赖，让 SSD 数据读取与 GPU 距离计算充分重叠，在保证召回率的同时大幅提升查询吞吐。
- 针对 GPU 核级 I/O 同步导致长尾延迟放大的问题——论文设计查询级并发 SSD 访问架构，以单个查询为独立单元执行 I/O，避免全局等待，有效降低长尾延迟并提升 I/O 效率。
- 针对图节点过小引发 I/O 放大与带宽浪费的问题——FlashANNS 采用基于采样的图度选择机制，自适应选取最优图出度，使计算与 I/O 耗时平衡，消除 I/O 放大并最大化流水线效率。

对比工作：diskann、spann、fusionann

`Xiao Y, Sun M, Song Z, et al. FlashANNS: GPU-Driven Asynchronous I/O Pipelining for Eliminating Storage-Compute Bottlenecks in Billion-Scale Similarity Search[J]. Proceedings of the ACM on Management of Data, 2026, 4(1 (SIGMOD): 1-27.`

论文阅读：https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-xi-du-%EF%BC%9BSIGMOD%2726%E3%80%91FlashANNS%EF%BC%9A-ji-yu-GPU-de-yi-bu-IO-liu-shui-xian-ji-shu-%EF%BC%8C-yong-yu-xiao-chu-shi-yi-ji-xiang-si-xing-sou-suo-zhong-de-cun-chu-yu-ji-suan-ping-jing.html
## 三、SVFusion（2026；开源；已读）
SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search

摘要：近似最近邻检索（ANNS）是信息检索、推荐系统等现代应用的底层支撑。随着向量数据规模飞速增长，面向实时向量检索的高效索引构建已成为刚需。
现有基于 CPU 的方案虽支持数据更新，但吞吐性能偏低；GPU 加速系统虽性能强劲，却难以适配动态更新场景，且受限于 GPU 显存容量。这就使得**兼顾精度与速度、面向大规模持续检索**的业务场景，存在显著的性能断层。
本文提出 **SVFusion**—— 一套面向实时向量检索的**GPU-CPU - 磁盘协同框架**，打通了高性能 GPU 计算与在线数据更新能力。
SVFusion 采用分层向量索引架构，依托 CPU-GPU 协同计算，结合**负载感知的向量缓存机制**，最大化利用有限的 GPU 显存资源；同时基于 CUDA 多流优化与自适应资源调度实现实时协同调度，并引入并发控制机制，保证查询与更新交错执行时的数据一致性。
实验结果表明：在各类流式负载下、面向大规模数据集并保持高召回率的前提下，相较于基线方法，SVFusion 在查询延迟与吞吐性能上均取得显著提升，**平均吞吐提升 20.9 倍**，查询延迟降低**1.3~50.7 倍**。

挑战-解决措施：
- 针对**GPU 显存容量不足，无法容纳大规模向量数据，频繁数据迁移抵消 GPU 加速收益**的挑战——SVFusion 设计了跨 GPU、CPU、磁盘的分层向量索引结构，并采用工作负载感知的向量放置机制，综合访问热度与图拓扑动态决定向量缓存位置与计算执行位置，在不压缩、不损失精度的情况下突破显存限制。
- 针对**高频向量插入与删除会导致图拓扑退化、检索精度下降，且更新开销巨大**的挑战——SVFusion 采用懒删除与局部拓扑感知修复相结合的策略，以轻量级逻辑标记降低删除延迟，并对受影响节点进行增量局部修复，配合异步全局整理，在持续更新下维持图质量与检索精度。
- 针对**查询与更新并发执行时难以保证数据一致性，且混合流工作负载下资源调度效率低**的挑战——SVFusion 提出细粒度锁与多版本机制的并发控制协议，并结合 CUDA 多流与自适应资源管理实现实时协同，保证跨层级存储的数据一致，同时让查询与更新高效并行。

对比工作：FreshDiskANN、HNSW、GGNN

`Peng Y, Yang D, Xie Z, et al. SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search[J]. arXiv preprint arXiv:2601.08528, 2026.`

论文阅读：https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-xi-du-%EF%BC%9B26%E3%80%91SVFusion%EF%BC%9A-mian-xiang-da-gui-mo-shi-shi-xiang-liang-jian-suo-de-%20CPU-GPU%20-xie-tong-chu-li-jia-gou.html
## 四、PilotANN（2026；开源）
PilotANN: Memory-Bounded GPU Acceleration for Vector Search

摘要：近似最近邻搜索（ANNS）已成为现代深度学习应用的核心技术，尤其在生成式模型中得到广泛应用。这类模型处理的数据集日益复杂、向量维度不断升高，对检索性能提出更高要求。现有纯 CPU 方案即便采用高效的图索引结构，也难以满足持续增长的算力需求；而纯 GPU 方案则面临严重的显存容量限制。
为此，本文提出 **PilotANN**—— 一种面向图索引 ANNS 的 CPU-GPU 混合协同系统，同时利用 CPU 大容量内存与 GPU 并行计算优势。系统核心创新在于将 top-k 检索流程分解为**精度递增、计算成本递减**的三级互补阶段：
1. 基于 SVD 降维向量，在 GPU 上加速子图遍历；
2. 在 CPU 上进行候选集精化；
3. 使用完整向量执行最终精确检索。
此外，本文提出**快速入口点选择**策略，优化检索起始点并最大化 GPU 利用率。实验结果表明，PilotANN 在亿级规模数据集上实现 **3.9–5.4 倍** 吞吐加速，可处理**超出 GPU 显存容量 12 倍**的超大规模数据集。

挑战-解决措施：
- 针对**GPU 显存容量有限，无法加载超大规模向量与图索引**的挑战——PilotANN 采用多阶段处理架构，通过子图采样与 SVD 降维将紧凑子图与压缩向量放入 GPU 执行快速遍历，完整图与原始向量保留在 CPU 内存，突破显存容量限制。
- 针对**ANNS 的距离计算计算密度低，难以充分发挥 GPU 算力**的挑战——PilotANN 提出快速入口点选择方法，利用类 GEMM 的高密度计算筛选高质量检索起点，提升 GPU 利用率并降低后续图遍历复杂度。
- 针对**混合 CPU-GPU 执行中数据迁移开销大、检索流程割裂**的挑战——PilotANN 将检索拆分为 GPU 子图导航、CPU 残差精修、全图精确检索三阶段，以极小的候选集传输实现高效协同，大幅提升整体吞吐。

对比工作：RUMMY、HNSW

`Gui Y, Yin P, Yan X, et al. Pilotann: Memory-bounded gpu acceleration for vector search[C]//Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2026: 348-358.`

论文阅读：未读
## 五、GustANN（2026；开源）
High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU

摘要：近似最近邻搜索（ANNS）已在各类场景中得到广泛应用。实际应用需要能够以高吞吐高效执行十亿级规模向量的检索。基于 SSD 的图索引 ANNS 系统具备实现这一目标的潜力，但 CPU 算力有限成为了性能瓶颈。
本文提出一种以 GPU 为核心、CPU 辅助的 ANNS 架构，并设计实现了**GustANN**系统 —— 一种面向十亿级规模、兼顾高吞吐与高性价比的图向量检索系统。本文通过三项关键技术达成设计目标：
1. 优化内存高效的 GPU 核函数，最大限度降低图检索中的 GPU 显存占用，提升 GPU 与 SSD 的并发处理能力；
2. 采用 CPU 辅助数据传输，缓解 GPU 端的 PCIe 带宽瓶颈；
3. 基于枢纽点检索实现多 SSD 间的负载均衡。
实验结果表明，与现有 ANNS 系统相比，GustANN 实现了**至少 2.50 倍的吞吐提升**，性价比（以美元 / QPS 衡量）提升**2.62 倍**。

挑战-解决措施：
针对 CPU 算力不足、GPU 架构存在缺陷的问题——GustANN 采用以 GPU 为核心、CPU 辅助的架构，将图遍历交给 GPU 执行，突破传统 CPU 架构的吞吐量瓶颈。
针对现有图检索算法占用显存过大、限制并发批次的问题——GustANN 设计了省显存的 GPU 图遍历核，放弃占用空间的访问表，改用先排序后并行去重的方式，以少量冗余计算换取大幅显存节省，提升 GPU 与 SSD 的并发能力。
针对 GPU 直接访问 SSD 导致 PCIe 带宽成为瓶颈、有效数据利用率低的问题——GustANN 提出 CPU 辅助传输机制，先将 SSD 页读到 CPU 内存，再只提取需要的节点发送到 GPU，节省超过 75% 的 PCIe 带宽。
针对多 SSD 上因单入口导致的负载不均衡、尾延迟差异大的问题——GustANN 提出枢纽点搜索，在 GPU 上维护小枢纽图，将查询分散到图的不同区域启动遍历，消除微观负载失衡。

对比工作：BANG、SPANN、DiskANN、Starling

`Jiang H, Guo H, Xie M, et al. High-Throughput, Cost-Effective Billion-Scale Vector Search with a Single GPU[J]. Proceedings of the ACM on Management of Data, 2025, 3(6): 1-27.`

论文阅读：未读
# GPU-only ANNS系统
## 一、Jasper（2026；开源；已读）
Jasper: ANNSQuantized for Speed, Built for Change on GPU

摘要：近似最近邻搜索（ANNS）是机器学习与信息检索领域的核心问题。GPU 为高性能 ANNS 提供了理想支撑：它具备海量并行计算能力，可高效完成距离计算，硬件易获取，且能与下游应用协同部署。
尽管具备这些优势，当前 GPU 加速的 ANNS 系统仍面临三大关键局限：
第一，真实场景中的数据集持续动态更新，需要快速批量插入数据，但多数 GPU 索引在新数据到来时，必须从零重建；
第二，高维向量会占用大量内存带宽，而现有 GPU 系统缺乏高效量化技术，无法在减少数据传输的同时，避免代价高昂的随机内存访问；
第三，贪心搜索中固有的数据依赖型内存访问，导致计算与内存操作难以并行，进一步拉低系统性能。
为此，本文提出**Jasper**—— 一款原生适配 GPU 的 ANNS 系统，兼顾高查询吞吐量与可更新性。Jasper 基于 Vamana 图索引构建，通过三大核心创新突破现有瓶颈：
1. 设计 CUDA 批量并行构建算法，支持无锁流式插入；
2. 实现 GPU 高效版 RaBitQ 量化，内存占用最高可压缩至原 1/8，且无随机访问开销；
3. 优化贪心搜索内核，提升计算利用率，实现更好的延迟隐藏、更高吞吐量。
在 5 个数据集上的实验结果表明：Jasper 的查询吞吐量较 CAGRA 最高提升 1.93 倍，屋顶线模型实测峰值利用率达 80%；索引构建速度平均比 CAGRA 快 2.4 倍，且支持 CAGRA 不具备的更新能力；对比此前最快的 GPU 版 Vamana 实现 BANG，Jasper 查询速度提升 19–131 倍。

挑战-解决措施：见摘要

对比工作：CAGRA、ParlayANN、GANNS、BANG

`McCoy H, Wang Z, Pandey P. Jasper: ANNS Quantized for Speed, Built for Change on GPU[J]. arXiv preprint arXiv:2601.07048, 2026.`

论文阅读：https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-yue-du-%EF%BC%9B26%E3%80%91Jasper%EF%BC%9A-mian-xiang-su-du-de-liang-hua-jin-si-zui-jin-lin-sou-suo-%EF%BC%8C-wei-%20GPU%20-dong-tai-geng-xin-er-gou-jian.html
# 流式ANNS负载
## 一、CANDOR-Bench（2026；开源；已读）
DB-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams

摘要
面向实时向量数据流的 **连续近似最近邻搜索（ANNS）** 是一个愈发关键却尚未得到充分研究的问题。在开放世界场景中 —— 数据分布发生偏移、噪声不断累积、并发访问频繁出现 —— 现有近似最近邻搜索算法最初为静态或简化流式场景设计，难以在数据写入延迟、检索精度与更新效率三者间实现平衡。尽管 ANN 基准测试集、大规模 ANN 基准测试集等工具已规范了静态或大规模场景下的评估标准，却无法刻画真实数据流中细微且高动态的数据更迭特性。
本文提出**CANDOR‑Bench（动态开放世界流下的连续近似最近邻搜索基准框架）**，该框架基于大规模 ANN 基准测试集构建，用于评估动态、开放世界环境下的内存型近似最近邻搜索性能。CANDOR‑Bench 支持高频数据写入（每秒最高数十万条向量）、自适应分布漂移建模（含模态偏移）、随机噪声注入与查询‑更新并发执行，且无需修改算法源码。
基于 12 个数据集与 19 种主流近似最近邻搜索算法开展评测后，我们发现：**尚无一种算法能在各类动态开放世界场景中持续兼顾高召回率、高吞吐量与高更新效率**，这对基于静态基准测试得出的既有研究假设提出了质疑。该性能差异本质上反映出流式场景固有的深层权衡关系。例如，更小的更新批次可提升数据时效性，却会带来更高的插入开销并降低检索精度。我们还发现，并发场景下的吞吐量往往受限于插入开销，而非查询延迟，这凸显出现有流式工作负载与原本针对离线构建优化的算法设计之间存在适配偏差。

`Wang M, Dong J, Wu Z, et al. CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams [Experiments & Analysis][J]. Proceedings of the ACM on Management of Data, 2026, 4(1 (SIGMOD): 1-27.`

论文阅读：https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-yue-du-%EF%BC%9BSIGMOD%2726%E3%80%91CANDOR-Bench%EF%BC%9A-dong-tai-kai-fang-shi-jie-liu-xia-nei-cun-xing-chi-xu-jin-si-zui-jin-lin-jian-suo-ji-zhun-ce-shi-%EF%BC%88-shi-yan-yu-fen-xi-%EF%BC%89.html
# ANNS图索引
## 一、CAGRA（2024；开源；已读）
Cagra: Highly parallel graph construction and approximate nearest neighbor search for gpus

摘要：近似最近邻搜索（ANNS）在数据挖掘与人工智能的诸多领域中发挥着关键作用，涵盖信息检索、计算机视觉、自然语言处理以及推荐系统等方向。近年来数据量急剧增长，穷举式精确最近邻搜索的计算代价往往高得难以承受，因此必须采用近似搜索技术。基于图的方法凭借均衡的性能与召回率表现，在近期的近似最近邻搜索算法研究中受到广泛关注。然而，尽管大规模并行通用计算已得到广泛应用，仅有少数研究探索了利用图形处理器（GPU）与多核处理器的算力优势。为弥补这一研究缺口，本文提出一种基于并行计算硬件的新型近邻图构建与搜索算法。借助现代硬件的高性能特性，本方法实现了显著的效率提升。
具体而言，本文方法在近邻图构建环节超越了现有基于中央处理器（CPU）与图形处理器（GPU）的同类方法，在大批量与小批量搜索场景下均实现更高吞吐率，同时保持相当的精度。在图构建耗时方面，本文所提方法CAGRA相较 CPU 领域顶尖实现之一的HNSW快2.2–27 倍；在90%–95% 召回率区间的大批量查询吞吐率上，本方法比 HNSW 快33–77 倍，比当前 GPU 领域顶尖实现快3.8–8.8 倍；针对单条查询，在 95% 召回率下本方法比 HNSW 快3.4–53 倍。

挑战-解决措施：
- 针对数据量激增、穷举精确最近邻搜索计算成本过高难以落地的问题 ——ANNS 领域需采用近似搜索技术替代精确搜索，在可接受精度损失下大幅降低计算开销。
- 针对图结构 ANNS 方法性能与召回率均衡，但 GPU、多核处理器等并行硬件利用不足的问题 —— 本文提出基于并行计算硬件的新型近邻图构建与搜索算法 CAGRA，充分释放现代硬件高性能算力。
- 针对现有 CPU/GPU 近邻图构建慢、查询吞吐低的问题 ——CAGRA 通过硬件感知的并行化设计，实现更快的图构建速度、更高的大小批量查询吞吐，同时保持精度与 SOTA 方法相当。

对比工作：GGNN、GANNS、HNSW、NSSG

`Ootomo H, Naruse A, Nolet C, et al. Cagra: Highly parallel graph construction and approximate nearest neighbor search for gpus[C]//2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 2024: 4236-4247.`

论文阅读：https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-yue-du-%EF%BC%9BICDE%2724%E3%80%91CAGRA%EF%BC%9A-gao-du-bing-xing-de-tu-gou-zao-yu-GPU-de-jin-si-zui-jin-lin-sou-suo.html
# 向量量化/压缩
## 一、TurboQuant（2025；开源）
Turboquant: Online vector quantization with near-optimal distortion rat

摘要：向量量化源于香农信源编码理论，旨在对高维欧几里得向量进行量化，同时最小化其几何结构的失真。我们提出 TurboQuant，可同时优化均方误差（MSE）与内积失真，解决了现有方法无法实现最优失真率的缺陷。该算法无需依赖数据、适配在线场景，在任意比特宽度和维度下，均能实现近似最优的失真率（与最优值仅相差一个小常数因子）。
TurboQuant 的核心原理为：对输入向量做随机旋转，使各坐标服从集中的贝塔分布；同时利用高维空间中不同坐标近似独立的特性，对每个坐标直接应用最优标量量化器。
研究发现，面向均方误差优化的量化器会导致内积估计产生偏差，为此我们提出两阶段方案：先用均方误差量化器处理向量，再对残差部分施加 1 比特量化约翰逊 - 林登斯特劳斯变换（QJL），最终得到无偏的内积量化器。
我们还从信息论角度，严格证明了所有向量量化器可达到的最优失真率下界，并证实 TurboQuant 的性能与该下界高度吻合，仅相差约 2.7 的小常数因子。
实验结果验证了理论结论：在 KV 缓存量化任务中，TurboQuant 采用每通道 3.5 比特可实现无损精度，2.5 比特仅造成轻微精度下降；在近邻搜索任务中，该方法的召回率优于现有乘积量化技术，且索引耗时几乎降为零。

挑战-解决措施：
* 现有向量量化方法难以同时优化均方误差与内积失真，达不到最优失真率，而且大多依赖数据，不适合在线场景。——为此提出 TurboQuant，采用不依赖数据的设计，适配在线应用；通过随机旋转输入向量，让坐标服从集中的贝塔分布，利用高维坐标近似独立的特性，对每个坐标单独使用最优标量量化器，在所有比特宽度和维度上实现接近最优的失真率。
* 面向均方误差优化的量化器会给内积估计带来偏差，无法满足无偏内积量化的需求。——对此设计两阶段量化策略，先用均方误差量化器压缩向量，再对残差部分做 1 比特量化约翰逊 - 林登斯特劳斯变换，得到无偏的内积量化器，消除估计偏差。
* 此前缺少向量量化最优失真率的理论证明，算法性能与理论最优之间的差距也不明确。——本文从信息论角度严格证明了所有向量量化器可达到的最优失真率下界，并验证 TurboQuant 与下界高度吻合，仅相差约 2.7 的常数因子，理论与性能都接近最优。
* KV 缓存量化容易造成精度损失，近邻搜索任务中现有乘积量化召回率偏低、索引耗时高。——实验表明，TurboQuant 在 KV 缓存量化中，每通道 3.5 比特可做到精度无损，2.5 比特仅轻微降质；在近邻搜索中，召回率优于现有乘积量化技术，索引耗时几乎降为零。

对比工作：
- **乘积量化（PQ）**：是高维近邻搜索的主流基线，依赖 k-means 离线训练码本，需要大量预处理，不适合在线场景；TurboQuant 在召回率上显著优于 PQ，且索引时间几乎为零。
- **RabitQ**：一种无预训练的网格类量化方法，缺乏向量化加速，在 GPU 上运行很慢，实际效率低于 TurboQuant。
- **KV 缓存量化相关对比方法**：包括 PolarQuant、SnapKV、PyramidKV、KIVI 等。这些方法或依赖数据校准、或属于 token 级压缩、或缺乏理论保证；而 TurboQuant 是在线、无数据依赖、有严格理论下界保证的量化方案，在长文本检索和生成任务上优于上述方法。

`Zandieh A, Daliri M, Hadian M, et al. Turboquant: Online vector quantization with near-optimal distortion rate[J]. arXiv preprint arXiv:2504.19874, 2025.`

论文阅读：未读
# 综述
## 一、SSD-based ANNS（2026；已读）
Disk-Resident Vector Similarity Search: A Survey

摘要：向量相似度检索（VSS）现已广泛应用于现代多模态数据检索系统，支撑文档检索、音乐识别、图像搜索、代码助手等各类应用场景。随着向量数据集规模飞速增长，且主存存储成本居高不下，向量相似度检索技术正逐步从**内存驻留**模式转向**磁盘驻留**模式。

内存驻留方案会将原始向量与索引结构全部存放于内存中，而磁盘驻留向量相似度检索则将原始向量与细粒度索引结构置于二级存储，仅在内存中保留轻量级组件。因此，磁盘驻留向量相似度检索必须充分考量数据访问开销（即 I/O 开销）与数据块层级布局，其索引组织方式与查询执行策略，与内存驻留检索系统存在本质差异。

本文首次对磁盘驻留向量相似度检索技术进行系统性综述，该技术是现代大规模数据系统中一项至关重要且极具挑战性的研究课题。文章提出统一分类框架，依据检索过滤所采用的核心索引结构，将现有技术划分为基于倒排聚类（IVF）、基于图、基于树形三大范式。针对每一类范式，进一步从索引构建、面向数据块的布局设计、查询执行策略、数据更新机制等关键技术模块拆解整体设计逻辑。此外，本文汇总了常用基准数据集，为可复现的性能评测提供支撑，并梳理了当前研究尚存的难点挑战与未来极具潜力的研究方向。

`Song Y, Li H, Wu Z, et al. Disk-Resident Vector Similarity Search: A Survey[J].`

论文阅读：https://waibibab-cs.github.io/post/%E3%80%90-lun-wen-xi-du-%EF%BC%9B26%E3%80%91-zong-shu-%EF%BC%9A-chang-zhu-ci-pan-de-xiang-liang-sou-suo.html
