# 论文目录：融合算子生成的三个挑战

2026-09-25。**本次共下载 6 篇，分为三类；EAGLE 是重点。** 主文见[三个挑战与解决思路](../融合算子生成_三个挑战与解决思路.md)。这里仅归档支撑论证的论文，没有下载各篇的全部参考文献。

建议先看 **Welder 的融合冲突例子 → EAGLE 的等价性验证 → OATest 的融合上下文案例**；其余三篇作为补充。

## 01 等价性与测试：Lin Tan 课题组

来源：[Lin Tan 论文主页](https://www.cs.purdue.edu/homes/lintan/papers.html)。[本类阅读笔记](01_等价性与测试/论文阅读笔记.md)。

1. **[EAGLE: Creating Equivalent Graphs to Test Deep Learning Libraries](01_等价性与测试/01_EAGLE_ICSE2022.pdf)**，ICSE 2022，13 页，**重点阅读**。用等价计算图检测优化、布局等引发的结果不一致。支撑挑战二：优化后的语义仍须检验。重点 §3.1、§5.2、§6，PDF 第 3–5、8、10 页。[作者原始 PDF](https://www.cs.purdue.edu/homes/lintan/publications/eagle-icse22.pdf)。
2. **[DocTer: Documentation-Guided Fuzzing for Testing Deep Learning API Functions](01_等价性与测试/02_DocTer_ISSTA2022.pdf)**，ISSTA 2022，13 页。依据文档约束生成输入，补充说明 shape、dtype、参数依赖与边界覆盖的重要性。辅助挑战二，非融合生成论文。[作者原始 PDF](https://www.cs.purdue.edu/homes/lintan/publications/docter-issta22.pdf)。

## 02 编译与优化测试：Junjie Chen 课题组

来源：[Junjie Chen 主页](https://sites.google.com/site/junjiechen08/)。[本类阅读笔记](02_编译与优化测试/论文阅读笔记.md)。

3. **[Optimization-Aware Test Generation for Deep Learning Compilers（OATest）](02_编译与优化测试/01_OATest_ICSE2026_2511.18918v1.pdf)**，ICSE 2026，13 页。把优化模式放进不同上下文，发现融合相关编译缺陷。支撑挑战二：局部正确不代表组合仍正确。重点 §4.2、图 5，PDF 第 8 页。[原始 PDF，2511.18918v1](https://arxiv.org/pdf/2511.18918v1)。
4. **[Fuzzing Deep Learning Compilers with HirGen](02_编译与优化测试/02_HirGen_ISSTA2023_2208.02193v5.pdf)**，ISSTA 2023，13 页。覆盖算子连接、shape、dtype 与等价改写，包含前序变换影响后续 FuseOps 的案例。辅助挑战二。[原始 PDF，2208.02193v5](https://arxiv.org/pdf/2208.02193v5)。

## 03 融合与资源权衡：补充的直接融合研究

这两篇不属于上述课题组，用于直接解释融合生成的执行与性能困难。[本类阅读笔记](03_融合与资源权衡/阅读笔记.md)。

5. **[Welder: Scheduling Deep Learning Memory Access via Tile-graph](03_融合与资源权衡/01_Welder_OSDI2023.pdf)**，OSDI 2023，19 页（含会议封面）。用分块数据流组织算子内部与算子之间的复用。直接支撑挑战一，并解释挑战三的资源权衡。重点 §2、图 2，PDF 第 4 页。[会议入口](https://www.usenix.org/conference/osdi23/presentation/shi)、[原始 PDF](https://www.usenix.org/system/files/osdi23-shi.pdf)。
6. **[FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads](03_融合与资源权衡/02_FusionStitching_arXiv2021v2.pdf)**，arXiv 2020，下载 2021 v2，14 页。讨论复用、通信、并行度与拆分融合的取舍。直接支撑挑战一，辅助挑战三。重点 §2.3，PDF 第 3 页。**此版本是预印本，不标为会议论文。**[原始 PDF，2009.10924v2](https://arxiv.org/pdf/2009.10924v2)。

## 使用边界

- 论文用于解释困难和已有解决方式，不能据此声称所有 Agent 都无法生成融合算子，或本项目已经解决全部问题。
- EAGLE 等测试论文不证明本项目性能更好；Welder 等融合论文不证明本项目的跨轮决策更可靠，均需自己的实验。
- PDF 已验证可读取；重点案例页已核对。页码统一指文件页序。未复现论文实验，也未运行作者代码。
- OATest 网页摘要与本地 PDF 的缺陷统计存在差异，笔记按本地版本定位，不混用数字。
