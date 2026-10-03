# PyPTO Pro → Triton Ascend 迁移计划

目标：保留现有 Loop Engineering harness，把开发目标改为昇腾 NPU 上的 Triton（triton-ascend）。本计划先盘点，再实施，最后记录检查结果。

## 1. 保持的流程

仍使用 Stage1 → 1.5 → 2，以及 Stage3–10 的原有路由。保留编译/精度/性能失败处理、Stage9 六类场景、P0/P1/P2、同轮修正、人工介入、history、最佳版本、±5%经验、x/y语义退出、退步回退及全部case判定规则。

## 2. 逐项修改

- [x] **运行入口与硬件检测**：`orchestrator.py`、`lib/agent_runner.py`、`lib/bench_parser.py` 等去掉 `pypto_pro` 编程模型探测和 `TILE_FWK` 依赖，保留 Ascend/CANN 设备参数读取。检查 Triton Ascend 后端，明确设备选择；更新日志、框架标签和实际依赖。
- [x] **示例工程**：默认 `example/` 改为 cann-bench 的 `examples/triton_ascend_cann_example/`。本项目提供可读的 Triton Ascend 示例与验证说明；仍按 `cann_bench` 导出任务规定的接口，task中的case/golden保持不动。
- [x] **Stage1/2/3**：需求、首版生成、修改提示词使用 `triton`、`triton.language as tl`、`@triton.jit`、grid、mask、stride与分块；去掉 `pl.*`、`vf.*`、TileGroup等专属要求。自测、连续调用、范围约束及交接文件保持。
- [x] **Stage4/5/6/7/8/9/10**：更新角色身份、安装/import验证、JIT编译解释、profiling和搜索建议。保留 CANN-Bench 报告接口、`kernel_details` 口径、Stage9裁剪与知识写入机制。
- [x] **融合知识与profiling skill**：改写 `knowledge/arch_programming_guide.md`、反作弊参考、profiling指南和融合方法，核对 Ascend Triton 实际可表达能力。F1–F10/变体ID、Jev评分/Top N结构保持；不能把GPU专用能力或PyPTO实测当作Triton事实。skill目录改为 `skills/triton-profiling-analysis/` 并同步引用。
- [x] **配置和使用文档**：更新README、环境迁移指南、示例说明、活动设计文档；配置仍只有 `config.yaml`，保留连接信息、密钥、x/y等参数。仅更改确有必要的框架路径或说明，不输出密钥。项目磁盘根目录暂保持原路径，避免破坏外部引用。
- [x] **旧材料与恢复边界**：旧 PyPTO 源码示例、报告、知识和教程按用户要求从仓库移除，由用户另存备份。新性能记录包含框架/后端标识，避免跨框架比较或恢复最佳；旧工作目录仍不能自动作为 Triton 任务续跑。
- [x] **测试与复查**：更新合成fixture的目标框架和示例路径；增加无PyPTO依赖、Triton语法/示例包、设备选择、跨框架隔离测试。运行完整离线回归，复核所有原有路由、人工介入、经验与退出规则；扫描活动文件中的PyPTO残留及断链。

## 3. 验证边界

本地Windows负责静态检查和离线回归，不把这些检查说成真实NPU评测。真正编译、精度和性能验收需在安装兼容版本CANN、torch_npu和triton-ascend的昇腾机器上运行；文档给出命令。采用本地cann-bench的Triton示例接口，外部cann-bench仓库本轮只读。

## 4. 资料依据

- 本地 `cann-bench/examples/triton_ascend_cann_example/`：本项目的标准部署接口。
- [Triton Ascend快速开始](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/quick_start.md)：昇腾环境与运行方式。
- [Triton Ascend安装指南](https://github.com/Ascend/triton-ascend/blob/main/docs/en/installation_guide.md)：按安装版本核对兼容矩阵，不沿用旧PyPTO环境假设。
- [Triton向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)：基本JIT/grid/load/store写法；设备相关操作按昇腾适配。

## 5. 实施与测试结果

已按上述清单实施，主要位置如下：

- **目标与启动**：`lib/framework_target.py` 集中声明 `Triton / triton-ascend`；`orchestrator.py` 刷新芯片信息，验证实际 driver 后端 `npu`，记录四项 Python 运行时版本和 CANN 版本文件指纹。
- **新任务与恢复**：新 work 保存 `workflow_target.json`。已有状态但没有 Triton 身份的 work、仍导入旧框架的初始实现、错误的 example 来源会明确报错；旧文件不自动改写。
- **代码与材料**：全部活动 roles、融合知识与 profiling skill 改为 Triton Ascend；`examples/triton_ascend_example/` 提供最小融合算子、标准 wheel 和真实 NPU 自测入口。默认模板来自外部 cann-bench，原 task/case/golden 不改。
- **记录与日志**：新经验保存 `framework=Triton`、`backend=triton-ascend`、运行时版本、芯片及 CANN 来源；旧报告保持原身份。比较口径加入这些信息，CANN 指纹每次比较重新读取。日志名统一为 `triton-ascend-*`，日志文件位置不变。
- **旧材料**：按用户要求移除仓库内的旧框架历史材料，由用户另存备份；旧实测数字不作为 Triton 成绩。项目物理目录名保留。`config.yaml` 仅更新说明文字，实际参数和连接信息不变。

专项复查修正了恢复时沿用旧芯片、CANN 变化未进入比较、安装与 import 使用不同环境、在源码目录误验安装包、过期反作弊编号、固定 case 数和仅靠核名判断来源等问题。Stage 路由与阈值没有更改。

**迁移时的离线验证**：完整回归 **548 项全部通过**（迁移前基线 521 项），无失败或跳过；网络、外部子进程和真实密钥访问拦截计数均为 0。报告中的源码指纹与当时的代码一致。Python 语法、文档链接、10 类/24 变体 ID 与章节引用及 `git diff --check` 均通过。测试报告保存于 `output/jev_tests/results.json`。

**真实 NPU 验证尚未执行**：按 [环境迁移指南](environment_migration_guide.md) 配环境，再按 [Triton 示例说明](../examples/triton_ascend_example/README.md) 执行 `self_test.py`，最后运行一次完整 workflow。离线测试不等于真实 JIT、精度或性能验收。

按用户要求删除历史目录后，迁移契约、融合方案及评测解析的 **37 项专项检查通过**；历史目录及相关失效链接已移除，测试不再依赖旧归档。结果见 `output/jev_tests/archive_removal_checks.json`。

## 6. 全面复查与补漏

活动目录为 `roles/`、`lib/`、`knowledge/`、`skills/triton-profiling-analysis/` 和两个 Triton/Jev 示例目录。每次运行的 `<work>/example` 使用外部 cann-bench 的 Triton Ascend 标准模板，本地最小融合示例另作自测参考。根目录的旧名字只表示磁盘位置。

这次复查发现并修复以下漏项：

- **生成目录**：Stage2 提示词仍指定 `impl/<算子>_impl.py`，与包结构冲突；现在统一要求在 `impl/cann_bench/` 实现并从 `__init__.py` 导出接口。Stage2/3 日志记录整个 `impl/` 工程，支持多文件实现。零分修复提示也统一为非 editable 安装。
- **配置实际接线**：各 Agent 现在使用本次 `--config` 中的 CLI、设备编号及 CANN 路径；Stage3/9 的辅助调用同样生效，Stage6 也收到该配置路径。
- **运行环境**：芯片探测、Agent 与性能评测统一使用完整 CANN 环境，保留可见设备映射；配置切换或再次启动会刷新缓存。`set_env.sh` 执行失败时明确报错，不再掩盖失败继续运行。
- **恢复目录**：恢复时检查 `work/task` 与 `--task-dir` 是否指向同一目录，防止开发读一套需求、评测跑另一套 case。
- **文档与目录说明**：清理 README 中已不存在的目录项、旧框架实测表和过期 Stage9 建议格式；TODO-001 改为当前已解决事项与待验证内容。环境指南不再要求切回旧开发分支；内部文档链接改为相对路径，并忽略新建的 `.venv/`。

这些修改没有改变 Stage 路由、语义退出阈值、人机介入、history 或经验记录规则。真实 `config.yaml` 参数未修改。

**本轮验证**：完整离线回归 **559 项全部通过**，无失败或跳过；网络、外部子进程和真实密钥访问拦截计数均为 0。新增 11 项测试，覆盖实际 Stage → Agent 调用的配置接线、设备编号、同路径重复启动、CANN 环境、task 目录与各开发场景的提示词/日志目录。已有只读 task 链接测试补齐“目标目录存在”的模拟，校验规则没有放宽。84 个 Python 文件语法检查、39 条本地文档链接与 `git diff --check` 均通过。最新报告为 `output/jev_tests/results.json`；真实 NPU 验证边界同第 3 节。
