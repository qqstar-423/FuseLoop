# Triton Ascend Workflow

> 多智能体协作的 Triton Ascend 算子自动开发与迭代优化框架

迁移说明见 [Triton Ascend 迁移计划](docs/triton_ascend_migration_plan.md)，环境配置见 [环境迁移指南](docs/environment_migration_guide.md)。当前生成目标固定为 Triton Ascend；磁盘上的 `pypto-pro-workflow` 只是保留的目录名，不决定生成框架，也不要求安装 PyPTO。

## 📖 项目简介

Triton Ascend Workflow 是一个面向昇腾 NPU 的**多智能体算子自动开发框架**，基于 Triton Ascend 编程范式，通过 3 个 CLI Agent（CANNBot 写代码、Kerminal 编译评测、Hermes 搜索优化）、Jev 融合方案选择节点和 1 个确定性控制平面（orchestrator）协同工作，实现算子从需求分析→融合方案选择→代码编写→编译部署→精度验证→性能优化→报告生成的**全流程自动化迭代**。

| 维度 | 说明 |
|------|------|
| 目标框架 | Triton Ascend（triton-ascend，昇腾 NPU） |
| 评测工具 | cann-bench（精度+性能评测） |
| Agent | CANNBot（GLM-5.3-Flash）、Kerminal（kernelcat1.0）、Hermes（GLM-5.3-Flash）、Jev（融合方案概率评估） |
| 迭代阶段 | 原 10 个 stage + Stage1.5，最多 20 轮自动迭代 |
| 核心特性 | 跨轮记忆（history）、Tech Lead 方向指导、反作弊知识库、代码版本追溯 |

## ✨ 核心特性

- **融合方案选择**：需求分析后新增 Stage1.5，Jev 根据需求和硬件为融合方法评估概率，程序选取前 n 个组成 JSON 融合方案库，传给 Stage2/3/7/8/9
- **阶段流水线**：需求分析→融合方案选择→首版实现→编译部署→精度评测→性能评测→Profiling→搜索优化→Tech Lead 总结→代码修改→最终报告
- **跨轮记忆（history 机制）**：Tech Lead 每轮提炼 insights/ledger，追踪最慢 case，指导下一轮方向，避免重复失败
- **融合方向追踪**：`fusion_kernel_strategy` 累加字段记录每轮融合方案的尝试（真融合/假融合/片上直传 vs HBM 中间读写），stage9 填写，stage3 读取
- **语义退出与最佳实现**：有效评测后保存代码、融合方案和报告快照；全部 case 达标且 x 次有效改进累计不足 5% 时交付最佳达标实现
- **已有实现应急重跑**：`--init-impl` 复制指定代码及其对应开发材料，跳过 Stage1/1.5/2，直接从 Stage4 编译评测；不影响正常新写流程
- **反作弊知识库**：实际错误处理表与开发自检清单，结合源码、kernel CSV 和正式报告分析
- **芯片自动检测**：从 CANN 编译器 tbe API 直接读取 UB/L1/L0A/L0B/L0C/核数真实值，注入所有 stage，禁止写死 tiling 参数
- **代码版本追溯**：每轮迭代前自动备份 impl 到 `operator_iter/`
- **性能评测代码直执**：stage6 由 orchestrator 直接 subprocess 调用 cann-bench，不经 agent，自动重试+旧报告过滤

stage6 显式选择 cann-bench 的 `kernel_details` 性能策略，直接复用工具产出的 baseline、HAP 和评分，不另行测量 baseline。该策略汇总 kernel 执行耗时，不包含 kernel 之间的同步与调度间隔，因此不代表完整调用耗时；旧 `trace_view` 成绩不能与此策略的成绩直接比较。

## Stage1.5：Jev 融合方案选择

**阶段顺序：[Stage1 需求分析](roles/n1_stage1_requirements_analysis.md) → [Stage1.5 Jev 融合方案选择](roles/n1_stage1.5_jev_fusion_selection.md) → [Stage2 首版实现](roles/n1_stage2_first_impl.md)。** Stage1.5 的 role 文件说明输入、职责和输出；实际英文请求由程序组织并调用 Jev。

[Jev 控制台](https://console.typesafe.ai/home)

Jev 已接入 Stage1 和 Stage2 之间。输入包括 `knowledge/fusion_method.md` 原文、该文档拆解成的 `knowledge/fusion_options.json`（F1–F10，含 24 个变体）、Stage1 新增的 `<work>/fusion_requirements.en.json` 和当前硬件信息。该需求 JSON 用紧凑英文保留算子语义、接口、case 特征、精度与融合约束；程序先估算本次可用字节数并传给 Stage1（绝对上限 6000 字节），完整需求分析仍保留在 ANALYSIS.md。Jev 对十类方法分别给出独立适用性概率，不要求概率之和为 1；程序按概率排序，取前 n 个保存为本次运行的 **JSON 融合算子库**。

每次运行的 `<work>/example` 指向 `paths.cannbench_repo` 下的 `examples/triton_ascend_cann_example/`；缺失时停止并提示安装完整示例。本仓库的 `examples/triton_ascend_example/` 是额外的最小融合演示和 NPU 自测入口，不是这个默认软链接的来源。任务的 `proto.yaml`、case、golden 和评测入口保留；需要改的是候选实现及运行环境。Stage4 构建 wheel，Stage3 自测和 Stage5 真实调用还要覆盖首次 JIT。

Stage1.5 的调用顺序是：**读取材料 → 复用 Kerminal 翻译 → 程序组织英文 JSON 并校验 → Jev 评分 → 选择前 n 个方案**。融合方法原文、方案库深层描述、需求和硬件中的非英文内容都进入翻译检查；已有英文内容保留。JSON 结构、数值和方案 ID 由程序保留，译文中的数字和技术标识也要校验。实际发送的请求必须通过英文检查和翻译后的大小检查；翻译失败、残留中文或超限时停止，不发送混合语言请求，也不使用旧概率冒充新结果。底层 Jev 客户端同样检查所有调用入口的英文输入。

在 `config.yaml` 的 `fusion_selection.top_n` 修改 n，默认 `3`；翻译复用 `agents.kerminal.cli`，超时由 `fusion_selection.translation_timeout_seconds` 配置（默认 240 秒）；Jev 连接参数仍使用 `jev` 段。独立调用与离线测试入口继续保留。

| 接收节点 | 方案库的用途 |
|---|---|
| Stage2 首版实现 | 按最高概率方案设计首版；如硬件/框架客观不支持，给出证据并选择后续可实现候选 |
| Stage3 修改优化 | 理解现有方案与备选方向，结合评测证据执行既有修改指令 |
| Stage7 性能分析 | 对照候选数据流和资源条件分析瓶颈 |
| Stage8 搜索优化 | 围绕瓶颈搜索候选方案的实现方法，核实当前硬件适用性 |
| Stage9 Tech Lead | 结合候选方案、历史尝试和实测证据审查融合方向 |

新任务的五个节点均读取 `<work>/fusion/fusion_library.json`。概率是初始选择参考，实测证据优先；跨核存储、同步等能力必须核实，共享 L2 Cache 不能直接视为 DSM。初始 Jev 库由程序维护；Stage2/3 在 `develop/iterN/fusion_library.json` 记录本轮实际选择、改动及未评分的新方案，不改写初始概率。Stage3 另写《融合方案选择决策依据.md》，与自测报告分开，传给 Stage7/8/9。

运行产物位于 `<work>/fusion/`：`fusion_library.json` 保存前 n 个完整候选及概率，`ranking.json` 保存所有方法的概率，`jev_request.json` 和 `jev_response.json` 保存请求与响应，便于追溯。

`<work>/fusion/translation/` 保存 `translation_input.json`（翻译前材料）、`english_inputs.json`（英文材料）、`translation_manifest.json`（输入与译文指纹）及 Kerminal 翻译记录。输入未变且译文校验通过时复用翻译缓存；原始方案库和后续五个节点使用的可读方案说明仍保留原文。

断点恢复会复用输入一致的概率结果；输入变化时重新评估，只修改 n 时直接重选。完整 ANALYSIS.md 的哈希也参与缓存检查。旧工作目录若已进入迭代，且既没有新增需求 JSON、也没有 `fusion/` 目录，保留原有迭代并记录警告，不追溯补跑 Stage1.5；若已存在 `fusion/` 却缺少需求，则停止并报错，避免误用不完整的旧结果。

| 文件 | 功能 |
|---|---|
| `lib/fusion_selection.py` | Stage1.5 请求构建、概率校验排序、前 n 候选持久化、缓存复用与下游 prompt 注入 |
| `lib/jev_translation.py` | 复用 Kerminal 翻译输入文本，校验英文、结构与数字，缓存英文材料 |
| `lib/jev_client.py` | 读取文件正文、组装JSON、检查请求大小、调用Jev并返回选项概率等原生结果 |
| `lib/kerminal_rpc.py` | 通过本地Kerminal CLI翻译文本字段 |
| `tools/jev_smoke_test.py` | 独立调用入口，支持自定义源码、英文证据和问题JSON |
| `tools/kerminal_translate_smoke.py` | 翻译与数值保真检查示例 |

连接参数和本机密钥统一放在 `config.yaml` 的 `jev` 段，密钥读取顺序为环境变量 → `jev.api_key`。不再需要 `config.local.yaml`；只有旧配置显式保留 `credentials_file` 时才兼容读取该旧文件。离线验证运行 `.venv-jev\Scripts\python.exe tools/run_jev_offline_tests.py`，结果写入 `output/jev_tests/results.json`，不会读取真实密钥配置。

## 🧭 适用场景

- 昇腾 NPU 上的 Triton Ascend 融合算子开发与性能优化
- cann-bench 评测驱动的算子迭代优化
- 多 Agent 协作的算子开发流程自动化研究

## 🔧 芯片自动检测

orchestrator 启动时自动检测当前 NPU 芯片的真实硬件参数，写入 `<work>/device_info.json`，并注入到所有 stage 的 prompt 中，确保 agent 按实际硬件规格设计 tiling 参数。**不使用任何硬编码映射表**——所有参数从设备实际读取，读不到则退出。

检测流程：

1. 通过 `torch_npu.npu.get_device_properties(device_id)` 获取配置中逻辑设备的 SoC 型号（如 `Ascend950PR_9579`）、Cube/Vector 核数、L2 Cache 大小
2. 通过 CANN 编译器内部接口 `tbe.common.platform.get_soc_spec` 读取片上内存真实值（UB/L1/L0A/L0B/L0C/CORE_NUM），这是编译器编译 kernel 时依赖的同一份数据
3. 保留设备报告的芯片名称，并补充架构代号（如 `dav-3510`，纯名称映射不涉及硬件参数）
4. 导入 Triton，验证实际运行后端为 `npu`，记录 torch、torch_npu、triton 与 triton-ascend 版本；仅能导入同名上游包不足以通过
5. 校验关键参数（soc_version/UB/L1/CORE_NUM/L0A）全部获取到，任一缺失则打印排查步骤并 `sys.exit(1)` 退出

手动验证命令：
```bash
# 查看 SoC 型号
python3 -c "import torch,torch_npu; print(torch_npu.npu.get_device_properties(0).name)"

# 查看片上内存参数（需要 CANN Toolkit）
python3 -c "
from tbe.common.platform import get_soc_spec, set_current_compile_soc_info
set_current_compile_soc_info('Ascend950PR_9579')  # 替换为你的 SoC 型号
for k in ['UB_SIZE','L1_SIZE','L0A_SIZE','L0B_SIZE','L0C_SIZE','CORE_NUM']:
    v = get_soc_spec(k)
    print(f'{k} = {v} ({v//1024} KB)' if isinstance(v,int) and v>1024 else f'{k} = {v}')
"
```

如果自动检测失败（如 CANN 未安装完整），可在 `config.yaml` 的 `hardware` 段手动填写全部参数。

## 运行指南

### 第一步：配置 config.yaml

**仓库直接提供唯一的 `config.yaml`，并由 Git 跟踪，日常只修改这一份运行配置。** 新机器拉取仓库后，调整其中的实际路径和 `jev.api_key` 即可；仍可设置 `TYPESAFE_API_KEY` 环境变量，优先于文件中的密钥。

```bash
# 一键获取当前机器的所有路径
echo "cannbot:      $(which cannbot)"
echo "kerminal:     $(which kerminal)"
echo "hermes:       $(which hermes)"
echo "cann toolkit: $(ls -d ~/Ascend/cann-*)"
echo "node bin:     $(dirname $(which node))"
echo "arch:         $(uname -m | sed 's/x86_64/x86_64-linux/;s/aarch64/aarch64-linux/')"
```

config.yaml 中需要改的字段（共 7 处）：

```yaml
agents:
  cannbot:
    cli: "<which cannbot>"
  kerminal:
    cli: "<which kerminal>"
  hermes:
    cli: "<which hermes>"

paths:
  cannbench_repo: "<cann-bench 仓库的绝对路径>"

cann:
  toolkit_path: "<ls ~/Ascend/cann-*>"
  arch: "<aarch64-linux 或 x86_64-linux>"
  node_bin: "<dirname $(which node)>"
```

`cann.driver_path` 通常不用改（默认 `/usr/local/Ascend/driver`）。`hardware` 段不用改（orchestrator 自动检测）。

另外确认 `jev` 段的服务连接与密钥配置；用 `fusion_selection.top_n` 控制 Stage1.5 保留的融合方案数，默认 3。

### 第二步：验证环境

```bash
# NPU 可用
npu-smi info
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"

# CANN Toolkit 完整
python3 -c "from tbe.common.platform import get_soc_spec; print('tbe OK')"

# 三个 Agent 可用
cannbot --version
kerminal --version
hermes --version
```

### 第三步：运行

**从零开始写算子**

```bash
cd pypto-pro-workflow
python3 orchestrator.py \
  --task-dir /path/to/cann-bench/bench_lab/<bench>/<level>/<operator>
```

**带融合方向提示（推荐）**

`--optimize-hint` 告诉 agent 用什么融合策略，会注入到 stage1 分析和 stage2 实现中：

```bash
python3 orchestrator.py \
  --task-dir /path/to/cann-bench/bench_lab/multimodal-fusion-bench/level3/fused_rmsnorm_pos_qkv_qknorm \
  --optimize-hint "用 Triton Ascend 实现 RMSNorm、位置编码、QKV 投影和 Q/K 归一化；先核对每个 case 的形状与精度要求，再选择可实现的融合方式。减少中间数据搬运，以实测选择单 kernel 或多 kernel 方案。"
```

hint 怎么写：说清楚计算步骤、精度约束和优化目标。是否单 kernel、如何切块和调度，由 agent 根据当前 Triton Ascend 能力、芯片资源和评测证据决定。

方向说明较长时保存为 UTF-8 文件，用 `--optimize-hint-file 方向说明.md` 替换 `--optimize-hint "正文"`；两者互斥，注入位置和作用相同。

**修改 workflow 后，用已有算子应急重跑**

```bash
python3 orchestrator.py \
  --task-dir /path/to/task \
  --init-impl /path/to/old_work/impl
```

`--init-impl` 以你指定的代码为准，不要求存在 `selection/best.json`，也不会自动改选历史最佳。适用于改动 workflow 工程后，复用已生成算子重新运行。首次直接走 **Stage4 → Stage5 → Stage6 → 原有后续流程**，不调用 Stage1、Jev 或 Stage2，也不在首次评测前让 cannbot 修改代码。

- 复制 `ANALYSIS.md`、`fusion_requirements.en.json`、`device_info.json`、整个 `fusion/`，以及指定 `impl/` 对应那一轮的开发材料（方案、融合选择、自测报告、结构化结果和日志），放到新 `develop/iter0/`。如果代码已经经过 Stage3 优化，就取与代码及证据哈希匹配的开发轮次，不能拿首版报告代替。
- 不复制旧 `eval/`、`build/`、`profile/`、`search/`、history、成功/失败经验、最佳记录、退出计数或人工意见。新任务的性能和历史从本次正式评测重新记录。
- 启动时仍检查实际硬件与运行环境，但只在本地核对导入评分，**不重新调用 Jev**。缺必要材料、任务或硬件不匹配时明确停止，不静默重新开发。候选概率和 Top N 沿用导入库。
- 已匹配的开发自测证据可以继承；代码或文档对不上时只作参考，不冒充自测已通过，也不放宽最佳实现的原有入库条件。
- `init_impl_manifest.json` 保存来源、复制文件及哈希；日志列出复制的目录和跳过的阶段。源目录不改动，同秒启动也会创建不同的新目录。

不能与 `--work-dir` 同时使用；中断后用 `--work-dir 新work目录` 恢复，仍保留应急模式。可选 `--optimize-hint` 只记录为后续优化参考，首次仍先评测指定实现。

**断点续跑**

如果 workflow 中途中断（Ctrl+C / 超时 / 容器重启），用 `--work-dir` 指定已有的 work 目录继续：

```bash
python3 orchestrator.py \
  --task-dir /path/to/task \
  --work-dir work/fused_rmsnorm_pos_qkv_qknorm_20260916_131832
```

orchestrator 会从 `.state.json` 记录的断点继续。impl / eval / history 等已有数据保留。新任务记录 `workflow_target.json` 标识 Triton Ascend；已有旧框架实现或状态的 work 不直接续跑。请创建新任务重新实现、评测，不能只修改身份文件来导入旧成绩。

**控制迭代轮数**

默认最多 20 轮（`config.yaml` 的 `workflow.max_iterations`）。可以用 `--max-iter` 覆盖：

```bash
# 先跑 1 轮验证环境没问题
python3 orchestrator.py --task-dir /path/to/task --max-iter 1

# 正式跑 20 轮
python3 orchestrator.py --task-dir /path/to/task
```

### --optimize-hint 在不同场景下的注入方式

| 场景 | 命令 | 处理方式 |
|------|------|---------|
| 新写算子 | `--optimize-hint "融合方向..."` | Stage1 分析可行性，Stage2 按该方向开发 |
| 已有实现应急重跑 | `--init-impl ... --optimize-hint "后续方向..."` | 跳过 Stage1/1.5/2；保存提示供后续分析与优化，先评测指定代码 |
| 不传提示 | — | 无额外方向 |

### 退出条件

性能达标（所有 case speedup ≥ 1.0）后**不直接退出**：

1. 有效成绩成为基线，再经历 x 次有效改进；窗口两端的历史最佳 avg_speedup 累计提升小于 5% 时，程序退出并将最佳达标快照交给 Stage10。
2. 仍有 case 未达标时保持 Stage6 → 7 → 8 → 9 → 3；连续 y 次有效改进累计不足 5% 提示 Stage9 重点审查候选方案。开启人工咨询时，第 3 个新有效轮次触发该条件后先咨询人，再形成最终修改意见。未达标 case 持续改善时可保留方案。
3. x/y 分别由 `workflow.semantic_exit.passed_window` / `underperforming_window` 配置，默认均为 3；需基线加 3 份有效成绩。达标状态切换重置窗口；失败轮不计数，退步后的恢复不当作新提升，恰好 5% 不属于停滞。
4. P0/P1/P2、编译/精度/反作弊失败路由、最大迭代上限保持。Stage9 的 `exit_decision` 不再控制退出，也不再因多 kernel 或 HBM 中间读写强制换方案。

Stage9 仍按原逻辑记录提升 ≥5% 的成功经验和退步 ≥5% 的失败教训，包括满足语义退出的那一轮。

配置中的 **x = `passed_window`（场景1，全部达标）**，**y = `underperforming_window`（场景2，至少一个 case 未达标）**。它们控制有效迭代窗口；日志中的“第几次进入场景”是触发次数，人工咨询的“第 3 次”又是独立计数，三者不要混淆。

三个判断的区别、代码现状和后续建议见[性能判断规则](docs/workflow_performance_decision_rules.md)。

单轮退步仍比较本轮与最近一轮 `avg_speedup > 0` 的结果，百分比保留一位小数后 `≤ -5%` 触发。**本轮全部 case ≥1** 时，Stage9 先写退步教训，程序再恢复同口径最佳达标代码，Stage3 从该基线继续修改；**本轮仍有 case <1** 时只记录知识，不自动恢复代码。恢复不新增评测、不改变原成绩或停滞窗口。若当前评测未通过版本绑定或快照校验，不自动恢复。

两类停滞及退步处置在 `log/workflow.log`、`log/state_transitions.log` 用 `=====` 醒目标记。停滞记录窗口、累计提升和“第几次进入本场景”；两个场景分别累计，同一轮恢复不重复计数，与人工咨询计数独立。次数保存在 `selection/semantic_events.json`；退步处置保存在 `selection/rollbacks/iterN/record.json`，包含前后性能、最佳来源、Stage9 决策和知识地址。实际恢复前的代码另存 `failed_impl/`，旧证据绑定保存在 `failed_binding.json`；Stage9/3 提示词会明确退步报告与恢复基线的区别。

### 人类参与与 Stage9 场景裁剪

公共配置已开启 `workflow.human_review.proactive_enabled` 和 `consultation_enabled`，可分别关闭；旧自定义配置缺少此段时保持无人值守行为。在项目根目录的**另一个终端**提交，不打断当前节点：

```bash
python tools/human_review.py --work-dir "本次 work 目录" --message "先保留融合方案，重点优化两个慢 case"
python tools/human_review.py --work-dir "本次 work 目录" --kind question --message "为什么推荐这个方案？"
python tools/human_review.py --work-dir "本次 work 目录" --status
```

停滞第 3 次会在日志中提示《问题文档.md》的位置，内含 2～3 个选项和推荐理由。同一入口直接回复选项；回复“请等待”仅在原 2 分钟期限上加一次 10 分钟，正式意见立即结束等待。超时按推荐方向继续，不能记作人工同意。恢复沿用原截止时间，同轮不重复计数。

Stage9 先处理人的原话，实质方向必须形成带意见编号的人工 P0；程序校验后才交给 Stage3。Stage3 在融合选择依据和 `develop/<iter>/human_feedback.json` 中写落实情况；已处理、已执行和退出导致未执行分开记录。纯问题先答疑，不自动当指令。

原话及状态在 `<work>/human_review/inbox/`、`human_review/state.json`；每次问题/反馈包/证据版本在 `human_review/<iter>/<请求编号>/`。Stage9 实际收到的 role、prompt 与最终决策在 `knowledge/stage9/<iter>/<请求编号>/`，便于核查人工意见是否进入 P0。

Stage9 按编译失败、精度失败、评测异常、普通优化、停滞审查、全部达标六类加载任务与默认输入；咨询只生成问题，回复后才提交最终账本。全部达标分支仍跳过 Stage7/8，最佳记录、退出窗口及 ±5% 经验规则保留。详情见 [人类参与方案](docs/workflow_human_review_design.md)。

### 最佳融合实现存在哪里、什么时候用

只有 Stage6 完成有效的完整性能评测后才记录。还须正式精度通过、给定 case 与连续调用自测通过，并确认自测、精度、性能对应同一代码版本。缺失材料或代码已变化时不入库，继续原流程修复和补测。

旧任务若从未生成 Stage1.5 需求和方案目录，仍沿用原恢复流程；Stage3 可补录实际方案并标记 `probability=null`，完成自测和新评测后再入库。已经开始 Stage1.5 却缺少方案库不按旧任务放行。同轮重跑必须更新选择依据、自测及日志，程序不会把旧材料重新绑定到新代码。

| 位置（相对工作目录） | 内容 |
|---|---|
| `selection/best.json` | 当前评测口径下的最佳记录：实现方案、融合方案、HAP 原始指标、avg_speedup/avg_speed、报告和代码快照路径 |
| `selection/records/iterN-指纹/impl/` | 该次已评测实现的独立副本，后续写代码不覆盖 |
| `selection/records/iterN-指纹/reports/` | 原始性能报告、性能/精度汇总和可用 profiler 数据的副本 |
| `selection/records/iterN-指纹/evidence/` | 对应融合选择依据、本轮方案库、自测报告及日志 |
| `selection/state.json` | 有效评测索引、比较口径和窗口配置；重复恢复同一轮不多计一次 |
| `selection/current_implementation.json` | 当前开发产物与代码的哈希绑定，本身不是性能记录 |

**如何选最好：精度和自测合格后，全部 case 达标优先；同一类再比较 avg_speedup，持平保留较早版本。**HAP 原样保存工具的 `performance_score` 和逐 case `perf_score/t_hw_us/op_times`，不编造新的综合值。尚无全部达标实现时保留最佳可用候选，并明确 `best_available`，不能称其达标。

只比较框架/后端及运行时版本、CANN 工具链指纹、硬件、任务/case 集合、评测工具及 baseline/计时策略相同的成绩；记录实际执行的 device_id，固定的 `metadata/*.json` 基准文件变化也会另开比较组，同一口径下报告数值的正常波动不另分组。历史不同口径结果须重测，不自动导入。Stage3/7/8/9 读取最佳记录和逐 case 历史趋势；语义退出或达到迭代上限时，Stage10 读取最佳快照的代码与报告，当前 `impl/` 可以继续保留最后一次尚未评测的修改。

## 项目结构

```
pypto-pro-workflow/
├── orchestrator.py          ← 主控制平面（唯一入口）
├── config.yaml              ← 唯一运行配置（路径、迭代参数、Jev 连接与密钥，由 Git 跟踪）
├── 测试用例.csv              ← 测试用例记录
├── roles/                   ← 各阶段角色定义（agent prompt 或程序执行说明）
│   ├── n1_stage1_requirements_analysis.md   ← 阶段1 需求分析（cannbot）
│   ├── n1_stage1.5_jev_fusion_selection.md   ← 阶段1.5 融合方案选择（Jev，由程序调用）
│   ├── n1_stage2_first_impl.md              ← 阶段2 首版实现（cannbot）
│   ├── n1_stage3_fix_and_optimize.md        ← 阶段3 修改优化（cannbot）
│   ├── n2_stage4_build.md                   ← 阶段4 编译部署（kerminal）
│   ├── n2_stage5_precision_eval.md          ← 阶段5 精度评测（kerminal）
│   ├── n2_stage6_perf_eval.md               ← 阶段6 性能评测说明（程序直接执行）
│   ├── n2_stage7_kerminal_profile.md        ← 阶段7 profiling（kerminal）
│   ├── n3_stage8_search.md                  ← 阶段8 搜索优化（hermes）
│   ├── n4_stage9_tech_lead_guide.md         ← 阶段9 经验提炼（kerminal/tech_lead）
│   └── n5_stage10_kerminal_report.md        ← 阶段10 最终报告（kerminal）
├── lib/                     ← 工具库
│   ├── framework_target.py  ← Triton Ascend 目标、后端与工作目录校验
│   ├── fusion_selection.py  ← Stage1.5 Jev 评分、候选排序与持久化、方案库注入
│   ├── fusion_evidence.py   ← 融合选择、自测结果与代码版本绑定
│   ├── semantic_exit.py     ← 最佳实现快照、有效性能窗口与逐 case 趋势
│   ├── jev_translation.py   ← 英文材料准备、保真校验与翻译缓存
│   ├── cann_env.py          ← CANN 环境变量构建（从 config.yaml 读取 + source set_env.sh）
│   ├── agent_runner.py      ← agent CLI 调用 + 环境注入
│   ├── bench_parser.py      ← cann-bench 评测结果解析 + 反作弊 kernel_csv 自动分析
│   ├── history_manager.py   ← 跨轮记忆管理（history/ledger/fix_plan/rounds 保护）
│   ├── state.py             ← .state.json 状态管理
│   ├── logger.py            ← 日志系统（workflow/state/history/N1-N5 共 8 个日志）
│   └── handoff.py           ← 文件读写工具
├── knowledge/               ← 知识库（不对 agent 直接展示，通过 orchestrator 注入）
│   ├── anti_cheat_reference.md  ← 实际错误处理表 + 开发自检清单
│   ├── fusion_method.md         ← 融合方法原始说明
│   ├── fusion_options.json      ← Stage1.5 的详细候选方法（F1–F10）
│   ├── arch_programming_guide.md ← Triton Ascend 编程与硬件边界
│   └── profiling_guide.md       ← Profiling 数据解读指南
├── docs/                    ← 设计、迁移计划与环境配置说明
├── examples/
│   ├── triton_ascend_example/   ← 最小融合算子、构建脚本与 NPU 自测
│   └── jev_smoke/               ← 独立 Jev 输入示例
└── skills/triton-profiling-analysis/ ← 昇腾 Triton 性能分析
```

## 运行时 work 目录

每次运行自动创建 `work/<op>_<timestamp>/`：

```
work/exp_20260828_232950/
├── task/                  ← 软链接 → cann-bench task 目录（只读）
├── example/               ← 软链接 → triton_ascend_cann_example（只读）
├── workflow_target.json   ← 程序记录的 Triton Ascend 目标身份
├── device_info.json       ← 当前芯片、运行时与 CANN 信息
├── impl/                  ← 全局：当前最新算子代码
│   ├── cann_bench/         ← Triton 实现与 __init__.py 导出接口
│   ├── setup.py           ← 安装包配置
│   └── build.sh           ← 构建入口
├── ANALYSIS.md            ← 全局：阶段1 需求分析
├── fusion_requirements.en.json ← 阶段1 给 Jev 的紧凑英文需求（≤6000字节）
├── WORK_RECORD.md         ← 全局：工作留痕
├── FINAL_REPORT.md        ← 全局：阶段10 最终报告
├── .state.json            ← 全局：状态机（iteration/stage/history）
├── fusion/                ← Stage1.5 产物，Stage2/3/7/8/9 读取方案库
│   ├── fusion_library.json    ← 概率最高的前 n 个完整融合候选
│   ├── ranking.json           ← 全部融合方法及概率
│   ├── jev_request.json       ← Jev 请求留证
│   ├── jev_response.json      ← Jev 响应留证
│   └── translation/          ← 原文、英文材料、缓存指纹与 Kerminal 翻译记录
├── knowledge/
│   ├── history.json           ← 跨轮记忆（insights/ledger/rounds/suggest_next/fusion_kernel_strategy）
│   ├── proven_patterns.md     ← 成功经验（性能提升≥5%时自动记录，stage9填内容+程序填数字）
│   ├── regression_patterns.md ← 失败教训（性能退步≥5%时自动记录，同上）
│   ├── tech_lead_pitfalls.md  ← 决策错题本（cannbot反馈→tech_lead裁定，✅误判/❌驳回）
│   ├── stage9/iterN/<请求编号>/ ← 每次 Stage9 的独立交接目录
│   │   ├── request.json       ← 程序记录的轮次、失败原因、性能对比与输出路径
│   │   ├── history_before.json ← 调用前完整历史快照
│   │   ├── decision.json      ← Stage9 只提交本轮账本、模型结论及经验分析
│   │   └── commit.json        ← 程序完成合并与知识落盘后的记录
│   └── _rounds_snapshot.json  ← Stage6 成绩备份（兼容旧工作目录）
├── selection/                ← 程序维护的开发证据与有效性能记录
│   ├── current_implementation.json ← 代码、方案选择和自测的版本绑定
│   ├── best.json             ← 首次有效评测后才创建的最佳记录
│   ├── state.json            ← 有效评测索引与窗口配置
│   └── records/iterN-指纹/   ← impl/、reports/、evidence/ 和 manifest.json
├── operator_iter/             ← 每轮 impl 备份（代码追溯用，不对 agent 展示）
│   ├── iter1/
│   └── iter{N}/
├── develop/
│   ├── iter0/
│   │   ├── design_rationale.md     ← 阶段2 首版设计思路
│   │   ├── fusion_library.json     ← 本轮实际选择及原 Jev 概率
│   │   ├── self_test_report.md     ← 自测报告（含同 shape 更换数据连续调用）
│   │   └── self_test_result.json   ← 可核验的结果摘要，引用实际测试日志
│   ├── iter1/
│   │   ├── design_rationale.md     ← 阶段3 第1轮优化设计思路
│   │   ├── 融合方案选择决策依据.md ← 单独说明融合方案选择和实际改动
│   │   ├── fusion_library.json     ← 本轮选择、改动及未评分的新方案
│   │   ├── self_test_report.md     ← 阶段3 第1轮自测报告
│   │   └── self_test_result.json   ← 给定 case 与连续调用测试结果及日志地址
│   └── iter{N}/...
├── build/
│   ├── iter1/build.log
│   └── iter2/build.log
├── eval/
│   ├── iter1/
│   │   ├── precision_result.json   ← 精度判定
│   │   ├── precision_binding.json  ← 精度结果、代码版本和评测口径绑定
│   │   ├── precision_reports/      ← 软链接到 cann-bench 精度报告
│   │   ├── perf_result.json        ← 性能判定
│   │   ├── perf_reports/           ← 软链接到 cann-bench 性能报告
│   │   └── prof_data/              ← 深拷贝：本次任务各 case 的 profiler 数据
│   │       ├── 1/...kernel_details.csv
│   │       ├── 2/...
│   │       └── <case_id>/...
│   └── iter2/
│       ├── ...（同上）
│       └── prof_data/              ← 每轮独立快照，不被下一轮覆盖
├── profile/
│   ├── iter1/bottleneck_analysis.md ← 每轮唯一报告，正文标注 Stage7 或 Stage9
│   └── iter2/bottleneck_analysis.md
├── search/
│   ├── iter1/
│   │   ├── SEARCH_REPORT.md
│   │   └── FIX_DIRECTIVE.md
│   └── iter2/...
└── log/
    ├── workflow.log             ← 全局日志
    ├── state_transitions.log   ← 状态切换专用
    ├── history.log             ← history 变化日志（每轮完整 JSON 快照）
    ├── N1.log                  ← cannbot 节点日志
    ├── N2.log                  ← kerminal 节点日志
    ├── N3.log                  ← hermes 节点日志
    ├── N4.log                  ← tech_lead 节点日志
    └── N5.log                  ← 报告节点日志
```

说明：
- `profile/iterN/bottleneck_analysis.md` 保持统一文件名。仍有 case 未达标时，Stage7 写报告，供 Stage8/9/3 读取；全部达标时，程序将 Stage9 已接收 JSON 的总体结论和逐 case 分析整理成同名报告，供 Stage3 读取，退出轮也保存。来源、轮次和 Stage9 决策路径写在报告内。重试失败不发布，人工复议成功覆盖同一文件；Stage9 对已有报告的直接改写会被撤销。Stage7 实际重跑前移除本轮旧报告，要求重新生成；从 Stage8/9 恢复则保留报告。发现本轮额外 Markdown 报告则停止交付，不静默删除。编译、精度或评测异常分支不补造性能分析。
- `develop/iter0/` 是阶段2 首版实现的设计思路和自测报告，`iter{N}/` 是阶段3 每轮优化后的设计思路和自测报告
- `eval/iter{N}/prof_data/` 是本次报告对应的完整 profiling 副本；程序先保存，再生成 `perf_result.json`。`source_csv_dir` 及所有有效 `kernel_csv` 均指向本轮 work 文件，原始采集目录被覆盖也不会改变这些引用。
- 每次评测及重试使用独立的 `eval/iter{N}/perf_reports/attempt_<次数>_<编号>/`；日志记录源目录、保存目录和 case 关联数量。批量采集 `_batched/` 同样保存，无法唯一对应 case 的 CSV 留空并说明原因，不关联旧文件或其他 case。
- 阶段7 读取的 design_rationale 是上一轮的（fallback 查找最近存在的，处理 build_fail 轮无产出的情况）
- `build_fail`/`precision_fail` 只触发 bug 修复，不产出 design_rationale（设计未变）
- 每次 Stage3（含修复轮）均更新独立的融合选择依据、本轮库和自测材料，注明保留方案或实际变更。

## 一次迭代流程（以 exp 算子为例）

```
启动 → 阶段1(cannbot分析task) → 阶段1.5(Jev评估融合方案、程序选前n个)
     → 阶段2(cannbot按最高概率方案写Triton Ascend代码)
     → 进入迭代循环：
       阶段4: kerminal 打包、安装 wheel，并在源码目录外验证 import
              → FAILED → stage9(tech_lead总结) → 阶段3(cannbot修复) → 重回阶段4
              → SUCCESS ↓
       阶段5: kerminal 跑 cann-bench --no-perf 精度评测
              → precision_fail → stage9(tech_lead总结) → 阶段3(cannbot修精度) → 重回阶段4
              → true ↓
       阶段6: orchestrator 直接执行 cann-bench 性能评测（不经 agent）
              → 反作弊零分 → stage9(tech_lead反作弊分析) → 阶段3 → 重回阶段4
              → 所有case speedup≥1.0 → stage9(证据审查和原有知识积累)
                   → 满足 x 窗口 → 阶段10(最佳已评测快照) → 结束
                   → 尚未满足 → 阶段3(按瓶颈继续优化) → 重回阶段4
              → 有case<1.0 ↓
       阶段7: kerminal 分析 profiling 数据，定位瓶颈
       阶段8: hermes 搜索优化方案，输出 FIX_DIRECTIVE.md
       阶段9: kerminal(tech_lead) 提交本轮 decision.json，程序校验后更新 history.json
       阶段3: cannbot 按 FIX_DIRECTIVE + insights + fusion_kernel_strategy 修改代码
     → 回到阶段4，下一轮迭代
```

以上说明的是流程，不是 Triton 实测成绩。真实 NPU 上的完整 JIT、精度和性能结果，需要配置环境后运行新任务验证。

## 跨轮记忆（history 机制）

每轮迭代结束后，阶段9 tech_lead 读取本轮结果和历轮设计，只写本次 `knowledge/stage9/<iter>/<请求编号>/decision.json`。程序核对请求编号、轮次和字段后，将 `ledger_entry` 合并到本轮账本，完整保留其他轮次和程序填写的成绩；Stage9 不直接覆盖 `history.json`。下一轮阶段3 cannbot 仍读取原有 history 摘要和 P0/P1/P2。

每次调用保留 `request.json`、调用前 `history_before.json`、原始 `decision.json` 和成功提交后的 `commit.json`。输出缺失、无效、属于其他轮次，或漏填本轮必须的经验/裁定时，程序汇总可独立检查的错误，让 Stage9 在同轮最多修正两次，分别保存在 `retry1/`、`retry2/`；首次加修正共三次，第三次仍不合格才停止，不重新评测、不增加迭代次数，也不将旧建议交给 Stage3。当前账本的 `stage9_decision_path` 指向实际采用的决策。

**history.json 主要字段**：

| 字段 | 谁写 | 更新方式 | 给谁看 |
|------|------|---------|--------|
| `rounds` | 程序（stage6后自动） | 每轮追加 | tech_lead |
| `ledger` | 程序写硬数据 + 合并 tech_lead 本轮方向/文件范围 | 按轮追加或更新，保留其他轮次 | tech_lead + cannbot |
| `insights` | tech_lead（阶段9） | 每轮整体重写 | cannbot（最关键） |
| `bottleneck_now` | tech_lead（阶段9） | 每轮替换 | cannbot |
| `suggest_next` | tech_lead（阶段9） | 每轮替换（**P0/P1/P2 分优先级列表**） | cannbot（最优先执行） |
| `worst_cases_tracker` | 程序合并 Stage9 的 case 分析 | 按 case 更新，保留其他历史 | cannbot |
| `fusion_kernel_strategy` | tech_lead（阶段9） | **每轮累加追加**，不覆盖旧条目 | cannbot + tech_lead |

**suggest_next 是结构化任务列表**：每条填写目标、证据、目标 case 及实际实现映射、逐文件修改步骤和验收方式；禁止修改的文件在本轮账本统一填写，允许修改的文件由程序从步骤汇总。完整示例见[Stage9 任务单说明](docs/workflow_stage9_plan_v2.md)。

P0 是必须处理的正确性问题、关键方向或人工指导；P1 是有依据的改进；P2 是待尝试优化。融合变更按证据判断，不固定排在所有问题前面。cannbot 收到的渲染效果：`🔴[P0]` > `🟡[P1]` > `🟢[P2]`。

**fusion_kernel_strategy 示例**（累加追踪融合方案演变）：
```json
"fusion_kernel_strategy": [
  {"iter": 1, "direction": "两段 kernel 经 HBM 传递", "evidence": "需结合完整评测判断收益", "status": "🔄待验证"},
  {"iter": 3, "direction": "片上直传", "evidence": "精度通过且全部 case 实测达标", "status": "✅有效"}
]
```

**cannbot 阶段3 收到的历史经验（按优先级排列）**：
1. `★ 本轮修改指令（P0/P1/P2）` — 优先正确性和未达标 case，再提高整体性能；融合变更按实际证据排序
2. `历史经验知识库(insights)` — 核对已否决方向的硬件、shape 和实现条件，避免重复失败
3. `当前性能瓶颈` — 优化要针对这个瓶颈
4. `融合算子策略追踪` — 复用适用条件内的有效方案；新证据与历史冲突时回查
5. `已验证的成功经验` — 延续有效方向（knowledge/proven_patterns.md）
6. `已验证的失败教训` — 避免重复同条件下的失败（knowledge/regression_patterns.md）
7. `决策错题本` — tech_lead 历轮误判，写代码时避开（knowledge/tech_lead_pitfalls.md）
8. `最慢case追踪` — 查看逐 case 趋势，硬件上限须有当前环境证据
9. `假设追踪账本(ledger)` — regression/no_change 的方向不要重复
10. `性能趋势(rounds)` — 每轮 avg_speedup 变化

## 知识积累（proven_patterns / regression_patterns）

程序在 stage6 后自动计算性能 diff（本轮 vs 上一轮有效数据），根据变化幅度触发知识积累：

| 条件 | 触发 | 文件 | 谁写内容 | 谁写数字 |
|------|------|------|---------|---------|
| avg_speedup 提升 ≥5% | 记录成功经验 | `knowledge/proven_patterns.md` | stage9 tech_lead 填 what_changed / why_it_worked | 程序填 speedup_before/after/delta_pct |
| avg_speedup 退步 ≤-5% | 记录失败教训 | `knowledge/regression_patterns.md` | stage9 tech_lead 填 what_changed / why_it_failed | 程序填 speedup_before/after/delta_pct |
| -5% < delta < +5% | 不记录 | — | — | — |

流程：程序判断涨跌 → 注入 diff 和证据路径 → Stage9 在本次 `decision.json` 填写 `proven_pattern` 或 `regression_pattern` → 程序直接读取同名字段、补上真实数字并写入对应 Markdown。同轮重复审查更新原条目，不重复追加。达到阈值却缺少分析时停止 Stage9，不再写“未填写”的占位经验。

详细记录包含：实际改动、有效/失败原因、受影响 case 及分析、代码/profiling 证据、适用的硬件和 shape 条件、结论局限、后续复用或修正建议。程序另附前后轮次、平均加速比及变化百分比、可对齐 case 的成绩变化、报告/方案依据路径和原始决策地址。未证明的归因应注明待验证。

`history.json` 保存跨轮结论与账本；上述两个 Markdown 保存详细涨跌经验；每次原始分析留在独立 `decision.json`。历史旧格式继续可读，不自动将旧占位内容补写为已经验证的经验。

这两个 knowledge 文件注入到 **stage2/3/7/8/9** 的 prompt 中，所有写代码和分析的角色都能看到。

## 下级反馈 + 错题本（question.md / tech_lead_pitfalls.md）

一个"下级反馈 → 上级裁定 → 沉淀 → 反哺"的闭环，防止 tech_lead 反复给出错误建议。

**触发场景**：stage3 cannbot 实施 tech_lead 建议时，发现某条建议在硬件/框架层面确实不可行（有硬证据）。

**完整闭环**：
```
iter N:  stage9(给建议) → stage3(实施, 发现建议有误)
                              ↓ 严格判断+硬证据
                            写 develop/iterN/question.md
iter N+1: stage9 读 iterN/question.md + 对应 history 意见
            ↓ 裁定
            ├─ ✅confirmed(确认误判) → 承认错误, 给正确做法
            └─ ❌rejected(驳回)      → cannbot 理解错了, 说明正确认知
          程序: 裁定结论写回 question.md + append 到错题本
          之后 stage9 提意见前必读错题本, 不再重犯
```

**严格把关**（防止 cannbot 拿 question 偷懒）：
- 必须真的尝试过 + 有硬证据（编译错误/日志/profiler/官方文档限制）
- 只能是客观不可行，不是"我觉得没必要"
- 提 question 前先查错题本，已收录的不重复提
- 无论建议对错，**仍需用替代方案完成本轮任务**——question 是附带反馈，不是拒绝执行

**错题本文件** `knowledge/tech_lead_pitfalls.md`：
| verdict | 含义 | 记录内容 |
|---------|------|---------|
| ✅confirmed | tech_lead 确实建议错了 | 误判原因 + 正确做法 |
| ❌rejected | cannbot 理解错了 | 驳回原因 + 正确认知 |

- 谁写：tech_lead 裁定后输出 `pitfall` 字段 → 程序 append（累加不覆盖）
- 注入范围：**stage2/3/7/9**（写代码的、分析的、决策的都能看到），文件不存在时返回空不影响流程
- 兜底：如果迭代提前退出导致 question.md 从未被裁定，stage10 会扫描并在报告中提示

## stage6 后路由判断

stage6 性能评测完成后，程序按以下顺序做 3 个判断（**顺序不能调换**）：

```
stage6 完成 → 程序写入 rounds/ledger → 程序算 perf_diff
  ↓
  ① 反作弊？(score_error_code 存在 或 avg_speedup=0)
  │  YES → stage9(反作弊分析) → stage3 → 回 stage4
  │         日志: [判断] 反作弊触发: no_npu_kernel_detected
  │
  ② 性能达标？(所有 case speedup ≥ 1.0)
  │  YES → 记录有效实现、计算 x 窗口 → stage9(审查和知识积累)
  │         ├─ 累计提升不足5%且窗口满足 → stage10(最佳快照) → 结束
  │         └─ 尚未满足 → stage3(perf_pass_optimize) → 回 stage4
  │
  ③ 正常性能不达标
     → 记录有效实现、计算 y 窗口（仅提示重新审查，不强制换方案）
     → stage7(profiling) → stage8(搜索) → stage9(perf_optimize) → stage3 → 回 stage4
       日志: [判断] 性能不达标: avg_speedup=0.5, perf_pass=False
```

perf_diff 数据（程序算一次，stage7/8/9 共享）：
- 提升 ≥5%：注入 `📊 性能提升检测` + 逐 case 对比 + "请总结成功经验"
- 退步 ≤-5%：注入 `⚠️ 性能退步检测` + 逐 case 对比 + "请分析退步原因"
- ±5% 以内：不注入 diff 文本

## state_transitions.log 日志格式

每条状态切换带原因，关键判断点带 `[判断]` / `[知识积累]` 标签：

```
iter1_stage4 → iter1_stage5 | 编译成功，开始精度评测
[判断] 精度失败: passed=18/24
iter1_stage5 → iter1_stage9 | tech_lead经验总结(precision_fail)
iter1_stage9 → iter1_stage3 | 回退修改代码(precision_fail)

iter2_stage6 后:
[判断] 性能提升: avg_speedup 0.5→1.5 (+200%), iter1→iter2
[判断] 性能不达标: avg_speedup=1.5, perf_pass=False → stage7→8→9→3
[知识积累] 成功经验写入 proven_patterns.md: +200%

iter5_stage6 后:
[判断] 性能退步: avg_speedup 3.0→2.5 (-16.7%), iter4→iter5
[知识积累] 退步教训写入 regression_patterns.md: -16.7%

iter9_stage6 后:
iter9_stage9 → iter9_stage10 | 程序语义退出，交付最佳达标实现
```
```bash
tail -10 work/exp_xxx/log/N4.log   # tech_lead
```

## 三个 Agent

| Agent | 节点 | 模型 | CLI 路径 | 配置文件 |
|-------|------|------|---------|---------|
| cannbot | N1（写代码） | GLM-5.3-Flash | `config.yaml → agents.cannbot.cli` | `~/.config/opencode/opencode.jsonc` |
| kerminal | N2（编译评测）| kernelcat1.0 | `config.yaml → agents.kerminal.cli` | `~/.kerminal/config.toml` |
| hermes | N3（搜索） | GLM-5.3-Flash | `config.yaml → agents.hermes.cli` | `~/.hermes/config.yaml` + `~/.hermes/.env` |

### 长提示词传递与启动失败恢复

Linux 的 `Argument list too long` 同时可能来自单个参数、参数总量或环境变量，不能只看约 2 MB 的总量。三个 Agent 均不再把完整提示词作为启动参数；阶段顺序、角色正文、历史和人工 P0 保持原样。

| Agent | 完整提示词如何传入 | 接口依据 |
|-------|------------------|----------|
| CANNBot（Stage1/2/3） | `run` 从文件接入的 stdin 读取全文 | [官方 Linux 1.1.2 发布包](https://registry.npmjs.org/cannbot-linux-x64/1.1.2) |
| Kerminal（Stage4/5/7/9/10） | `-a never exec --skip-git-repo-check -C <cwd> -` 从 stdin 读取全文，返回实际退出码 | 本机 0.8.12 的 `help exec` 与完整命令的参数解析已验证；不再需要模拟终端 |
| Hermes（Stage8） | 同一 Python 环境先启动 `lib/hermes_prompt.py`，再读取文件，在进程内部把全文交给原 `-z` 入口 | [官方 0.20.4 入口](https://github.com/NousResearch/hermes-agent/blob/e624e9fde561e1add9388384012b295fde669ade/pyproject.toml#L372)；该版本的 `-z` 不读取 stdin |

每次调用保留 `work/<任务>/log/prompts/<agent>_<角色文件名>_<唯一编号>.txt`，包含完整角色、任务和公共约束。日志打印传递方式、文件路径和 UTF-8 字节数；同轮重试保留独立文件，不截断或压缩。若系统仍报 E2BIG，日志再给出参数及环境的总字节数和最大单项字节数，不打印环境值。模型自身的上下文容量仍是另一项限制。

Hermes 要使用 `agents.hermes.cli` 指向的官方 pip Python 入口，支持直接入口或符号链接；保持原 venv 解释器和 `-t web,file --yolo --in <cwd>` 参数。未知二进制或自定义 shell 启动器会明确报错，不猜测、不退回长参数。若 Hermes 自身启用了宿主管理容器或二次启动，并试图把同一长提示词再次放进系统命令行，桥接会明确停止；本项目使用在目标环境内直接安装的 Python CLI。CLI 接口已核对，实际模型、工具权限和 NPU 仍需在部署环境联调。

其他入口已核对：Jev 翻译走 stdin JSON-RPC，Jev 评分走 SDK 请求体，评测和硬件探测只传路径、固定选项或短脚本。手动提供长文本时，用 `--optimize-hint-file 方向.md` 代替 `--optimize-hint "正文"`（两者互斥、后续作用相同）；人工建议用 `python3 tools/human_review.py --work-dir <work> --file 意见.md`。

若旧版本在 Stage3 启动时因此中断，同步本轮 runner 和桥接文件后，使用原 `--task-dir` 和原 `--work-dir` 启动即可回到同轮 Stage3；原命令有 `--config` 时也保留。不需要另开 work，不会因此重做已经完成的评测或重复写经验。启动前仍按现有规则检查环境及恢复材料。

## 新环境安装指南

### 1. cannbot（Node.js）

```bash
# 前提：Node.js >= 18
npm install -g cannbot @cannbot-ai/install-helper

# 验证
cannbot --version   # 应显示 1.1.2+

# 配置 API key
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/opencode.jsonc << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM (ZhipuAI)",
      "options": {
        "baseURL": "https://open.bigmodel.cn/api/paas/v4",
        "apiKey": "<你的智谱API key>"
      },
      "models": {
        "glm-5.3-flash": { "name": "GLM-5.3-Flash" }
      }
    }
  },
  "model": "glm/glm-5.3-flash"
}
EOF
```

### 2. kerminal（二进制）

```bash
# 安装（从官方安装脚本，会下载到 ~/.local/bin/kerminal）
curl -fsSL https://kerminal.cn/install.sh | bash

# 验证
kerminal --version

# 配置
mkdir -p ~/.kerminal
cat > ~/.kerminal/config.toml << 'EOF'
model = "kernelcat1.0"
model_provider = "autokernel"
thinking_enabled = true
thinking_budget_tokens = 10000
show_thinking = true
show_raw_agent_reasoning = true

[features]
skills = true

[model_providers.autokernel]
name = "autokernel"
wire_api = "messages"
experimental_bearer_token = "<你的 autokernel token>"
EOF
```

### 3. hermes（Python venv）

```bash
# 安装（官方一键脚本，自带 Python 3.11 venv）
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# 或从源码安装（如果有 hermes-agent-main/）
cd hermes-agent-main && pip install -e .

# 验证
hermes status

# 如果 /usr/local/bin/hermes 报 ModuleNotFoundError，做软链接：
ln -sf ~/.hermes/venvs/hermes-dev/bin/hermes /usr/local/bin/hermes

# 配置模型
cat > ~/.hermes/config.yaml << 'EOF'
model:
  default: glm/glm-5.3-flash
EOF

# 配置 API key + 搜索引擎
cat > ~/.hermes/.env << 'EOF'
GLM_API_KEY=<你的智谱API key>
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
TAVILY_API_KEY=<你的tavily key，可选，没有则用免费ddgs>
EOF
```

### 4. workflow 配置

安装完三个 Agent 后，按"运行指南"中的"第一步：配置 config.yaml"修改路径，然后按"第二步：验证环境"确认所有组件可用。
