# CANNBot 接入检查与改动建议

检查日期：2026-09-29。依据：当前 workflow、你提供的两个官方源码目录，以及 CANNBot CLI 1.1.2 官方 Linux 包内代码。**910 服务器已确认 CLI 为 1.1.2；改用文件读取后成功解析了 156061 字节的技能 JSON，本次查询未发现名称以 `triton-` 开头的技能，也未发现 `npu-arch`。其他位置是否装有这些资源、官方 AGENTS.md 是否加载及历史调用情况仍待核实。**

**最新进度：** 用户随后通过 install-helper 1.2.0 选择 OpenCode / global / Triton 算子开发，安装器报告在 `/home/developer/.config/opencode` 成功安装 8 skills、0 agents。下面的“未发现”是安装前记录；安装后仍需用 workflow 的 CANNBot CLI 1.1.2 重新核验技能可见性及 AGENTS.md、模板文件，不能用安装器检测到的 OpenCode 1.18.32 代替这一步。

## 1. 先回答你的疑问

**现在确实是 Stage2/3 直接调用 CANNBot CLI。它能够执行我们的开发任务，但程序没有保证它会加载官方 Triton skills。**

- **CANNBot CLI 是干活的程序，不是大模型本身。** 核验过的 1.1.2 使用 OpenCode 系运行引擎，负责调用配置的大模型、读写文件、运行命令和调用工具。我们的 Python 程序直接启动 `cannbot run`，没有再单独启动一个 `opencode` 命令。
- **cannbot-skills 是专业知识和工作方法。** CLI 能发现、模型实际加载、写代码时正确采用，是三件不同的事。
- **你提供的 CANNBot 主仓与当前 CLI 不是同一个安装包。** 本地 `cannbot-master/script/package.json` 是 `@cannbot-plugin/cannbot@1.3.2`，提供插件安装功能；它的命令也叫 `cannbot`，但只接受 `install`。不要拿它直接替换支持 `run` 的执行程序。
- **当前 workflow 明确指定了 Triton Ascend。** 不是让 CANNBot 自己猜框架；但是“知道要写 Triton”不等于“已经用上官方 Triton 技能”。

所以，目前不能断言“CLI 调错导致性能差”，也不能说“既然叫 CANNBot，就已经用上了官方全部能力”。

### CLI 和 CANNBot Skills 能否划等号

**不能。CLI 是执行程序，CANNBot Skills 是可选装的专业能力资料；CLI 可以加载这些资料，但两者不是同一个东西。**

- CLI 1.1.2 的技能加载代码先提供内置的 `customize-opencode`，再合并从外部目录等来源发现的技能。因此，“CLI 完全不带技能”也不准确；它内置少量技能，不代表内置了整套 Triton 开发资料。
- 技能仓的 [install.sh](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/install.sh:82>) 安装的是 `install-helper`。之后通过安装向导或明确命令选择插件/技能，再部署对应资源。安装一个执行程序或安装辅助工具，不等于已选装全部技能。
- Triton 插件的 [init.sh](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/plugins-official/triton-op-generator/init.sh:35>) 明确列出自己的 8 个技能，再建立技能、模板和 AGENTS.md 的链接。这是独立、具体的部署步骤。
- 安装向导的 [Triton 插件清单](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/plugins-community/install-helper/plugins.d/triton-op-generator.yml:10>) 同样明确写了 8 个 `installSkills`，并写明 `installAgents: []`；helper 对这种清单走专门的安装逻辑，不一定执行 `init.sh`。两种入口都需要明确选中要部署的资源，不能用 CLI 已安装来代替这一步。
- 服务器当前能发现 10 个技能，证明技能功能可用且已有部分技能；其中没有目标 Triton 技能。那 9 个外部文档技能的名字在你提供的这份技能仓源码中未找到，因此不能据此认定整仓已安装，也不能推断其安装来源。

当前更准确的结论是：**已安装并运行 CANNBot CLI，当前可发现部分技能，但没有发现官方 Triton 算子开发技能。** 是否曾在别处安装 Triton 资源、CLI 顶层安装器或外部插件曾部署哪些内容，现有证据不足。后续应明确接入 Triton 资源，而不是把“能运行 cannbot”当成“已经加载 Triton 技能”的证明。

## 2. 我们现在到底怎样调用

```text
Stage2 / Stage3
  → 拼接本阶段 role、任务、芯片、融合方案、已有经验等内容
  → 保存本次完整 prompt，通过 stdin 交给 cannbot run
  → CANNBot 使用自己的模型配置、内部 agent 和可用工具执行
  → 在 work 中生成或修改实现，交回原 workflow 继续评测
```

实际启动参数：`cannbot run --format json --dangerously-skip-permissions`。

| 你关心的内容 | 当前代码现状 |
|---|---|
| 怎样指定 Triton？ | Stage2/3 的角色开头明确写 Triton Ascend；另有真实 NPU 环境检查、芯片信息、Triton 工程示例和开发限制。 |
| 使用哪个内部 agent？ | 没传 `--agent`。核验的 CLI 使用自己的 `default_agent`，未配置时取可见的默认 primary，通常是 `build`；服务器实际是哪一个仍待确认。 |
| 使用哪个大模型？ | 没传 `--model`，沿用 CLI 的模型配置。README 的推荐模型不等于实际运行模型。 |
| 我们的 role 是系统角色吗？ | 程序把 role 内容放进本次消息正文，没有把它注册为 CANNBot 内部 agent。虽然变量名叫 `system_prompt`，这里实际不是独立的 system 消息。 |
| 会话会一直积累吗？ | 每次新进程、新会话；历史靠 workflow 本次提供的材料恢复。 |
| 官方 Triton skills 接上了吗？ | workflow 没有负责安装、检查或要求调用这些 skills；若服务器全局或项目目录已经安装，仍可能被 CLI 发现和使用。 |
| 能从现有日志证明吗？ | 保存了输入 prompt、cwd、退出码和部分工具名称；没有完整的 skill 加载记录，不能可靠证明用了哪些技能。 |

来源：[实际调用代码](<C:/Users/<user>/Desktop/ai agent/pypto-pro-workflow/lib/agent_runner.py:112>)、[Stage2 角色](<C:/Users/<user>/Desktop/ai agent/pypto-pro-workflow/roles/n1_stage2_first_impl.md:1>)、[Stage3 角色](<C:/Users/<user>/Desktop/ai agent/pypto-pro-workflow/roles/n1_stage3_fix_and_optimize.md:1>)、[框架与环境检查](<C:/Users/<user>/Desktop/ai agent/pypto-pro-workflow/lib/framework_target.py:14>)。

## 3. AGENTS.md、skill、subagent 到底是什么关系

**AGENTS.md 是工作说明；skill 是按需加载的专业说明和配套资源；subagent 是另外启动的代理会话。三者不能混为一谈。**

你指出的 `triton-op-generator/AGENTS.md` 虽然写着“多 Agent 团队”，并列出了 6 个 skills，但这份插件的实际安装脚本明确写着 **“This plugin has no subagents”**。

这份插件的运行方式是：**当前会话按流程调用设计、编码、验证、优化等 skills。不能把名单中的 6 个 skills 理解成一定会并行启动的 6 个专家。** 其他插件可能确实有子代理，但不能套用到这个插件。

该插件自己的 `init.sh` 在 OpenCode 项目安装模式下主要做两件事：

1. 把选定的技能链接到 `.opencode/skills/`，连同模板资源一起安装。白名单实际有 8 个技能，还包括精度调试和硬件资料。
2. 把整份官方 `AGENTS.md` 放到项目根，作为整套任务流程的说明；没有因此创建名为 `triton-op-generator` 的原生 agent。

所以，**不能只看见 `name: triton-op-generator`，就在命令里直接加 `--agent triton-op-generator`。** 必须先确认它已注册；核验过的 CLI 遇到不存在的名称会警告并回退，仍可能用默认 agent。

来源：[插件主说明](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/plugins-official/triton-op-generator/AGENTS.md:1>)、[无子代理的安装说明](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/plugins-official/triton-op-generator/init.sh:248>)、[技能与主说明安装逻辑](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/plugins-official/triton-op-generator/init.sh:363>)。

## 4. 现在能不能用到官方 skills

**能不能发现，取决于服务器上的实际安装位置和运行目录；下载在另一个源码文件夹里，不等于已接入。**

**CLI 对话和加载技能可以同时发生。** 官方的“安装后直接对话”，前提是已经把插件资料安装到了这个会话可以发现的位置，而非只安装执行程序。

这份 Triton 插件的 `init.sh project opencode <目标项目目录>` 会在目标目录建立：

```text
目标项目目录/
├── AGENTS.md                  ← 官方 Triton 整套流程说明
└── .opencode/
    ├── skills/                ← Triton 专属技能及配套资源的链接
    └── template/              ← 算子类别经验模板的链接
```

核验的 CLI 1.1.2 会自动读取从 cwd 到项目边界之间的 `AGENTS.md` 并注入上下文；还支持全局指令，默认在 `~/.config/opencode/AGENTS.md`，以及配置 `instructions` 显式指定的文件。全局路径可由相关环境变量改变。

**它不会开局遍历整台电脑，去另一个源码目录找到 `plugins-official/triton-op-generator/AGENTS.md`。** 我们的 Stage2/3 默认从 `work/<任务目录>` 启动，也没有在提示词中明确提供这份官方文件。若服务器没有额外安装、链接或配置，就没有通过这条调用接入这份专属说明；Windows 上下载的源码也不会自动同步给 Linux 上的 CLI。

核验的 CLI 1.1.2 会扫描 OpenCode 配置目录下的 `skill/skills`，也支持发现 `.agents/skills`、`.claude/skills` 等兼容位置和配置的额外路径。具体结果受运行账号、cwd、项目边界、权限和关闭技能扫描的开关影响。

因此有两种可能：

- 服务器只装了 CLI，没有另外安装或配置相关资源：不能假定官方 Triton skills 已存在。
- 服务器已经安装了相关 skills，且本次 cwd 可以发现：CLI 可以把技能描述提供给模型，再由模型调用 `skill` 工具加载正文；**发现了也不等于本轮实际加载了。**

补充一个容易混淆的地方：新主仓的统一安装器把 OpenCode 技能放在 `.agents/skills/`；上面这个 Triton 插件自己的脚本使用 `.opencode/skills/`。二者是不同安装入口，路径不同。我们的工作目录目前只由程序建立 `task`、`example` 等链接，没有主动部署这套 Triton 插件。

官方 [quickstart.md](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/plugins-official/triton-op-generator/quickstart.md:17>) 写到安装后用 `agent list` 查看 `triton-op-generator`，但当前安装脚本没有注册对应的原生 agent，这两处存在不一致。`agent list` 不能证明某份 `AGENTS.md` 是否已加载。上面的安装结构用于说明原理；整套安装会替换目标根指令，如何适配本 workflow 见第 5、6 节。

## 5. 是否需要改代码，怎么改

**建议改，但改的是“明确接入哪些技能、保证路径可用、记录实际使用情况”，不换掉现有 Stage 流程。**

| 顺序 | 要改什么 | 对应位置与结果 |
|---|---|---|
| 1 | 增加 CANNBot 启动检查 | 在 runner 配套检查中确认 CLI 版本、是否支持 `run`、实际 cwd、所需技能是否可发现。若指定内部 agent，先验证已注册，不能静默回退。所需资源缺失时明确报错。 |
| 2 | 建立适合本 workflow 的 Triton 技能入口 | 保留官方设计、编码、精度调试和定点优化资料，适配我们的接口、目录和任务范围。通过项目级技能目录或经核实支持的配置路径接入；固定来源版本，检查引用文件可读。 |
| 3 | 让 Stage2/3 明确使用这些技能 | 修改两个角色和提示词组装，给出技能名称、用途、资源路径及本轮需要加载的内容；要求通过实际 skill 调用加载适配后的入口。不能只写一句“你是 Triton 专家”。 |
| 4 | 补齐调用证据与测试 | runner 按结构解析事件，记录成功加载的 skill、读取的关键参考文件，以及实际发生的子代理调用。现有进程结束后残余 JSON 没有完整解析，也一起补齐。测试缺技能、坏路径、不同 cwd 和断点恢复。 |

分配方式建议如下：

- **Stage2：** 加载 Triton 设计和编码资料，结合真实芯片与 Jev 候选生成首版。卷积融合应按需读卷积、矩阵乘和访存资料，而不是只参考通用 Triton 示例。
- **Stage3 编译/精度修复：** 加载编码或精度调试资料，围绕报错和允许修改的文件工作。
- **Stage3 性能优化：** 按 Stage9 本轮任务选择对应的切块、分核、访存、融合等优化资料，服从人工 P0 和文件范围，不自行启动另一套完整优化循环。

适配后的技能仍须输出现有 `impl/cann_bench` 工程及 `develop/<iter>` 材料。正式成绩继续来自 cannbench；Stage9、经验记录、最佳版本和语义退出仍由当前 workflow 管理。资源版本随 work 留记录，保证续跑时知道用的哪份资料。

日志要分清“发现技能”“成功加载技能”“读取参考文件”，不能把模型一句“我用了该技能”当作执行证据。实际模型或 agent 若当前 CLI 事件没有返回，也应写“未取得”，不能用配置推测成实际值；日志不输出密钥或整份私人配置。

## 6. 为什么不直接安装官方整套 triton-op-generator

它提供的是另一套端到端工作流，原样叠加存在明确冲突：

- **接口不同：** 官方通常生成 `ModelNew` 单文件，我们需要 cannbench 可安装工程和指定函数接口。
- **主控重复：** 官方自带生成、验证、性能优化循环和退出条件，会与 Stage3/6/9、人工指令及语义退出重叠。
- **成绩不同：** 官方报告使用 `speedup_vs_torch` 等指标，其中汇总采用几何平均；不能直接作为我们的 `hap`、`avg_speedup` 或“每个 case ≥ 1”结论。
- **安装和路径要适配：** 插件脚本会替换目标根 `AGENTS.md`；部分说明硬编码 `.claude/template`，而 OpenCode 安装的资源在 `.opencode/template`。

官方资料也有少量陈旧表述，例如优化索引标题写 31 项，表格已列到 33 项。因此应按具体资料和实测选用，不能把全部规则不加区分地塞进提示词。

最值得接入的是：[设计资料](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/ops/triton-op-designer/SKILL.md:49>)、[编码资料](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/ops/triton-op-coding/SKILL.md:145>)、[性能优化索引](<C:/Users/<user>/Desktop/ai agent/cann源码/cannbot-skills-master/ops/triton-latency-optimizer/references/Index.md:22>)。

## 7. 服务器上怎样确认

使用与 workflow **相同的账号、环境、CLI 路径和 cwd**。下面按之前日志中的 CLI 路径、work 路径举例；若配置了其他 `agents.cannbot.cwd`，应进入那个目录。

```bash
cd /mnt/workspace/pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622
CANNBOT_BIN=/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot
"$CANNBOT_BIN" --version
"$CANNBOT_BIN" --help
"$CANNBOT_BIN" agent list
CANNBOT_CHECK_DIR=$(mktemp -d /tmp/cannbot-skill-check.XXXXXX)
"$CANNBOT_BIN" debug skill > "$CANNBOT_CHECK_DIR/skills.json" 2> "$CANNBOT_CHECK_DIR/stderr.log"
printf 'CLI退出码：%s\n检查目录：%s\n' "$?" "$CANNBOT_CHECK_DIR"

python3 - "$CANNBOT_CHECK_DIR/skills.json" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
print(f"输出文件：{path}，大小：{len(raw)} 字节")
skills = json.loads(raw.decode("utf-8-sig"))
found = [s for s in skills if s.get("name", "").startswith("triton-") or s.get("name") == "npu-arch"]
for s in found:
    print(s["name"], "->", s.get("location"))
if not found:
    print("未发现 Triton 专属技能")
PY
```

这些是核验过的 CLI 1.1.2 的查询接口，不会发起算子生成；其他版本先以 `--help` 为准。`agent list` 展示已注册角色；`debug skill` 的筛选结果展示能发现的技能及来源，**不是本轮已调用清单**。若命令只显示 `install` 用法，应先检查是否指向了同名插件安装器。

910 服务器最初使用管道查询时，Python 报 `UnicodeDecodeError`，位置为 65535，原因是 `unexpected end of data`。静态核对 CLI 1.1.2，`debug skill` 一次写出包含技能正文的大段 JSON，没有等待输出排空，主入口结束时调用 `process.exit()`；结合报错位置，高度怀疑输出在约 64 KiB 处被截断。改用上面的普通文件读取后，用户提供的结果为：CLI 退出码 0、输出文件 156061 字节、JSON 成功解析、未发现 Triton 专属技能。本次检查的读取问题已解决，不需要忽略编码错误或补齐 JSON。

这个结果说明：在本次测试账号、cwd 和配置下，CLI 没有发现我们期望的 `triton-*` 与 `npu-arch` 技能。若 Stage2/3 使用相同运行环境，就不能通过这些技能名加载官方设计、编码和优化说明。它仍能根据我们的角色提示词生成 Triton；不能据此断言机器上完全没有技能、其他位置没有安装，或官方 AGENTS.md 一定未加载。下一步应列出完整技能名称与来源，并完成第 3 项指令和模板路径检查，以区分未安装、安装位置不对及发现配置问题。

实测依据：[910 服务器第 2 项输出](<C:/Users/<user>/Desktop/ai agent/pypto-pro-workflow/docs/cannbot使用说明/910环境cannbot终端测试输出/2 检查这次 CLI 能发现哪些 Triton skills.md>)。

随后提供的完整清单确认共发现 10 个技能：1 个内置 `customize-opencode`，以及 9 个来自 `/home/developer/.config/opencode/skills/` 的文档生成、文档质量检查、AscendC API 检索技能，没有官方 Triton 设计、编码、精度调试或优化技能。这说明技能发现功能能够读取该全局目录，但目标 Triton 技能没有出现在本次清单中。不能把这些 AscendC 文档技能当成 Triton 算子开发技能。

注意：[标题为第 3 项的文件](<C:/Users/<user>/Desktop/ai agent/pypto-pro-workflow/docs/cannbot使用说明/910环境cannbot终端测试输出/3 检查 AGENTS.md 和模板是否在可发现的位置.md>) 实际保存的是上述技能清单，并非 AGENTS.md 和模板路径检查；因此这两类文件的存在与加载情况仍未确认。

后续 [安装记录](<C:/Users/<user>/Desktop/ai agent/pypto-pro-workflow/docs/cannbot使用说明/cannbotskill安装.md>) 已显示 Triton 插件安装成功。仍需分别确认技能发现、配套文件和实际调用：本地 helper 1.1.17 的 manifest 安装路径会部署全局 AGENTS.md，但没有看到与旧 init.sh 相同的 template 安装步骤；用户安装的是 1.2.0，不能直接推断其落盘结果。若全局 AGENTS.md 已部署，新的 CANNBot 会话可能加载官方完整流程，还需处理与本 workflow 的 Stage 编排冲突。安装日志中的 Node 18 低于 helper 声明的 Node 20 要求是兼容性警告，本次日志明确显示安装完成，不应把警告直接当成安装失败。

## 8. 这能解释性能差吗

**目前只能确认“官方 Triton 技能的接入和使用缺乏保证”，不能确认它就是性能差的原因。** 本地没有本次 Triton 实验的完整实现、评测和工具事件，尚未核实实际采用的算法、tiling、模型与技能。

建议先补启动核验和加载记录，再用同芯片、同 case、同基准、同模型及相近开发预算对比。除了平均加速比，还看慢 case、最差 case 和精度。若技能确实已加载但性能仍差，就应根据代码和 profiling 检查算法、分块、数据搬运等具体问题。

**本轮产物：本检查文档和改动建议。尚未安装插件、修改运行代码或重新评测。**
