# Triton Ascend Workflow 环境迁移指南

> 本文档指导如何在一台新机器上从零配置 Triton Ascend Workflow 的完整运行环境。
>
> **目标**：在 Triton Ascend 支持的 Linux 昇腾环境中，配置兼容依赖、路径和 API Key 后运行 workflow，不改 Stage 流程。

---

## 当前参考环境（基准机器）

表中 Agent 与 SDK 版本来自已有环境记录；算子运行环境现改为 Triton Ascend，必须按所安装 release 的兼容矩阵核对。这里不表示已在当前 Windows 机器完成 NPU 运行。

| 组件 | 版本 | 查询命令 |
|------|------|---------|
| 操作系统 | Ubuntu 24.04.3 LTS (x86_64) | `cat /etc/os-release` |
| 内核 | 6.6.0-132.0.0.111 | `uname -r` |
| Python | 按 Triton Ascend wheel 支持范围选择，建议先核对 Python 3.11 | `python3 --version` |
| pip | 26.0.1 | `pip3 --version` |
| Jev Python SDK（typesafe-sdk） | 0.7.0（项目固定依赖） | `python3 -m pip show typesafe-sdk` |
| PyYAML | 6.0.3（项目固定依赖） | `python3 -m pip show PyYAML` |
| CANN Toolkit | 9.1.0 | `ls /home/developer/Ascend/cann-*` |
| CANN 驱动 | 25.7.rc1 | `cat /usr/local/Ascend/driver/version.info` |
| Triton Ascend | 与 CANN / PyTorch / torch_npu 配套安装 | `python3 -m pip show triton-ascend triton` |
| PyTorch | 2.7.1+cpu | `pip3 show torch` |
| torch_npu | 2.7.1.post8 | `pip3 show torch_npu` |
| Node.js | 旧机 18.19.1；新环境使用 ≥20，以满足 install-helper 1.2.0 的声明要求 | `node --version` |
| npm | 10.2.4 | `npm --version` |
| Kerminal | 0.8.12 | `kerminal --version` |
| Hermes | 0.20.4 (2026.8.18) | `hermes --version` |
| Hermes venv Python | 3.11.16 | `~/.hermes/venvs/hermes-dev/bin/python3 --version` |
| CANNBot CLI | 1.1.2（910 环境已确认） | `cannbot --version` |
| OpenCode 独立 CLI | 安装器检测到 1.18.32；workflow 不调用这个命令 | `opencode --version`（另行安装时） |
| @cannbot-ai/install-helper | 1.2.0（2026-09-29 安装记录） | `install-helper --version`、`npm list -g --depth=0` |
| CANNBot Triton 技能 | 8 个；安装后已用 CANNBot 查询得到 `8/8` | 按第 4.7 节查询实际技能及路径 |

910 环境的最新核验：技能可发现，官方全局 `AGENTS.md` 已链接；`template/convolution.md` 的预期链接缺失，模板引用和现有 Stage 流程的适配仍待完成，不能把 `8/8` 当作全流程验收通过。

---

## 前置条件

| 依赖 | 要求 | 查询命令 | 当前机器版本 |
|------|------|---------|------------|
| 昇腾 NPU | 在线可用 | `npu-smi info` | Ascend 950 (驱动 25.7.rc1) |
| CANN Toolkit | 与 Triton Ascend 所选版本匹配 | `ls /home/developer/Ascend/cann-*` | 9.1.0 |
| Python（运行 Workflow） | ≥3.10，且该版本有配套 Triton Ascend / torch_npu wheel | `python3 --version` | 按实际安装 |
| PyTorch | ≥ 2.1 | `pip3 show torch` | 2.7.1 |
| torch_npu | 与 PyTorch 匹配 | `pip3 show torch_npu` | 2.7.1.post8 |
| Node.js | 新环境 ≥20（install-helper 1.2.0 要求） | `node --version` | 旧机 18.19.1，本次安装出现兼容性警告 |
| Git | 已安装 | `git --version` | — |

---

## 第 1 步：CANN + Triton Ascend 环境

> 目标是昇腾 NPU 后端的 `triton-ascend`，Python 导入名称仍是 `triton`。须单独安装兼容的发行包；不能用 CUDA Triton 代替。

### 1.1 确认 CANN 安装

```bash
# 查看已安装的 CANN 版本
ls -d /home/developer/Ascend/cann-*
# 当前机器输出: /home/developer/Ascend/cann-9.1.0

# 查看驱动版本
cat /usr/local/Ascend/driver/version.info | head -1
# 当前机器输出: Version=25.7.rc1

# 查看 NPU 设备（注意：需要 LD_LIBRARY_PATH 已配置才能运行）
npu-smi info
```

### 1.2 配置环境变量

在 `~/.bashrc` 末尾添加（**路径按你的实际 CANN 安装目录调整**）：

```bash
# === CANN 环境变量 ===
# 当前机器 CANN 安装在 /home/developer/Ascend/cann-9.1.0，按实际修改
export CANN_PATH=/home/developer/Ascend/cann-9.1.0

export LD_LIBRARY_PATH=$CANN_PATH/lib64:$CANN_PATH/x86_64-linux/lib64:$CANN_PATH/lib64/plugin/opskernel:$CANN_PATH/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PYTHONPATH=$CANN_PATH/python/site-packages:$CANN_PATH/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
export PATH=$CANN_PATH/bin:$CANN_PATH/tools/ccec_compiler/bin:$CANN_PATH/tools/profiler/bin:$PATH
```

```bash
source ~/.bashrc
```

> **当前机器的 ~/.bashrc 实际配置**（供参考）：
> ```
> export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.1.0/lib64:/home/developer/Ascend/cann-9.1.0/x86_64-linux/lib64:/home/developer/Ascend/cann-9.1.0/lib64/plugin/opskernel:/home/developer/Ascend/cann-9.1.0/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
> export PYTHONPATH=/home/developer/Ascend/cann-9.1.0/python/site-packages:/home/developer/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe:
> export PATH=/home/developer/Ascend/cann-9.1.0/bin:/home/developer/Ascend/cann-9.1.0/tools/ccec_compiler/bin:...:$PATH
> ```

### 1.3 验证

```bash
# 验证 PyTorch + NPU
python3 -c "import torch; print(f'torch={torch.__version__}')"
# 期望: torch=2.7.1+cpu（或其他版本）

python3 -c "import torch_npu; print(f'torch_npu={torch_npu.__version__}')"
# 期望: torch_npu=2.7.1.post8（或其他版本）

python3 -c "import torch, torch_npu; print(f'npu_available={torch.npu.is_available()}')"
# 期望: npu_available=True

# 查看安装版本与包位置
python3 -m pip show triton-ascend triton
python3 -c "import triton; print(triton.__version__); print(triton.__file__)"

# 检查实际后端；要显示 npu，导入成功本身不代表装对了后端
python3 -c "import torch,torch_npu,triton; torch.npu.set_device(0); print(triton.runtime.driver.active.get_current_target())"
```

### 1.4 安装与真实 JIT 自测

按 [Triton Ascend 安装指南](https://github.com/Ascend/triton-ascend/blob/main/docs/en/installation_guide.md) 和 [对应版本快速开始](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/quick_start.md) 选择完整组合。不同分支文档出现 CANN 9.0.0 与 9.1.0，不能只按包名强行升级或降级。老版本和上游 Triton 的共存限制也不同，以所选发行版为准。

在装好匹配的 CANN、torch、torch_npu 后，安装对应的 Triton Ascend wheel。以下命令里的版本号应替换为兼容矩阵选定的值：

```bash
python3 -m pip install "triton-ascend==<选定版本>" --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
python3 -m pip check
python3 -c "import triton; from importlib.metadata import version; print('triton module=' + triton.__version__); print({p:version(p) for p in ['triton-ascend','torch','torch_npu']})"
```

运行仓库中的 `examples/triton_ascend_example/` 自测（命令见该目录 README）：必须真实启动 NPU kernel，验证非整块输入、连续换输入调用和精度。wheel 打包成功只说明 Python 工程可安装，不代表 JIT 成功，更不能证明性能达标。

`hardware.device_id` 是当前可见 NPU 中的逻辑设备编号。Workflow 传递 `WORKFLOW_NPU_DEVICE_ID`，评测仍使用 `--device-id`；不会强行覆盖已有设备可见性映射。不要继续设置旧框架专属设备变量。

程序还会从活动 CANN 目录读取 `compiler/version.info` 或 `version.info`，保存版本文件路径和内容指纹，避免工具链变更被误记成算子优化收益。查不到有效文件会停止并提示检查环境。可先查询：

```bash
printf '%s\n' "$ASCEND_HOME_PATH" "$ASCEND_TOOLKIT_HOME" "$CANN_PATH"
cat "$ASCEND_HOME_PATH/compiler/version.info"  # 按实际安装布局，也可能是根目录 version.info
```

每次启动都会重新读取当前芯片，断点恢复也不会把旧 `device_info.json` 当作当前硬件。显式手填的 hardware 参数仍保留，并标明配置来源。

---

## 第 2 步：Kerminal 安装

### 2.1 安装

```bash
curl -fsSL https://kerminal.cn/install.sh | bash
```

安装后位于 `~/.local/bin/kerminal`。确保 `~/.local/bin` 在 PATH 中：
```bash
export PATH="$HOME/.local/bin:$PATH"  # 加到 ~/.bashrc
```

### 2.2 配置

配置文件位置：`~/.kerminal/config.toml`

```bash
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
experimental_bearer_token = "<你的 Kerminal API Key>"
EOF
```

> **API Key 获取**：联系 Autokernel 团队获取，或使用已有的 Key。
> **API Key 位置**：`~/.kerminal/config.toml` 中的 `experimental_bearer_token` 字段。

### 2.3 验证

```bash
# 版本
kerminal --version
# 当前机器输出: kerminal 0.8.12

# 安装路径
which kerminal
# 当前机器输出: /root/.local/bin/kerminal

# 测试 CLI 是否可调用（非交互模式）
echo "回复OK" | kerminal -a never exec --skip-git-repo-check -C "$PWD" -
```

---

## 第 3 步：Hermes 安装

### 3.1 安装

当前机器使用源码安装方式：

```bash
# 方式 A：从源码安装（当前机器使用的方式）
cd hermes-agent-main && pip install -e .

# 方式 B：官方一键安装脚本
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

安装后会创建 `~/.hermes/` 目录和 Python 3.11 venv。

如果 `/usr/local/bin/hermes` 不存在或报 ModuleNotFoundError，做软链接：

```bash
ln -sf ~/.hermes/venvs/hermes-dev/bin/hermes /usr/local/bin/hermes
```

> **当前机器实际路径链**：
> ```
> /usr/local/bin/hermes → ~/.hermes/venvs/hermes-dev/bin/hermes
> 安装源码目录: /workspace/workflow_mutilagent_operator_develop/hermes-agent-main
> venv Python: 3.11.16
> ```

### 3.2 配置

需要配置两个文件：

**~/.hermes/config.yaml**（模型配置）：

```bash
cat > ~/.hermes/config.yaml << 'EOF'
model:
  default: glm/glm-5.3-flash
EOF
```

**~/.hermes/.env**（API Key + 搜索引擎）：

```bash
cat > ~/.hermes/.env << 'EOF'
GLM_API_KEY=<你的智谱 API Key>
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
TAVILY_API_KEY=<你的 Tavily API Key，可选，没有则用免费 ddgs>
EOF
```

### 3.3 迁移：能否直接复制配置到另一台机器？

| 文件 | 能否复制 | 说明 |
|------|---------|------|
| `~/.hermes/config.yaml` | ✅ 可以 | 模型配置，跨机器通用 |
| `~/.hermes/.env` | ✅ 可以 | API Key，只要没过期就能用 |
| `~/.hermes/venvs/` | ❌ 不行 | Python venv 含编译的 .so，跨系统不兼容 |
| `/usr/local/bin/hermes` | ❌ 不行 | 软链接，目标路径可能不同 |

**迁移步骤**：在新机器上重新运行安装脚本 → 把 `config.yaml` 和 `.env` 复制过去。

### 3.4 验证

```bash
# 版本
hermes --version
# 当前机器输出: Hermes Agent v0.20.4 (2026.8.18)

# 安装路径
which hermes
# 当前机器输出: /usr/local/bin/hermes
readlink -f $(which hermes)
# 当前机器输出: /root/.hermes/venvs/hermes-dev/bin/hermes

# venv Python 版本
~/.hermes/venvs/hermes-dev/bin/python3 --version
# 当前机器输出: Python 3.11.16

# 检查配置文件
cat ~/.hermes/config.yaml
cat ~/.hermes/.env

# 测试 CLI 是否可调用（非交互模式）
hermes -z "回复OK" --cli
```

---

## 第 4 步：CANNBot CLI 与 Triton 技能安装

**本项目的环境要求包括 CANNBot CLI 和官方 Triton 专属技能。** 只有 CLI 可以运行对话和编码任务，但不会因此自动拥有整套 Triton 开发资料，不能作为本项目安装完成的标准。

### 4.0 先分清 CANNBot、OpenCode 和 Skills

| 名称 | 在本项目中干什么 |
|------|----------------|
| OpenCode 系运行引擎 | 编程 agent 的执行底座，组织模型调用、会话、文件和命令工具。核验过的 CANNBot 1.1.2 使用这套引擎。 |
| CANNBot CLI（`cannbot`） | workflow 实际启动的执行程序。Stage2/3 把角色和任务交给它，由其内部引擎执行。 |
| CANNBot Skills | 专业提示词、参考资料、脚本和模板；本项目需要选装其中的 Triton 技能。 |
| `install-helper` | 技能和插件安装工具，负责把选中的资源部署到相应目录，不负责替 Stage2/3 写代码。 |

实际调用关系是 `Stage2/3 → cannbot run → 内部 OpenCode 系引擎 → 配置的大模型与工具`。程序没有再启动一个独立的 `opencode` 命令。两种 CLI 可以同时存在，版本号也不同；使用 `~/.config/opencode` 是配置与技能发现的兼容关系。

安装向导选择 **OpenCode**，表示使用这种配置和技能目录格式，不表示把 workflow 的执行命令换成 `opencode`；没有另装 OpenCode 时，也可以在向导中手动选择此目标。下文的最终验证必须使用 `config.yaml` 中 `agents.cannbot.cli` 指向的 CANNBot。

### 4.1 前置：Node.js + nvm

```bash
# 检查 Node.js
node --version
# 当前机器输出: v18.19.1

which node
# 当前机器输出: /home/developer/.nvm/versions/node/v18.19.1/bin/node

# 检查 nvm 管理的 Node 版本
ls /home/developer/.nvm/versions/node/
# 当前机器输出: v18.19.1  v20.20.2

# 新环境用 Node.js 20；先确保 nvm 可用
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
```

910 旧机使用 Node 18.19.1 安装 helper 1.2.0 时出现了 `required: node >=20` 警告，随后仍报告安装成功；新机器应满足声明要求。切换 nvm 的 Node 版本会改变全局 npm 命令的位置，后续用 `command -v cannbot` 查询实际路径，并同步检查 `agents.cannbot.cli` 和 `cann.node_bin`，不要机械复制旧机的 `v18.19.1` 路径。

### 4.2 安装 CANNBot CLI

```bash
npm install -g cannbot@1.1.2

# 验证安装
cannbot --version
cannbot --help
command -v cannbot
# 参考版本为 1.1.2；帮助中必须包含 run、debug
```

这里安装的是支持 `cannbot run` 的执行程序。另一个包 `@cannbot-plugin/cannbot` 提供插件安装功能，命令也叫 `cannbot`，不能用它直接替换当前执行程序。技能安装见第 4.6 节。

### 4.3 配置

配置文件位置：`~/.config/opencode/opencode.jsonc`

```bash
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
        "apiKey": "<你的智谱 API Key>"
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

### 4.4 启动方式

```bash
# 交互界面（TUI）— 日常使用
cannbot

# 带初始消息的交互
cannbot --prompt "帮我分析这个算子"

# 非交互模式（CLI 调用，workflow 框架用的就是这个）
cannbot run "你好，回复OK"

# Web 界面
cannbot web
```

### 4.5 验证 CLI 和模型连接

```bash
# 版本
cannbot --version
# 当前机器输出: 1.1.2

# 安装路径
which cannbot
# 当前机器输出: /home/developer/.nvm/versions/node/v18.19.1/bin/cannbot

# 实际文件
readlink -f $(which cannbot)
# 当前机器输出: /home/developer/.nvm/versions/node/v18.19.1/lib/node_modules/cannbot/bin/cannbot

# 确认配置文件可读
test -r ~/.config/opencode/opencode.jsonc && echo "模型配置文件可读"

# 测试 CLI 是否可调用
cannbot run "回复OK"
```

这一步只证明 CLI 和模型连接可用，不证明 Triton 技能已安装或实际被调用。

### 4.6 安装 CANNBot Triton 技能

**先装安装工具，再用它选择 Triton 插件。** 以下是 910 环境实际执行的安装入口：

```bash
curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash
```

该脚本安装的是 `install-helper`，不是默认安装整个技能仓。看到“请运行：install-helper”后，还要继续执行：

```bash
install-helper --version
install-helper
```

本次脚本安装到了 helper 1.2.0。该链接默认随发布更新；若需要固定安装器版本，可用 `npm install -g @cannbot-ai/install-helper@1.2.0` 代替上面的脚本安装。固定安装器版本不代表固定技能仓内容，技能来源路径仍须记录。

本次向导按以下顺序选择：

1. **安装 Plugin — 完整开发工作流**。
2. **OpenCode**。
3. **global — 所有项目共享**。
4. **Triton 算子开发**。
5. 确认安装。

安装器报告：`8 skills, 0 agents`，目录为 `/home/developer/.config/opencode`。`0 agents` 正常，这个插件没有独立子代理；当前会话按需调用 skills。

**这是已执行的完整插件安装方式，也会部署官方流程说明。** 全局 `AGENTS.md` 会影响该配置下的新会话，官方自身的生成、验证和优化循环可能与我们的 Stage 流程重叠。部署后按第 4.8 节核对；安装成功不等于 Stage2/3 的使用方式已经适配完成。

### 4.7 用 workflow 的 CANNBot 验证 8 个技能

使用运行 workflow 的同一个账号、环境、CLI 和 cwd。下面保留了已实测的 910 路径；新机器替换为 `config.yaml` 的 `agents.cannbot.cli`，并进入日志中 `[N1 CANNBot] 启动，cwd=...` 对应目录。新任务尚未建立 work 时可先在项目目录预检，建立后再从实际 cwd 复验。

```bash
cd /mnt/workspace/pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622
CANNBOT_BIN=/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot
CANNBOT_CHECK_DIR=$(mktemp -d /tmp/cannbot-skill-check.XXXXXX)

"$CANNBOT_BIN" --version
"$CANNBOT_BIN" debug skill > "$CANNBOT_CHECK_DIR/skills.json" 2> "$CANNBOT_CHECK_DIR/stderr.log"
printf 'CLI退出码：%s\n检查目录：%s\n' "$?" "$CANNBOT_CHECK_DIR"

python3 - "$CANNBOT_CHECK_DIR/skills.json" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
print(f"技能输出：{len(raw)} 字节，文件：{path}")
skills = json.loads(raw.decode("utf-8-sig"))
available = {s["name"]: s for s in skills}
expected = """
triton-task-extractor
triton-op-designer
triton-op-coding
triton-op-verifier
triton-latency-optimizer
triton-precision-debug
npu-arch
triton-simulator-optimizer
""".split()

ok = 0
for name in expected:
    item = available.get(name)
    location = item.get("location", "") if item else ""
    readable = bool(location) and Path(location).is_file()
    ok += int(readable)
    print("OK" if readable else "缺失或路径失效", name, "->", location)
print(f"目标技能：{ok}/8")
sys.exit(0 if ok == 8 else 1)
PY
```

正常结果：CLI 退出码为 `0`、JSON 完整解析、8 项均为 `OK`、最后为 `8/8`。路径应指向实际 `SKILL.md`；`debug skill` 能返回正文表示 CLI 已能读取技能文件，但不证明某轮 Stage2/3 已调用它。

**不要直接把 `debug skill` 接到 Python 管道。** 1.1.2 在本次实测中曾于第 65535 字节报 UTF-8 数据不完整；源码存在输出未排空便退出的风险。改成先重定向普通文件再解析后，成功读到完整 JSON。不要用忽略解码错误或手动补 JSON 的方式凑出结果。

### 4.8 核对官方说明、模板及接入边界

下面检查的是本次安装的默认全局路径；若改变了配置目录或安装位置，替换为实际路径。

```bash
ls -l "$HOME/.config/opencode/AGENTS.md"
readlink -f "$HOME/.config/opencode/AGENTS.md"
test -r "$HOME/.config/opencode/AGENTS.md" && echo "官方说明可读"

# 单独检查模板：8/8 不覆盖这些配套资源
ls -l "$HOME/.config/opencode/template/convolution.md"

# 若模板链接缺失，再确认源仓库中有没有该模板
ls -l "$HOME/.cannbot/repo/plugins-official/triton-op-generator/template/convolution.md"
```

910 环境截至本次记录的结果：

| 项目 | 实测结果 |
|------|----------|
| CANNBot 技能查询 | 用户已确认 `8/8` |
| 全局 `AGENTS.md` | 链接到 `/home/developer/.cannbot/repo/plugins-official/triton-op-generator/AGENTS.md` |
| 全局 `template/convolution.md` | 文件不存在，尚未修复 |
| Stage2/3 实际技能调用及流程适配 | 尚未验证，不能由 `8/8` 推断完成 |

模板需要补齐链接或修正技能引用。本地技能源码还存在 `.claude/template/...` 这样的固定引用，因此只补 `~/.config/opencode/template`（项目安装时为 `.opencode/template`）并不自动解决所有引用；应以本次安装的技能内容和实际 cwd 核对，避免覆盖已有目录。

本 workflow 仍以 Stage1～9、cannbench 正式评测、Stage9 指令和语义退出为准。官方完整插件的 `ModelNew` 接口、内部评测和优化循环需要适配，不能直接代替当前工程接口及正式成绩。后续应明确让 Stage2/3 加载所需技能，并记录实际调用；本节记录安装与验证，没有修改程序或完成这部分适配。

迁移时不要直接复制旧机器的技能软链接；它们可能指向旧用户的 `~/.cannbot/repo`。在新机器安装资源，并重新执行 `8/8` 和配套路径检查。

---

## 第 5 步：Workflow 配置

### 5.1 克隆项目

```bash
git clone https://gitcode.com/AI4SE/pypto-pro-workflow.git
cd pypto-pro-workflow
# 使用已包含本次 Triton Ascend 迁移的分支或提交，不切回旧框架版本
git branch --show-current
```

### 5.2 安装 Workflow + Jev 的 Python 依赖

Stage1.5 通过 Jev 给融合方案评分，调用库是 `typesafe-sdk`；`PyYAML` 用于读取工作流配置。原独立依赖清单已合并到本节，两个固定版本保持不变：

| 依赖 | 固定版本 | 用途 | 查询命令 |
|------|---------|------|---------|
| typesafe-sdk | 0.7.0 | 调用 Jev 评分服务 | `python3 -m pip show typesafe-sdk` |
| PyYAML | 6.0.3 | 读取 `config.yaml` | `python3 -m pip show PyYAML` |

**安装到用于启动 `orchestrator.py` 的 Python 环境中**，不要装进 Hermes 的独立 venv。Jev SDK 要求 Python ≥3.10；下面统一使用 `python3 -m pip`，确保安装和启动使用同一个解释器。

```bash
# 确认 Python 版本和实际路径
python3 --version
python3 -c "import sys; print(sys.executable)"
python3 -m pip --version

# 安装固定版本（在项目根目录执行）
python3 -m pip install "typesafe-sdk==0.7.0" "PyYAML==6.0.3"
```

如果系统提示 `externally-managed-environment`，先用已经能运行 PyTorch/NPU 的 Python 创建虚拟环境，再执行上面的安装命令：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

`--system-site-packages` 让新环境可以使用该 Python 已安装的 PyTorch 等包；激活后仍须按第 1.3 节确认 `torch_npu` 和 `triton` 可导入，且实际后端为 `npu`。迁移时重新安装依赖，不直接复制旧机器的 venv。

项目目录名仍是 `pypto-pro-workflow`，仅为原仓库路径；生成目标由程序固定为 Triton Ascend。换机器时以 `config.yaml` 中的实际部署路径为准，不需要让 Linux 路径在 Windows 本机也存在。

**查询版本和验证导入：**

```bash
# 查看 Version 和 Location，确认没有装到另一个 Python 环境
python3 -m pip show typesafe-sdk PyYAML

# 直接输出两个包的版本
python3 -c "from importlib.metadata import version; print('typesafe-sdk=' + version('typesafe-sdk')); print('PyYAML=' + version('PyYAML'))"
# 期望输出:
# typesafe-sdk=0.7.0
# PyYAML=6.0.3

# 只验证本地导入，不调用 Jev API
python3 -c "from typesafe_sdk import TypeSafeClient; import yaml; print('Jev SDK / YAML import OK')"
```

### 5.3 修改 config.yaml

**仓库直接提供唯一的 `config.yaml`，并由 Git 跟踪。** 新机器克隆后，直接调整其中的实际路径和 `jev.api_key`，日常也只修改这一份运行配置；仍支持通过环境变量提供 Jev 密钥。

```yaml
agents:
  cannbot:
    cli: "/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot"  # which cannbot
  kerminal:
    cli: "/root/.local/bin/kerminal"                                 # which kerminal
  hermes:
    cli: "/usr/local/bin/hermes"                                     # which hermes

paths:
  cannbench_repo: "/workspace/workflow_mutilagent_operator_develop/cann-bench"  # cann-bench 本地路径

jev:
  base_url: "https://api.typesafe.ai"
  model: "jev-1.13.0"
  api_key_env: "TYPESAFE_API_KEY"
  api_key: "<你的 Jev API Key>"
```

上面是需要检查的字段，保留 `config.yaml` 中的其余配置。密钥优先读取 `TYPESAFE_API_KEY` 环境变量，未设置时读取 `jev.api_key`。`jev-1.13.0` 是服务端模型名称，`typesafe-sdk 0.7.0` 是本地 Python 库版本，两者不是同一个版本号。

Stage1.5 翻译材料使用 `agents.kerminal.cli` 指向的 Kerminal，确认第 2 步已完成配置。

快速查找各工具路径：

```bash
echo "cannbot:  $(which cannbot)"
echo "kerminal: $(which kerminal)"
echo "hermes:   $(which hermes)"
```

### 5.4 在 config.yaml 中配置 CANN 路径

当前程序由 `lib/cann_env.py` 读取 `config.yaml` 的 `cann` 段，并加载 Toolkit 的 `set_env.sh`。**迁移时无需修改 `lib/agent_runner.py`**，只调整配置：

```yaml
cann:
  toolkit_path: "/home/developer/Ascend/cann-9.1.0"
  driver_path: "/usr/local/Ascend/driver"
  arch: "x86_64-linux"  # uname -m 为 aarch64 时填 aarch64-linux
  node_bin: "/home/developer/.nvm/versions/node/v18.19.1/bin"
```

交互终端仍按第 1.2 节加载对应 CANN 环境，保证手动验证与程序配置指向同一个 Toolkit。

### 5.5 安装 cann-bench

保留 cann-bench 评测框架及任务 case。确认仓库含 `examples/triton_ascend_cann_example/`，程序会把它链接为新任务的 `work/example`；这是候选实现模板，与任务的 golden/baseline 不同。

本项目另有 `examples/triton_ascend_example/`，用于最小融合示例和 NPU 自测，不替代外部 cann-bench 的默认模板。实际工作目录中的链接是 `<work>/example`（如 `work/<算子>_<时间>/example`）。

cann-bench 是算子评测工具，workflow 依赖它跑精度和性能评测。不要求和 workflow 在同一目录，但**推荐放在同级目录**（默认 fallback 路径是 `../cann-bench`）：

```
你的工作目录/
├── pypto-pro-workflow/     ← workflow
└── cann-bench/             ← 推荐放这里，config.yaml 中填写实际路径
```

```bash
# 和 pypto-pro-workflow 同级目录下 clone
cd ..
git clone https://gitcode.com/cann/cann-bench.git
```

如果放在其他位置，修改 `config.yaml` 中的路径即可：
```yaml
paths:
  cannbench_repo: "/你的实际路径/cann-bench"
```

---

## 第 6 步：验证全流程

### 6.1 逐个验证（复制粘贴即可）

```bash
echo "=== 1. 系统 ==="
python3 --version
node --version

echo "=== 2. CANN ==="
ls /home/developer/Ascend/cann-*
cat /usr/local/Ascend/driver/version.info | head -1

echo "=== 3. PyTorch + NPU ==="
pip3 show torch | grep Version
pip3 show torch_npu | grep Version

echo "=== 4. Triton Ascend ==="
python3 -m pip show triton-ascend triton
python3 -c "import torch,torch_npu,triton; torch.npu.set_device(0); print(triton.runtime.driver.active.get_current_target())"

echo "=== 5. Kerminal ==="
kerminal --version

echo "=== 6. Hermes ==="
hermes --version

echo "=== 7. CANNBot ==="
cannbot --version

echo "=== 8. 路径确认 ==="
echo "cannbot:  $(which cannbot)"
echo "kerminal: $(which kerminal)"
echo "hermes:   $(which hermes)"

echo "=== 9. Workflow + Jev 依赖 ==="
python3 -c "import sys; print(sys.executable)"
python3 -m pip show typesafe-sdk PyYAML
python3 -c "from typesafe_sdk import TypeSafeClient; import yaml; print('Jev SDK / YAML import OK')"

echo "=== 10. Jev 配置读取（不调用 API，不输出密钥） ==="
python3 -c "from lib.jev_client import load_settings; s=load_settings(); print('Jev config OK; model=' + s.model)"
```

上面的依赖和配置检查在 `pypto-pro-workflow` 项目根目录、准备启动 Workflow 的 Python 环境中执行；它们不验证远端 Jev 服务是否可访问。CANNBot 版本检查也不能代替技能检查：还须完成第 4.7 节的 `8/8` 和第 4.8 节的配套资源、流程适配核对。

### 6.2 运行 Workflow

```bash
cd pypto-pro-workflow
python3 orchestrator.py --task-dir /实际路径/cann-bench/bench_lab/实际基准/level1/exp --max-iter 1
```

先跑 1 轮验证环境没问题，再放开 `--max-iter`。

---

## API Key 清单

迁移时需要配置以下 API Key（**不要硬编码在代码中**）：

| API Key | 用途 | 配置文件位置 | 获取方式 |
|---------|------|-------------|---------|
| Kerminal (Autokernel) | Kerminal Agent 模型调用 | `~/.kerminal/config.toml` → `experimental_bearer_token` | 联系 Autokernel |
| 智谱 GLM | CANNBot + Hermes 模型调用 | CANNBot: `~/.config/opencode/opencode.jsonc` → `apiKey`<br>Hermes: `~/.hermes/.env` → `GLM_API_KEY` | https://open.bigmodel.cn |
| Tavily（可选） | Hermes 搜索引擎 | `~/.hermes/.env` → `TAVILY_API_KEY` | https://tavily.com |
| CANNBench（可选） | Kerminal 的 benchsite MCP | `~/.kerminal/config.toml` → `[mcp_servers.benchsite.env]` → `BENCHSITE_API_TOKEN` | cann-bench 平台 |
| Jev（TypeSafe） | Stage1.5 融合方案评分 | `config.yaml` → `jev.api_key`，或环境变量 `TYPESAFE_API_KEY`（优先） | TypeSafe/Jev 服务账号的 API Key |

> **注意**：CANNBot 和 Hermes 共用同一个智谱 GLM API Key，但配置在不同文件中，迁移时两处都要改。

---

## 配置文件路径速查

| 工具 | 配置文件 | 内容 |
|------|---------|------|
| CANN | `~/.bashrc` | LD_LIBRARY_PATH / PYTHONPATH / PATH |
| Kerminal | `~/.kerminal/config.toml` | 模型 + API Key + MCP 配置 |
| Hermes | `~/.hermes/config.yaml` | 模型配置 |
| Hermes | `~/.hermes/.env` | API Key + 搜索引擎 |
| CANNBot | `~/.config/opencode/opencode.jsonc` | 模型 + API Key |
| CANNBot Triton 技能 | `~/.config/opencode/skills/<技能名>/SKILL.md` | 全局技能入口，通常链接到 `~/.cannbot/repo`；以 `debug skill` 的实际 location 为准 |
| CANNBot 全局说明 | `~/.config/opencode/AGENTS.md` | 本次链接到官方 Triton 完整流程；影响该配置下的新会话 |
| Triton 模板 | `~/.config/opencode/template/` 及技能实际引用的位置 | 算子类别经验；本次预期全局链接缺失，不能由技能安装成功推断已齐全 |
| Workflow | `pypto-pro-workflow/config.yaml` | 唯一运行配置：Agent/CANN/cann-bench 路径、迭代参数、Jev 连接与密钥 |

---

## 迁移检查清单

在新机器上逐项确认：

- [ ] 与 Triton Ascend 匹配的 CANN Toolkit 已安装（`ls /home/developer/Ascend/cann-*`）
- [ ] CANN 驱动已安装（`cat /usr/local/Ascend/driver/version.info`）
- [ ] `~/.bashrc` 中 CANN 环境变量已配置（`echo $LD_LIBRARY_PATH | grep Ascend`）
- [ ] PyTorch 已安装（`pip3 show torch`）
- [ ] torch_npu 已安装（`pip3 show torch_npu`）
- [ ] NPU 可用（`python3 -c "import torch, torch_npu; print(torch.npu.is_available())"` 输出 True）
- [ ] Triton Ascend 可用，真实后端为 `npu`，示例 JIT 和连续调用自测通过
- [ ] 新环境 Node.js ≥20，满足 helper 1.2.0 声明要求；旧机 Node 18 的兼容性警告已核对
- [ ] Kerminal 已安装（`kerminal --version`）
- [ ] `~/.kerminal/config.toml` 已配置
- [ ] Hermes 已安装（`hermes --version`）
- [ ] `~/.hermes/config.yaml` + `~/.hermes/.env` 已配置
- [ ] CANNBot 已安装（`cannbot --version`）
- [ ] `~/.config/opencode/opencode.jsonc` 已配置
- [ ] install-helper 已安装，并已实际选择安装 Triton 插件/所需技能，而非只装安装器
- [ ] 用 workflow 相同账号、CLI 和实际 cwd 查询 Triton 技能，8 项可读，结果为 `8/8`
- [ ] AGENTS.md 的实际来源已确认，已处理官方完整流程与 Stage2/3 职责的冲突
- [ ] 模板及技能引用的参考资料可读，没有失效软链接或未处理的固定路径；本次模板缺失仍待修复
- [ ] 首次联调已检查 Stage2/3 的实际技能调用证据，未把可发现清单当作已调用记录
- [ ] cann-bench 已克隆
- [ ] 运行 Workflow 的 Python ≥3.10，依赖安装在同一解释器中（`python3 -c "import sys; print(sys.executable)"`）
- [ ] Jev SDK 0.7.0、PyYAML 6.0.3 已安装（`python3 -m pip show typesafe-sdk PyYAML`）
- [ ] Jev SDK / YAML 导入正常，Jev 配置可读取（见第 6.1 节）
- [ ] 仓库中的 `config.yaml` 已按本机调整，Jev 密钥已确认或通过环境变量配置
- [ ] `config.yaml` 中 3 个 CLI 路径 + 1 个 cann-bench 路径已修改
- [ ] `config.yaml` 的 `cann` 段已按本机路径和架构调整
- [ ] `python3 orchestrator.py --task-dir ... --max-iter 1` 能跑通

---

## 常见问题

### Q: `npu-smi info` 报 `libc_sec.so: cannot open shared object file`

LD_LIBRARY_PATH 没配置。执行 `source ~/.bashrc` 或检查 CANN 环境变量。

### Q: `import torch` 报 `undefined symbol: _PyErr_SetLocaleString`

Python 与 torch 的 ABI 或动态库环境不匹配。必须在 Kerminal 实际使用的 shell 中修复并重验 `python3`、`torch`、`torch_npu` 和 Triton 导入；Stage4 的安装验证及后续评测依赖它，不能忽略。统一使用 `python3 -m pip`，确认安装包与评测使用同一解释器。

### Q: `hermes` 命令报 ModuleNotFoundError

```bash
ln -sf ~/.hermes/venvs/hermes-dev/bin/hermes /usr/local/bin/hermes
```

### Q: `cannbot run` 报连接错误

检查 `~/.config/opencode/opencode.jsonc` 中的 `apiKey` 和 `baseURL` 是否正确：
```bash
cat ~/.config/opencode/opencode.jsonc
curl -s https://open.bigmodel.cn/api/paas/v4/models -H "Authorization: Bearer <你的Key>" | head -1
```

### Q: CANNBot 能回复，但查不到 Triton skills

CLI 与技能是两项安装内容。执行技能仓的 `install.sh` 只完成安装工具准备，还需运行 `install-helper` 选择 Triton。安装后在 workflow 的实际 cwd 用它配置的 CANNBot 查询，不能只看另一个 `opencode` 命令。步骤见第 4.6、4.7 节。

### Q: 查询技能报 `UnicodeDecodeError`，位置在 65535

本次 CLI 1.1.2 遇到过管道输出疑似截断，不能据此判断技能未安装。按第 4.7 节先写普通文件再解析，同时保留退出码和 stderr；不要忽略解码错误。910 环境改用此方法后已成功读出清单。

### Q: 已经 `8/8`，为什么还缺 `convolution.md`

技能目录和类别模板不是同一批文件，安装器报告 8 个技能不保证模板入口齐全。按第 4.8 节核对源文件、链接和技能正文引用，再补齐缺失项；也不代表官方全局 AGENTS.md 与现有 Stage 流程已经适配。

### Q: orchestrator 报 "CANN 环境检测失败"

确保 `~/.bashrc` 中的 CANN 环境变量已 source，且以下命令不报错：
```bash
python3 -c "import torch_npu; print(torch_npu.__version__)"
```

### Q: 某个 Agent 超时无响应

检查对应的 API Key 是否过期：
```bash
# 测试智谱 API
curl -s https://open.bigmodel.cn/api/paas/v4/models -H "Authorization: Bearer <你的GLM Key>" | head -1

# 测试 Kerminal API（发一条消息看是否有响应）
echo "回复OK" | kerminal -a never exec --skip-git-repo-check -C "$PWD" -
```

### Q: 想换芯片型号（如从 950 换到 910B）

不需要改代码——orchestrator 启动时会自动检测当前芯片型号并注入到所有 stage。只需确保 CANN 环境变量指向正确版本的 Toolkit。

### Q: 新机器的 CANN 装在不同路径

需要配置两个地方，无需修改代码：
1. `~/.bashrc` 中的环境变量
2. `config.yaml` 中的 `cann.toolkit_path` 等路径
