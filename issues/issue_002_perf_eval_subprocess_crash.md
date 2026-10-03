# Issue #002: cann-bench 性能评测 subprocess 崩溃（returncode=-11 / returncode=1）

## 问题现象

stage6 性能评测时，orchestrator 直接 `subprocess.run` 调 cann-bench，连续 2 次失败后 workflow 停止：

```
[ITER1-阶段6-性能评测] 第1次评测：success=False, report=无
[ITER1-阶段6-性能评测] 性能评测崩溃（第1次），等待10秒后重试...
[ITER1-阶段6-性能评测] 第2次评测：success=False, report=无
RuntimeError: cann-bench 性能评测连续2次崩溃
```

N2.log 中的详细错误：
```
ImportError: /opt/buildtools/python-3.12.9/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so: undefined symbol: _PyErr_SetLocaleString
```

## 问题定位

### 根因

与 Issue #001 相同——`/opt/buildtools/python-3.12.9` 的 `_ctypes.so` 损坏。

但 stage6 的表现和 stage5 不同：
- **stage5（精度评测）**：由 kerminal agent 跑，kerminal 遇到 ctypes 错误后**自己排障**，找到绕过方式成功执行
- **stage6（性能评测）**：由 orchestrator 直接 `subprocess.run`，没有 agent 的自动排障能力，直接失败

### 排查过程中的误导

1. **误以为是 LD_PRELOAD 问题**：尝试在 subprocess env 中 `env.pop("LD_PRELOAD")`，但 _ctypes.so 本身就坏了，去不去 LD_PRELOAD 都不行
2. **误以为是 source ~/.bashrc 问题**：尝试改成 `bash -c "source ~/.bashrc; python3 ..."` 执行，反而引入了 Python 版本冲突（`/opt/buildtools` 和 `/usr/local` 的 `sys.path` 混乱）
3. **误以为是环境变量继承问题**：反复切换 `env=os.environ.copy()` / `env=_build_eval_env()` / `bash -c`，都不能解决根因

### 为什么 kerminal agent（stage5）能成功

kerminal 是 AI agent，它在 shell 里遇到 ctypes 错误后：
1. 自己执行 `which python3`、`ls /usr/bin/python3*` 排查可用的 Python
2. 尝试了多种方式调用 cann-bench
3. 最终找到了能成功执行的方式（通过 kerminal 内部的 shell 环境）

这是 agent 的自适应能力，subprocess 做不到。

## 历次修复尝试对比（为什么之前改的不对）

这个问题反复修了 4 次才根治，前 3 次都没对症下药：

| 尝试 | 改法 | 为什么不对 |
|------|------|-----------|
| ❌ 第1次 | 芯片检测改成主进程直接 `import torch` | 主进程 import torch + LD_PRELOAD 触发 **SIGSEGV**，直接崩主进程 |
| ❌ 第2次 | 改成 `bash -c "source ~/.bashrc; python3"` | `source ~/.bashrc` 把 `/opt/buildtools` 的 python 引入 sys.path，**加载了坏的 _ctypes.so → ImportError** |
| ❌ 第3次 | 直接 `cp` 系统好的 `_ctypes.so` 覆盖坏的 | 能跑几分钟，但**监控进程（clabagent）定期还原 /opt/buildtools 下的文件**，改完就被改回去 |
| ✅ 第4次 | 把好的 ctypes 目录放 PYTHONPATH 最前面，让 Python 优先加载 | 不碰会被还原的系统文件，从**另一个可用目录**加载，稳定 |

**核心误区**：
1. 前两次一直在折腾"subprocess 怎么执行"（直接调 / bash -c / env 切换），**方向就错了**——问题不在执行方式，在于用的那个 `python3` 的 `_ctypes.so` 本身是坏的
2. 第三次找对了根因（文件坏），但**修错了地方**——改系统文件会被监控还原，治标不治本
3. 第四次才真正对症：**不修坏文件，绕开它**——环境里同时存在坏的（`/opt/buildtools/.../lib-dynload`）和好的（`/usr/lib/python3.12/lib-dynload`）两份 `_ctypes.so`，把好的目录放 PYTHONPATH 最前面，Python 导入 C 扩展时先扫 PYTHONPATH，找到好的就不会去加载坏的

**根因本质**：`/opt/buildtools/python-3.12.9` 的解释器是 3.12.3，但目录里的 `_ctypes.so` 是为 3.12.9 编译的（缺 `_PyErr_SetLocaleString` 符号），版本错配。这是环境本身的缺陷，workflow 代码只能绕开，不能真正修复环境。

## 解决方案

### 根本修复（已执行）

同 Issue #001，替换坏的 `_ctypes.so`：

```bash
cp /usr/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so \
   /opt/buildtools/python-3.12.9/lib/python3.12/lib-dynload/_ctypes.cpython-312-x86_64-linux-gnu.so
```

### ⚠️ 复发：替换的文件被监控进程还原

**现象**：手动替换 `_ctypes.so` 后能跑，但过一段时间又坏了（文件 md5 变回损坏版本，时间戳被更新）。说明环境有监控进程（疑似 clabagent）定期还原 `/opt/buildtools` 下的文件。**改系统文件不是稳定方案**。

**稳定修复（已实现，不改系统文件）**：

`/usr/lib/python3.12/lib-dynload/` 下有一份**可用**的 `_ctypes.so`（md5 与损坏版本不同，不被还原）。让 Python 优先从这里加载，绕开损坏的那个：

1. `lib/bench_parser.py` 新增 `_find_good_ctypes_dir()`，返回含可用 `_ctypes.so` 的目录（`/usr/lib/python3.12/lib-dynload`）
2. `_build_eval_env()` 把该目录放到 `PYTHONPATH` **最前面**——Python 导入 C 扩展时先扫 PYTHONPATH，找到好的就不会加载 buildtools 里坏的
3. `orchestrator.py` 芯片检测的 `bash -c` 命令加 `export PYTHONPATH=/usr/lib/python3.12/lib-dynload:$PYTHONPATH; unset LD_PRELOAD`
4. `lib/agent_runner.py` 的 `_build_env()` 同样把好的 ctypes 目录放 PYTHONPATH 最前

**验证**：
```bash
PYTHONPATH="/usr/lib/python3.12/lib-dynload:$PYTHONPATH" python3 -c "from _ctypes import Union; import torch; print('OK')"
# 输出 OK，不再 ImportError
```

### 代码层面（已实现）

- `_build_eval_env()` 构造干净的 env（PYTHONPATH 含好的 ctypes 目录，去掉 LD_PRELOAD）
- 性能评测最多重试 2 次，间隔 10 秒
- 连续崩溃时 `raise RuntimeError` 停在 stage6，便于断点续跑
- N2.log 记录每次失败的 returncode + stderr，方便定位

### 验证

```bash
# 确认 python3 能正常 import torch
python3 -c "import torch, torch_npu; print('OK')"

# 确认 cann-bench 能直接跑
cd /workspace/workflow_mutilagent_operator_develop/cann-bench/src && \
python3 -m kernel_eval.cli eval --bench-name cann \
  --task-dir /path/to/task --device-id 0 --warmup 2 --repeat 3
```

## 影响范围

- `lib/bench_parser.py` 的 `run_perf_eval()` 和 `run_precision_eval()`
- stage6 性能评测（代码直接执行）
- stage5 精度评测不受影响（kerminal agent 执行）

## 教训

1. **subprocess 失败时必须打印 stderr**——之前 stderr 被吃掉了，看不到 ImportError 详情
2. **不要盲目换 subprocess 执行方式**（直接调/bash -c/env 切换），先看 stderr 定位根因
3. **agent 的自适应能力是 subprocess 不可替代的**——stage5 能过 stage6 不能过就是因为 agent 会排障
