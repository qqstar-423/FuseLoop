# Issue #003: exit_decision 死循环——tech_lead 写了但程序读不到

## 问题现象

性能达标后（所有 case speedup ≥ 1.0），workflow 进入 stage9 融合验证。tech_lead 在 `fusion_kernel_strategy` 里明确写了"exit_decision=true, 必须退出"，但 orchestrator 每次读到的都是 `None`，判定为"假融合继续迭代"——**连续 8 轮死循环**（iter7-iter14），每轮零代码改动。

```
[判断] Tech Lead exit_decision=None: 假融合 → 继续迭代优化融合方案  （iter7）
[判断] Tech Lead exit_decision=None: 假融合 → 继续迭代优化融合方案  （iter8）
...连续8轮...
[判断] Tech Lead exit_decision=None: 假融合 → 继续迭代优化融合方案  （iter14）
```

## 问题定位

### 根因

**不是 agent 的问题，是代码的字段名不一致**。

tech_lead 在 history.json 顶层写了 `"exit_decision": True`（无下划线前缀），但程序读的是 `"_last_exit_decision"`（有下划线前缀）——永远对不上。

```
tech_lead 写的:  history["exit_decision"] = True          ← role 里定义的字段名
程序读的:        history.get("_last_exit_decision", None)  ← 代码期望带下划线前缀
```

**为什么不一致**：`update_from_tech_lead()` 设计的是 `exit_decision` → 存为 `_last_exit_decision`（加前缀区分来源）。但 tech_lead 是 kerminal agent 直接覆盖写 history.json 文件，不经过 `update_from_tech_lead()` Python 函数。所以 agent 按 role 的字段名写了无前缀版本，程序按内部约定找有前缀版本。

### 为什么连续 8 轮才发现

- 每轮 stage9 只有 ~4 分钟（tech_lead 很快就确认了"真融合"），但 stage3 要 ~10 分钟（cannbot 零改动但还是要自测），加上 stage4/5/6 各几分钟——每轮空转 ~20 分钟
- 8 轮 × 20 分钟 = 约 2.5 小时浪费
- tech_lead 自己也意识到了问题（第 7 次确认时写了"workflow 持续误读 exit_decision"），但它没有能力修代码

## 解决方案

在 orchestrator 读 `exit_decision` 时增加兜底解析：如果顶层字段 `_last_exit_decision` 为 None，从 `fusion_kernel_strategy` 最后一条的 `status` 文本里解析：

```python
if exit_decision is None:
    fks = _h.get("fusion_kernel_strategy", [])
    if fks:
        last_status = str(fks[-1].get("status", "")).lower()
        if "exit_decision=true" in last_status or ("✅" in ... and "必须退出" in ...):
            exit_decision = True
```

## 教训

1. **不要假设 agent 会按 API 规范写 JSON 字段**——它是大模型，可能写在描述文本里而不是独立字段
2. **关键判断必须有兜底解析**——从多个位置尝试读取，不能只依赖一个字段
3. **死循环必须有硬上限**——连续 N 轮 exit_decision=None 应该自动上报，不能无限空转
