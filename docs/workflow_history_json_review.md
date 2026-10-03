# history.json 字段审阅：代码现状

文件位置：`<work>/knowledge/history.json`，不是 `.state.json` 里的同名 `history` 数组。以下按当前代码整理，数值、文件名和人工消息编号都是示例。

**先看完整字段，再看各场景切片。** `#` 后是解释；这是审阅写法，不能直接当 JSON 输入程序。已同步 v2 任务单：程序先定义字段，Stage9 按 case、文件、方法、验收填写；简短原始决策例子见 [任务单 v2](workflow_stage9_plan_v2.md)。

## 1. 完整字段示例

假设第3轮平均加速比从2.0提升到2.2，但 case3 仍只有0.97；人要求保留现有融合方向，优先修慢 case。为方便阅读，只展示这一轮，真实文件会保留其他轮次。

```text
{
  "plan_version": 2, # 程序：已接收v2任务；初始空历史或旧历史可能没有此字段
  "insights": [ # Stage9：当前汇总的经验结论；更新时提交完整列表
    "[分块] iter1–iter3 | 证据: case3 从0.80到0.97 | 结论: 小分块有效 | 状态: 待继续"
  ],
  "ledger": [ # 每轮账本：程序填成绩、汇总修改范围，Stage9 填分析及任务
    {
      "iter": 3, # 程序：这是第几轮
      "reason": "perf_optimize", # 程序：为什么进入 Stage9；各种取值见下文
      "evaluation_summary": "本轮缩小分块后均值从2.0到2.2，case3到0.97；尾块根因仍待对照验证", # Stage9必填：已评测实现的改动、结果及依据
      "direction": "case3 尾块开销仍高；保留融合方案，继续缩小尾块", # Stage9：下一步方向概览；具体执行以 suggest_next 为准
      "modify_files": ["impl/kernel.py"], # 程序：从本轮任务 changes 的 modify/create 汇总，模型不另填
      "readonly_files": ["impl/reference.py"], # Stage9：禁止修改的文件；不能与上一项重合
      "action_plan": { # 程序：每轮保存完整原始任务，旧轮不会被新建议覆盖
        "version": 2, # 当前任务协议
        "tasks": [
          {
            "task_id": "T1", # 与下方最新suggest_next的同名任务一致
            "priority": "P0", # 人工方向优先
            "task_type": "modify", # inspect仅检查；modify含修改或新建
            "action": "人工建议：保留融合方式，优化 case3 尾块", # 概括，不另藏方法
            "reason": "其他 case 已达标，先解决局部慢点", # 依据
            "case_scope": "cases", # 针对具体case；operator表示有理由的整体工程任务
            "target_cases": ["case3"], # 从程序case目录选完整编号
            "operator_reason": "", # cases时为空；operator时填写理由，target_cases为空
            "case_bindings": [ # 每个目标case一条映射
              {
                "case_id": "case3", # 与目标编号一致
                "implementation_files": ["impl/kernel.py"], # 实际实现路径
                "route_evidence": [
                  {"file": "impl/dispatcher.py", "location": "dispatch float32分支", "explanation": "case3为float32，此分支调用kernel.py"} # 证据文件、位置、说明
                ]
              }
            ],
            "changes": [ # 操作是允许范围的唯一来源
              {"file": "impl/kernel.py", "operation": "modify", "location": "kernel尾块分支", "method": "比较两种尾块大小，保留完整块分支及原数学定义"} # 文件、操作、位置、方法
            ],
            "acceptance_checks": ["全部给定case精度自测通过并留日志", "正式评测对照case3及其余case耗时"], # 未评测不能谎称通过
            "source": "human", # 人工来源
            "human_message_id": "human-001" # 对应人的原始意见
          }
        ]
      },
      "avg_speedup_before": 2.0, # 程序：已通过口径核对的上一有效报告均值；不可比时为null，不用旧 state.speedup
      "avg_speedup_after": 2.2, # 程序：本轮平均加速比
      "delta": 0.2, # 程序：after－before 的差值；不是百分比，这里实际提升10%
      "verdict": "big_win", # 程序：本轮成绩变化标签；完整取值见第3节
      "comparison_status": "comparable", # 程序：comparable可比 / new_baseline新基线 / unavailable当前不能比较
      "comparison_reason": "两轮硬件、任务、case、基准及评测协议一致", # 程序：为什么可比或不可比
      "comparison_previous_iter": 2, # 程序：核对了哪一轮；没有前轮时为null，不表示已通过比较
      "environment": { # 程序：本轮环境；以下芯片数值仅为示例，不是本机检测结果
        "framework": "Triton", # 新任务目标框架，性能记录从当轮归档环境读取
        "backend": "triton-ascend", # 使用昇腾适配后端
        "framework_source": "performance_report", # 性能经验用当轮报告；故障审查用workflow_target，不让Stage9猜填
        "runtime_versions": {"triton": "3.2.0", "triton_ascend": "3.2.1", "torch": "2.7.1", "torch_npu": "2.7.1.post8"}, # 仅展示字段；实际版本从运行环境读取，不代表推荐组合
        "toolchain": {"verified": true, "cann_roots": {"ASCEND_HOME_PATH": "/opt/ascend/cann"}, "version_files": [{"path": "/opt/ascend/cann/compiler/version.info", "sha256": "<真实文件指纹>", "version": "<实际版本>"}]}, # CANN工具链来源与版本指纹，参与比较
        "chip_model": "Ascend 910B", # 设备检测保存的芯片名称
        "soc_version": "Ascend910B4", # 精确SoC标识，与芯片名称一同核对
        "npu_arch": "Dav-C220", # 设备保存的架构；未提供则unknown
        "programming_model": "Triton block program (SPMD)", # 编程模型，实际运行后端应是 npu
        "device_id": 0, # 评测设备号；未记录为null
        "source_kind": "performance_report", # 性能用正式报告；故障审查用workflow_context或device_info
        "source_path": "<work>/eval/iter3/perf_result.json" # 实际存储为完整路径；也可能指向已验证归档
      },
      "case_analysis": [ # Stage9：性能场景必填，覆盖程序列出的最慢case；编译/精度/评测异常场景不填
        {
          "case_id": "case3", # 报告中的完整编号，不得缩写；本例报告编号本来就是case3
          "observation": "speedup=0.97，仍未达标", # 本轮现象
          "explanation": "推测尾块补齐开销较高，尚需对照验证", # 结论；明确已证实/推测/待验证
          "evidence": "eval/iter3/perf_result.json cases[case3]；impl/kernel.py 尾块分支", # 具体依据位置
          "next_action": "比较两种尾块大小，重新检查精度和性能" # 研究线索；不额外授权修改，执行须列入 suggest_next
        },
        {
          "case_id": "case1", # 第二个最慢case；示例总共只有2例
          "observation": "speedup=3.43，已达标", # 本轮现象
          "explanation": "该路径当前有效，没有足够证据要求修改", # 结论
          "evidence": "eval/iter3/perf_result.json cases[case1]", # 依据
          "next_action": "保留路径，验证其他改动不会使其退步" # 不修改也要说明如何保持
        }
      ],
      "human_responses": [ # 有人工消息时出现：Stage9 对人的逐条答复
        {
          "message_id": "human-001", # 对应收到的人工消息编号
          "kind": "direction", # 本条是方向建议；另外三类见第4节
          "answer": "采纳，保留现有融合方案，优先修改 case3" # Stage9 的处理说明，不代表 Stage3 已执行
        }
      ],
      "stage9_decision_path": "C:/workflow-work/demo/knowledge/stage9/iter3/req-003/decision.json" # 程序：原始决策文件，含详细分析；路径是示例
    }
  ],
  "rounds": [ # 程序：走到 Stage6 写入步骤的评测摘要；不等于最佳版本列表
    {
      "iter": 3, # 轮次，与 ledger 中对应轮次相同
      "avg_speedup": 2.2, # 平均加速比，保留4位小数
      "overall_score": 75.0, # cannbench 报告的总分，保留2位小数；不是平均加速比
      "performance_score": 50.8, # 报告的性能评分（HAP相关），保留2位小数；不含全部逐 case HAP 数据
      "improved": true, # 同口径下均值是否上涨；不是“提升≥5%”；新基线或不可比时为null
      "comparison_status": "comparable", # 程序：含义同ledger的对应字段
      "comparison_reason": "两轮硬件、任务、case、基准及评测协议一致", # 程序：比较依据
      "comparison_previous_iter": 2, # 程序：前轮编号
      "worst_case": { # 本轮加速比最低的 case
        "case_id": "case3", # case 编号
        "speedup": 0.97 # 加速比；小于1表示慢于基准
      },
      "worst_6_speedups": [ # 按加速比从低到高，最多6条；示例只有2个 case
        {
          "case_id": "case3", # 最慢的 case
          "speedup": 0.97 # 该 case 的加速比
        },
        {
          "case_id": "case1", # 第二慢的 case
          "speedup": 3.43 # 该 case 的加速比
        }
      ]
    }
  ],
  "bottleneck_now": "case3 的尾块补齐和搬运开销较高", # Stage9：当前主要瓶颈
  "suggest_next": [ # Stage9：最新一批可执行事项，唯一指令清单；不是历轮建议的完整列表
    {
      "task_id": "T1", # 本轮唯一编号，Stage3按编号报告落实情况
      "priority": "P0", # 必须落实的关键建议；人工方向须明确为 P0
      "task_type": "modify", # 有修改；仅检查用inspect
      "action": "人工建议：保留融合方式，优化 case3 尾块", # 与本轮action_plan原任务相同
      "reason": "其他 case 已达标，先解决局部慢点", # 建议依据
      "case_scope": "cases", # 具体case任务
      "target_cases": ["case3"], # 完整编号
      "operator_reason": "", # 本条不是整体工程任务
      "case_bindings": [
        {
          "case_id": "case3", # 一个case一条映射
          "implementation_files": ["impl/kernel.py"], # 实际实现文件
          "route_evidence": [
            {"file": "impl/dispatcher.py", "location": "dispatch float32分支", "explanation": "case3为float32，此分支调用kernel.py"} # 不能只凭case名猜文件
          ]
        }
      ],
      "changes": [
        {"file": "impl/kernel.py", "operation": "modify", "location": "kernel尾块分支", "method": "比较两种尾块大小，保留完整块分支及原数学定义"} # Stage3按这个方法实施
      ],
      "acceptance_checks": ["全部给定case精度自测通过并留日志", "正式评测对照case3及其余case耗时"], # 验收要求
      "inspect_files": [], # 程序从inspect操作派生；空不代表禁止读相关证据
      "modify_files": ["impl/kernel.py"], # 程序从modify/create派生，模型不填写
      "source": "human", # 人工建议专用标记；普通模型建议可不带
      "human_message_id": "human-001" # 对应 ledger.human_responses 中的 message_id
    }
  ],
  "worst_cases_tracker": { # 程序从本轮 case_analysis 按case更新，不删除其他case；旧字符串仍兼容
    "case3": {
      "case_id": "case3", # 与 ledger.case_analysis 的完整编号一致
      "observation": "speedup=0.97，仍未达标", # 本轮现象
      "explanation": "推测尾块补齐开销较高，尚需对照验证", # 当前结论
      "evidence": "eval/iter3/perf_result.json cases[case3]；impl/kernel.py 尾块分支", # 依据
      "next_action": "比较两种尾块大小，重新检查精度和性能", # 下一步
      "iteration": 3 # 程序标记最后更新轮次，避免把旧结论当作本轮
    },
    "case1": {
      "case_id": "case1", # 另一case的最新结论
      "observation": "speedup=3.43，已达标", # 本轮现象
      "explanation": "该路径当前有效，没有足够证据要求修改", # 当前结论
      "evidence": "eval/iter3/perf_result.json cases[case1]", # 依据
      "next_action": "保留路径，验证其他改动不会使其退步", # 下一步
      "iteration": 3 # 最后更新轮次
    }
  },
  "fusion_kernel_strategy": [ # Stage9：历轮融合尝试；程序去重追加，不覆盖旧轮次
    {
      "iter": 3, # 必填：这次尝试所属轮次
      "direction": "大 shape 合并卷积与 sigmoid，小 shape 保留两段执行", # 当前 role 推荐：描述实际融合方向和数据流
      "strategy": "按 shape 路由的混合融合", # 可选：程序也接受这个描述字段；不要求与 direction 同时存在
      "evidence": "develop/iter3/融合方案选择决策依据.md；eval/iter3/perf_result.json", # 依据：实际实现及评测文件
      "status": "已验证，仍有慢 case" # 尝试状态；自由文本，不是程序限定的枚举
    }
  ]
}
```

新决策不再接收 `fix_plan`，避免出现第二套修改指令；旧账本中可能保留该字段，只作历史参考。旧 `worst_cases_tracker` 自由内容仍可读取，新逐case记录采用上述结构。上面列出当前程序使用的主要字段。

## 2. 各场景切片：哪些地方会变

下面只展示相应位置的字段差异，不重复整份文件。

### 2.1 尚无历史 / Stage9 尚未完成

还没有历史文件时，程序加载的默认值如下；首次保存后才落盘：

```text
{
  "insights": [], # 暂无汇总结论
  "ledger": [], # 暂无轮次账本
  "rounds": [], # 暂无性能摘要
  "bottleneck_now": "", # 暂无瓶颈判断
  "suggest_next": [], # 暂无修改建议
  "worst_cases_tracker": {}, # 暂无慢 case 趋势
  "fusion_kernel_strategy": [] # 暂无融合尝试记录
}
```

Stage6 已写成绩、Stage9 尚未完成时，本轮 `ledger` 的 `reason` 和 `direction` 都可能是 `"(pending tech_lead)"`，两个文件列表为空，还没有本次 `stage9_decision_path`。这表示待决策，不表示账本已完整。

若接收 Stage9 决策时还没有本轮账本，合并函数会补建该轮条目：`avg_speedup_before/after`、`delta`、`verdict` 都为 `null`，再填入决策内容，不凭空补性能成绩。

### 2.2 编译失败 / 精度失败

只新增本轮 `ledger`，**不新增本轮 `rounds`**。以前轮次的成绩仍保留。编译失败的本轮条目例如：

```text
{
  "iter": 4, # 当前失败轮次
  "reason": "build_fail", # 编译失败；精度失败时为 precision_fail
  "evaluation_summary": "本轮参数类型不匹配导致编译失败，尚未评测性能", # 本轮实际检查结果
  "direction": "修正 kernel.py 的参数类型", # Stage9 的修复方向；精度失败则写精度根因与改法
  "modify_files": ["impl/kernel.py"], # 程序从本轮v2任务自动汇总；action_plan结构同第1节，本切片省略
  "readonly_files": ["impl/reference.py"], # 本轮不能改的文件
  "avg_speedup_before": null, # 本轮不做性能对比
  "avg_speedup_after": null, # 没有本轮性能成绩；不要把 null 理解为0分
  "delta": null, # 无性能差值
  "verdict": "build_fail", # 精度失败时为 precision_fail
  "stage9_decision_path": "C:/workflow-work/demo/knowledge/stage9/iter4/req-004/decision.json" # 接收成功后的决策路径
}
```

### 2.3 性能评测异常 / 零分

如果已经走到 Stage6 的历史写入步骤，**异常成绩也可能进入 `rounds` 和 `ledger`**。例如上轮2.0、本轮0分，最终账本中：

```text
{
  "reason": "score_zero: 报告缺失", # 原因只保留传给 Stage9 的第一行；实际文字随异常变化
  "avg_speedup_before": null, # 异常报告不参与性能涨跌比较
  "avg_speedup_after": 0.0, # 本轮异常或零分结果
  "delta": null, # 不把评测失败算成优化退步
  "verdict": null, # 不生成涨跌标签
  "comparison_status": "unavailable", # 当前报告不可比较
  "comparison_reason": "当前报告有评测错误", # 实际文字会指出具体错误或缺失内容
  "comparison_previous_iter": null # 当前报告已不合格，无须与前轮比较
}
```

对应 `rounds` 中 `avg_speedup=0.0`、`improved=null`；若没有 case，则 `worst_case={"case_id":"?","speedup":null}`、`worst_6_speedups=[]`。若保留了部分 case，则仍会保存其最慢摘要。case 的空值/非法数值保留为 `null` 并优先展示，不伪造0；缺失或非法的均值、评分也记 `null`，无法比较时 `delta/improved/verdict` 为 `null`。

如果评测进程重试后仍失败且没有报告，程序会在写历史前报错，**这次不会新增上述条目**。所以 `rounds` 既不是所有失败尝试的日志，也不是“全部有效成绩”的清单。

### 2.4 有 case <1：普通优化 / 累计停滞

这两个场景的历史字段结构相同：

```text
{
  "reason": "perf_optimize", # 普通优化、达到 y 次停滞窗口都使用这个原因
  "direction": "case3 仍在改善，继续优化尾块，暂不换融合方案" # Stage9 结合证据决定；也可以选择换方案
}
```

未停滞时正常给下一轮建议；停滞时重点审查融合方向。**没有专门的 `stagnation` 或触发次数字段写在 history 里**，次数和窗口另存于 `selection/`。第三次停滞触发人工咨询，也不增加新的 history 顶层字段。

### 2.5 所有 case ≥1：继续优化 / 语义退出

两个场景本轮 `ledger.reason` 都是 `"perf_pass_semantic_review"`。继续优化时仍有 P0/P1/P2 建议；程序确认满足退出条件后，Stage9 才可以给：

```text
{
  "suggest_next": [] # history 顶层：程序已确认语义退出，无后续修改建议时可为空
}
```

**不能仅凭空建议判断已退出**，因为初始值也是空列表。退出原因在 `.state.json`，最终最佳版本在 `selection/best.json`；history 没有当前标准的退出控制字段。最大迭代次数结束也不新增专用 history 字段。

## 3. 成绩标签：最容易混淆的三个字段

`ledger.verdict` 当前可能是：

```text
"verdict": "baseline" # 本轮完整有效，但没有可比前轮；包括首次测量、口径变化或前轮缺口径，before为null
"verdict": "big_win" # 提升≥5%；例如2.0→2.2，提升10%
"verdict": "small_win" # 提升≥1%且<5%；例如2.0→2.04，提升2%
"verdict": "no_change" # 变化率≥-1%且<1%；例如2.0→2.0
"verdict": "regression" # 变化率<-1%；例如2.0→1.96，退步2%
"verdict": "build_fail" # 编译失败，无性能对比
"verdict": "precision_fail" # 精度失败，无性能对比
"verdict": null # 尚无评测成绩，或当前报告无效/口径不明，不能给涨跌标签
```

- **`delta` 是绝对差值。** 2.0→2.2 时为0.2，百分比提升是10%。
- **`improved` 只在可比时判断是否上涨。** 2.0→2.001 也是 `true`；持平是 `false`。新基线、口径不明或无效报告为 `null`。
- **`verdict` 只是账本标签。** 退步2%就会标为 `regression`，不代表触发“退步≥5%写经验/回退”。这里的1%分界也不控制语义退出。

现在账本和单轮±5%经验使用同一份比较结论：报告包含完整有效case，并核对两轮硬件、case、基准及计时方式等。口径不一致或旧报告缺口径时，新完整结果作为基线；不跨过最近的异口径报告找更早成绩，不从旧 state 或缺口径的 history 均值补出涨跌。结论另存 `eval/iter<iter>/perf_comparison.json`，恢复Stage9会重新核对。`verdict` 仍用未取整变化率；单轮±5%经验仍先保留一位小数，阈值边界附近可能不同。正式三条规则见 [性能判断规则](workflow_performance_decision_rules.md)。

## 4. 人工参与的另外三种答复

第1节已展示 `direction`。下面是 `ledger[].human_responses[]` 另外三类条目，字段没有另起一套：

```text
[
  {
    "message_id": "human-002", # 人问“为什么不换融合方案”的消息编号
    "kind": "question", # 纯提问，不作为执行指令
    "answer": "case3 从0.80改善到0.97，现有证据支持继续局部优化" # Stage9 的回答
  },
  {
    "message_id": "human-003", # 人说“请等待”的消息编号
    "kind": "wait", # 等待要求，不转成人工 P0
    "answer": "已处理等待要求，现根据随后收到的意见继续决策" # Stage9 的说明；实际等待状态由程序另存
  },
  {
    "message_id": "human-004", # 人的方向与正确性或资源约束冲突
    "kind": "conflict", # 明确说明冲突，不能静默忽略人工意见
    "answer": "整个输入不能一次放入当前片上内存", # 为什么原建议不能直接执行
    "alternative": "保留减少搬运的目标，采用分块驻留和循环处理" # conflict 必填：可执行的替代方案
  }
]
```

`direction` 和 `conflict` 都必须在**本次** `suggest_next` 中对应一条人工 P0。人工任务同样填完整 v2 结构，下面只展示与人工关联有关的字段切片，其他字段同第1节：

```text
{
  "priority": "P0", # 人工方向的可执行替代方案仍明确为 P0
  "action": "人工建议的替代方案：分块驻留，减少重复搬运", # 交给 Stage3 的动作
  "reason": "满足片上内存容量限制，同时保留人的优化目标", # 替代理由
  "changes": [ # 替代方案也要写明确操作和方法，不能突破只读范围
    {"file": "impl/kernel.py", "operation": "modify", "location": "kernel分块循环", "method": "按片上容量分块驻留，在循环内处理后续块"}
  ],
  "source": "human", # 来源标记
  "human_message_id": "human-004" # 对应本次冲突处理的消息
}
```

主动介入和 Stage9 咨询后的回复，都用这套记录。咨询出题和等待阶段不提交最终账本；最终决策通过校验后才合并。超时使用推荐方向，不能伪造为人的批准。人的原话、问题选项、等待时间、Stage3 是否执行，分别保存在人工参与目录和执行回执中，不能只看这里的 `answer`。

## 5. 哪些内容另存，哪些是旧字段

以下路径均相对 `<work>`：

- **详细进步/退步经验：** `knowledge/proven_patterns.md`、`knowledge/regression_patterns.md`，包含改了什么、为什么、成绩变化、证据和程序补入的 `environment`（Markdown中的“框架与芯片”JSON块）。模型原始结构保存在本轮 `decision.json`，程序补充的环境以最终经验文件为准。history 留成绩摘要、方向、汇总结论及本轮环境。
- **Stage3 异议的裁定：** `knowledge/tech_lead_pitfalls.md`，同样保存环境、问题文件与决策文件路径。`pitfall` 不是当前 history 的顶层字段。
- **最佳实现及融合方案：** `selection/best.json` 和 `selection/records/<记录编号>/manifest.json`，包含实现方案、HAP、平均加速比、代码快照目录、性能文件和融合方案。它们才是最佳版本的完整记录。
- **停滞窗口和进入场景的次数：** `selection/state.json`、`selection/semantic_events.json`。回退记录在 `selection/rollbacks/iter<iter>/record.json`。
- **本次 Stage9 场景和输入：** `knowledge/stage9/iter<iter>/<请求编号>/request.json`、`prompt.md`；原始输出在同目录 `decision.json`，合并前历史在 `history_before.json`。
- **程序给出的填写规范：** 同目录 `decision_schema.json`（字段规则）、`decision_template.json`（本次骨架）、`case_catalog.json`（case编号与输入线索，complete标明是否完整）；完整时按case_ids校验编号，不完整时observed_case_ids仅参考。修正时 `retry1/` 同样保存。
- **决策校验失败：** 同目录 `validation_error.json` 记录具体错误；同轮最多修正两次，首次加修正共三次；两次修正的完整输入与输出分别保存在 `retry1/`、`retry2/`，`errors` 汇总能独立检查的错误。成功时账本的 `stage9_decision_path` 指向最终接收的文件。
- **人工参与：** 原话在 `human_review/inbox/`，咨询问题在 `human_review/iter<iter>/<请求编号>/问题文档.md`，状态在 `human_review/state.json`，Stage3 回执在 `develop/iter<iter>/human_feedback.json`。
- **Jev 候选概率库：** `fusion/fusion_library.json`。history 的 `fusion_kernel_strategy` 是历轮实际尝试记录，不是候选概率表。

三类经验及账本保存后，`log/workflow.log` 和 `log/state_transitions.log` 都以 `[知识积累]` 记录类型、轮次、框架、芯片、SoC、编程模型、完整输出目录和文件、决策路径及环境来源；性能经验还记录前后平均加速比和变化率。

性能经验的芯片信息只取实际比较报告的 `comparison_context.hardware`，不会因当前设备文件改变而给历史成绩换标签。错题本记录本次审查的硬件上下文。旧经验缺少环境时显示“未记录”，不猜填；框架版本目前未自动检测。Stage8只写 `search/<iter>/` 下的搜索报告和指令，已有输入/输出路径日志，不直接保存上述经验。

Stage9原始decision.json的 `plan_version=2` 会保存到history顶层；`iteration`、`request_id`、`ledger_entry`、顶层 `human_responses` 则按上面的结构合并，不原样放到history顶层。原始决策不填 `ledger_entry.modify_files/action_plan`，不填每条任务的 `inspect_files/modify_files`；这些都由程序生成。

旧文件可能见到以下内容，**不要按它们理解当前主流程**：

```text
"exit_decision": {} # 旧文件可能遗留，内部省略；当前不接收该模型字段，也不据此退出
"_pending_proven_pattern": {} # 旧成功经验中转字段；当前合并会删除
"_pending_regression_pattern": {} # 旧退步经验中转字段；当前合并会删除
"_pending_pitfall": {} # 旧异议中转字段；当前合并会删除
"proven_pattern": {} # 若遗留在旧 history 顶层，当前合并会删除；现在直接消费 decision 中的内容
"regression_pattern": {} # 同上，写入退步经验文档
"pitfall": {} # 同上，写入错题本文档
```

旧文件的其他未知字段可能继续保留，并非当前程序主动生成。旧建议仍能展示，但会标注未按 v2 校验；即使旧建议有 inspect_files/modify_files，恢复执行前也须交 Stage9 重填任务单，不猜 case、方法或验收。旧 `worst_cases_tracker` 数组仍兼容。

如果读入旧数组形式的 tracker，再写入新逐case结论，原数组保存在可选 `legacy_worst_cases_tracker` 字段，新的 `worst_cases_tracker` 改为按完整 case_id 索引。旧 case_18 与新完整编号的对应关系不猜测、不自动改写；原文仍可查。

## 6. 谁维护历史，怎么防漏

- **程序写成绩。** Stage9 只能提交本轮决策，不能改 `rounds`、旧轮账本或本轮硬指标。同轮重新执行 Stage6 时，程序会重建该轮成绩和账本，其他轮不动。
- **程序合并本轮账本。** 同轮人工答复按 `message_id` 更新并保留其他已处理消息；决策路径指向最终接收的决策。新决策禁止 `fix_plan`，本轮旧值在更新时清除；旧轮限制只作回顾。本轮评测回顾与方向概览分开展示。
- **任务单检查与存档。** 程序检查 task_id、case编号/映射覆盖、每文件操作/方法、验收与只读冲突，从 changes 生成允许列表，并把本轮原始任务保存在 ledger.action_plan。Stage3 直接看到完整任务；direction 和慢case的 next_action 只供理解，不另加指令。
- **恢复也复核。** 从 Stage3 断点恢复时检查完整 v2 任务；旧建议或冲突先回同轮 Stage9 重填，不直接沿用。inspect是重点检查对象，不限制读取其他相关证据。若将回退最佳实现，按实际回退基底核对文件，恢复后再次核对，不能用退步版本的路径假设代替最佳版本。
- **融合尝试追加。** 完全相同的对象去重；同轮不同尝试可以各存一条。
- **慢case结论逐轮留档。** 性能场景逐一检查最慢case覆盖及结论、证据、后续动作；保存到本轮 `ledger.case_analysis`，并按case更新长期tracker，未提交的其他case保留。详细原因是否正确仍需要核对证据，格式校验不能替代真实分析。
- **部分当前结论仍是替换。** `insights`、`bottleneck_now` 提供时整体替换，省略时保留；`suggest_next` 每次最终决策替换。这些汇总结论仍依赖 Stage9 整理，不能视作完整逐轮档案。历史不设新增硬字数上限，也不自动压缩或截断。
- **未通过校验不下发。** 决策校验失败时保留原 history，把独立错误汇总送回 Stage9，同轮最多修正两次；第三次仍不合格则停在 Stage9，不发给 Stage3、不写本次经验、不增加性能迭代。中断或调用故障不会被当成格式错误盲目重试；完整过程沿 `knowledge/stage9/` 追溯。

代码依据：[成绩与账本](../lib/history_manager.py)、[决策字段及合并](../lib/tech_lead.py)、[主程序调用与落盘](../orchestrator.py)、[Stage9 输出约定](../roles/stage9/decision.md)。
