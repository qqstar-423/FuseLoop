## 条件任务：详细记录本轮性能经验

仅依据程序注入的本轮可比成绩：单轮平均加速比提升 ≥5% 必填 `proven_pattern`；退步 ≥5% 必填 `regression_pattern`。未触发的字段省略；y 窗口停滞不代替单轮条件。

对象包含：

- `what_changed`：具体代码改动，引用绑定的选择依据并核对代码。
- 成功用 `why_it_worked`、退步用 `why_it_failed`：对应计算、搬运或调度证据解释原因；假设明确标注。
- `fusion_related`：布尔值，是否与融合方案有关。
- `case_analysis`：非空列表，每项都填非空文字 `case_id`、`observation`、`explanation`，既写受益项也写受损项。`explanation` 单独说明该 case 变化的原因；不能只写现象，也不能用总体 `why_it_worked/why_it_failed` 或 `ledger_entry.case_analysis` 代替。原因不确定时写明推测或待验证及验证方法，不编造结论。
- `evidence`：证据路径与该路径的 case/字段/实际测量；`<iter>` 替换为真实绑定轮次。
- `applicability`：一段非空文字，说明适用硬件、shape、参数、未验证范围和局限；不要写成对象或数组。
- `next_action`：非空文字，说明后续复用、避让或对照验证方法。

本次 `decision_schema.json` 和 `decision_template.json` 已列出触发的经验对象及完整字段；模板的空值须填写。提交前同时核对账本和经验各自的 `case_analysis`，两者用途不同，不能只补其中一份。同轮修正时逐项处理 `validation_error.json` 的 `errors` 列表，不只修第一条，也不能删除受影响 case 来绕过检查。

程序直接消费公共字段，补前后均值、差值、轮次、全部 case 数字差分及性能来源，写入 `knowledge/proven_patterns.md` 或 `knowledge/regression_patterns.md`。不输出 `_pending_*`，不写占位原因；缺少本轮必填经验会停止 Stage9。同轮复议更新原经验，原始 decision 各自留档。

框架与芯片 `environment` 由程序按本次性能报告自动补入，无须你填写。`applicability` 仍需分析硬件、shape 等适用条件；复用旧经验先核对其环境，旧记录未注明的内容不得自行当作当前环境。
