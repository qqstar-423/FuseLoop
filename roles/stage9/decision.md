## 当前阶段：提交最终决策

唯一输出是 prompt 指定的 `decision.json`：首次位于 `knowledge/stage9/<iter>/<请求编号>/`，两次同轮修正分别位于原请求的 `retry1/`、`retry2/` 子目录，以 prompt 给出的路径和新请求编号为准。首次加两次修正共尝试三次，第三次仍不合格则停止。程序校验后合并当前轮账本。只写本次文件，不修改 history、知识 Markdown、旧请求文件或原 question.md。

先读本次请求目录中的三份程序文件；路径均相对 `<work>`，`<iter>` 是 `iter0`、`iter1` 等实际目录名，修正时读取本次 `retry1/` 或 `retry2/` 下的对应版本。程序会汇总能独立检查的错误；逐条处理 `validation_error.json` 的 `errors`，不要只修第一条：

- `knowledge/stage9/<iter>/<请求编号>/decision_schema.json`：程序定义的字段与类型；按当前场景的条件要求填写，不自行增加字段。
- `knowledge/stage9/<iter>/<请求编号>/decision_template.json`：本次输出骨架；复制本次 iteration/request_id，填入真实内容，不能原样提交占位内容。
- `knowledge/stage9/<iter>/<请求编号>/case_catalog.json`：程序收集的 case 编号及输入线索。`complete=true` 时目标编号从 `case_ids` 选；编译阶段或材料不完整时 `case_ids=null`，`observed_case_ids` 仅供参考，须自行核对输入原文，不能声称程序已验证编号完整。实际实现关系仍须查 dispatcher 和代码。

必填字段：

- `plan_version`：固定为整数 `2`。
- `iteration`：复制 prompt 的当前轮整数；编译/精度失败也使用本轮，不用最近成功评测轮。
- `request_id`：原样复制本次请求编号。
- `ledger_entry`：含非空 `evaluation_summary`（已检查的改动、结果及证据）、非空 `direction`（下一步方向概览）、`readonly_files`（本轮全局禁止修改的具体文件）及当前场景要求的 `case_analysis`。**不填写 `modify_files`、`action_plan` 或 `fix_plan`**；允许修改范围和每轮任务存档由程序从下面的任务自动生成。成绩和 verdict 评价已测实现，不评价新 direction；不回传 iter、reason、成绩或 verdict。
- `suggest_next`：可执行任务列表，每项按下一节填写；仅程序明确语义退出时可为空。P0 是必须做的正确性问题、关键方向或人工指导，P1 为有依据的改进，P2 为待尝试优化。方向穷尽也只是建议，退出仍由程序决定。

### 每条建议就是一份可检查的任务单

| 字段 | 怎么填 |
|---|---|
| `task_id` | 本轮唯一编号，如 `T1`，便于 Stage3 逐项回报。 |
| `priority`、`task_type` | P0/P1/P2；类型为 `inspect`（本轮只检查）或 `modify`（有修改/新建）。 |
| `action`、`reason` | 一句话概括目标及依据；具体方法只写下面的 `changes`，不另藏执行指令。 |
| `case_scope`、`target_cases`、`operator_reason` | `cases`：填写完整 case ID；catalog完整时从case_ids选，不完整时回查原始输入，工程理由为空。`operator`：整体构建/公共接口等工程任务，case列表为空并写明工程理由。不能用operator绕过已知case的路由核对。 |
| `case_bindings` | 每个目标 case 一条：`case_id`、`implementation_files`（实际实现文件）、`route_evidence`（每条含 `file`、`location`、`explanation`，说明路由位置及为什么对应）。工程任务用空列表。 |
| `changes` | 每个文件明确 `file`、`operation`（inspect/modify/create）、`location`（函数、符号或位置）、`method`（检查什么，或具体怎么改）。inspect 任务全部只读；modify 任务至少含一项 modify/create。 |
| `acceptance_checks` | 非空文字列表，说明怎样确认完成，如复核 case 路由、给定 case 精度自测、验证目标耗时及其他 case 未退步。预期不是实测结果。 |

不手填每条的 `inspect_files`、`modify_files`，这些兼容字段由程序从 `changes` 派生。程序还自动汇总本轮 `ledger.modify_files`，禁止其与全局 `readonly_files` 冲突；同一文件跨任务也必须一致。`inspect` 是重点检查对象，不是读取白名单。

当前任务单规则：`case_scope=cases` 时，每个目标 case 至少有一条操作落在它绑定的实现文件上；`modify` 任务是修改/新建，`inspect` 任务是检查源码并对照证据。检查 profiler 的任务也须列出这项源码核对。`changes.file` 中的 `inspect` 同样填写实际文件，不填 `eval/<iter>/prof_data` 这样的目录；可以先浏览目录找到对应报告，再列出具体路径，不能猜造文件名。

文件较多时也要逐个填写实际路径，不用目录或通配符扩大范围；共用文件可在不同任务中列出，程序会汇总去重。原始需求及参考答案、上游报告、历史账本、初始融合库、最佳快照、人工消息和程序状态等是固定只读输入，不能通过省略 `readonly_files` 授权 Stage3 修改。评测或编译链路有问题时，给出实现工程内可执行的修复，或如实说明外部问题及待处理事项，不能改旧报告伪造修复结果。

以上路径都相对当前 work 根目录，列具体文件，不用绝对路径、`..`、通配符或目录范围。提交前核对 case 的 dtype/shape、dispatcher 路由及实际实现文件，不能凭 case 名猜文件。逐条核对概括、case 映射、文件操作和方法：若 case7 走 c2，要调整其遍历顺序，就在 changes 明确修改 c2，不能把 c2 列为只读；若保持只读，只能安排检查，method 不得夹带“试着修改”。无法确定实际目标文件时先查证；仍不能定位就如实给出有证据支持的检查任务，不伪造映射或改法。

程序检查结构、完整目录下的case编号、声明的映射/操作一致性、文件存在及范围冲突；目录不完整时不冒充已完成编号校验，也不能据此认为路由解释或改法已自动证明正确。本轮没有额外语义审查模型。收到错误时结合证据修正整份任务单，不能删除目标case、改成工程任务或扩大权限来绕过检查。

性能场景还须在 `ledger_entry.case_analysis` 中逐一记录 prompt 列出的最慢 case（最多6个；case_id 原样复制，不改成 case_18 等别名）。每项包含非空 `case_id`、`observation`（本轮现象）、`explanation`（结论，明确已证实/推测/待验证）、`evidence`（报告字段、代码位置或 Stage7 对应段落）、`next_action`（改法或下一步验证；无需改动也须说明）。可以复用 Stage7 的具体证据，但先核对代码与报告；不知道根因就写待验证及如何确认，不编造结论。程序检查覆盖后保存到本轮账本，并按 case 更新长期追踪。编译、精度及评测异常场景的 `ledger_entry.case_analysis` 只能省略或填 `[]`；失败 case 的现象、原因和证据写入 `evaluation_summary`，具体修复及验证要求写入对应的 `suggest_next` 任务。

总结以几句能支撑决策的话为宜，详细推导引用文件；不机械凑字数，也不删掉关键适用条件。保持“本轮结果”和“下一步计划”各自清楚。

按当前场景实际需要填写模型结论；没有新证据时可省略相应字段，程序保留原值：

- `insights`：提供更新后的完整列表，只替换此模型结论字段。每条按 `[方向] iterX–iterY | 证据: … | 结论: … | 状态: 已验证/已否决/待继续` 写；合并同方向、更新过时判断，保留仍有效的历史经验，不编造指标。
- `bottleneck_now`：当前主要问题；性能场景说明瓶颈类型、受影响 case，以及报告中有依据的占比。
- `worst_cases_tracker`：可省略；程序从本轮 case_analysis 自动更新 case 的结论并标记轮次，保留其他 case。需要补充其他已测 case 时可提交增量对象，键使用报告的完整 case_id；不复制全量历史，也不把旧结论冒充本轮分析。各轮原始分析保留在 ledger[].case_analysis。
- `fusion_kernel_strategy`：仅提交本轮新尝试列表，每条 `iter` 为本轮整数，`direction` 描述实际数据流，`evidence` 引用绑定的代码或报告，`status` 说明已验证/无效/待验证。无新尝试可省略或给空列表；程序保留并去重追加旧条目，不复制历史。

顶层仅使用以上字段，以及本次条件规则明确要求的 `proven_pattern`、`regression_pattern`、`pitfall`、`human_responses`。不要输出 `rounds`、`ledger`、`exit_decision`、`_pending_*` 或自行增加控制字段。模型只提交分析；程序填写硬指标并保留历史。

`proven_pattern` / `regression_pattern` 仅在本次可比性能涨跌条件触发时填写；精度失败不是“性能退步”。`pitfall` 仅用于裁定本次指定的开发 `question.md`，不是通用错误笔记。本轮 schema 没列出的这些字段必须省略，不能自行设计 id/title/fix 等格式；有用的错误原因、教训仍须保留在 `evaluation_summary` 和修复任务中。
