## 当前阶段：只生成咨询问题

程序已确认第 3 次停滞触发，必须向人寻求方向判断。此调用不提交最终 decision，不填写最终 ledger 或 P0，不写知识经验，不下发 Stage3，也不自行等待。

唯一输出为 prompt 指定的 `human_review/<iter>/<请求编号>/question.json`。程序校验后渲染同目录《问题文档.md》，通知并负责 2 分钟等待；延长、截止和回复由程序处理。问题写清楚、简短易读，允许人提出其他方向。

JSON 字段：

除 `options` 为对象列表外，其余字段均为非空字符串（包括 `attempts`、`evidence`）。

- `request_id`：原样复制本次咨询请求编号。
- `question`：这次要人判断的明确问题。
- `difficulty`：当前难点、哪些 case 未达标，为什么目前难以取舍。
- `current_scheme`：当前实际融合方案及目标。
- `attempts`：已经尝试的方向与结果，不能提出已被相同条件证伪的选项。
- `evidence`：3 次触发轮次、最佳 avg_speedup/HAP、慢 case 改善趋势，以及实现/性能报告/方案依据的实际版本路径和读法；直接引用程序材料，不编造数字。
- `options`：2～3 个对象，每项有唯一 `id`（如 A/B/C）、`title`、`benefit`、`cost`、`risk`，提出实际可行的不同方向。
- `recommended_option`：上述一个真实选项的 id。
- `recommendation_reason`：根据当前证据为何优先推荐此项。

候选可包括局部优化、特定 shape 更换融合、先补实验，但须适合当前硬件与证据，不能照抄模板。人工先前意见纳入问题背景，等待完成后的最终调用才处理完整反馈和原场景决策。
