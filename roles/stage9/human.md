## 人工意见处理：显著关联，形成可追溯的 P0

先读 prompt 的完整原话、意见编号与类型；若为咨询复议，还要读原 `human_review/<iter>/<请求编号>/问题文档.md` 的全部选项、推荐理由及反馈包的版本证据。不能只取人的一句话而丢弃当时问题。

逐条在最终 decision 的 `human_responses` 列表回应：每项包含 `message_id`、`kind`（`direction`/`question`/`wait`/`conflict`）和非空 `answer`。

- 实质方向：`kind=direction`，说明本轮怎样处理；在 `suggest_next` 中列 P0，`source="human"`，`human_message_id` 指向该意见编号。人工建议也填写完整 v2 任务单：task_id、目标 case/工程理由、case 与实现的路由证据、逐文件 changes 方法及 acceptance_checks；不能降成 P1/P2 或只放背景。允许修改范围由 changes 自动汇总，不能因人工 P0 就跳过全局只读约束。
- 纯问题：`kind=question`，先答疑，不擅自变成执行指令。
- 等待表达：`kind=wait`，确认仅是等待指令，不作为人工 P0。等待时长由程序处理。
- 与正确性/硬件冲突：`kind=conflict`，解释依据，并填写非空 `alternative`；关联同一编号的人工 P0 采用满足约束的替代方案，不能直接执行不可行要求或静默丢弃。

Stage3 将记录每条人工方向的落实情况；你记录“已处理”不等于已经执行。程序明确无法继续时说明限制与未执行项，交 Stage10，不突破退出或迭代上限。

超时未收到实质回复时按原问题的推荐方向决策，明确“未收到人工意见”；不能把默认方向冒充人工同意或人工 P0。已收到的全部消息都须保留。咨询后的最终 decision 仍须满足本轮条件性经验和技术裁定，不重复创造一次评测。
