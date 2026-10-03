## 条件任务：裁定开发节点的 question.md

prompt 给出的 `develop/<iter>/question.md` 是尚未裁定的技术异议。对照原建议、开发反馈的硬证据和已知约束，判断确认误判、驳回或部分成立，不能仅凭异议更换方向。

必须在本次 decision 输出 `pitfall` 对象：`verdict` 为 `confirmed` 或 `rejected`，其余必填非空字符串 `topic`、`target_advice`、`feedback`、`root_cause`、`correct_approach`。部分成立用 `confirmed`，在原因和正确做法中分别说明成立与不成立的部分。

程序写入 `knowledge/tech_lead_pitfalls.md` 并回写原问题，你不直接修改。新建议对照已有错题本，避免重犯已确认误判。此字段缺失将停止交付，不会下发旧计划。

程序同时记录本次审查的框架、芯片和决策文件来源；环境事实由程序补入，不由模型猜填。复用旧裁定时核对适用硬件、框架和后端版本。
