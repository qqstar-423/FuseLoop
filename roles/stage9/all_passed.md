## 当前场景：全部 case 达标，继续优化或收尾

本轮每个 case 的 speedup ≥1，实际路由没有经过本轮 Stage7/8，不要求读取不存在的本轮分析与搜索结论。

仍只提交本次 decision JSON。程序在校验、提交成功后，将 `ledger_entry.evaluation_summary` 和 `case_analysis` 整理成本轮唯一 Markdown 分析报告，标注“分析来源：Stage9”，再交给 Stage3；无需另写报告文件。写清现象、原因、证据和下一步，未读取的原始 profiler 数据不能写成已验证事实。

读 `eval/<iter>/perf_result.json`、对应 `eval/<iter>/prof_data/` 和 `eval/<iter>/perf_reports/`，结合绑定代码、方案选择依据和历史评估上轮目标及本轮收益。保留必要瓶颈分析：按 case 核对 kernel、搬运、计算和固定开销；需要时读项目根目录 `skills/triton-profiling-analysis/SKILL.md` 的五文件流程。

比较 `fusion/fusion_library.json` 的方法条件时以实测为准，已验证有效的多 kernel 方案可以继续。用 `selection/state.json` 的 x 窗口和 `selection/best.json` 的最佳清单理解程序决定，不自行宣布退出。最佳代码和报告通过 `selection/records/<iter>-<指纹>/manifest.json` 绑定，不混用版本。

程序明确语义退出时完成本轮经验与人工意见处理；否则给下一轮方向。退出前人工意见若受现有退出条件或迭代上限限制不能执行，要记录原因交 Stage10，不以人工意见自行突破限制。本场景不触发未达标停滞咨询。
