## 当前场景：未全部达标，停滞审查

当前程序确认 y 次有效性能迭代的最佳平均加速比累计提升不足 5%。先读 Stage7 的 `profile/<iter>/bottleneck_analysis.md`、Stage8 的 `search/<iter>/FIX_DIRECTIVE.md`、绑定的方案理由及相关历史，再比较 `fusion/fusion_library.json` 的候选条件和概率。

结合 `eval/<iter>/perf_result.json` 的均值、HAP 和未达标 case、`selection/state.json` 的窗口与 case 趋势、`selection/best.json` 及其 `selection/records/<iter>-<指纹>/manifest.json` 快照，判断局部问题还是结构性限制。需要时回查对应版本的代码、性能来源和 `eval/<iter>/prof_data/`，不能拿当前代码解释旧成绩。

平均提升小但慢 case 持续接近 1 时，可以继续当前方向。保留、局部优化或更换均须有证据、目标 case 和下一步验证；停滞不等于单轮退步 ≥5%，两类知识记录不能混淆。

程序决定这是第几次触发，以及本次是最终决策、只生成咨询问题还是反馈后复议。只完成下方指定阶段；等待和计数不由你执行。
