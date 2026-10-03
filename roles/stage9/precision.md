## 当前场景：精度失败

读 `eval/<iter>/precision_result.json` 的失败 case，再查 `eval/<iter>/precision_reports/` 原始误差、`task/cases.yaml` 和 `task/golden.py`，结合编译日志、代码与绑定的自测结果。

解释自测与正式检查的差异，重点区分计算/边界输入错误、旧数据复用、连续调用未覆盖和环境问题。检查上轮正确性修复目标是否落实，明确修复方向、文件范围、下一轮验证 case。没有本轮正式性能结果，不要求性能优化、收益归因或涨跌经验。
