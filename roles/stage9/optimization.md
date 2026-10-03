## 当前场景：未全部达标，普通性能优化

优先读 `profile/<iter>/bottleneck_analysis.md`（Stage7 瓶颈结论）与 `search/<iter>/FIX_DIRECTIVE.md`（Stage8 改法及依据），再对照实际方案、上轮建议和相关历史。具体归因有疑点时才回查 `eval/<iter>/prof_data/` 的原始数据，不重复整套 profiling 分析。

用 `eval/<iter>/perf_result.json` 核对均值、HAP 和实际列出的最多 6 个最慢有效 case，结合 `selection/state.json` 的窗口、case 趋势及 `selection/best.json` 的版本清单，看哪些改动有效。最佳快照在 `selection/records/<iter>-<指纹>/manifest.json`，按清单读取代码和报告，避免混用版本。

少数慢 case 先看 shape 分块、尾块和固定开销等局部原因；有结构性瓶颈证据时再比较 `fusion/fusion_library.json` 的候选条件及概率。决定下一轮保留、局部优化或更换方向，说明目标 case 和验证方法。不能仅因非单 kernel 强制更换融合方案。

维护本轮实际融合尝试与跨轮结论；按程序给出的单轮涨跌条件记录经验。本场景不主动求助，但须处理传入的人工意见。
