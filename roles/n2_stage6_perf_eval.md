# Triton Ascend 算子性能评测工程师

> **注意**：本文件为 stage6 的角色定义文档，用于说明性能评测的流程和数据格式。
> **stage6 的评测执行已由 orchestrator 代码自动完成**（直接 subprocess 调用 cann-bench），不再注入到 agent。
> 本文件保留作为框架设计参考，不会被 agent 加载执行。

你是 Triton Ascend 算子的性能评测工程师。性能评测由 orchestrator 自动执行 cann-bench 完成。

**评测执行流程（由 orchestrator 代码完成，非 agent）**：
1. orchestrator 调用 `python3 -m kernel_eval.cli eval` 执行性能评测，显式传入 `--perf-metric-strategy kernel_details` 与本次独立的 `--reports-dir`
2. 评测崩溃时自动重试（最多 2 次，间隔 10 秒）；每次 attempt 使用新目录，避免混入旧数据
3. 将本次报告同目录的完整 `prof_data/` 深拷贝到 `eval/<iter>/prof_data/`
4. 从这份 work 内数据解析生成 `perf_result.json`，使 `source_csv_dir` 和有效 `kernel_csv` 均指向本轮副本
5. 检测反作弊（score_error_code），自动解析 kernel_csv

## 程序输入与产物（说明文档，不是 agent 提示词）

以下路径相对本次工作目录；`<iter>` 表示 `iter0`、`iter1` 等目录名，以程序实际评测轮次为准。

- 输入 `task/`：链接到程序传入的 `task_dir`；读取接口、用例及参考实现，确定本轮评测对象，不修改测试定义。
- 输入 `impl/`：已安装且通过本轮精度评测的实现；性能数据必须对应同一代码版本。
- 输入 `device_info.json`：硬件参数和 Triton Ascend 后端信息；与程序记录的任务、评测配置及计时策略一起确定成绩是否可比较。
- 输入 `eval/<iter>/precision_result.json`、`eval/<iter>/precision_binding.json`：前者给出精度结果，后者绑定代码、报告及评测环境；程序校验二者后判断性能记录是否可入最佳库。
- 产物 `eval/<iter>/perf_result.json`：性能汇总；先看全部 case 是否达标、`avg_speedup` 和 HAP，再用 `source_json` 追溯原始报告。
- 产物 `eval/<iter>/perf_reports/`：每次 attempt 的独立原始输出目录及 JSON/MD/HTML 报告入口；以本轮 `source_json` 指定的报告核对 baseline、候选耗时及评分。
- 产物 `eval/<iter>/prof_data/`：本轮 profiler 数据；按汇总中 `kernel_csv` 指定的具体文件分析 kernel 耗时，不混入旧轮次数据。
- 保留完整目录层级，如 `prof_data/level3/fused/<算子>/<case号>/.../kernel_details.csv`。批量采集的 `_batched/` 一并保存；缺少独立 CSV 或找到多份时，该 case 的 `kernel_csv` 留空并记录日志，不把共享 CSV 强行对应给 case。
- 最佳选择相关的 `selection/current_implementation.json`、`selection/state.json`、`selection/best.json`：依次为代码与自测证据绑定、有效评测历史及窗口状态、当前最佳清单；程序结合这些记录判定入库、排序与退出，具体实现和报告按最佳清单引用的 `selection/records/<iter>-<指纹>/` 快照读取。

## 关键指标含义

baseline、HAP、耗时和评分均由 cann-bench 原评测流程提供，workflow 直接复用报告，不独立重测 baseline。`kernel_details` 汇总 kernel 执行耗时，不包含 kernel 之间的同步/调度间隔；不能将其称为完整调用耗时，也不能与旧 `trace_view` 成绩直接比较。

- **speedup**（每个 case）= `baseline_perf_us / elapsed_us`。>1 表示比 baseline 快，<1 表示慢
- **score_i**（每个 case 的性能得分）= `(T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))`。饱和型指标（0~1）
- **performance_score** = `(Σ score_i / total_cases) × 0.5 × 100`。满分 50
- **overall_score** = `compilation_score(满分20) + function_score(满分30) + performance_score(满分50)`。满分 100
- **avg_speedup** = 所有通过 case 的 speedup 平均值
- **perf_pass** = 所有 case 的 speedup ≥ 1.0 才为 true

## perf_result.json 字段

```jsonc
{
  "perf_pass": true/false,
  "overall_score": <float>,
  "performance_score": <float>,
  "avg_speedup": <float>,
  "score_error_code": "<反作弊错误码，无则 null>",
  "score_error": "<反作弊错误信息，无则 null>",
  "source_json": "<cann-bench JSON 报告路径>",
  "source_csv_dir": "<work>/eval/<iter>/prof_data",
  "source_md": "<cann-bench Markdown 报告路径>",
  "worst_6_cases": [
    {
      "case_id": "level3/fused_conv_sigmoid_1",
      "case_num": "1",
      "baseline_us": 3.75,
      "elapsed_us": 35.3,
      "speedup": 0.1062,
      "kernel_csv": "<work>/eval/<iter>/prof_data/level3/fused_conv_sigmoid/1/.../kernel_details.csv"
    }
  ]
}
```
