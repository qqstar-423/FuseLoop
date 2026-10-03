工作目录：/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914
算子：fused_patch_concat_layernorm
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/task`；相对工作目录：`task`；用途：只读算子需求和评测基准；怎么看：对照 desc.md、proto.yaml、cases.yaml、golden.py 判断建议是否保持语义和完整 case 覆盖
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/ANALYSIS.md`；相对工作目录：`ANALYSIS.md`；用途：Stage1 需求分析；怎么看：核对接口、精度、硬件约束和实现难点，避免偏离需求
- 文件：`/mnt/workspace/pypto-pro-workflow/knowledge/anti_cheat_reference.md`；相对项目根目录：`knowledge/anti_cheat_reference.md`；用途：反作弊判定参考；怎么看：按实际错误处理表对照错误码、kernel CSV 和真实代码定位违规或零分原因
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/impl`；相对工作目录：`impl`；用途：当前代码；怎么看：按本场景问题核对入口和相关实现
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/history.json`；相对工作目录：`knowledge/history.json`；用途：跨轮经验、性能轮次与修改账本；怎么看：只读：先读 suggest_next、insights、ledger 和 rounds；仅向本次 decision 提交当前轮账本，程序负责合并历史
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/device_info.json`；相对工作目录：`device_info.json`；用途：本次硬件参数来源；怎么看：先核对芯片、编程模型、核数及 UB/L1 容量，再判断方案和 tiling 是否可行
🔧 当前昇腾 NPU 与 Triton Ascend 环境（所有 stage 共享）
  芯片型号: Ascend910B3
  SoC 版本: Ascend910B3
  NPU 架构: dav-c220
  框架: Triton；后端: triton-ascend；运行时版本: {"triton": "3.2.0", "triton_ascend": "3.2.1", "torch": "2.7.1+cpu", "torch_npu": "2.7.1.post8"}
  编程模型: Triton block program (SPMD)
  后端 target: npu/Ascend910B3
  AI Core 数: 20（Cube 20 + Vector 40）
  UB: 192 KB；L1: 512 KB
  L0A/L0B/L0C: 64/64/128 KB
  L2 Cache: 196608 KB
  Device ID: 0（现有可见设备映射中的逻辑编号）
  检测方式: tbe_device_query
  使用 import triton、import triton.language as tl、@triton.jit 与显式 grid。
  BLOCK 分块、mask、stride 和矩阵计算参数须结合这些资源与本机后端支持；片上缓冲区由编译器管理。
  不照搬 CUDA warp、shared-memory、PTX 或特定 GPU 的异步指令假设；以已安装 triton-ascend 的 API 为准。
- 文件：`/mnt/workspace/pypto-pro-workflow/knowledge/arch_programming_guide.md`；相对项目根目录：`knowledge/arch_programming_guide.md`；用途：Triton Ascend 架构编程指南；怎么看：核对本机 API、分块、边界 mask 和内存访问约束
  example/ 示例只用于接口与基本写法参考；分块参数仍需针对当前算子和芯片验证。

本轮失败原因：perf_optimize

===== 本轮不比较性能涨跌 =====
没有上一轮有效性能结果，不计算涨跌；本轮作为新基线，后续仅与同口径完整结果比较。不得用本次均值差推断优化收益或退步，不填写本次 proven_pattern/regression_pattern。
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/perf_comparison.json`；相对工作目录：`eval/iter1/perf_comparison.json`；迭代模板：`eval/<iter>/perf_comparison.json`；用途：程序的性能比较口径检查；怎么看：查看不比较的原因和两轮报告来源，不跨口径归因
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/fusion_library.json`；相对工作目录：`fusion/fusion_library.json`；用途：JSON 融合算子库；怎么看：比较候选数据流、前提与初始概率，以实测证据决定保留或更换
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/perf_result.json`；相对工作目录：`eval/iter1/perf_result.json`；迭代模板：`eval/<iter>/perf_result.json`；用途：本轮性能结果；怎么看：看 avg_speedup、各 case 的 speedup 和最慢用例，不只看平均值
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/prof_data`；相对工作目录：`eval/iter1/prof_data`；迭代模板：`eval/<iter>/prof_data`；用途：本轮 profiler 数据（按需回查）；怎么看：仅在结论矛盾或缺少证据时按 case 回查
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/perf_reports`；相对工作目录：`eval/iter1/perf_reports`；迭代模板：`eval/<iter>/perf_reports`；用途：本轮性能报告；怎么看：回查原始计时与评分，核实性能汇总及反作弊信号
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/profile/iter1/bottleneck_analysis.md`；相对工作目录：`profile/iter1/bottleneck_analysis.md`；迭代模板：`profile/<iter>/bottleneck_analysis.md`；用途：本轮瓶颈分析；怎么看：优先读 Stage7 结论，区分已证实根因和推测
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter1/FIX_DIRECTIVE.md`；相对工作目录：`search/iter1/FIX_DIRECTIVE.md`；迭代模板：`search/<iter>/FIX_DIRECTIVE.md`；用途：本轮修改指令；怎么看：优先读 Stage8 结论，转成原有 P0/P1/P2 优先级
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter1/SEARCH_REPORT.md`；相对工作目录：`search/iter1/SEARCH_REPORT.md`；迭代模板：`search/<iter>/SEARCH_REPORT.md`；用途：本轮搜索报告；怎么看：按需检查候选方法、依据及适用条件
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/build/iter1/build.log`；相对工作目录：`build/iter1/build.log`；迭代模板：`build/<iter>/build.log`；用途：本轮编译日志；怎么看：看 STATUS 和首个真实错误，定位构建/接口问题
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/precision_result.json`；相对工作目录：`eval/iter1/precision_result.json`；迭代模板：`eval/<iter>/precision_result.json`；用途：本轮精度结果；怎么看：核对总数、通过数及失败 case，优先处理正确性
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/precision_reports`；相对工作目录：`eval/iter1/precision_reports`；迭代模板：`eval/<iter>/precision_reports`；用途：本轮精度报告详情；怎么看：按失败 case 查误差、输入和原始报错
当前场景缺失材料（不能当作已完成结论）：
历轮设计思路：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/design_rationale.md`；相对工作目录：`develop/iter0/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter0 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
请分析本轮结果，回顾历轮设计思路；只读 /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/history.json，不要覆盖它。
Stage9 输出文件：/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/decision.json
Stage9 请求编号：508b8258a96b4adc89fcc0462b664058
Stage9 当前轮次：1
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/decision.json`；相对工作目录：`knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/decision.json`；迭代模板：`knowledge/stage9/<iter>/508b8258a96b4adc89fcc0462b664058/decision.json`；用途：本次 Stage9 决策输出（待你生成）；怎么看：按 role 写 iteration、request_id、ledger_entry 和模型结论；经验使用 proven_pattern/regression_pattern/pitfall 公共字段

本次 ledger_entry.case_analysis 必须逐一覆盖以下 case_id（原样复制，不能缩写；每例写 observation、explanation、evidence、next_action；根因未证实须注明待验证）：
["level3/fused_patch_concat_layernorm_19", "level3/fused_patch_concat_layernorm_21", "level3/fused_patch_concat_layernorm_1", "level3/fused_patch_concat_layernorm_20", "level3/fused_patch_concat_layernorm_3", "level3/fused_patch_concat_layernorm_10"]


## JSON 融合算子库（Stage1.5 Jev）
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/fusion_library.json`；相对工作目录：`fusion/fusion_library.json`；用途：只读初始融合候选库，提供 Jev 选出的 Top N 方法及概率；怎么看：先看 candidates 的 rank、method 和 probability；概率用于选择初始方向，后续以实测为准
Stage9：结合本节点已有职责参考以下融合方案及其概率，实际精度和性能证据优先。
概率表示各方案独立的初始适用性估计，不要求加和为1，也不代表实测加速比。保留原始 Jev 概率；不得自行修改、归一化或伪造。本轮接入不改变已有评测、P0/P1/P2 审查或退出路由。
- 文件：`/mnt/workspace/pypto-pro-workflow/knowledge/fusion_method.md`；相对项目根目录：`knowledge/fusion_method.md`；用途：融合方法原始说明；怎么看：需要追溯评分时，核对方法适用条件和限制
- 文件：`/mnt/workspace/pypto-pro-workflow/knowledge/fusion_options.json`；相对项目根目录：`knowledge/fusion_options.json`；用途：完整融合方法选项表；怎么看：按 method id 对照方法及变体，区别于筛选后的 Top N 库
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/ANALYSIS.md`；相对工作目录：`ANALYSIS.md`；用途：Stage1 需求分析；怎么看：核对算子语义、输入范围和实现约束
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion_requirements.en.json`；相对工作目录：`fusion_requirements.en.json`；用途：提供给 Jev 的结构化算子需求；怎么看：核对 operator_summary、case_groups 和 implementation_constraints
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/device_info.json`；相对工作目录：`device_info.json`；用途：当前硬件信息；怎么看：核对设备型号、内存容量及方法所需硬件能力
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/translation/english_inputs.json`；相对工作目录：`fusion/translation/english_inputs.json`；用途：实际用于评分的英文材料副本；怎么看：仅在追溯 Jev 输入时查看，按原需求、硬件和方法逐项核对
{
  "schema_version": 1,
  "protocol": "fusion-suitability/v2-en",
  "fingerprint_sha256": "ad1acfcea11c0e71dd1821fbb8e023e7aa98a1fc633160e04978ccaa0fb040d2",
  "response_sha256": "4ad68049de7f0b5b015e17ceb5764cac5f9cb4765bafe817939294f47a18fce4",
  "sources": {
    "methods": {
      "path": "/mnt/workspace/pypto-pro-workflow/knowledge/fusion_method.md",
      "sha256": "0f4551524c7736322930fb5dea5d7f370f92ca10ad78514d69c82b5d8f515ae5"
    },
    "options": {
      "path": "/mnt/workspace/pypto-pro-workflow/knowledge/fusion_options.json",
      "sha256": "aa69dd32252e0cf35f29947d2412a8a897783c7bac0e2ff74328a006797257a2"
    },
    "stage1_analysis": {
      "path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/ANALYSIS.md",
      "sha256": "1f35661220edc12e55e5038ee53972124e9b1e5100f7503c5b42d738f401adcb"
    },
    "stage1_requirements": {
      "path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion_requirements.en.json",
      "sha256": "79b9ef114a7266a6df7c0dae38d19897d728821a235d98cab9dde09d9417d22e"
    },
    "hardware": {
      "path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/device_info.json",
      "sha256": "a87c5ed65c87b6b653692c4463dedab1d637c782fba1e2863ca87c862a7837e1"
    }
  },
  "english_inputs_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/translation/english_inputs.json",
  "jev_config": {
    "base_url": "https://api.typesafe.ai",
    "model": "jev-1.13.0",
    "timeout_seconds": 30,
    "max_retries": 1,
    "max_state_question_bytes": 28000,
    "max_request_bytes": 56000
  },
  "budget": {
    "method": "conservative_utf8_json_bytes_not_exact_tokens",
    "state_bytes": 19367,
    "state_plus_longest_question_bytes": 22117,
    "request_bytes": 45405
  },
  "cache_reused": false,
  "probability_semantics": "Independent probability that this method is a feasible and promising implementation direction for the supplied operator requirements on the current hardware. These are initial Jev estimates, not measured performance, and need not sum to one.",
  "total_methods": 10,
  "top_n": 3,
  "candidates": [
    {
      "rank": 1,
      "method": {
        "id": "F2",
        "name": "Vertical Fusion",
        "name_zh": "垂直融合／单 kernel 全融合",
        "summary": "Fuse a producer-consumer chain and retain legal intermediate tiles on chip within one kernel.",
        "graph_conditions": [
          "A consumer depends on a producer.",
          "Tiling permits consumption of ready intermediate data."
        ],
        "case_features": [
          "Compute plus bias, activation, casting or compatible layout work.",
          "Traffic/launch savings are material and live tiles fit resources."
        ],
        "hardware_requirements": [
          "Backend supports all operations in one kernel.",
          "Adequate registers/local memory and supported transfers/synchronization."
        ],
        "execution_organization": "One kernel coordinates producer and consumer tiles, possibly across compute and vector sections.",
        "dataflow": "Producer tile -> registers/local memory -> consumer tile -> output, removing selected intermediate materialization.",
        "benefits": [
          "Remove intermediate traffic and launches.",
          "Reuse locally available producer data."
        ],
        "risks": [
          "Resource pressure, lower parallelism or spills.",
          "Transfer/synchronization cost and pipeline imbalance."
        ],
        "variants": [
          {
            "id": "F2.compute_epilogue",
            "name": "Compute with fused epilogue",
            "description": "Apply bias, activation and conversion to completed compute tiles before final output stores."
          },
          {
            "id": "F2.loop_chain",
            "name": "Compatible loop-chain fusion",
            "description": "Combine compatible dependent loops while preserving reduction boundaries."
          },
          {
            "id": "F2.tiled_pipeline",
            "name": "Tiled producer-consumer pipeline",
            "description": "Overlap successive tiles with supported buffering and synchronization; tune live resource use."
          }
        ],
        "implementation_checks": [
          "Count all live tiles, buffers and compiler storage.",
          "Verify transfer layout, synchronization and precision.",
          "Inspect spills, stalls and occupancy; one kernel is not automatically optimal.",
          "Verify the required Triton Ascend operations and compiler options on the installed backend; preserve the task interface, strides, masks, dtype and measured work scope."
        ],
        "compatible_with": [
          "F1",
          "F3",
          "F4",
          "F5",
          "F7",
          "F8",
          "F9"
        ],
        "source_sections": [
          "F2: Vertical Fusion（垂直融合）"
        ]
      },
      "probability": 0.79
    },
    {
      "rank": 2,
      "method": {
        "id": "F5",
        "name": "Hybrid Routing",
        "name_zh": "混合路由／按 shape 分发",
        "summary": "Route metadata-defined case groups to validated concrete fusion or tiling variants.",
        "graph_conditions": [
          "Several implementations satisfy one complete contract.",
          "A legal dispatcher can inspect relevant metadata."
        ],
        "case_features": [
          "Shape, dtype, layout or parameters change the best plan.",
          "A small case group needs specialized tiling or fusion."
        ],
        "hardware_requirements": [
          "Supported dispatch and compilable variants.",
          "Complete permitted-case coverage or a valid fallback."
        ],
        "execution_organization": "Dispatch to a concrete implementation such as F1/F2; routing itself is not a low-level fusion primitive.",
        "dataflow": "Permitted metadata -> selected variant -> complete variant execution -> contractual outputs.",
        "benefits": [
          "Adapt to heterogeneous cases.",
          "Optimize difficult groups locally."
        ],
        "risks": [
          "Variant growth, compilation cost and fragile boundaries.",
          "Noise overfitting, benchmark-identity routing or stale cached data."
        ],
        "variants": [
          {
            "id": "F5.fusion_boundary_routing",
            "name": "Fusion-boundary routing",
            "description": "Select plans such as F1/F2 using measured metadata-defined case groups."
          },
          {
            "id": "F5.schedule_routing",
            "name": "Schedule and tile routing",
            "description": "Keep the fusion family while selecting tiles, layouts or buffers for different case groups."
          },
          {
            "id": "F5.specialized_with_fallback",
            "name": "Specializations with fallback",
            "description": "Use justified specializations plus a correct fallback; record predicates and affected cases."
          }
        ],
        "implementation_checks": [
          "Document metadata predicates; test routes and boundaries.",
          "Test fresh tensor values across calls; reuse allocations, not stale contents.",
          "Measure dispatch and mandatory preprocessing consistently.",
          "Validate precision on every selected path.",
          "Verify the required Triton Ascend operations and compiler options on the installed backend; preserve the task interface, strides, masks, dtype and measured work scope."
        ],
        "compatible_with": [
          "F1",
          "F2",
          "F3",
          "F4",
          "F6",
          "F7",
          "F8",
          "F9",
          "F10"
        ],
        "source_sections": [
          "F5: Hybrid Routing（按形状路由）"
        ]
      },
      "probability": 0.58
    },
    {
      "rank": 3,
      "method": {
        "id": "F8",
        "name": "Joint Compute-Memory Fusion",
        "name_zh": "计算与访存联合融合",
        "summary": "Jointly schedule compute-heavy stages and memory-heavy subgraphs to reduce materialization or overlap work.",
        "graph_conditions": [
          "Compute connects to elementwise, reduction or layout subgraphs.",
          "Tile dependencies allow a staged schedule."
        ],
        "case_features": [
          "Alternating compute/memory stages leave traffic or idle resources.",
          "Back-to-back compute has a transformable intervening subgraph."
        ],
        "hardware_requirements": [
          "Backend supports the proposed staged operations/schedule.",
          "Supported overlap, buffers and dependency synchronization."
        ],
        "execution_organization": "Assign compute and memory work to supported stages/streams; compiler scheduling still enforces dependencies.",
        "dataflow": "Ready tiles feed memory and later compute stages through valid buffers; distinguish removed from merely overlapped materialization.",
        "benefits": [
          "Reduce intermediates or idle resources.",
          "Jointly optimize layout, reuse and overlap."
        ],
        "risks": [
          "Unsupported schedule or hidden materialization.",
          "Buffering, contention and synchronization exceed gains.",
          "Serial dependencies limit concurrency."
        ],
        "variants": [
          {
            "id": "F8.compute_memory_pipeline",
            "name": "Compute / memory-subgraph pipeline",
            "description": "Pipeline compute tiles with connected elementwise, layout or reduction work and valid dependencies."
          },
          {
            "id": "F8.back_to_back_compute",
            "name": "Back-to-back compute stages",
            "description": "Jointly schedule consecutive compute stages and transforms when layouts/reductions permit."
          }
        ],
        "implementation_checks": [
          "Inspect generated execution and traffic, not stream names.",
          "Verify buffer lifetime and all dependencies.",
          "Compare utilization and complete-scope latency with F1/F2.",
          "Verify the required Triton Ascend operations and compiler options on the installed backend; preserve the task interface, strides, masks, dtype and measured work scope."
        ],
        "compatible_with": [
          "F1",
          "F2",
          "F5",
          "F6",
          "F7",
          "F9"
        ],
        "source_sections": [
          "F8: Joint Compute-Memory Fusion（计算访存联合融合）"
        ]
      },
      "probability": 0.44
    }
  ]
}

## 与当前代码绑定的融合选择依据
实现 SHA256：f9c75b8658b1f0d769b4752ccb1cd1361f194f3c70f884ed668c4d19fc8d4324
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/fusion_library.json`；相对工作目录：`develop/iter0/fusion_library.json`；迭代模板：`develop/<iter>/fusion_library.json`；用途：当前轮实际融合选择与方案库；怎么看：看 selection 和新增方法；原始 Jev probability 只是先验参考
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/design_rationale.md`；相对工作目录：`develop/iter0/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：当前代码的融合选择决策依据；怎么看：先看选定方案、未选其他方案的原因及目标 case，再与实测核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/self_test_report.md`；相对工作目录：`develop/iter0/self_test_report.md`；迭代模板：`develop/<iter>/self_test_report.md`；用途：当前代码的自测报告；怎么看：看给定 case 与连续调用是否实际执行，区分失败、未执行和通过
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/self_test_result.json`；相对工作目录：`develop/iter0/self_test_result.json`；迭代模板：`develop/<iter>/self_test_result.json`；用途：程序校验使用的自测 JSON；怎么看：看 provided_cases、continuous_calls 的通过状态及 evidence_path
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/self_test.log`；相对工作目录：`develop/iter0/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：给定 case 的原始自测日志；怎么看：核对实际命令、case 数量及结果是否支持自测报告
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/self_test.log`；相对工作目录：`develop/iter0/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：连续调用的原始自测日志；怎么看：核对相同 shape 更换参数后逐次对照参考实现的结果
最佳实现入库资格：True；Source, fusion decision and self-tests are bound.
本轮库与只读初始 Jev 库区分；真实评测优先于初始概率。
{
  "method_ids": [
    "F2"
  ],
  "implementation_plan": "F2.compute_epilogue / F2.loop_chain 变体：单 @triton.jit kernel 端到端融合 Slice×4(strided gather) → Concat(寄存器/UB 内拼接，不落 GM) → LayerNorm。grid = B*(H/2)*(W/2)，每 program 串行处理 1 个输出 token；通道向量 BLOCK_C = next_power_of_2(C)（最大 512）。4 段按 golden 顺序 TL(+0)/BL(+W*C)/TR(+C)/BR(+W*C+C) 带 mask 载入并升精度到 fp32；统计采用两遍法 + 均值修正轮：round1 求 sum→mean0，round2 在同一遍 diff 上同时累加 sum(diff)（修正量 corr，mean=mean0+corr）与 sum(diff²)（var=Σdiff²/n−corr²，平行轴定理），归约轮数 3→2；rstd=1/sqrt(var+eps)，仿射 (diff−corr)*rstd*w+b 按段延迟加载 w/b 并回写原 dtype。host 侧仅做元数据/连续性守护、torch.empty 分配输出与 kernel 启动。",
  "reason": "概率最高项 F2(0.79) 与算子结构完全匹配：Concat 是纯跨行/列 strided gather、LayerNorm 是逐 token 全量归约，二者天然可在单 kernel 内以寄存器/UB tile 衔接，中间 concat 张量无需在 HBM 物化；满足 F2 的硬件前提（dav-c220 上 tl.load/store/sum/sqrt/where 均可用，实测冒烟通过）。概率第二的 F5(0.58) 本轮不作为独立实现：所有 case 共享同一数学定义，仅 C 尺寸不同且由 BLOCK_C=next_power_of_2(C) 一个 constexpr 自然覆盖，无需要路由的异质 case 组，引入 dispatcher 只增加分支与编译成本；其精神（按 shape 选择分块）已由 BLOCK_C 的 host 侧计算体现。F8(0.44) 的计算/访存交错流水在本算子上无交错段（load→reduce→store 单链），不适用。概率是初始方向参考而非性能结论，后续以 cann-bench 实测为准。",
  "target_cases": [
    "全部 30 个 case（10 个 shape 组 × fp16/fp32/bf16），含 C∈{3,65} 非 2 幂尾块、零方差（case 25-27）、近常数强相消（case 28-30）、矩形输入（case 19-21）、grid=1（case 22）与最大 grid=2304（case 10-12）"
  ],
  "actual_changes": [
    "新增 impl/cann_bench/fused_patch_concat_layernorm.py：单 kernel 垂直融合实现（_fused_patch_concat_layernorm_kernel + host 包装函数）",
    "均值统计采用两遍法 + corr 修正轮，diff 通道向量用 tl.where(mask,·,0) 防尾块污染",
    "x/w/b 全部升精度 fp32 参与统计与仿射，输出前转回 x.dtype",
    "基址与输出基址使用 int64（pid.to(tl.int64)）防大张量溢出，通道偏移保持 int32",
    "impl/cann_bench/__init__.py 导出 fused_patch_concat_layernorm；setup.py/build.sh 按 example 工程"
  ],
  "expected_benefits": [
    "相比未融合的三段实现（slice/cat kernel + layer_norm kernel）：省去 4C 中间张量的一次 HBM 写 + 一次 HBM 读（最大 case 约 2.4 MB fp16 / 4.7 MB fp32 往返），并省一次 kernel 启动与同步开销",
    "四段 gather 与归约在 program 内完成，无跨 program 依赖、无 workspace、无原子操作",
    "修正轮合并使归约轮数保持 2 轮，同时保住近常数 case（rstd≈171）的相消精度"
  ],
  "methods": [
    {
      "rank": 1,
      "method": {
        "id": "F2",
        "name": "Vertical Fusion",
        "name_zh": "垂直融合／单 kernel 全融合",
        "summary": "Fuse a producer-consumer chain and retain legal intermediate tiles on chip within one kernel.",
        "graph_conditions": [
          "A consumer depends on a producer.",
          "Tiling permits consumption of ready intermediate data."
        ],
        "case_features": [
          "Compute plus bias, activation, casting or compatible layout work.",
          "Traffic/launch savings are material and live tiles fit resources."
        ],
        "hardware_requirements": [
          "Backend supports all operations in one kernel.",
          "Adequate registers/local memory and supported transfers/synchronization."
        ],
        "execution_organization": "One kernel coordinates producer and consumer tiles, possibly across compute and vector sections.",
        "dataflow": "Producer tile -> registers/local memory -> consumer tile -> output, removing selected intermediate materialization.",
        "benefits": [
          "Remove intermediate traffic and launches.",
          "Reuse locally available producer data."
        ],
        "risks": [
          "Resource pressure, lower parallelism or spills.",
          "Transfer/synchronization cost and pipeline imbalance."
        ],
        "variants": [
          {
            "id": "F2.compute_epilogue",
            "name": "Compute with fused epilogue",
            "description": "Apply bias, activation and conversion to completed compute tiles before final output stores."
          },
          {
            "id": "F2.loop_chain",
            "name": "Compatible loop-chain fusion",
            "description": "Combine compatible dependent loops while preserving reduction boundaries."
          },
          {
            "id": "F2.tiled_pipeline",
            "name": "Tiled producer-consumer pipeline",
            "description": "Overlap successive tiles with supported buffering and synchronization; tune live resource use."
          }
        ],
        "implementation_checks": [
          "Count all live tiles, buffers and compiler storage.",
          "Verify transfer layout, synchronization and precision.",
          "Inspect spills, stalls and occupancy; one kernel is not automatically optimal.",
          "Verify the required Triton Ascend operations and compiler options on the installed backend; preserve the task interface, strides, masks, dtype and measured work scope."
        ],
        "compatible_with": [
          "F1",
          "F3",
          "F4",
          "F5",
          "F7",
          "F8",
          "F9"
        ],
        "source_sections": [
          "F2: Vertical Fusion（垂直融合）"
        ]
      },
      "probability": 0.79
    }
  ]
}

# FusedPatchConcatLayernorm 首版设计思路（develop/iter0）

本轮实现对应的融合方案选择决策见同目录 `fusion_library.json` 的 `selection` 字段；本文给出完整设计依据。自测证据见 `self_test_report.md` 与 `self_test.log`。

## 1. 整体算法方案

选择 **F2 垂直融合（单 kernel 全融合）**：一个 `@triton.jit` kernel 端到端完成 Slice×4 → Concat → LayerNorm。

选择理由：

1. **算子结构与 F2 天然匹配**。Concat 在数学上是纯跨行/跨列的 strided gather（4 个间隔 `0 / W*C / C / W*C+C` 的等长通道段），LayerNorm 是逐 token 的 4C 通道全量归约。两者以「program 内寄存器/UB tile」衔接即可，中间 concat 张量（`[B, (H/2)(W/2), 4C]`）完全不需要在 HBM 物化——这正是 F2 的定义（producer tile → 寄存器/本地存储 → consumer tile → 输出，消除中间物化）。
2. **硬件前提已实测核实**。dav-c220 / triton-ascend 3.2.1 上 `tl.load/tl.store`（含 bf16）、`tl.sum`、`tl.sqrt`、`tl.where`、masked load/store、int64 基址均可用（/tmp 冒烟测试与 30 case 全量自测均编译运行通过），满足 F2 的 "backend supports all operations in one kernel"。
3. **无跨 program 依赖**：每个输出 token 的 4C 通道归约只依赖 4 个输入 token 的数据，program 之间无数据交换，无需 workspace、原子操作或第二 kernel。

统计精度方案（决定实现路径的关键点）：

- **所有 load 后立即升精度 fp32**。近常数 case 28–30（x∈[0.99,1.01]，fp16 波动仅 ~10 ulp）若在 fp16 内累加，mean/var 全部失真；golden 的 `F.layer_norm` 对 fp16/bf16 输入也在内部以 fp32 统计。
- **两遍法 + 均值修正轮**，且修正与方差合并为一轮归约：
  1. round1：`total = Σ(s00+s10+s01+s11)`，`mean0 = total / (4C)`；
  2. round2：`d0 = where(cmask, s − mean0, 0)`；同一遍同时累加 `Σd0`（修正量 `corr = Σd0/(4C)`，`mean = mean0 + corr`）与 `Σd0²`（按平行轴定理 `var = Σd0²/(4C) − corr²`）；
  3. `rstd = 1/sqrt(var + eps)`（eps>0 保证零方差 case 安全，无除以 var、无 rsqrt(0)）。
  - corr 修正把 mean 的树归约舍入误差（~1e-7）压到 ~1e-10 量级：case 28 中 `rstd≈171` 会把 mean 误差放大 171 倍传入 y，未修正时对近零输出通道的绝对误差可达 ~1e-4，修正后 <1e-6。corr² ≈ 1e-13 远小于 var ≈ 3.3e-5，合并轮在数学上与三遍法等价（平行轴定理），仅少一轮归约。
- **尾块防护**：非 2 幂 C（3、65）用 `BLOCK_C = next_power_of_2(C)` + `mask = c < C`；`d0 = where(cmask, s − mean0, 0)` 在减法后立刻强制归零，防止 masked load 的填充值污染统计量（填充值为 0，减去 mean0 后变成 −mean0，不清零会直接毁掉 mean/var）。
- **拼接顺序**：按 golden 切片顺序逐段核对为 TL/BL/TR/BR，段偏移依次 `+0 / +W*C / +C / +W*C+C`；weight/bias 与输出通道同样按 4 段对齐（段 k 的 w/b 偏移 `k*C + c`）。
- 输出 `y = (d0 − corr) · rstd · w + b`（`d0 − corr` 等价于 `s − mean`），逐段延迟加载 w/b 后立即回写，控制 UB 驻留。

## 2. Tiling 策略

- **通道方向**：`BLOCK_C = next_power_of_2(C)`，全 case 上界 512（C=384/512）。fp32 驻留估算（BLOCK_C=512）：4 段 x（4 KB×4）+ 4 段 d0（4 KB×4）= 32 KB 常驻，w/b 分段按需加载（2 KB×8 瞬态）≈ 48 KB，低于 192 KB UB 上限，无溢出风险；C ≤ 256 时余量更大。`BLOCK_C` 作为 `tl.constexpr` 由 host 用 `triton.next_power_of_2(C)` 计算，共 5 个编译变体（4/32/128/256/512）× 3 dtype。
- **token 方向**：每 program 处理 1 个输出 token（4C 通道），不沿 token 再分块——4C ≤ 2048 元素一次载入即完成，无循环。
- **空间方向**：无 mask。H、W 为不小于 2 的偶数保证四段全部在界内（proto.yaml 约束），host 侧仍校验 `height/width` 偶数性与 `length == height*width`。
- **地址位宽**：x/y 的 token 基址用 int64（`pid.to(tl.int64) * ...`）防御大张量；通道偏移 int32。给定 case 最大张量 4×576×512 ≈ 1.18M 元素，int32 本已安全，int64 为一般化兜底。
- **shape 全部为运行时标量参数**（height/width/channels/height_half/width_half/eps），不写死任何 shape。

## 3. 数据流设计

```text
HBM(x GM) ──4×masked load（段偏移 0 / WC / C / WC+C）──▶ 寄存器/UB：s00..s11 (fp32)
                                                            │
                                          round1: Σ(s00+..+s11) ─▶ mean0
                                                            │
                              d0_k = where(cm, s_k − mean0, 0)（×4，驻留）
                                                            │
                  round2: Σ(d0) ─▶ corr ─▶ mean；Σ(d0²) ─▶ var（平行轴）─▶ rstd
                                                            │
HBM(w/b GM) ──分段延迟 load（段 k：k*C + c）──▶ (d0_k − corr)·rstd·w_k + b_k
                                                            │
                                     ──cast 回 x.dtype──▶ HBM(y GM，pid*4C + k*C + c)
```

- 中间 concat 张量、mean/var/rstd 均在 program 内寄存器/UB 完成，**不落 GM**。
- 输入保证连续（proto 明示），host 保留轻量 `.contiguous()` 守护（不触发实际拷贝）；stride 由 H、W、C 推导，无需通用 stride 传递。
- 输出为 `torch.empty` 新分配连续张量，由 kernel 在读取前完整写入（反作弊要求）。

## 4. 多核切分方案

- **grid = (B × (H/2) × (W/2),)**，一维展开，1 token/program；`pid` 解码 `b = pid / (Hh·Wh)`、`i = (rem) / Wh`、`j = rem % Wh`。
- 输出 token 的扁平索引恰好等于 `pid`（输入 `(b,2i,2j)` 与输出 `(b,i,j)` 同为行主序），省一次索引映射。
- 20 AI Core / 40 Vector 调度单元：大 grid case（784–2304）可充分占用；小 grid case（22: grid=1、25: grid=6、28: grid=40）天然欠饱和，但绝对工作量仅 12–5200 元素（微秒级），首版不引入跨 program 通道切分（需 workspace + 第二 kernel 做归约，净亏损）。
- 大 grid + 小 BLOCK_C 的 `TOKENS_PER_PROG` 分组（复用 w/b tile）留待后续性能阶段实测裁决，首版优先保证正确性与结构简单。

## 5. 融合算子方案

### 当前融合方式

- **单 kernel 内完成**：四路 strided gather（Slice×4）、通道拼接（Concat）、均值/方差统计（LayerNorm 前半）、标准化 + 仿射（LayerNorm 后半）、输出写回。
- **无多 kernel 部分**：整个算子只有一个 kernel，无 workspace 初始化 kernel、无辅助 kernel。
- 融合变体对应 F2.compute_epilogue（gather 完成后逐段做仿射 epilogue 再 store）与 F2.loop_chain（统计轮合并）的组合。

### 数据流向图

见第 3 节。四段 x 载入后常驻同一 program 的寄存器/UB（以 `s00..s11` → `d00..d11` 复用），统计量（mean0/corr/var/rstd）为标量，w/b 分段瞬态加载——**所有中间张量留在同一 program 内**，没有任何中间量通过 GM/HBM 工作区跨 kernel 传递（也不存在第二个 kernel）。

### 是否有 HBM 中间读写

**无**。中间 concat 张量与归约统计均不经过 HBM；kernel 只访问 HBM 一次读 x（4 段）+ 一次读 w/b + 一次写 y。这是该算子与「slice/cat kernel + layer_norm kernel」两段式实现的最本质差异。

### 融合收益估算

相比未融合版本（4 路 slice → cat → F.layer_norm，至少 2 个 kernel + 中间物化）：

- 省 1 次 kernel 启动 + host 端 dispatch/同步开销（数据量小、调度主导的负载上占比可观）；
- 省中间 concat 张量 `[B, (H/2)(W/2), 4C]` 的一次 HBM 写 + 一次 HBM 读：最大 case（16–18，B=4、4C=2048、fp16）约 4×576×2048×2 B ≈ 2.4 MB 往返，fp32 时 ≈ 4.7 MB；case 1–3 亦约 1.2 MB（fp32）；
- 省中间张量的分配/释放与 L2 压力（输入 x 可直接被 L2 命中复用）。

注：以上为结构性收益推算，非实测结论；实测以 cann-bench 正式性能报告为准。

## 6. 已知风险和待优化点

1. **小 grid 欠饱和**（case 22/25/28，grid=1/6/40）：40 个 Vector 单元吃不满。绝对耗时微秒级，首版接受；若后续实测占比显著，可评估 program 内多 token 分组之外的方案（如通道维跨 program 拆分 + 二段归约，但需 workspace，预期净亏损）。
2. **大 grid 的 token 分组未做**：`TOKENS_PER_PROG∈{2,4}` 可减少 program 调度轮数并复用 w/b tile，但拉长 per-program 串行链且 UB 驻留随分组线性增长，需 A/B 实测后再引入。
3. **归约轮数**：当前 2 轮（合并修正后）。若实测归约指令占比高，可评估 Welford 单轮方案，但 fp32 下 Welford 的数值行为需重新验证（近常数 case 风险）。
4. **int64 基址的标量开销**：仅为防御性设计，当前 case 规模下 int32 足够；若实测地址计算占比明显，可在保证元素数 < 2^31 的前提下回退 int32。
5. **bf16 路径**：已通过全部 bf16 case 自测（30 case 中 bf16 组全部 PASS），`tl.load/store` bf16 在本机后端编译运行无问题，风险关闭。
6. **精度余量**：全 30 case 的 worst_ratio ≤ 0.42（判据 |a−b| ≤ T·(|b|+M)），与 golden 自身对 fp64 参考的误差水平（0.001–0.21）相当；fp32 组最优（~0.001），fp16 近常数 case 28（0.41）为主要消耗点——corr 修正已将其从潜在越界压到 golden 同量级。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/state.json`；相对工作目录：`selection/state.json`；用途：有效评测索引与语义窗口配置；怎么看：以下 window 和 case_trends 由程序按同口径历史快照计算，不将失败轮计入窗口
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/impl`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/impl`；迭代模板：`selection/records/<iter>-<指纹>/impl`；用途：最佳已验证实现的独立代码快照目录；怎么看：读取该目录的实现，勿把当前工作代码当作历史最佳版本
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/manifest.json`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/manifest.json`；迭代模板：`selection/records/<iter>-<指纹>/manifest.json`；用途：最佳实现快照清单；怎么看：核对 iteration、融合方案、avg_speedup、hap 及代码和评测证据路径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/reports/performance_source.json`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/reports/performance_source.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/performance_source.json`；用途：最佳实现对应的原始性能报告；怎么看：按 case 看 speedup、耗时和 HAP，确认使用同一评测口径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/reports/perf_result.json`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/reports/perf_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/perf_result.json`；用途：最佳实现的结构化性能结果；怎么看：看 avg_speedup、cases 和最慢用例，配合窗口和趋势判断
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/reports/precision_result.json`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/reports/precision_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/precision_result.json`；用途：最佳实现对应的精度结果；怎么看：核对 precision_overall、通过数量和原始精度报告定位
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/evidence/0_decision_rationale.md`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/evidence/0_decision_rationale.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/0_decision_rationale.md`；用途：最佳实现的融合选择依据快照；怎么看：看当时的选择理由，与当前方案区别及实测结果核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/evidence/1_fusion_library.json`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/evidence/1_fusion_library.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/1_fusion_library.json`；用途：最佳实现的当轮融合方案库快照；怎么看：看 selection 中实际方法和实现方案，不把初始概率当性能
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/evidence/2_self_test_report.md`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/evidence/2_self_test_report.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/2_self_test_report.md`；用途：最佳实现的自测报告快照；怎么看：核对给定 case 及连续调用的执行范围与结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/evidence/3_self_test_result.json`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/evidence/3_self_test_result.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/3_self_test_result.json`；用途：最佳实现的自测资格 JSON 快照；怎么看：核对两类自测状态；其中历史日志路径以本清单中的归档日志为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/evidence/4_selftest_log_continuous.log`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/evidence/4_selftest_log_continuous.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/4_selftest_log_continuous.log`；用途：最佳实现的连续调用原始日志快照；怎么看：核对同 shape 更换参数后逐次比较参考实现的结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/evidence/5_selftest_log_provided.log`；相对工作目录：`selection/records/iter1-e2ccbd45852863d85f17/evidence/5_selftest_log_provided.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/5_selftest_log_provided.log`；用途：最佳实现的给定 case 原始自测日志快照；怎么看：核对原始执行输出是否支持报告结论
{
  "eligible": true,
  "reason": "Valid measured implementation recorded",
  "should_exit": false,
  "review_fusion": false,
  "best": {
    "iteration": 1,
    "selection_label": "best_available",
    "all_cases_pass": false,
    "avg_speedup": 1.208110829133419,
    "hap": {
      "performance_score": 22.712252457219726,
      "cases": [
        {
          "case_id": "level3/fused_patch_concat_layernorm_19",
          "perf_score": 0.1437505967726535,
          "t_hw_us": 3.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 182.7
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_21",
          "perf_score": 0.14911732166049355,
          "t_hw_us": 3.45,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 180.34
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_1",
          "perf_score": 0.17512988101223395,
          "t_hw_us": 3.48,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 151.14
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_20",
          "perf_score": 0.18199136398272797,
          "t_hw_us": 3.18,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 132.0
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_3",
          "perf_score": 0.18310014931150806,
          "t_hw_us": 3.68,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 151.4
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_10",
          "perf_score": 0.2023585711844129,
          "t_hw_us": 5.26,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 191.94
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_11",
          "perf_score": 0.21176979662483775,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 187.6
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_12",
          "perf_score": 0.2222478545722986,
          "t_hw_us": 5.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 173.92
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_2",
          "perf_score": 0.22376283131607172,
          "t_hw_us": 3.25,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 104.58
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_4",
          "perf_score": 0.2536269239266514,
          "t_hw_us": 3.83,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 105.18
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_6",
          "perf_score": 0.2670687055782899,
          "t_hw_us": 4.13,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 106.22
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_15",
          "perf_score": 0.3118811881188119,
          "t_hw_us": 4.27,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 89.06
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_13",
          "perf_score": 0.3532299540247799,
          "t_hw_us": 5.04,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 88.04
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_5",
          "perf_score": 0.37526145655067505,
          "t_hw_us": 4.39,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 70.1
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_14",
          "perf_score": 0.3854880642874025,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 83.44
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_7",
          "perf_score": 0.45659912376779854,
          "t_hw_us": 3.71,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 43.4
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_9",
          "perf_score": 0.4753386820991714,
          "t_hw_us": 4.01,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 43.9
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_18",
          "perf_score": 0.5267138326421078,
          "t_hw_us": 4.8,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 43.6
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_17",
          "perf_score": 0.5307781649245065,
          "t_hw_us": 5.08,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 45.48
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_16",
          "perf_score": 0.5409513825186738,
          "t_hw_us": 4.59,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 39.62
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_8",
          "perf_score": 0.5827067669172933,
          "t_hw_us": 4.31,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 32.06
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_30",
          "perf_score": 0.6897050318102951,
          "t_hw_us": 2.65,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 13.38
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_29",
          "perf_score": 0.7090962787950383,
          "t_hw_us": 2.67,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 12.52
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_28",
          "perf_score": 0.7124542124542125,
          "t_hw_us": 3.03,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 14.02
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_22",
          "perf_score": 0.7259414225941423,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.62
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_24",
          "perf_score": 0.7294372294372294,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.5
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_23",
          "perf_score": 0.8244444444444444,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 2.58
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_25",
          "perf_score": 0.8170515097690941,
          "t_hw_us": 1.53,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.62
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_26",
          "perf_score": 0.825136612021858,
          "t_hw_us": 1.34,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.9
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_27",
          "perf_score": 0.8412121212121212,
          "t_hw_us": 1.54,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.16
            }
          }
        }
      ]
    },
    "implementation_dir": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/impl",
    "manifest_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/manifest.json",
    "performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/reports/performance_source.json",
    "precision_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter1-e2ccbd45852863d85f17/reports/precision_result.json",
    "fusion_scheme": {
      "method_ids": [
        "F2"
      ],
      "implementation_plan": "F2.compute_epilogue / F2.loop_chain 变体：单 @triton.jit kernel 端到端融合 Slice×4(strided gather) → Concat(寄存器/UB 内拼接，不落 GM) → LayerNorm。grid = B*(H/2)*(W/2)，每 program 串行处理 1 个输出 token；通道向量 BLOCK_C = next_power_of_2(C)（最大 512）。4 段按 golden 顺序 TL(+0)/BL(+W*C)/TR(+C)/BR(+W*C+C) 带 mask 载入并升精度到 fp32；统计采用两遍法 + 均值修正轮：round1 求 sum→mean0，round2 在同一遍 diff 上同时累加 sum(diff)（修正量 corr，mean=mean0+corr）与 sum(diff²)（var=Σdiff²/n−corr²，平行轴定理），归约轮数 3→2；rstd=1/sqrt(var+eps)，仿射 (diff−corr)*rstd*w+b 按段延迟加载 w/b 并回写原 dtype。host 侧仅做元数据/连续性守护、torch.empty 分配输出与 kernel 启动。",
      "reason": "概率最高项 F2(0.79) 与算子结构完全匹配：Concat 是纯跨行/列 strided gather、LayerNorm 是逐 token 全量归约，二者天然可在单 kernel 内以寄存器/UB tile 衔接，中间 concat 张量无需在 HBM 物化；满足 F2 的硬件前提（dav-c220 上 tl.load/store/sum/sqrt/where 均可用，实测冒烟通过）。概率第二的 F5(0.58) 本轮不作为独立实现：所有 case 共享同一数学定义，仅 C 尺寸不同且由 BLOCK_C=next_power_of_2(C) 一个 constexpr 自然覆盖，无需要路由的异质 case 组，引入 dispatcher 只增加分支与编译成本；其精神（按 shape 选择分块）已由 BLOCK_C 的 host 侧计算体现。F8(0.44) 的计算/访存交错流水在本算子上无交错段（load→reduce→store 单链），不适用。概率是初始方向参考而非性能结论，后续以 cann-bench 实测为准。",
      "target_cases": [
        "全部 30 个 case（10 个 shape 组 × fp16/fp32/bf16），含 C∈{3,65} 非 2 幂尾块、零方差（case 25-27）、近常数强相消（case 28-30）、矩形输入（case 19-21）、grid=1（case 22）与最大 grid=2304（case 10-12）"
      ],
      "actual_changes": [
        "新增 impl/cann_bench/fused_patch_concat_layernorm.py：单 kernel 垂直融合实现（_fused_patch_concat_layernorm_kernel + host 包装函数）",
        "均值统计采用两遍法 + corr 修正轮，diff 通道向量用 tl.where(mask,·,0) 防尾块污染",
        "x/w/b 全部升精度 fp32 参与统计与仿射，输出前转回 x.dtype",
        "基址与输出基址使用 int64（pid.to(tl.int64)）防大张量溢出，通道偏移保持 int32",
        "impl/cann_bench/__init__.py 导出 fused_patch_concat_layernorm；setup.py/build.sh 按 example 工程"
      ],
      "expected_benefits": [
        "相比未融合的三段实现（slice/cat kernel + layer_norm kernel）：省去 4C 中间张量的一次 HBM 写 + 一次 HBM 读（最大 case 约 2.4 MB fp16 / 4.7 MB fp32 往返），并省一次 kernel 启动与同步开销",
        "四段 gather 与归约在 program 内完成，无跨 program 依赖、无 workspace、无原子操作",
        "修正轮合并使归约轮数保持 2 轮，同时保住近常数 case（rstd≈171）的相消精度"
      ],
      "methods": [
        {
          "rank": 1,
          "method": {
            "id": "F2",
            "name": "Vertical Fusion",
            "name_zh": "垂直融合／单 kernel 全融合",
            "summary": "Fuse a producer-consumer chain and retain legal intermediate tiles on chip within one kernel.",
            "graph_conditions": [
              "A consumer depends on a producer.",
              "Tiling permits consumption of ready intermediate data."
            ],
            "case_features": [
              "Compute plus bias, activation, casting or compatible layout work.",
              "Traffic/launch savings are material and live tiles fit resources."
            ],
            "hardware_requirements": [
              "Backend supports all operations in one kernel.",
              "Adequate registers/local memory and supported transfers/synchronization."
            ],
            "execution_organization": "One kernel coordinates producer and consumer tiles, possibly across compute and vector sections.",
            "dataflow": "Producer tile -> registers/local memory -> consumer tile -> output, removing selected intermediate materialization.",
            "benefits": [
              "Remove intermediate traffic and launches.",
              "Reuse locally available producer data."
            ],
            "risks": [
              "Resource pressure, lower parallelism or spills.",
              "Transfer/synchronization cost and pipeline imbalance."
            ],
            "variants": [
              {
                "id": "F2.compute_epilogue",
                "name": "Compute with fused epilogue",
                "description": "Apply bias, activation and conversion to completed compute tiles before final output stores."
              },
              {
                "id": "F2.loop_chain",
                "name": "Compatible loop-chain fusion",
                "description": "Combine compatible dependent loops while preserving reduction boundaries."
              },
              {
                "id": "F2.tiled_pipeline",
                "name": "Tiled producer-consumer pipeline",
                "description": "Overlap successive tiles with supported buffering and synchronization; tune live resource use."
              }
            ],
            "implementation_checks": [
              "Count all live tiles, buffers and compiler storage.",
              "Verify transfer layout, synchronization and precision.",
              "Inspect spills, stalls and occupancy; one kernel is not automatically optimal.",
              "Verify the required Triton Ascend operations and compiler options on the installed backend; preserve the task interface, strides, masks, dtype and measured work scope."
            ],
            "compatible_with": [
              "F1",
              "F3",
              "F4",
              "F5",
              "F7",
              "F8",
              "F9"
            ],
            "source_sections": [
              "F2: Vertical Fusion（垂直融合）"
            ]
          },
          "probability": 0.79
        }
      ]
    }
  },
  "window": {
    "status": "underperforming",
    "required_improvements": 3,
    "valid_samples": 1,
    "completed_improvements": 0,
    "enough_samples": false,
    "start_iteration": 1,
    "end_iteration": 1,
    "start_best_avg_speedup": 1.208110829133419,
    "end_best_avg_speedup": 1.208110829133419,
    "cumulative_improvement": 0.0,
    "threshold": 0.05,
    "stagnated": false
  },
  "case_trends": [
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "current_speedup": 0.23044859071059945,
      "currently_underperforming": true,
      "gap_to_one": 0.7695514092894006,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.23044859071059945,
          "gap_to_one": 0.7695514092894006
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "current_speedup": 0.27414817130353236,
      "currently_underperforming": true,
      "gap_to_one": 0.7258518286964677,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.27414817130353236,
          "gap_to_one": 0.7258518286964677
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "current_speedup": 0.2898720682302772,
      "currently_underperforming": true,
      "gap_to_one": 0.7101279317697228,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.2898720682302772,
          "gap_to_one": 0.7101279317697228
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "current_speedup": 0.30772769089236435,
      "currently_underperforming": true,
      "gap_to_one": 0.6922723091076357,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.30772769089236435,
          "gap_to_one": 0.6922723091076357
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "current_speedup": 0.5721263062244434,
      "currently_underperforming": true,
      "gap_to_one": 0.4278736937755566,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.5721263062244434,
          "gap_to_one": 0.4278736937755566
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "current_speedup": 0.6516059443911792,
      "currently_underperforming": true,
      "gap_to_one": 0.34839405560882075,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.6516059443911792,
          "gap_to_one": 0.34839405560882075
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "current_speedup": 0.4794520547945206,
      "currently_underperforming": true,
      "gap_to_one": 0.5205479452054794,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.4794520547945206,
          "gap_to_one": 0.5205479452054794
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "current_speedup": 1.157748611812216,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.157748611812216,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "current_speedup": 1.1165347405452948,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.1165347405452948,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "current_speedup": 1.1004587155963301,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.1004587155963301,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "current_speedup": 0.1831417624521073,
      "currently_underperforming": true,
      "gap_to_one": 0.8168582375478927,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.1831417624521073,
          "gap_to_one": 0.8168582375478927
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "current_speedup": 0.3103843947217441,
      "currently_underperforming": true,
      "gap_to_one": 0.6896156052782558,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.3103843947217441,
          "gap_to_one": 0.6896156052782558
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "current_speedup": 0.2412121212121212,
      "currently_underperforming": true,
      "gap_to_one": 0.7587878787878788,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.2412121212121212,
          "gap_to_one": 0.7587878787878788
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "current_speedup": 0.19102805811245427,
      "currently_underperforming": true,
      "gap_to_one": 0.8089719418875457,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.19102805811245427,
          "gap_to_one": 0.8089719418875457
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "current_speedup": 2.1933701657458564,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.1933701657458564,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "current_speedup": 3.2635658914728682,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.2635658914728682,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "current_speedup": 2.2114285714285713,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.2114285714285713,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "current_speedup": 3.318181818181818,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.318181818181818,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "current_speedup": 3.4410256410256412,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.4410256410256412,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "current_speedup": 3.706730769230769,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.706730769230769,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "current_speedup": 2.1583452211126963,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.1583452211126963,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "current_speedup": 2.1309904153354635,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.1309904153354635,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "current_speedup": 0.24299867899603697,
      "currently_underperforming": true,
      "gap_to_one": 0.757001321003963,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.24299867899603697,
          "gap_to_one": 0.757001321003963
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "current_speedup": 1.9805680119581464,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.9805680119581464,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "current_speedup": 0.3638524434303099,
      "currently_underperforming": true,
      "gap_to_one": 0.6361475565696901,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.3638524434303099,
          "gap_to_one": 0.6361475565696901
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "current_speedup": 0.6256776034236805,
      "currently_underperforming": true,
      "gap_to_one": 0.3743223965763195,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.6256776034236805,
          "gap_to_one": 0.3743223965763195
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "current_speedup": 0.38909809828657504,
      "currently_underperforming": true,
      "gap_to_one": 0.610901901713425,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.38909809828657504,
          "gap_to_one": 0.610901901713425
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "current_speedup": 0.8539170506912444,
      "currently_underperforming": true,
      "gap_to_one": 0.14608294930875565,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.8539170506912444,
          "gap_to_one": 0.14608294930875565
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "current_speedup": 1.3431066749844043,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.3431066749844043,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "current_speedup": 0.9145785876993167,
      "currently_underperforming": true,
      "gap_to_one": 0.08542141230068334,
      "change_from_previous": null,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.9145785876993167,
          "gap_to_one": 0.08542141230068334
        }
      ]
    }
  ]
}
程序定义的填写格式（先读后填）：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/decision_schema.json`；相对工作目录：`knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/decision_schema.json`；迭代模板：`knowledge/stage9/<iter>/508b8258a96b4adc89fcc0462b664058/decision_schema.json`；用途：程序定义的 Stage9 JSON 格式（只读）；怎么看：按本轮字段、类型及层级填写；未列出的经验字段必须省略；case_analysis 按本轮要求填写；修改文件由 changes 自动汇总
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/decision_template.json`；相对工作目录：`knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/decision_template.json`；迭代模板：`knowledge/stage9/<iter>/508b8258a96b4adc89fcc0462b664058/decision_template.json`；用途：本请求的决策填写模板（只读）；怎么看：复制 iteration/request_id，填写真实结论、任务和已列出的性能经验；经验 case 按实测增补受益及受损项；空白及 null 必须按 schema 填写
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/case_catalog.json`；相对工作目录：`knowledge/stage9/iter1/508b8258a96b4adc89fcc0462b664058/case_catalog.json`；迭代模板：`knowledge/stage9/<iter>/508b8258a96b4adc89fcc0462b664058/case_catalog.json`；用途：本轮可核对的完整 case ID 清单（只读）；怎么看：已列出 ID 时按原名选择；没有清单时回查 task，不把最慢六例当成全部用例
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/impl`；相对工作目录：`impl`；用途：本轮建议实际实施的代码基底（只读查阅）；怎么看：在此核对 case 路由与待改函数；changes 仍填写实施时的 impl/... 相对路径。若本轮需回退最佳版本，不按退步代码臆造修改位置
