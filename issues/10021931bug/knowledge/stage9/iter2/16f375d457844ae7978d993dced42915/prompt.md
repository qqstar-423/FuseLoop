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

📊 性能提升检测（程序自动计算，数据可信）
  avg_speedup: 1.208110829133419 → 1.3213246705259825 (+9.4%)
  对比轮次: iter1 → iter2
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/design_rationale.md`；相对工作目录：`develop/iter1/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：上轮评测附近的历史设计参考；怎么看：用于回顾改动意图；实际被评测版本以代码绑定的选择依据和快照为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/design_rationale.md`；相对工作目录：`develop/iter1/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：本轮评测附近的历史设计参考；怎么看：与上轮参考比较；先用代码绑定依据或快照确认版本，再归因成功经验
  逐 case 对比（提升最大的在前）:
    case_level3/fused_patch_concat_layernorm_11: 0.2898720682302772 → 0.5354470263883419 (+84.7%)
    case_level3/fused_patch_concat_layernorm_13: 0.5721263062244434 → 1.025030525030525 (+79.2%)
    case_level3/fused_patch_concat_layernorm_15: 0.4794520547945206 → 0.7811928283937066 (+62.9%)
    case_level3/fused_patch_concat_layernorm_14: 0.6516059443911792 → 0.9896250455041864 (+51.9%)
    case_level3/fused_patch_concat_layernorm_17: 1.1165347405452948 → 1.663826998689384 (+49.0%)
    case_level3/fused_patch_concat_layernorm_18: 1.1004587155963301 → 1.6231393775372125 (+47.5%)

⚡ 本轮性能大幅提升！请对照上面两轮的 design_rationale，总结这次改动为什么有效，填写 proven_pattern 字段。- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/fusion_library.json`；相对工作目录：`fusion/fusion_library.json`；用途：JSON 融合算子库；怎么看：比较候选数据流、前提与初始概率，以实测证据决定保留或更换
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/perf_result.json`；相对工作目录：`eval/iter2/perf_result.json`；迭代模板：`eval/<iter>/perf_result.json`；用途：本轮性能结果；怎么看：看 avg_speedup、各 case 的 speedup 和最慢用例，不只看平均值
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/prof_data`；相对工作目录：`eval/iter2/prof_data`；迭代模板：`eval/<iter>/prof_data`；用途：本轮 profiler 数据（按需回查）；怎么看：仅在结论矛盾或缺少证据时按 case 回查
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/perf_reports`；相对工作目录：`eval/iter2/perf_reports`；迭代模板：`eval/<iter>/perf_reports`；用途：本轮性能报告；怎么看：回查原始计时与评分，核实性能汇总及反作弊信号
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/profile/iter2/bottleneck_analysis.md`；相对工作目录：`profile/iter2/bottleneck_analysis.md`；迭代模板：`profile/<iter>/bottleneck_analysis.md`；用途：本轮瓶颈分析；怎么看：优先读 Stage7 结论，区分已证实根因和推测
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter2/FIX_DIRECTIVE.md`；相对工作目录：`search/iter2/FIX_DIRECTIVE.md`；迭代模板：`search/<iter>/FIX_DIRECTIVE.md`；用途：本轮修改指令；怎么看：优先读 Stage8 结论，转成原有 P0/P1/P2 优先级
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter2/SEARCH_REPORT.md`；相对工作目录：`search/iter2/SEARCH_REPORT.md`；迭代模板：`search/<iter>/SEARCH_REPORT.md`；用途：本轮搜索报告；怎么看：按需检查候选方法、依据及适用条件
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/build/iter2/build.log`；相对工作目录：`build/iter2/build.log`；迭代模板：`build/<iter>/build.log`；用途：本轮编译日志；怎么看：看 STATUS 和首个真实错误，定位构建/接口问题
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/precision_result.json`；相对工作目录：`eval/iter2/precision_result.json`；迭代模板：`eval/<iter>/precision_result.json`；用途：本轮精度结果；怎么看：核对总数、通过数及失败 case，优先处理正确性
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/precision_reports`；相对工作目录：`eval/iter2/precision_reports`；迭代模板：`eval/<iter>/precision_reports`；用途：本轮精度报告详情；怎么看：按失败 case 查误差、输入和原始报错
当前场景缺失材料（不能当作已完成结论）：
历轮设计思路：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/design_rationale.md`；相对工作目录：`develop/iter0/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter0 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/design_rationale.md`；相对工作目录：`develop/iter1/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter1 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
请分析本轮结果，回顾历轮设计思路；只读 /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/history.json，不要覆盖它。
Stage9 输出文件：/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision.json
Stage9 请求编号：16f375d457844ae7978d993dced42915
Stage9 当前轮次：2
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision.json`；相对工作目录：`knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision.json`；迭代模板：`knowledge/stage9/<iter>/16f375d457844ae7978d993dced42915/decision.json`；用途：本次 Stage9 决策输出（待你生成）；怎么看：按 role 写 iteration、request_id、ledger_entry 和模型结论；经验使用 proven_pattern/regression_pattern/pitfall 公共字段

本次 ledger_entry.case_analysis 必须逐一覆盖以下 case_id（原样复制，不能缩写；每例写 observation、explanation、evidence、next_action；根因未证实须注明待验证）：
["level3/fused_patch_concat_layernorm_19", "level3/fused_patch_concat_layernorm_21", "level3/fused_patch_concat_layernorm_1", "level3/fused_patch_concat_layernorm_3", "level3/fused_patch_concat_layernorm_20", "level3/fused_patch_concat_layernorm_12"]


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
实现 SHA256：d176d86e49ca2027eebbcab24a5b56cc2fa02b4b87eb9f66c8bc7af7b55b726b
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/fusion_library.json`；相对工作目录：`develop/iter1/fusion_library.json`；迭代模板：`develop/<iter>/fusion_library.json`；用途：当前轮实际融合选择与方案库；怎么看：看 selection 和新增方法；原始 Jev probability 只是先验参考
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/融合方案选择决策依据.md`；相对工作目录：`develop/iter1/融合方案选择决策依据.md`；迭代模板：`develop/<iter>/融合方案选择决策依据.md`；用途：当前代码的融合选择决策依据；怎么看：先看选定方案、未选其他方案的原因及目标 case，再与实测核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/self_test_report.md`；相对工作目录：`develop/iter1/self_test_report.md`；迭代模板：`develop/<iter>/self_test_report.md`；用途：当前代码的自测报告；怎么看：看给定 case 与连续调用是否实际执行，区分失败、未执行和通过
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/self_test_result.json`；相对工作目录：`develop/iter1/self_test_result.json`；迭代模板：`develop/<iter>/self_test_result.json`；用途：程序校验使用的自测 JSON；怎么看：看 provided_cases、continuous_calls 的通过状态及 evidence_path
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/self_test.log`；相对工作目录：`develop/iter1/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：给定 case 的原始自测日志；怎么看：核对实际命令、case 数量及结果是否支持自测报告
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/self_test.log`；相对工作目录：`develop/iter1/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：连续调用的原始自测日志；怎么看：核对相同 shape 更换参数后逐次对照参考实现的结果
最佳实现入库资格：True；Source, fusion decision and self-tests are bound.
本轮库与只读初始 Jev 库区分；真实评测优先于初始概率。
{
  "method_ids": [
    "F2"
  ],
  "implementation_plan": "保持 iter0 的 F2 垂直融合单 kernel 骨架（Slice×4 strided gather → UB 内 Concat → LayerNorm，fp32 统计 + corr 修正轮），本轮按 Tech Lead 任务单做两项调度/指令级修改：(1) kernel 签名新增运行时参数 tokens 与 tl.constexpr MULTI_TOKEN，全部计算体包进 token 内循环：MULTI_TOKEN=True 时 token_lo,token_hi,token_step = pid, tokens, tl.num_programs(0)（grid=min(tokens,40)=40，每 program 步进 40 串行处理多个 token，调度波从 17-58 波降为 1 波）；MULTI_TOKEN=False 时 token_lo,token_hi,token_step = pid, pid+1, 1（与 iter0 逐 token 路径编译等价，保护 tokens<=64 的已达标小 grid case）。host 侧路由：grid=(min(tokens,40),) if tokens>64 else (tokens,)，MULTI_TOKEN=(tokens>64)。(2) 统计掩码 tl.where 的比较值由 int32 channel_mask 改为 fp32 向量比较 offs_f=offs.to(tl.float32); tl.where(offs_f < channels, diff, 0.0)（4 处），消除 i32 cmp 标量退化；load/store 的 channel_mask 保持不变（编译器自动向量化）。数值路径（fp32 升精度、两遍法 + corr 修正、tl.maximum(var,0)、1/sqrt(var+eps)、回写原 dtype）与融合结构不变。",
  "reason": "融合结构维持 F2：iter0/iter1 实测已证明融合本身有效（小 grid 组 speedup 1.98-3.71，大 C 中等 grid 组 1.10-1.16），瓶颈在 1 token/program 的调度粒度而非融合方案，故不更换 method，仅在 F2 骨架内做 T1 调度改造。tokens>64 阈值来自 Tech Lead 裁定：case 22-30（tokens<=40）当前 speedup 1.98-3.71 已达标，保持原路径避免回归；case 1-21（tokens>=392）全部切入 MULTI_TOKEN 路径。fp32 向量比较来自官方 profiling guide 同构修复案例（i32 cmp 在 NPU 退化为 scalar，fp32 走 vec_cast+vec_cmp），作为 P1 辅助改动与 T1 叠加。本轮未引入新融合方法，无新增 method 定义；概率沿用初始库，仅作先验参考。",
  "target_cases": [
    "T1 主目标：全部 17 个未达标大 grid case（tokens>=392 且 C<=256）：level3/fused_patch_concat_layernorm_19/21/1/20/3/10/11/12/2/4/6/15/13/5/14/7/9，调度波从 17-58 波降为 1 波",
    "T1 保护目标：已达标小 grid case 22-30（tokens<=40，走原 single 路径）与大 C 中等 grid case 16-18（tokens=576，切入 MULTI_TOKEN 路径）",
    "T2 覆盖目标：非 2 幂 C（C=80/96/65/3）与全部 case 的统计掩码向量化",
    "全部 30 个 case 的精度回归（含 C=3/65 尾块、零方差 25-27、近常数 28-30、grid=1 的 case 22）"
  ],
  "actual_changes": [
    "impl/cann_bench/fused_patch_concat_layernorm.py kernel 签名新增 tokens（运行时 int）与 MULTI_TOKEN: tl.constexpr（插在 eps 之后、BLOCK_C 之前）",
    "kernel 计算体整体包进 for token in range(token_lo, token_hi, token_step) 内循环；pid 解码、4 段 gather、统计、仿射、写回全部随循环变量 token 索引；out_base = token.to(tl.int64) * (4 * channels)",
    "host 侧路由：grid=(min(tokens,40),) if tokens>64 else (tokens,)；调用传入 tokens 与 MULTI_TOKEN=(tokens>64)；40 为 device_info.json 的 Vector Core 数",
    "4 处 tl.where(channel_mask, seg-mean0, 0.0) 改为 tl.where(offs_f < channels, seg-mean0, 0.0)，offs_f = offs.to(tl.float32)；load/store 的 channel_mask 保留",
    "输入校验、contiguous 守护、torch.empty 分配、tokens==0 早退、函数签名与导出均未改动"
  ],
  "expected_benefits": [
    "大 grid case（tokens 690-2304）：调度波从 17-58 波降为 1 波，per-program 固定开销被 17-58 个 token 摊薄；kernel 级 A/B 冒烟显示 case 19/21/1/20/3/10 kernel 耗时从 132-192us 降至 min 7.8-8.9us / median 22.5-27.3us（含采集开销），预期 speedup 从 0.18-0.31 显著回升",
    "小 grid case（tokens<=64）零回归：MULTI_TOKEN=False 路径与 iter0 编译等价（循环区间长度 1），控制组冒烟 min 值与 iter1 正式评测同量级",
    "T2 使统计掩码选择操作留在向量管线，非 2 幂 C case（C=80/96/65/3）的标量指令占比下降",
    "中间 concat 张量仍不在 HBM 物化（融合程度不变），HBM 读写维持每 token 一次 4 段读 + 一次 4 段写"
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

# 融合方案选择决策依据 — develop/iter1

本轮实现绑定 `develop/iter1/fusion_library.json` 的 `selection` 字段；本文为其完整决策依据。自测证据见同目录 `self_test_report.md` / `self_test.log` / `self_test_result.json`，性能正式数字以 cann-bench 评测为准。

## 1. 本轮选择：维持 F2（Vertical Fusion），在其骨架内做调度粒度改造

- **选中 method_ids**: `["F2"]`（初始 Jev 概率 0.79，候选库 rank 1）
- **变体**: F2.compute_epilogue + F2.loop_chain（沿用 iter0），叠加本轮新增的 **多 token/program 内循环调度**（F2 骨架内的执行组织优化，不构成新融合方法，未向库中追加新 method 定义）
- **本轮实际改动**（唯一修改文件 `impl/cann_bench/fused_patch_concat_layernorm.py`）：
  1. kernel 新增 `tokens` 运行时参数与 `MULTI_TOKEN: tl.constexpr`；全部计算体包进 `for token in range(token_lo, token_hi, token_step)` 内循环（MULTI_TOKEN=True 时步长 `tl.num_programs(0)`，False 时与 iter0 逐 token 编译等价）；`out_base` 跟随 `token`。
  2. host 侧路由：`grid=(min(tokens,40),) if tokens>64 else (tokens,)`，`MULTI_TOKEN=(tokens>64)`；40 = device_info.json 的 Vector Core 数。
  3. 4 处统计掩码 `tl.where` 的比较值由 int32 `channel_mask` 改为 fp32 向量 `offs_f < channels`（消除 i32 cmp 标量退化）；load/store 掩码保持不变。
- **未改动**：数值路径（fp32 升精度统计、两遍法 + corr 修正轮、`tl.maximum(var,0)`、`1/sqrt(var+eps)`、回写原 dtype）、融合结构（单 kernel、无 workspace、无跨 program 归约）、host 校验/分配/早退逻辑、接口签名与导出。

## 2. 为什么不更换融合方案

- **实测证据支持 F2 结构本身**：iter1 评测（selection/state.json 的 best 记录与 knowledge/history.json ledger）显示 13/30 case 达标——小 grid 组（22-30，tokens≤40）speedup 1.98-3.71、大 C 中等 grid 组（16-18，C=512，tokens=576）1.10-1.16。未达标的 17 个 case 全部为 tokens≥392 且 C≤256，瓶颈被 profiler 证实为 **1 token/program 导致 17-58 个调度波、per-program 固定开销主导**（如 case 10 grid=2304 kernel 191.94us vs baseline 52.62us），而非融合数据流问题。
- **Tech Lead 裁定**：fusion_kernel_strategy 记录"融合结构有效，调度粒度需优化为多 token/program"；FIX_DIRECTIVE 明确"不要拆分为多 kernel、不要引入 workspace/跨 program 归约"。本轮严格遵守。
- **概率仅作先验**：F2(0.79) 的先验与实测方向一致；F5(0.58) 的"按 shape 路由"精神已由本轮 host 侧 tokens>64 阈值路由体现（元数据谓词 + 合法 fallback，两条路径编译特化），不需要引入独立 dispatcher 层；F8(0.44) 的计算/访存交错流水在本算子（load→reduce→store 单链）上无适用场景，维持不选。

## 3. 目标 case 与预期收益（预期，非实测）

| 组 | case | 本轮改动 | 预期 |
|---|---|---|---|
| 大 grid 未达标组（17 个） | 1-7、9-15、19-21（tokens 392-2304） | MULTI_TOKEN 路径：调度波 17-58 → 1 波，固定开销被每 program 17-58 个 token 摊薄；T2 统计掩码向量化 | kernel 耗时从 130-192us 回落到 baseline 同量级（31-54us 附近），speedup 从 0.18-0.31 显著回升 |
| 大 C 中等 grid 组 | 16-18（C=512，tokens=576，原达标 1.10-1.16） | 切入 MULTI_TOKEN 路径，每 program ~14 token | 持平或小幅提升 |
| 小 grid 已达标组 | 22-30（tokens≤40，原达标 1.98-3.71） | single 路径（与 iter0 编译等价，循环区间长度 1） | 零回归 |
| 非幂 C 精度回归 | 22-24、28-30（C=3/65）、19-21（C=80） | T2 fp32 比较（语义等价） | 精度不变（自测已证逐位一致） |

**已获得的非正式方向性证据**（kernel 级 A/B 冒烟，torch_npu profiler，10 次调用/40 事件，临时脚本在 /tmp）：case 19/21/1/20/3/10 的 kernel 事件 min 值 7.8-8.9us、median 22.5-27.3us（含采集与发射间隙），对照 iter1 正式评测的 132-192us 方向性改善显著；小 grid 控制组 case 22/28 的 min 值 3.30/8.48us 与其 iter1 正式评测 3.62/14.02us 同量级，无回归迹象。**该冒烟口径与 cann-bench 不同，不作为收益结论，仅证明方向正确、可进入正式评测。**

## 4. 融合算子方案（结构现状）

### 当前融合方式

- **单 kernel 内完成**（与 iter0 相同）：四路 strided gather（Slice×4）、通道拼接（Concat，UB 内）、均值/方差统计（LayerNorm 前半）、标准化 + 仿射（LayerNorm 后半）、输出写回。
- **无多 kernel 部分**：整个算子仍只有一个 kernel，无 workspace 初始化、无辅助 kernel、无跨 program 归约。
- 本轮变化仅在执行组织：原"1 program = 1 token"变为"40 program 各串行处理一段 token"（仅 tokens>64 时），融合边界与数据流拓扑不变。

### 数据流向

```text
HBM(x GM) ──4×masked load（段偏移 0 / WC / C / WC+C）──▶ 寄存器/UB：s00..s11 (fp32)
                                                            │
                                          round1: Σ(s00+..+s11) ─▶ mean0
                                                            │
                              d0_k = where(offs_f<C, s_k − mean0, 0)（×4，驻留）
                                                            │
                  round2: Σ(d0) ─▶ corr ─▶ mean；Σ(d0²) ─▶ var（平行轴）─▶ rstd
                                                            │
HBM(w/b GM) ──分段延迟 load（段 k：k*C + c）──▶ (d0_k − corr)·rstd·w_k + b_k
                                                            │
                                     ──cast 回 x.dtype──▶ HBM(y GM，token*4C + k*C + c)
```

- **program 内中间张量**：4 段 x、4 段 diff、统计标量（mean0/corr/var/rstd）全部驻留同一 program 的寄存器/UB；每个循环迭代驻留 1 个 token 的 tile（BLOCK_C=512 时约 20-24KB fp32），随迭代释放重用，不随 token 数增长。
- **跨 kernel 工作区与 GM/HBM 读写**：不存在第二个 kernel，无 workspace。kernel 对 HBM 的访问 = 每 token 一次 4 段 x 读 + 一次 4C 输出写，外加 w/b 的小量重复读（Multi-Token 下 w/b 每 token 迭代重新加载一次；因 40 program 摊薄后每 program 加载次数 = token 数，总量与 iter0 相同，可被 L2 命中吸收；本轮未做 w/b tile 跨迭代复用，属后续可选优化，未越界实施）。上述为源码级数据流陈述；"UB 具体驻留布局"由编译器管理，本轮未做片上布局级 profiler 断言。
- **输出完整性**：`torch.empty` 分配的输出由 kernel 在读取前完整写入；strided 步进保证每个 token 恰好被一个 program 写一次，无重叠无遗漏（TC5 连续调用换参验证无旧值复用）。

### 本轮修改对融合的影响

- **融合程度不变**：中间 concat 张量仍不在 HBM 物化（每 token 省一次 4C 中间写 + 一次 4C 读，相对未融合三段实现的结构性收益保持）。
- **改善点在调度层**：同一融合数据流从"多波小任务"变为"单波长任务"，HBM 中间读写没有增加，且 w/b 的 L2 复用机会随 per-program token 数增加而略升。
- **无融合退化**：未引入 HBM 中间物化、未拆 kernel。

### 历史否决方向的规避

- knowledge/history.json 中无 ❌ 融合方向记录（fusion_kernel_strategy 唯一一条为 ✅"融合结构有效，调度粒度需优化"），本轮无重复失败风险。
- insights 中 ❌ 性质的历史教训本轮均未触碰：未改统计轮数/Welford（数值路径冻结）、未加 num_warps/num_stages、未取非 2 幂 BLOCK_C、未动 BLOCK_C 语义。
- iter0 设计文档中"大 grid 的 token 分组留待后续"的遗留项（TOKENS_PER_PROG）正是本轮 T1 的实施内容——按 Tech Lead 裁定以 `grid=40 + 步进内循环` 形式落地，而非 per-program 连续分组；若后续实测提示连续分组（相邻 token 同 program、w/b tile 可跨迭代复用）更优，需由下一轮任务单授权后另行 A/B。

## 5. 结论

本轮在 F2 骨架内以最小改动完成了 Tech Lead 指定的调度粒度修复（T1）与指令级向量化修复（T2）：融合结构、数值路径、接口语义零改动，精度自测 30/30 与 9/9 全通过且误差与 iter0 基线逐位一致；kernel 级冒烟显示 worst case 方向性大幅改善、小 grid 控制组无回归。方案选择的先验（F2=0.79）与 iter1 实测证据、Tech Lead 裁定三方一致。正式性能提升幅度以 cann-bench 同口径评测为准。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/state.json`；相对工作目录：`selection/state.json`；用途：有效评测索引与语义窗口配置；怎么看：以下 window 和 case_trends 由程序按同口径历史快照计算，不将失败轮计入窗口
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/impl`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/impl`；迭代模板：`selection/records/<iter>-<指纹>/impl`；用途：最佳已验证实现的独立代码快照目录；怎么看：读取该目录的实现，勿把当前工作代码当作历史最佳版本
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/manifest.json`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/manifest.json`；迭代模板：`selection/records/<iter>-<指纹>/manifest.json`；用途：最佳实现快照清单；怎么看：核对 iteration、融合方案、avg_speedup、hap 及代码和评测证据路径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/reports/performance_source.json`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/reports/performance_source.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/performance_source.json`；用途：最佳实现对应的原始性能报告；怎么看：按 case 看 speedup、耗时和 HAP，确认使用同一评测口径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/reports/perf_result.json`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/reports/perf_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/perf_result.json`；用途：最佳实现的结构化性能结果；怎么看：看 avg_speedup、cases 和最慢用例，配合窗口和趋势判断
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/reports/precision_result.json`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/reports/precision_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/precision_result.json`；用途：最佳实现对应的精度结果；怎么看：核对 precision_overall、通过数量和原始精度报告定位
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/0_decision_rationale.md`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/evidence/0_decision_rationale.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/0_decision_rationale.md`；用途：最佳实现的融合选择依据快照；怎么看：看当时的选择理由，与当前方案区别及实测结果核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/1_fusion_library.json`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/evidence/1_fusion_library.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/1_fusion_library.json`；用途：最佳实现的当轮融合方案库快照；怎么看：看 selection 中实际方法和实现方案，不把初始概率当性能
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/2_self_test_report.md`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/evidence/2_self_test_report.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/2_self_test_report.md`；用途：最佳实现的自测报告快照；怎么看：核对给定 case 及连续调用的执行范围与结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/3_self_test_result.json`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/evidence/3_self_test_result.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/3_self_test_result.json`；用途：最佳实现的自测资格 JSON 快照；怎么看：核对两类自测状态；其中历史日志路径以本清单中的归档日志为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/4_selftest_log_continuous.log`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/evidence/4_selftest_log_continuous.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/4_selftest_log_continuous.log`；用途：最佳实现的连续调用原始日志快照；怎么看：核对同 shape 更换参数后逐次比较参考实现的结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/5_selftest_log_provided.log`；相对工作目录：`selection/records/iter2-844fc6b40708c857f98b/evidence/5_selftest_log_provided.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/5_selftest_log_provided.log`；用途：最佳实现的给定 case 原始自测日志快照；怎么看：核对原始执行输出是否支持报告结论
{
  "eligible": true,
  "reason": "Valid measured implementation recorded",
  "should_exit": false,
  "review_fusion": false,
  "best": {
    "iteration": 2,
    "selection_label": "best_available",
    "all_cases_pass": false,
    "avg_speedup": 1.3213246705259825,
    "hap": {
      "performance_score": 24.32685560065395,
      "cases": [
        {
          "case_id": "level3/fused_patch_concat_layernorm_19",
          "perf_score": 0.13438364723734714,
          "t_hw_us": 3.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 197.3
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_21",
          "perf_score": 0.16500771810294354,
          "t_hw_us": 3.45,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 160.32
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_1",
          "perf_score": 0.16525222708344314,
          "t_hw_us": 3.48,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 161.84
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_3",
          "perf_score": 0.17283499504097719,
          "t_hw_us": 3.68,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 162.14
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_20",
          "perf_score": 0.18526179702650292,
          "t_hw_us": 3.18,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 129.22
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_12",
          "perf_score": 0.19597233523189586,
          "t_hw_us": 5.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 202.98
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_10",
          "perf_score": 0.21323728050427732,
          "t_hw_us": 5.26,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 180.0
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_2",
          "perf_score": 0.24957279562542722,
          "t_hw_us": 3.25,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 91.08
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_4",
          "perf_score": 0.2786182347706496,
          "t_hw_us": 3.83,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 93.0
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_6",
          "perf_score": 0.2895166939061405,
          "t_hw_us": 4.13,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 95.42
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_11",
          "perf_score": 0.33737763683992833,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 101.56
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_5",
          "perf_score": 0.3995748127151245,
          "t_hw_us": 4.39,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 63.7
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_15",
          "perf_score": 0.4326728214366134,
          "t_hw_us": 4.27,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 54.66
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_7",
          "perf_score": 0.46396772398441855,
          "t_hw_us": 3.71,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 42.24
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_9",
          "perf_score": 0.48309049592300496,
          "t_hw_us": 4.01,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 42.68
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_14",
          "perf_score": 0.4971045412983846,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 54.94
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_13",
          "perf_score": 0.5068768869506877,
          "t_hw_us": 5.04,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 49.14
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_16",
          "perf_score": 0.6165795369678865,
          "t_hw_us": 4.59,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 30.26
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_18",
          "perf_score": 0.6355607889314101,
          "t_hw_us": 4.8,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 29.56
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_17",
          "perf_score": 0.6423952769187518,
          "t_hw_us": 5.08,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 30.52
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_8",
          "perf_score": 0.6530165149983148,
          "t_hw_us": 4.31,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 24.9
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_30",
          "perf_score": 0.691304347826087,
          "t_hw_us": 2.65,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 13.3
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_22",
          "perf_score": 0.7139917695473251,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.78
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_24",
          "perf_score": 0.7200854700854702,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.62
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_28",
          "perf_score": 0.7242021276595745,
          "t_hw_us": 3.03,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 13.4
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_29",
          "perf_score": 0.7275757575757575,
          "t_hw_us": 2.67,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 11.66
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_23",
          "perf_score": 0.8082788671023965,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 2.76
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_25",
          "perf_score": 0.8219178082191781,
          "t_hw_us": 1.53,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.52
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_26",
          "perf_score": 0.8296703296703297,
          "t_hw_us": 1.34,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.82
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
    "implementation_dir": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/impl",
    "manifest_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/manifest.json",
    "performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/reports/performance_source.json",
    "precision_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/reports/precision_result.json",
    "fusion_scheme": {
      "method_ids": [
        "F2"
      ],
      "implementation_plan": "保持 iter0 的 F2 垂直融合单 kernel 骨架（Slice×4 strided gather → UB 内 Concat → LayerNorm，fp32 统计 + corr 修正轮），本轮按 Tech Lead 任务单做两项调度/指令级修改：(1) kernel 签名新增运行时参数 tokens 与 tl.constexpr MULTI_TOKEN，全部计算体包进 token 内循环：MULTI_TOKEN=True 时 token_lo,token_hi,token_step = pid, tokens, tl.num_programs(0)（grid=min(tokens,40)=40，每 program 步进 40 串行处理多个 token，调度波从 17-58 波降为 1 波）；MULTI_TOKEN=False 时 token_lo,token_hi,token_step = pid, pid+1, 1（与 iter0 逐 token 路径编译等价，保护 tokens<=64 的已达标小 grid case）。host 侧路由：grid=(min(tokens,40),) if tokens>64 else (tokens,)，MULTI_TOKEN=(tokens>64)。(2) 统计掩码 tl.where 的比较值由 int32 channel_mask 改为 fp32 向量比较 offs_f=offs.to(tl.float32); tl.where(offs_f < channels, diff, 0.0)（4 处），消除 i32 cmp 标量退化；load/store 的 channel_mask 保持不变（编译器自动向量化）。数值路径（fp32 升精度、两遍法 + corr 修正、tl.maximum(var,0)、1/sqrt(var+eps)、回写原 dtype）与融合结构不变。",
      "reason": "融合结构维持 F2：iter0/iter1 实测已证明融合本身有效（小 grid 组 speedup 1.98-3.71，大 C 中等 grid 组 1.10-1.16），瓶颈在 1 token/program 的调度粒度而非融合方案，故不更换 method，仅在 F2 骨架内做 T1 调度改造。tokens>64 阈值来自 Tech Lead 裁定：case 22-30（tokens<=40）当前 speedup 1.98-3.71 已达标，保持原路径避免回归；case 1-21（tokens>=392）全部切入 MULTI_TOKEN 路径。fp32 向量比较来自官方 profiling guide 同构修复案例（i32 cmp 在 NPU 退化为 scalar，fp32 走 vec_cast+vec_cmp），作为 P1 辅助改动与 T1 叠加。本轮未引入新融合方法，无新增 method 定义；概率沿用初始库，仅作先验参考。",
      "target_cases": [
        "T1 主目标：全部 17 个未达标大 grid case（tokens>=392 且 C<=256）：level3/fused_patch_concat_layernorm_19/21/1/20/3/10/11/12/2/4/6/15/13/5/14/7/9，调度波从 17-58 波降为 1 波",
        "T1 保护目标：已达标小 grid case 22-30（tokens<=40，走原 single 路径）与大 C 中等 grid case 16-18（tokens=576，切入 MULTI_TOKEN 路径）",
        "T2 覆盖目标：非 2 幂 C（C=80/96/65/3）与全部 case 的统计掩码向量化",
        "全部 30 个 case 的精度回归（含 C=3/65 尾块、零方差 25-27、近常数 28-30、grid=1 的 case 22）"
      ],
      "actual_changes": [
        "impl/cann_bench/fused_patch_concat_layernorm.py kernel 签名新增 tokens（运行时 int）与 MULTI_TOKEN: tl.constexpr（插在 eps 之后、BLOCK_C 之前）",
        "kernel 计算体整体包进 for token in range(token_lo, token_hi, token_step) 内循环；pid 解码、4 段 gather、统计、仿射、写回全部随循环变量 token 索引；out_base = token.to(tl.int64) * (4 * channels)",
        "host 侧路由：grid=(min(tokens,40),) if tokens>64 else (tokens,)；调用传入 tokens 与 MULTI_TOKEN=(tokens>64)；40 为 device_info.json 的 Vector Core 数",
        "4 处 tl.where(channel_mask, seg-mean0, 0.0) 改为 tl.where(offs_f < channels, seg-mean0, 0.0)，offs_f = offs.to(tl.float32)；load/store 的 channel_mask 保留",
        "输入校验、contiguous 守护、torch.empty 分配、tokens==0 早退、函数签名与导出均未改动"
      ],
      "expected_benefits": [
        "大 grid case（tokens 690-2304）：调度波从 17-58 波降为 1 波，per-program 固定开销被 17-58 个 token 摊薄；kernel 级 A/B 冒烟显示 case 19/21/1/20/3/10 kernel 耗时从 132-192us 降至 min 7.8-8.9us / median 22.5-27.3us（含采集开销），预期 speedup 从 0.18-0.31 显著回升",
        "小 grid case（tokens<=64）零回归：MULTI_TOKEN=False 路径与 iter0 编译等价（循环区间长度 1），控制组冒烟 min 值与 iter1 正式评测同量级",
        "T2 使统计掩码选择操作留在向量管线，非 2 幂 C case（C=80/96/65/3）的标量指令占比下降",
        "中间 concat 张量仍不在 HBM 物化（融合程度不变），HBM 读写维持每 token 一次 4 段读 + 一次 4 段写"
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
    "valid_samples": 2,
    "completed_improvements": 1,
    "enough_samples": false,
    "start_iteration": 1,
    "end_iteration": 2,
    "start_best_avg_speedup": 1.208110829133419,
    "end_best_avg_speedup": 1.3213246705259825,
    "cumulative_improvement": 0.09371146972813088,
    "threshold": 0.05,
    "stagnated": false
  },
  "case_trends": [
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "current_speedup": 0.21521255561047947,
      "currently_underperforming": true,
      "gap_to_one": 0.7847874443895205,
      "change_from_previous": -0.015236035100119977,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.23044859071059945,
          "gap_to_one": 0.7695514092894006
        },
        {
          "iteration": 2,
          "speedup": 0.21521255561047947,
          "gap_to_one": 0.7847874443895205
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "current_speedup": 0.29233333333333333,
      "currently_underperforming": true,
      "gap_to_one": 0.7076666666666667,
      "change_from_previous": 0.018185162029800972,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.27414817130353236,
          "gap_to_one": 0.7258518286964677
        },
        {
          "iteration": 2,
          "speedup": 0.29233333333333333,
          "gap_to_one": 0.7076666666666667
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "current_speedup": 0.5354470263883419,
      "currently_underperforming": true,
      "gap_to_one": 0.46455297361165815,
      "change_from_previous": 0.24557495815806463,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.2898720682302772,
          "gap_to_one": 0.7101279317697228
        },
        {
          "iteration": 2,
          "speedup": 0.5354470263883419,
          "gap_to_one": 0.46455297361165815
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "current_speedup": 0.2636712976647946,
      "currently_underperforming": true,
      "gap_to_one": 0.7363287023352054,
      "change_from_previous": -0.04405639322756977,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.30772769089236435,
          "gap_to_one": 0.6922723091076357
        },
        {
          "iteration": 2,
          "speedup": 0.2636712976647946,
          "gap_to_one": 0.7363287023352054
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "current_speedup": 1.025030525030525,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.45290421880608156,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.5721263062244434,
          "gap_to_one": 0.4278736937755566
        },
        {
          "iteration": 2,
          "speedup": 1.025030525030525,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "current_speedup": 0.9896250455041864,
      "currently_underperforming": true,
      "gap_to_one": 0.010374954495813604,
      "change_from_previous": 0.33801910111300715,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.6516059443911792,
          "gap_to_one": 0.34839405560882075
        },
        {
          "iteration": 2,
          "speedup": 0.9896250455041864,
          "gap_to_one": 0.010374954495813604
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "current_speedup": 0.7811928283937066,
      "currently_underperforming": true,
      "gap_to_one": 0.21880717160629337,
      "change_from_previous": 0.30174077359918605,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.4794520547945206,
          "gap_to_one": 0.5205479452054794
        },
        {
          "iteration": 2,
          "speedup": 0.7811928283937066,
          "gap_to_one": 0.21880717160629337
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "current_speedup": 1.515862524785195,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.3581139129729789,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.157748611812216,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 1.515862524785195,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "current_speedup": 1.663826998689384,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.5472922581440893,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.1165347405452948,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 1.663826998689384,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "current_speedup": 1.6231393775372125,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.5226806619408824,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.1004587155963301,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 1.6231393775372125,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "current_speedup": 0.16958945767866193,
      "currently_underperforming": true,
      "gap_to_one": 0.830410542321338,
      "change_from_previous": -0.013552304773445378,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.1831417624521073,
          "gap_to_one": 0.8168582375478927
        },
        {
          "iteration": 2,
          "speedup": 0.16958945767866193,
          "gap_to_one": 0.830410542321338
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "current_speedup": 0.35638998682476947,
      "currently_underperforming": true,
      "gap_to_one": 0.6436100131752305,
      "change_from_previous": 0.04600559210302535,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.3103843947217441,
          "gap_to_one": 0.6896156052782558
        },
        {
          "iteration": 2,
          "speedup": 0.35638998682476947,
          "gap_to_one": 0.6436100131752305
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "current_speedup": 0.24640148583810556,
      "currently_underperforming": true,
      "gap_to_one": 0.7535985141618944,
      "change_from_previous": 0.0051893646259843496,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.2412121212121212,
          "gap_to_one": 0.7587878787878788
        },
        {
          "iteration": 2,
          "speedup": 0.24640148583810556,
          "gap_to_one": 0.7535985141618944
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "current_speedup": 0.21488273453093815,
      "currently_underperforming": true,
      "gap_to_one": 0.7851172654690619,
      "change_from_previous": 0.023854676418483878,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.19102805811245427,
          "gap_to_one": 0.8089719418875457
        },
        {
          "iteration": 2,
          "speedup": 0.21488273453093815,
          "gap_to_one": 0.7851172654690619
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "current_speedup": 2.100529100529101,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.0928410652167555,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.1933701657458564,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 2.100529100529101,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "current_speedup": 3.0507246376811596,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.2128412537917086,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.2635658914728682,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 3.0507246376811596,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "current_speedup": 2.138121546961326,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.07330702446724535,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.2114285714285713,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 2.138121546961326,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "current_speedup": 3.3915929203539825,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.07341110217216462,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.318181818181818,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 3.3915929203539825,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "current_speedup": 3.513089005235602,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.07206336420996085,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.4410256410256412,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 3.513089005235602,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "current_speedup": 3.706730769230769,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.0,
      "history": [
        {
          "iteration": 1,
          "speedup": 3.706730769230769,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 3.706730769230769,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "current_speedup": 2.2582089552238807,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.09986373411118432,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.1583452211126963,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 2.2582089552238807,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "current_speedup": 2.2881646655231562,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.1571742501876927,
      "history": [
        {
          "iteration": 1,
          "speedup": 2.1309904153354635,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 2.2881646655231562,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "current_speedup": 0.2269026766991489,
      "currently_underperforming": true,
      "gap_to_one": 0.7730973233008511,
      "change_from_previous": -0.016096002296888068,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.24299867899603697,
          "gap_to_one": 0.757001321003963
        },
        {
          "iteration": 2,
          "speedup": 0.2269026766991489,
          "gap_to_one": 0.7730973233008511
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "current_speedup": 1.9924812030075187,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.011913191049372385,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.9805680119581464,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 1.9924812030075187,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "current_speedup": 0.41150537634408607,
      "currently_underperforming": true,
      "gap_to_one": 0.5884946236559139,
      "change_from_previous": 0.04765293291377615,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.3638524434303099,
          "gap_to_one": 0.6361475565696901
        },
        {
          "iteration": 2,
          "speedup": 0.41150537634408607,
          "gap_to_one": 0.5884946236559139
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "current_speedup": 0.6885400313971742,
      "currently_underperforming": true,
      "gap_to_one": 0.31145996860282577,
      "change_from_previous": 0.06286242797349373,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.6256776034236805,
          "gap_to_one": 0.3743223965763195
        },
        {
          "iteration": 2,
          "speedup": 0.6885400313971742,
          "gap_to_one": 0.31145996860282577
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "current_speedup": 0.4331377069796688,
      "currently_underperforming": true,
      "gap_to_one": 0.5668622930203312,
      "change_from_previous": 0.04403960869309376,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.38909809828657504,
          "gap_to_one": 0.610901901713425
        },
        {
          "iteration": 2,
          "speedup": 0.4331377069796688,
          "gap_to_one": 0.5668622930203312
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "current_speedup": 0.8773674242424243,
      "currently_underperforming": true,
      "gap_to_one": 0.12263257575757569,
      "change_from_previous": 0.02345037355117996,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.8539170506912444,
          "gap_to_one": 0.14608294930875565
        },
        {
          "iteration": 2,
          "speedup": 0.8773674242424243,
          "gap_to_one": 0.12263257575757569
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "current_speedup": 1.7293172690763055,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.38621059409190117,
      "history": [
        {
          "iteration": 1,
          "speedup": 1.3431066749844043,
          "gap_to_one": 0.0
        },
        {
          "iteration": 2,
          "speedup": 1.7293172690763055,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "current_speedup": 0.9407216494845361,
      "currently_underperforming": true,
      "gap_to_one": 0.05927835051546393,
      "change_from_previous": 0.026143061785219412,
      "history": [
        {
          "iteration": 1,
          "speedup": 0.9145785876993167,
          "gap_to_one": 0.08542141230068334
        },
        {
          "iteration": 2,
          "speedup": 0.9407216494845361,
          "gap_to_one": 0.05927835051546393
        }
      ]
    }
  ]
}
程序定义的填写格式（先读后填）：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision_schema.json`；相对工作目录：`knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision_schema.json`；迭代模板：`knowledge/stage9/<iter>/16f375d457844ae7978d993dced42915/decision_schema.json`；用途：程序定义的 Stage9 JSON 格式（只读）；怎么看：按本轮字段、类型及层级填写；未列出的经验字段必须省略；case_analysis 按本轮要求填写；修改文件由 changes 自动汇总
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision_template.json`；相对工作目录：`knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision_template.json`；迭代模板：`knowledge/stage9/<iter>/16f375d457844ae7978d993dced42915/decision_template.json`；用途：本请求的决策填写模板（只读）；怎么看：复制 iteration/request_id，填写真实结论、任务和已列出的性能经验；经验 case 按实测增补受益及受损项；空白及 null 必须按 schema 填写
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/case_catalog.json`；相对工作目录：`knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/case_catalog.json`；迭代模板：`knowledge/stage9/<iter>/16f375d457844ae7978d993dced42915/case_catalog.json`；用途：本轮可核对的完整 case ID 清单（只读）；怎么看：已列出 ID 时按原名选择；没有清单时回查 task，不把最慢六例当成全部用例
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/impl`；相对工作目录：`impl`；用途：本轮建议实际实施的代码基底（只读查阅）；怎么看：在此核对 case 路由与待改函数；changes 仍填写实施时的 impl/... 相对路径。若本轮需回退最佳版本，不按退步代码臆造修改位置
