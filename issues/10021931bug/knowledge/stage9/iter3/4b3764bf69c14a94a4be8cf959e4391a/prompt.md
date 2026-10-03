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

📚 已验证的成功优化经验（knowledge/proven_patterns.md）
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/proven_patterns.md`；相对工作目录：`knowledge/proven_patterns.md`；用途：以下成功经验摘要的来源；怎么看：先核对经验中的框架与芯片，再核对改动、适用条件和提升证据；旧记录未注明环境时须先验证再复用
这是历轮迭代中性能大幅提升（≥5%）时自动记录的成功经验。经验内容由 stage9 tech_lead 填写，orchestrator 验证提升幅度后才写入。
参考方式：
  - 写代码（stage3）：延续有效方向，避免回退已验证的改动
  - 分析性能（stage7）：对照成功经验判断当前瓶颈是否已有解法
  - 搜索方案（stage8）：搜索时优先搜和已验证方向相关的深入优化
  - Tech Lead（stage9）：引用成功经验作为 insights 和 suggest_next 的证据

  ✅ iter2: speedup 1.208110829133419 → 1.3213246705259825 (+9.4%)
     改了什么: 在 F2 单 kernel 骨架内新增 MULTI_TOKEN serial loop（grid 固定为 min(tokens,40)=40 个 Vector Core，tokens>64 时每 program 从 pid 起步、步长 40 串行处理多 token）+ 4 处 tl.where 统计掩码比较从 int32 改为 fp32 向量比较。代码：impl/cann_bench/fused_patch_concat_layernorm.py 行 28-36（循环路由）、行 60-64（fp32 比较）、行 145（host grid 路由）。选择依据：develop/iter1/融合方案选择决策依据.md。
     框架=Triton；后端=triton-ascend；芯片=Ascend910B3；SoC=Ascend910B3；编程模型=Triton block program (SPMD)
     为什么有效: 对大 C（≥256）case 有效的根因：(1) 调度波从 10-58 波降为 1 波，消除了多波调度的启动/同步开销；(2) per-token 有效计算量大（BLOCK_C=512 时每 token 4×512×4B=8KB fp32 load+reduce+store），serial loop 的 per-iteration 固定开销（token 解码、w/b 重载）占比 <15%，净收益显著。 对中等 C（128-256 fp32）也有效：case 11（C=128 fp32）per-token≈1.75us，无 cast 开销，固定开销占比低。 但对小 C（80-128）+ fp16/bf16 case 无效甚至有害：per-token 有效计算量极低（BLOCK_C=128 时仅 2KB），w/b 逐次重载（8 次 masked load/迭代）+ token 解码除法/取模 + fp16/bf16 cast 的 per-iteration 开销超过了减波收益。 case 11 fp32 +85% vs case 12 bf16 -14% 的对比（同 shape、同 tokens=2304、同 C=128）直接证明 cast 是 small C serial loop 的关键恶化因素。
     适用条件: 适用：Ascend910B3（40 Vector Core）上 F2 单 kernel 融合算子，MULTI_TOKEN serial loop（grid=物理核数 + 内循环步进）在以下条件有效：(1) C≥256（BLOCK_C≥256，per-token 计算充足）或 C≥128 + fp32（无 cast 开销）；(2) tokens>64（保证分流阈值以上走 MULTI_TOKEN）；(3) w/b 较小（<UB 上限）。 不适用：C≤128 + fp16/bf16（per-iteration 的 cast + w/b 重载固定开销主导），此时减波收益被循环开销抵消甚至导致退步。 未验证范围：C=192（case 5/7/9，borderline，本轮小幅改善 2-6%）；更大 grid（tokens>2304）；不同 BLOCK_C 取值。 硬件限制：grid=40 对应 40 Vector Core，其他核数设备需调整。

⚠️ 性能退步检测（程序自动计算，数据可信）
  avg_speedup: 1.3213246705259825 → 0.8224094689497551 (-37.8%)
  对比轮次: iter2 → iter3
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/design_rationale.md`；相对工作目录：`develop/iter2/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：上轮评测附近的历史设计参考；怎么看：用于回顾退步前的设计意图；实际被评测版本以绑定依据和快照为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/design_rationale.md`；相对工作目录：`develop/iter2/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：本轮评测附近的历史设计参考；怎么看：先用代码绑定依据或快照确认版本，再结合慢 case 核对改动并归因失败教训
  逐 case 对比（退步最大的在前）:
    case_level3/fused_patch_concat_layernorm_18: 1.6231393775372125 → 0.010621106737916775 (-99.3%)
    case_level3/fused_patch_concat_layernorm_16: 1.515862524785195 → 0.011364143118338709 (-99.3%)
    case_level3/fused_patch_concat_layernorm_17: 1.663826998689384 → 0.012641778114139473 (-99.2%)
    case_level3/fused_patch_concat_layernorm_15: 0.7811928283937066 → 0.00944460418924598 (-98.8%)
    case_level3/fused_patch_concat_layernorm_8: 1.7293172690763055 → 0.02144123329416217 (-98.8%)
    case_level3/fused_patch_concat_layernorm_13: 1.025030525030525 → 0.012946522662197799 (-98.7%)

🚨 本轮性能退步！请对照上面两轮的 design_rationale，分析原因并填写 regression_pattern，明确适用条件；在已验证条件下避免重复失败，改变条件后须重新验证。- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/fusion_library.json`；相对工作目录：`fusion/fusion_library.json`；用途：JSON 融合算子库；怎么看：比较候选数据流、前提与初始概率，以实测证据决定保留或更换
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/perf_result.json`；相对工作目录：`eval/iter3/perf_result.json`；迭代模板：`eval/<iter>/perf_result.json`；用途：本轮性能结果；怎么看：看 avg_speedup、各 case 的 speedup 和最慢用例，不只看平均值
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/prof_data`；相对工作目录：`eval/iter3/prof_data`；迭代模板：`eval/<iter>/prof_data`；用途：本轮 profiler 数据（按需回查）；怎么看：仅在结论矛盾或缺少证据时按 case 回查
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/perf_reports`；相对工作目录：`eval/iter3/perf_reports`；迭代模板：`eval/<iter>/perf_reports`；用途：本轮性能报告；怎么看：回查原始计时与评分，核实性能汇总及反作弊信号
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/profile/iter3/bottleneck_analysis.md`；相对工作目录：`profile/iter3/bottleneck_analysis.md`；迭代模板：`profile/<iter>/bottleneck_analysis.md`；用途：本轮瓶颈分析；怎么看：优先读 Stage7 结论，区分已证实根因和推测
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter3/FIX_DIRECTIVE.md`；相对工作目录：`search/iter3/FIX_DIRECTIVE.md`；迭代模板：`search/<iter>/FIX_DIRECTIVE.md`；用途：本轮修改指令；怎么看：优先读 Stage8 结论，转成原有 P0/P1/P2 优先级
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter3/SEARCH_REPORT.md`；相对工作目录：`search/iter3/SEARCH_REPORT.md`；迭代模板：`search/<iter>/SEARCH_REPORT.md`；用途：本轮搜索报告；怎么看：按需检查候选方法、依据及适用条件
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/build/iter3/build.log`；相对工作目录：`build/iter3/build.log`；迭代模板：`build/<iter>/build.log`；用途：本轮编译日志；怎么看：看 STATUS 和首个真实错误，定位构建/接口问题
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/precision_result.json`；相对工作目录：`eval/iter3/precision_result.json`；迭代模板：`eval/<iter>/precision_result.json`；用途：本轮精度结果；怎么看：核对总数、通过数及失败 case，优先处理正确性
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/precision_reports`；相对工作目录：`eval/iter3/precision_reports`；迭代模板：`eval/<iter>/precision_reports`；用途：本轮精度报告详情；怎么看：按失败 case 查误差、输入和原始报错
当前场景缺失材料（不能当作已完成结论）：
历轮设计思路：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/design_rationale.md`；相对工作目录：`develop/iter0/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter0 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/design_rationale.md`；相对工作目录：`develop/iter1/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter1 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/design_rationale.md`；相对工作目录：`develop/iter2/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter2 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
请分析本轮结果，回顾历轮设计思路；只读 /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/history.json，不要覆盖它。
Stage9 输出文件：/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision.json
Stage9 请求编号：4b3764bf69c14a94a4be8cf959e4391a
Stage9 当前轮次：3
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision.json`；相对工作目录：`knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision.json`；迭代模板：`knowledge/stage9/<iter>/4b3764bf69c14a94a4be8cf959e4391a/decision.json`；用途：本次 Stage9 决策输出（待你生成）；怎么看：按 role 写 iteration、request_id、ledger_entry 和模型结论；经验使用 proven_pattern/regression_pattern/pitfall 公共字段

本次 ledger_entry.case_analysis 必须逐一覆盖以下 case_id（原样复制，不能缩写；每例写 observation、explanation、evidence、next_action；根因未证实须注明待验证）：
["level3/fused_patch_concat_layernorm_15", "level3/fused_patch_concat_layernorm_18", "level3/fused_patch_concat_layernorm_16", "level3/fused_patch_concat_layernorm_12", "level3/fused_patch_concat_layernorm_17", "level3/fused_patch_concat_layernorm_13"]


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
实现 SHA256：de822d69e99dea8f391de1f33929ea3c7f1a0d79c2e0bcba8eaf61da4c3e8b70
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/fusion_library.json`；相对工作目录：`develop/iter2/fusion_library.json`；迭代模板：`develop/<iter>/fusion_library.json`；用途：当前轮实际融合选择与方案库；怎么看：看 selection 和新增方法；原始 Jev probability 只是先验参考
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/融合方案选择决策依据.md`；相对工作目录：`develop/iter2/融合方案选择决策依据.md`；迭代模板：`develop/<iter>/融合方案选择决策依据.md`；用途：当前代码的融合选择决策依据；怎么看：先看选定方案、未选其他方案的原因及目标 case，再与实测核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/self_test_report.md`；相对工作目录：`develop/iter2/self_test_report.md`；迭代模板：`develop/<iter>/self_test_report.md`；用途：当前代码的自测报告；怎么看：看给定 case 与连续调用是否实际执行，区分失败、未执行和通过
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/self_test_result.json`；相对工作目录：`develop/iter2/self_test_result.json`；迭代模板：`develop/<iter>/self_test_result.json`；用途：程序校验使用的自测 JSON；怎么看：看 provided_cases、continuous_calls 的通过状态及 evidence_path
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/self_test.log`；相对工作目录：`develop/iter2/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：给定 case 的原始自测日志；怎么看：核对实际命令、case 数量及结果是否支持自测报告
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/self_test.log`；相对工作目录：`develop/iter2/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：连续调用的原始自测日志；怎么看：核对相同 shape 更换参数后逐次对照参考实现的结果
最佳实现入库资格：True；Source, fusion decision and self-tests are bound.
本轮库与只读初始 Jev 库区分；真实评测优先于初始概率。
{
  "method_ids": [
    "F2"
  ],
  "implementation_plan": "保持 iter0-iter2 的 F2 垂直融合单 kernel 骨架（Slice×4 strided gather → UB 内 Concat → LayerNorm，fp32 统计 + corr 修正轮）与 tokens>64 → grid=min(tokens,40) MULTI_TOKEN 路由，本轮按 Tech Lead 任务单在 MULTI_TOKEN 分支内做三项执行组织修改：(1) w/b 的 8 次 masked load 移出 token 循环、循环前一次加载驻留 UB（两种分支共用；case 12 每 program 从 464 次降为 8 次）；(2) kernel 新增 TILE: tl.constexpr，MULTI_TOKEN=True 分支改 2D tile——每迭代处理 TILE 个 strided token 的 (TILE,BLOCK_C) 大 tile，解码/load/store/归约启动次数降为 1/TILE，fp16/bf16 cast 向量利用率随大 tile 提升；MULTI_TOKEN=False 分支保持逐 token 逻辑（循环区间长度 1，与 iter2 编译等价）；(3) host 计算 TILE=min(16, max(1, 1024//next_power_of_2(C))) 传入。另含实施中发现的必要修复：尾块行 token 钳位 token=where(tmask,raw_token,0)——后端对 2D gather 的所有行（含 masked 行）生成访存描述符，masked 行越界地址在特定组合（TILE=8+C=80+fp32+tokens=690，即 case 20）触发 MTE 硬异常；钳位使全部行地址有效，副作用仍由 tile_mask 抑制。数值路径（fp32 统计、两遍法 + corr 修正、tl.maximum(var,0)、1/sqrt(var+eps)、回写原 dtype）与融合结构不变。",
  "reason": "融合结构维持 F2：iter2 实测证明融合 + MULTI_TOKEN 对大 C 组有效（case 16-18 speedup 1.52-1.66、case 8 1.73、case 11 +85%），16 个未达标 case 的瓶颈被 profiler 证实为 serial loop 的 per-iteration 固定开销（w/b 逐次重载 8 次 masked load、token 解码除法/取模、小 tile cast 利用率低），而非融合数据流问题，故不更换 method，仅在 F2 骨架内实施任务单 T1 的 w/b 提取 + 2D tile。Tech Lead 裁定与 FIX_DIRECTIVE 一致：不拆多 kernel、不引入 workspace、不回退小 C 组到 grid=tokens 路径、不添加 num_warps/autotune。尾块钳位是 T2 验收项「tmask 尾块正确兜底」的完整落实（lane mask 之外后端还需要有效描述符地址），属于让 T1 可交付的正确性修复，不是方案变更。本轮未引入新融合方法，无新增 method 定义；初始 Jev 概率仅作先验参考，实测与 profiler 证据优先。",
  "target_cases": [
    "T1 主目标：6 个最慢 case level3/fused_patch_concat_layernorm_19/21/1/3/20/12（C=80-128, tokens 690-2304, speedup 0.17-0.26）：w/b 重载与解码按 1/TILE 削减、cast 向量化",
    "T1 覆盖目标：其余 10 个未达标 MULTI_TOKEN case 2/4/5/6/10/11/14/15/7/9（tokens 392-2304）",
    "保护目标：已达标 14 case——大 C 组 8/16-18 与 case 13（2D 路径 TILE=2/4，需零回归）与小 grid 组 22-30（single 路径编译等价，零改动语义）",
    "精度回归目标：全部 30 case（含 C=3/65 尾块、零方差 25-27、近常数 28-30、grid=1 的 case 22、修复钳位后的 case 20）"
  ],
  "actual_changes": [
    "impl/cann_bench/fused_patch_concat_layernorm.py kernel：offs/channel_mask/offs_f 与 8 次 w/b masked load（w0..w3/b0..b3，段偏移 0/C/2C/3C）上提到分支之前；循环内 8 处 w/b load 删除，store 改引用 w0-b3",
    "kernel 签名在 MULTI_TOKEN 之后新增 TILE: tl.constexpr",
    "MULTI_TOKEN=True 分支改 2D tile：for k in range(0, tokens-pid, TILE*step)；token=pid+k+arange(TILE)*step（strided）；尾块钳位 token=tl.where(tmask,raw_token,0)；batch/row/col (TILE,) 向量化；4 段 x 2D load 与 4 段 2D store（mask=tmask[:,None]&channel_mask[None,:]）；统计 axis=1 归约（mean0/corr/var/rstd 为 (TILE,)）；diff 的 tl.where 保留 fp32 向量比较（offs_f[None,:]<channels）",
    "MULTI_TOKEN=False 分支保持逐 token 逻辑：循环区间 (pid,pid+1,1)、1D load/axis=0 归约/1D store 不变，仅循环不变量上提",
    "host：tile=min(16, max(1, 1024//triton.next_power_of_2(channels)))，以 TILE=tile 传入；grid 与 MULTI_TOKEN=(tokens>64) 路由不变；输入校验/contiguous/torch.empty/tokens==0 早退未动"
  ],
  "expected_benefits": [
    "大 token 小 C 组（19-21/1-3/10-12）：w/b 重载（case 12 从 58×8=464 次降为 8 次）、解码与 load/store 发起按 1/TILE 削减（case 12：58 token→~8 tile），(TILE,BLOCK_C) 大 tile 提升 cast 向量利用率；预期 kernel 耗时显著回落，speedup 从 0.17-0.26 明显回升（正式幅度待评测）",
    "中等 token 组（2/4-6/13-15）：同样受益于固定开销削减（冒烟方向性信号：case 13 median -14.8%、case 2 -16.6%）",
    "已达标组零回归：大 C 组 TILE=2 改动温和（冒烟 -11%~+14% 波动为噪声）；case 22-30 single 路径与 iter2 编译等价（自测误差逐位一致）",
    "中间 concat 张量仍不在 HBM 物化（融合程度不变）；w/b 从每 token GM 重载改为每 program 一次驻留，冷缓存（CacheClean）正式口径下的访存消除收益大于热缓存冒烟所见"
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
  ],
  "measured_evidence": {
    "self_test": "develop/iter2/self_test_report.md（30/30 + 18/18，修复后完整重测）",
    "ab_smoke": "informal hot-cache A/B vs iter2 frozen implementation (selection/records/iter2-844fc6b40708c857f98b/impl)：23 shapes no-crash post-fix; directional gains case 12/10/13/2; protected cases noise-level; formal eval pending",
    "code_sha256": "1ee063d49389dc34c4dafe3982473f65c237378afcb5681437cb145fc8fdcb66"
  }
}

# 融合方案选择决策依据 — develop/iter2

本轮实现绑定 `develop/iter2/fusion_library.json` 的 `selection` 字段；本文为其完整决策依据。自测证据见同目录 `self_test_report.md` / `self_test.log` / `self_test_result.json`，性能正式数字以 cann-bench 下一轮评测为准。绑定代码 SHA256=1ee063d49389dc34c4dafe3982473f65c237378afcb5681437cb145fc8fdcb66。

## 1. 本轮选择：维持 F2（Vertical Fusion），在其 MULTI_TOKEN 分支内做 w/b 提取 + 2D tile 调度改造

- **选中 method_ids**: `["F2"]`（初始 Jev 概率 0.79，候选库 rank 1，与 iter0-iter2 一致）
- **变体**: F2.compute_epilogue + F2.loop_chain（沿用），本轮叠加任务单 T1 的两项执行组织改造：
  1. **w/b 提取**：8 次 masked load（w0..w3/b0..b3，段偏移 0/C/2C/3C）移出 token 循环、循环前一次加载驻留 UB——消除 serial loop 最大 per-iteration 访存项（case 12 每 program 从 58×8=464 次重载降为 8 次）。
  2. **2D tile**：MULTI_TOKEN=True 分支每迭代处理 TILE 个 strided token 的 (TILE, BLOCK_C) 大 tile（host 计算 TILE=min(16, max(1, 1024//next_power_of_2(C)))），解码/load/store/归约启动次数降为 1/TILE，fp16/bf16 cast 的向量利用率随大 tile 提升。MULTI_TOKEN=False 分支保持逐 token 逻辑不动。
  3. **必要修复（计划外，正确性）**：2D 尾块行地址钳位 `token = tl.where(tmask, raw_token, 0)`。实施后自测+A/B 发现 case 20（fp32, C=80, tokens=690, TILE=8）确定性触发 vector core 硬异常（`MTE accesses an invalid GM address`，独立进程复现 3+ 次）；分解实验证实后端对 (TILE,BLOCK_C) 2D gather 的**所有行**（含 tmask 掩掉的行）生成访存描述符，masked 行越界地址在特定组合下触发 MTE 异常（TILE=1/2、无尾块的 tokens=640、C=96 同尾块结构均不触发）。钳位后该配置稳定通过且数值正确（vs golden 7.15e-07）。这是 T2 验收项「tmask 尾块正确兜底」的完整落实，位于授权文件与分支内，共 3 行。
- **未改动**：数值路径（fp32 升精度统计、两遍法 + corr 修正轮、`tl.maximum(var,0)`、`1/sqrt(var+eps)`、回写原 dtype）、融合结构（单 kernel、无 workspace、无跨 program 归约）、host 校验/分配/早退、grid 与 tokens>64 路由、接口签名与导出。

## 2. 为什么不更换融合方案

- **实测证据支持 F2 结构本身**：iter2 评测显示 14/30 case 达标——大 C 组（16-18，C=512）speedup 1.52-1.66（较 iter1 +36-49%）、case 8（C=384）1.73、case 11（C=128 fp32）+85%、case 13（C=256）+79% 新达标；小 grid 组（22-30）1.99-3.71 两轮稳定。未达标的 16 个 case 全部为 MULTI_TOKEN=True 且 C≤256，瓶颈被 profiler 证实为 **serial loop 的 per-iteration 固定开销**（w/b 逐次重载、token 解码、小 tile cast），而非融合数据流问题。
- **Tech Lead 裁定与 FIX_DIRECTIVE 一致**：fusion_kernel_strategy 记录「对大 C 有效，对小 C 需 w/b 提取 + 2D tile」；FIX_DIRECTIVE 明确「不拆多 kernel、不引入 workspace/跨 program 归约、不回退小 C 组到 grid=tokens 路径（实测同 0.17-0.31 无收益）、不添加 num_warps/num_stages/autotune、BLOCK_C 不取非 2 幂」。本轮严格遵守。
- **概率仅作先验**：F2(0.79) 的先验与两轮实测方向一致。F5(0.58) 的「按 shape 路由」精神已由现行 host 侧 tokens>64 阈值路由体现（元数据谓词 + 合法 fallback，两条路径编译特化），本轮 TILE 公式进一步按 C 分档（128→8、256→4、512→2），仍属 F2 骨架内的调度特化，不需要引入独立 dispatcher 层；F8(0.44) 的计算/访存交错流水在本算子（load→reduce→store 单链）上无适用场景，维持不选。本轮未选「更高概率」的问题不存在（F2 即最高概率且实测支持）。

## 3. 目标 case 与收益（预期与已测严格区分）

| 组 | case | 本轮改动 | 预期（未实测） | 已测（同口径证据） |
|---|---|---|---|---|
| 最慢 6 case（C=80-128） | 19/21/1/3/20/12（tokens 690-2304，speedup 0.17-0.26） | TILE=8（17-58 token/program → 2-8 tile）+ w/b 驻留 + 大 tile cast | kernel 耗时显著回落，speedup 明显回升；冷缓存下 w/b 重载消除的收益大于热缓存冒烟所见 | 冒烟（热缓存 profiler，噪声大）：case 12 median −14.7%、case 10 min 13.46→8.98us、case 1 −9.4%、case 21 −8.1%；case 19/20/3 在噪声内。**正式提升待 cann-bench 评测** |
| 其余未达标（C=192-256） | 2/4/5/6/7/9/14/15（+10 已在左列） | TILE=4/2，固定开销削减比例略低 | 部分接近或达到 1.0 | 冒烟：case 2 median −16.6%、case 13 −14.8%（13 为已达标 case）；其余噪声内。**待评测** |
| 已达标大 C 组 | 8/13/16-18 | TILE=2/4 改动温和 | 零回归或小幅变化 | 冒烟高样本：case 7/8/14/16 min 差异 ≤0.4us；case 13/17/18 median −14.8%/−4.9%/−11.1%（改善方向）。**回归结论待评测** |
| 已达标小 grid 组 | 22-30（tokens≤40） | single 路径不动（编译等价） | 零回归 | 精度 ratio/max_abs 与 iter0/iter1 基线逐位一致；冒烟 case 22/28 min 3.42/8.32us 与对照 3.26/7.95us 噪声内 |

**已测正确性证据**（与收益无关，为交付门槛）：修复后完整重测 30/30 PASS + 连续调用 18/18 PASS；修复前 case 20 必崩（已修复并重测）。

## 4. 融合数据流现状（本轮改动后）

- **单 kernel 内完成**：四路 strided gather → UB 内 Concat → fp32 统计（两遍法 + corr）→ 仿射 → 写回，与 iter0-iter2 相同；无第二 kernel、无 workspace。
- **数据流**：w/b 循环前一次加载驻留 UB（每 program 8 段）；x 每 tile 4 次 (TILE,BLOCK_C) 2D 读，统计/仿射在片上完成，输出每 token 一次 4C 写。program 内每 tile 迭代驻留 8×TILE×BLOCK_C×4B + w/b ≈ 36-48KB（<< 192KB UB），随迭代释放重用。上述为源码级陈述，片上具体布局由编译器管理。
- **对融合的影响**：融合程度不变（中间 concat 仍不物化，HBM 中间读写无增减）；唯一数据流实质改动是 w/b 由「每 token GM 重载」改为「每 program 一次驻留」——在 cann-bench 冷缓存（CacheClean）口径下消除的是真 HBM 访问，预期收益大于热缓存冒烟所见。
- **历史否决规避**：history.json 无 ❌ 融合方向记录；insights ❌ 项（改统计轮数/Welford、num_warps/num_stages、非 2 幂 BLOCK_C、BLOCK_C 语义变更）均未触碰；token 分布保持 strided（连续分组变体仍未获授权，未实施）；TILE 逻辑未引入 single 分支。

## 5. 结论

本轮在 F2 骨架内以最小改动完成任务单 T1（w/b 提取 + MULTI_TOKEN 2D tile + host TILE）与 T2 结构核对，并修复实施中暴露的尾块描述符越界硬异常（钳位修复）。方案选择先验（F2=0.79）、iter2 实测瓶颈证据、Tech Lead 裁定三方一致。精度自测 30/30 + 18/18 通过（修复后完整重测），single 路径误差逐位一致。性能收益方向性证据良好但均属非正式口径，**正式提升幅度与零回归结论以 cann-bench 同口径评测为准**。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.

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
    "valid_samples": 3,
    "completed_improvements": 2,
    "enough_samples": false,
    "start_iteration": 1,
    "end_iteration": 3,
    "start_best_avg_speedup": 1.208110829133419,
    "end_best_avg_speedup": 1.3213246705259825,
    "cumulative_improvement": 0.09371146972813088,
    "threshold": 0.05,
    "stagnated": false
  },
  "case_trends": [
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "current_speedup": 0.03352455387221591,
      "currently_underperforming": true,
      "gap_to_one": 0.9664754461277841,
      "change_from_previous": -0.18168800173826355,
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
        },
        {
          "iteration": 3,
          "speedup": 0.03352455387221591,
          "gap_to_one": 0.9664754461277841
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "current_speedup": 0.013386247424253987,
      "currently_underperforming": true,
      "gap_to_one": 0.986613752575746,
      "change_from_previous": -0.27894708590907935,
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
        },
        {
          "iteration": 3,
          "speedup": 0.013386247424253987,
          "gap_to_one": 0.986613752575746
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "current_speedup": 0.013995840882885852,
      "currently_underperforming": true,
      "gap_to_one": 0.9860041591171141,
      "change_from_previous": -0.521451185505456,
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
        },
        {
          "iteration": 3,
          "speedup": 0.013995840882885852,
          "gap_to_one": 0.9860041591171141
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "current_speedup": 0.01183013412790338,
      "currently_underperforming": true,
      "gap_to_one": 0.9881698658720967,
      "change_from_previous": -0.2518411635368912,
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
        },
        {
          "iteration": 3,
          "speedup": 0.01183013412790338,
          "gap_to_one": 0.9881698658720967
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "current_speedup": 0.012946522662197799,
      "currently_underperforming": true,
      "gap_to_one": 0.9870534773378022,
      "change_from_previous": -1.012084002368327,
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
        },
        {
          "iteration": 3,
          "speedup": 0.012946522662197799,
          "gap_to_one": 0.9870534773378022
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "current_speedup": 0.013897764394934741,
      "currently_underperforming": true,
      "gap_to_one": 0.9861022356050653,
      "change_from_previous": -0.9757272811092517,
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
        },
        {
          "iteration": 3,
          "speedup": 0.013897764394934741,
          "gap_to_one": 0.9861022356050653
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "current_speedup": 0.00944460418924598,
      "currently_underperforming": true,
      "gap_to_one": 0.990555395810754,
      "change_from_previous": -0.7717482242044607,
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
        },
        {
          "iteration": 3,
          "speedup": 0.00944460418924598,
          "gap_to_one": 0.990555395810754
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "current_speedup": 0.011364143118338709,
      "currently_underperforming": true,
      "gap_to_one": 0.9886358568816613,
      "change_from_previous": -1.5044983816668562,
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
        },
        {
          "iteration": 3,
          "speedup": 0.011364143118338709,
          "gap_to_one": 0.9886358568816613
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "current_speedup": 0.012641778114139473,
      "currently_underperforming": true,
      "gap_to_one": 0.9873582218858605,
      "change_from_previous": -1.6511852205752446,
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
        },
        {
          "iteration": 3,
          "speedup": 0.012641778114139473,
          "gap_to_one": 0.9873582218858605
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "current_speedup": 0.010621106737916775,
      "currently_underperforming": true,
      "gap_to_one": 0.9893788932620833,
      "change_from_previous": -1.6125182707992958,
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
        },
        {
          "iteration": 3,
          "speedup": 0.010621106737916775,
          "gap_to_one": 0.9893788932620833
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "current_speedup": 0.043574516851591395,
      "currently_underperforming": true,
      "gap_to_one": 0.9564254831484086,
      "change_from_previous": -0.12601494082707054,
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
        },
        {
          "iteration": 3,
          "speedup": 0.043574516851591395,
          "gap_to_one": 0.9564254831484086
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "current_speedup": 0.03161154610260606,
      "currently_underperforming": true,
      "gap_to_one": 0.9683884538973939,
      "change_from_previous": -0.3247784407221634,
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
        },
        {
          "iteration": 3,
          "speedup": 0.03161154610260606,
          "gap_to_one": 0.9683884538973939
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "current_speedup": 0.041165671140071886,
      "currently_underperforming": true,
      "gap_to_one": 0.9588343288599281,
      "change_from_previous": -0.20523581469803368,
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
        },
        {
          "iteration": 3,
          "speedup": 0.041165671140071886,
          "gap_to_one": 0.9588343288599281
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "current_speedup": 0.03176051923147841,
      "currently_underperforming": true,
      "gap_to_one": 0.9682394807685216,
      "change_from_previous": -0.18312221529945974,
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
        },
        {
          "iteration": 3,
          "speedup": 0.03176051923147841,
          "gap_to_one": 0.9682394807685216
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "current_speedup": 2.1344086021505375,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.0338795016214366,
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
        },
        {
          "iteration": 3,
          "speedup": 2.1344086021505375,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "current_speedup": 2.7516339869281046,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.2990906507530551,
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
        },
        {
          "iteration": 3,
          "speedup": 2.7516339869281046,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "current_speedup": 1.9545454545454546,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.18357609241587136,
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
        },
        {
          "iteration": 3,
          "speedup": 1.9545454545454546,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "current_speedup": 3.720873786407767,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.3292808660537845,
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
        },
        {
          "iteration": 3,
          "speedup": 3.720873786407767,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "current_speedup": 3.60752688172043,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.09443787648482793,
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
        },
        {
          "iteration": 3,
          "speedup": 3.60752688172043,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "current_speedup": 3.4419642857142856,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.2647664835164836,
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
        },
        {
          "iteration": 3,
          "speedup": 3.4419642857142856,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "current_speedup": 2.3714733542319753,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.1132643990080946,
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
        },
        {
          "iteration": 3,
          "speedup": 2.3714733542319753,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "current_speedup": 2.344463971880492,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.0562993063573356,
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
        },
        {
          "iteration": 3,
          "speedup": 2.344463971880492,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "current_speedup": 0.028739044167044227,
      "currently_underperforming": true,
      "gap_to_one": 0.9712609558329558,
      "change_from_previous": -0.19816363253210467,
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
        },
        {
          "iteration": 3,
          "speedup": 0.028739044167044227,
          "gap_to_one": 0.9712609558329558
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "current_speedup": 1.9064748201438848,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.08600638286363393,
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
        },
        {
          "iteration": 3,
          "speedup": 1.9064748201438848,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "current_speedup": 0.018800168990283063,
      "currently_underperforming": true,
      "gap_to_one": 0.981199831009717,
      "change_from_previous": -0.392705207353803,
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
        },
        {
          "iteration": 3,
          "speedup": 0.018800168990283063,
          "gap_to_one": 0.981199831009717
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "current_speedup": 0.02195261118952521,
      "currently_underperforming": true,
      "gap_to_one": 0.9780473888104748,
      "change_from_previous": -0.666587420207649,
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
        },
        {
          "iteration": 3,
          "speedup": 0.02195261118952521,
          "gap_to_one": 0.9780473888104748
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "current_speedup": 0.019257650874119358,
      "currently_underperforming": true,
      "gap_to_one": 0.9807423491258807,
      "change_from_previous": -0.41388005610554945,
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
        },
        {
          "iteration": 3,
          "speedup": 0.019257650874119358,
          "gap_to_one": 0.9807423491258807
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "current_speedup": 0.018381842350653734,
      "currently_underperforming": true,
      "gap_to_one": 0.9816181576493462,
      "change_from_previous": -0.8589855818917705,
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
        },
        {
          "iteration": 3,
          "speedup": 0.018381842350653734,
          "gap_to_one": 0.9816181576493462
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "current_speedup": 0.02144123329416217,
      "currently_underperforming": true,
      "gap_to_one": 0.9785587667058379,
      "change_from_previous": -1.7078760357821432,
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
        },
        {
          "iteration": 3,
          "speedup": 0.02144123329416217,
          "gap_to_one": 0.9785587667058379
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "current_speedup": 0.018581425054147612,
      "currently_underperforming": true,
      "gap_to_one": 0.9814185749458524,
      "change_from_previous": -0.9221402244303885,
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
        },
        {
          "iteration": 3,
          "speedup": 0.018581425054147612,
          "gap_to_one": 0.9814185749458524
        }
      ]
    }
  ]
}
【性能退步处置】
本轮仍有 case < 1，保留当前实现，只记录退步教训。
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/rollbacks/iter3/record.json`；相对工作目录：`selection/rollbacks/iter3/record.json`；迭代模板：`selection/rollbacks/<iter>/record.json`；用途：本轮退步处置与版本对应关系；怎么看：核对 action/status、当前评测版本与最佳来源；本轮退步成绩不会被恢复后的成绩覆盖

程序定义的填写格式（先读后填）：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision_schema.json`；相对工作目录：`knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision_schema.json`；迭代模板：`knowledge/stage9/<iter>/4b3764bf69c14a94a4be8cf959e4391a/decision_schema.json`；用途：程序定义的 Stage9 JSON 格式（只读）；怎么看：按本轮字段、类型及层级填写；未列出的经验字段必须省略；case_analysis 按本轮要求填写；修改文件由 changes 自动汇总
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision_template.json`；相对工作目录：`knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision_template.json`；迭代模板：`knowledge/stage9/<iter>/4b3764bf69c14a94a4be8cf959e4391a/decision_template.json`；用途：本请求的决策填写模板（只读）；怎么看：复制 iteration/request_id，填写真实结论、任务和已列出的性能经验；经验 case 按实测增补受益及受损项；空白及 null 必须按 schema 填写
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/case_catalog.json`；相对工作目录：`knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/case_catalog.json`；迭代模板：`knowledge/stage9/<iter>/4b3764bf69c14a94a4be8cf959e4391a/case_catalog.json`；用途：本轮可核对的完整 case ID 清单（只读）；怎么看：已列出 ID 时按原名选择；没有清单时回查 task，不把最慢六例当成全部用例
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/impl`；相对工作目录：`impl`；用途：本轮建议实际实施的代码基底（只读查阅）；怎么看：在此核对 case 路由与待改函数；changes 仍填写实施时的 impl/... 相对路径。若本轮需回退最佳版本，不按退步代码臆造修改位置
