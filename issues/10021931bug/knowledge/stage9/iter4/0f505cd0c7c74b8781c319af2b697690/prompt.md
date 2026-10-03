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

🚨 已验证的失败教训（knowledge/regression_patterns.md）
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/regression_patterns.md`；相对工作目录：`knowledge/regression_patterns.md`；用途：以下退步教训摘要的来源；怎么看：先核对经验中的框架与芯片，再查看退步改动、根因和受影响 case；旧记录未注明环境时须先验证再复用
这是历轮迭代中性能大幅退步（≥5%）时自动记录的失败教训。先核对适用条件，在已验证条件下避免重复失败；改变条件后须重新验证。
参考方式：
  - 写代码（stage2/3）：避免重复相同条件下已导致退步的改动
  - 分析性能（stage7）：核对当前瓶颈和记录中的失败条件是否一致
  - 搜索方案（stage8）：排除已验证条件下的失败改法，条件改变须提供新证据
  - Tech Lead（stage9）：引用失败教训作为否决 insight 的证据

  ❌ iter3: speedup 1.3213246705259825 → 0.8224094689497551 (-37.8%)
     改了什么: 在 F2 单 kernel MULTI_TOKEN=True 分支内，将 1D 逐 token serial loop 改为 (TILE, BLOCK_C) 2D tile 路径：新增 TILE: tl.constexpr，token 解码改为 (TILE,) 向量化，4 段 x 读改为 2D masked strided gather，tl.sum 从 axis=0 改为 axis=1，store 改为 2D masked store，新增 tmask 尾块钳位。w/b 8 段 masked load 移出循环。Host 新增 tile 计算。代码：impl/cann_bench/fused_patch_concat_layernorm.py 行 47-106（2D 分支）、行 27（TILE 签名）、行 38-45（w/b 提取）、行 207（tile 计算）。依据：develop/iter2/融合方案选择决策依据.md。
     框架=Triton；后端=triton-ascend；芯片=Ascend910B3；SoC=Ascend910B3；编程模型=Triton block program (SPMD)
     为什么退步: triton-ascend 3.2.1 后端对 (TILE, BLOCK_C) 2D tensor 操作生成灾难性低效编译代码。证据：(1) 全部 21 个 MULTI_TOKEN=True case 退步 22-153x，kernel 768-4524us vs iter2 24-203us；(2) MULTI_TOKEN=False case 22-30 不受影响（speedup 1.9-3.7）；(3) fp32 case 17 退步 132x 排除 cast；(4) TILE=2 退步 79-153x 排除 TILE 取值/UB 溢出；(5) kernel_details Duration 证实退步在 kernel 内部。2D strided gather/axis=1 reduction/2D broadcast mask 的具体低效点为推测，需编译产物验证。
     适用条件: 适用于 Ascend910B3 + triton-ascend 3.2.1。2D tensor 操作（tl.load/tl.store 2D 索引、tl.sum axis=1、2D broadcast mask）在当前后端生成极低效代码，退步 22-153x。覆盖所有测试的 BLOCK_C(128/256/512)、dtype(fp16/bf16/fp32)、TILE(2/4/8)。未验证：未来后端版本；reshape+1D 变通。

📊 性能提升检测（程序自动计算，数据可信）
  avg_speedup: 0.8224094689497551 → 1.5389794201417895 (+87.1%)
  对比轮次: iter3 → iter4
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/design_rationale.md`；相对工作目录：`develop/iter3/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：上轮评测附近的历史设计参考；怎么看：用于回顾改动意图；实际被评测版本以代码绑定的选择依据和快照为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/design_rationale.md`；相对工作目录：`develop/iter3/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：本轮评测附近的历史设计参考；怎么看：与上轮参考比较；先用代码绑定依据或快照确认版本，再归因成功经验
  逐 case 对比（提升最大的在前）:
    case_level3/fused_patch_concat_layernorm_18: 0.010621106737916775 → 1.7730968218773095 (+16594.1%)
    case_level3/fused_patch_concat_layernorm_17: 0.012641778114139473 → 1.8849294729027468 (+14810.3%)
    case_level3/fused_patch_concat_layernorm_16: 0.011364143118338709 → 1.6814516129032255 (+14696.1%)
    case_level3/fused_patch_concat_layernorm_15: 0.00944460418924598 → 1.0460558549730525 (+10975.7%)
    case_level3/fused_patch_concat_layernorm_14: 0.013897764394934741 → 1.399125064333505 (+9967.3%)
    case_level3/fused_patch_concat_layernorm_13: 0.012946522662197799 → 1.248017839444995 (+9539.8%)

⚡ 本轮性能大幅提升！请对照上面两轮的 design_rationale，总结这次改动为什么有效，填写 proven_pattern 字段。- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/fusion_library.json`；相对工作目录：`fusion/fusion_library.json`；用途：JSON 融合算子库；怎么看：比较候选数据流、前提与初始概率，以实测证据决定保留或更换
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/perf_result.json`；相对工作目录：`eval/iter4/perf_result.json`；迭代模板：`eval/<iter>/perf_result.json`；用途：本轮性能结果；怎么看：看 avg_speedup、各 case 的 speedup 和最慢用例，不只看平均值
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/prof_data`；相对工作目录：`eval/iter4/prof_data`；迭代模板：`eval/<iter>/prof_data`；用途：本轮 profiler 数据（按需回查）；怎么看：仅在结论矛盾或缺少证据时按 case 回查
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/perf_reports`；相对工作目录：`eval/iter4/perf_reports`；迭代模板：`eval/<iter>/perf_reports`；用途：本轮性能报告；怎么看：回查原始计时与评分，核实性能汇总及反作弊信号
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/profile/iter4/bottleneck_analysis.md`；相对工作目录：`profile/iter4/bottleneck_analysis.md`；迭代模板：`profile/<iter>/bottleneck_analysis.md`；用途：本轮瓶颈分析；怎么看：优先读 Stage7 结论，区分已证实根因和推测
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter4/FIX_DIRECTIVE.md`；相对工作目录：`search/iter4/FIX_DIRECTIVE.md`；迭代模板：`search/<iter>/FIX_DIRECTIVE.md`；用途：本轮修改指令；怎么看：优先读 Stage8 结论，转成原有 P0/P1/P2 优先级
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter4/SEARCH_REPORT.md`；相对工作目录：`search/iter4/SEARCH_REPORT.md`；迭代模板：`search/<iter>/SEARCH_REPORT.md`；用途：本轮搜索报告；怎么看：按需检查候选方法、依据及适用条件
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/build/iter4/build.log`；相对工作目录：`build/iter4/build.log`；迭代模板：`build/<iter>/build.log`；用途：本轮编译日志；怎么看：看 STATUS 和首个真实错误，定位构建/接口问题
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/precision_result.json`；相对工作目录：`eval/iter4/precision_result.json`；迭代模板：`eval/<iter>/precision_result.json`；用途：本轮精度结果；怎么看：核对总数、通过数及失败 case，优先处理正确性
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/precision_reports`；相对工作目录：`eval/iter4/precision_reports`；迭代模板：`eval/<iter>/precision_reports`；用途：本轮精度报告详情；怎么看：按失败 case 查误差、输入和原始报错
当前场景缺失材料（不能当作已完成结论）：
历轮设计思路：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/design_rationale.md`；相对工作目录：`develop/iter0/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter0 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/design_rationale.md`；相对工作目录：`develop/iter1/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter1 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/design_rationale.md`；相对工作目录：`develop/iter2/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter2 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/design_rationale.md`；相对工作目录：`develop/iter3/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter3 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
请分析本轮结果，回顾历轮设计思路；只读 /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/history.json，不要覆盖它。
Stage9 输出文件：/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision.json
Stage9 请求编号：0f505cd0c7c74b8781c319af2b697690
Stage9 当前轮次：4
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision.json`；相对工作目录：`knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision.json`；迭代模板：`knowledge/stage9/<iter>/0f505cd0c7c74b8781c319af2b697690/decision.json`；用途：本次 Stage9 决策输出（待你生成）；怎么看：按 role 写 iteration、request_id、ledger_entry 和模型结论；经验使用 proven_pattern/regression_pattern/pitfall 公共字段

本次 ledger_entry.case_analysis 必须逐一覆盖以下 case_id（原样复制，不能缩写；每例写 observation、explanation、evidence、next_action；根因未证实须注明待验证）：
["level3/fused_patch_concat_layernorm_19", "level3/fused_patch_concat_layernorm_10", "level3/fused_patch_concat_layernorm_1", "level3/fused_patch_concat_layernorm_12", "level3/fused_patch_concat_layernorm_3", "level3/fused_patch_concat_layernorm_11"]


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
  "cache_reused": true,
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
实现 SHA256：a71e1bc42fd1e0a8f8683834824716cb8db2aaa7f3b602a5cbb07a486c3e712f
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/fusion_library.json`；相对工作目录：`develop/iter3/fusion_library.json`；迭代模板：`develop/<iter>/fusion_library.json`；用途：当前轮实际融合选择与方案库；怎么看：看 selection 和新增方法；原始 Jev probability 只是先验参考
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/融合方案选择决策依据.md`；相对工作目录：`develop/iter3/融合方案选择决策依据.md`；迭代模板：`develop/<iter>/融合方案选择决策依据.md`；用途：当前代码的融合选择决策依据；怎么看：先看选定方案、未选其他方案的原因及目标 case，再与实测核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/self_test_report.md`；相对工作目录：`develop/iter3/self_test_report.md`；迭代模板：`develop/<iter>/self_test_report.md`；用途：当前代码的自测报告；怎么看：看给定 case 与连续调用是否实际执行，区分失败、未执行和通过
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/self_test_result.json`；相对工作目录：`develop/iter3/self_test_result.json`；迭代模板：`develop/<iter>/self_test_result.json`；用途：程序校验使用的自测 JSON；怎么看：看 provided_cases、continuous_calls 的通过状态及 evidence_path
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/self_test.log`；相对工作目录：`develop/iter3/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：给定 case 的原始自测日志；怎么看：核对实际命令、case 数量及结果是否支持自测报告
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/self_test.log`；相对工作目录：`develop/iter3/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：连续调用的原始自测日志；怎么看：核对相同 shape 更换参数后逐次对照参考实现的结果
最佳实现入库资格：True；Source, fusion decision and self-tests are bound.
本轮库与只读初始 Jev 库区分；真实评测优先于初始概率。
{
  "method_ids": [
    "F2"
  ],
  "implementation_plan": "保持 iter0-iter2 已验证的 F2 垂直融合单 kernel 骨架（Slice×4 strided gather → UB 内 Concat → LayerNorm，fp32 统计 + corr 修正轮）与 tokens>64 → grid=min(tokens,40) 的 MULTI_TOKEN 路由。本轮按 Tech Lead 任务单 T1 执行零新增风险回退：(1) 删除 MULTI_TOKEN=True 分支的整个 (TILE, BLOCK_C) 2D tile 实现（lanes/hw/tmask/tile_mask/2D load/axis=1 归约/2D store 全部移除），恢复 iter2 best 的 1D 逐 token serial loop（token_lo=pid, token_hi=tokens, token_step=tl.num_programs(0)，1D load/store，axis=0 归约）；(2) kernel 签名删除 TILE: tl.constexpr；(3) host 删除 tile=min(16, max(1, 1024//next_power_of_2(C))) 计算与 TILE 实参，grid 与 MULTI_TOKEN=(tokens>64) 路由不动；(4) 保留 w/b 循环外提取（w0..b3 8 段 masked load 在循环前一次加载驻留 UB，store 引用 w0-b3），该改动独立于 2D tile，是相对 iter2 best 的唯一净增量；offs/channel_mask/offs_f 等循环不变量按 FIX_DIRECTIVE 骨架一并保留在分支外。数值路径（fp32 升精度、两遍法 + corr 修正、tl.maximum(var,0)、1/sqrt(var+eps)、回写原 dtype）与融合结构不变，不叠加任何新优化。",
  "reason": "维持初始库 rank-1 方法 F2（probability 0.79）不变：iter0-iter2 三轮实测证明融合数据流本身有效（小 grid 组 speedup 1.98-3.71，大 C 组 1.52-1.73），iter3 的灾难退步源于执行组织（2D tile 在 triton-ascend 3.2.1 编译质量极差）而非融合方案，Tech Lead 裁定与 FIX_DIRECTIVE 均要求回退而非更换 method。不选 F5（0.58）/F8（0.44）：本轮以恢复 iter2 水平为最高优先级，禁止叠加新优化或方案变体；2D tile 尝试已作为失败教训记录于 attempts[0]，本轮避免重复。初始 Jev 概率仅作先验参考，实测与 profiler 证据优先。",
  "target_cases": [
    "T1 主目标：6 个最慢 case level3/fused_patch_concat_layernorm_15/18/16/12/17/13（eval/iter3 speedup 0.009-0.013，kernel 3891-4524us）：回退 1D 后恢复 iter2 量级（30-203us），w/b 提取额外收益",
    "T1 覆盖目标：其余 15 个 MULTI_TOKEN case 1-11/14/19/20/21（tokens>=392）",
    "保护目标：9 个 MULTI_TOKEN=False case 22-30（tokens<=40，single 路径语义与 iter2 编译等价，零回归要求）",
    "精度回归目标：全部 30 case（含 C=3/65 尾块、零方差 25-27、近常数 28-30、grid=1 的 case 22）"
  ],
  "actual_changes": [
    "impl/cann_bench/fused_patch_concat_layernorm.py MULTI_TOKEN=True 分支：删除 2D tile 全部代码，替换为 1D 逐 token serial loop（token_lo=pid, token_hi=tokens, token_step=tl.num_programs(0)）；4 段 x 1D load、tl.sum axis=0、1D store",
    "kernel 签名删除 TILE: tl.constexpr；签名恢复为 (..., eps, BLOCK_C: tl.constexpr, MULTI_TOKEN: tl.constexpr)",
    "host 函数删除 tile=min(16, max(1, 1024//triton.next_power_of_2(channels))) 与 TILE=tile 实参；grid=(min(tokens,40),) if tokens>64 else (tokens,) 与 MULTI_TOKEN=(tokens>64) 原样保留",
    "w/b 循环外 8 段 masked load（w0/b0/.../w3/b3，段偏移 0/C/2C/3C）保留在循环前；store 表达式引用 w0-b3（未恢复 iter2 快照的循环内逐 token w/b 加载）",
    "offs/channel_mask/offs_f 循环不变量按 FIX_DIRECTIVE 骨架保留在分支外定义；输入校验/contiguous 守护/torch.empty/tokens==0 早退/函数签名与导出均未改动"
  ],
  "expected_benefits": [
    "21 个 MULTI_TOKEN case（tokens 392-2304）kernel 耗时从 iter3 被评版的 768-4524us 恢复到 iter2 量级（iter2 HAP 24.9-202.98us）；TC6 热缓存冒烟 median 19.2-25.0us 与 iter2 快照同水平（22.8-28.4us），证明回退生效（正式幅度待 cann-bench 冷缓存评测确认）",
    "w/b 循环外提取的独立收益：case 12（tokens=2304）每 program 从 58 次循环×8 次 masked load 降为 8 次，冒烟 median -16.1%；case 8 -15.8%；其余 case 在热缓存噪声（±14%）内；冷缓存（CacheClean）正式口径下收益预期更大（5-15% 量级，Tech Lead 估计，待评测）",
    "9 个小 grid case 22-30 零回归：MULTI_TOKEN=False 路径与 iter2 编译等价（TC4 误差与 iter0-iter2 基线同量级，case 22 逐位一致）",
    "中间 concat 张量仍不在 HBM 物化（融合程度不变）；HBM 读写维持每 token 一次 4 段读 + 一次 4 段写，w/b 从每 token GM 重载改为每 program 一次"
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

# 融合方案选择决策依据 — fused_patch_concat_layernorm（develop/iter3）

> 本轮为 iter3 恢复重做轮：前次会话已按任务单完成代码回退（impl 修改于 07:05），但未产出文档与自测；本轮核对代码与 FIX_DIRECTIVE 骨架一致后全量重测并补齐全部交付物。自测证据见 `self_test_report.md` 与 `self_test.log`；本文只陈述方案选择，与测试报告分开。

## 一、本轮实际选择

- **选中方法**：`F2 Vertical Fusion（垂直融合／单 kernel 全融合）`，初始 Jev probability=0.79（初始库 rank-1），无新增方法，无新概率。
- **实现形态**：F2 单 kernel 骨架 + `tokens>64 → grid=min(tokens,40)` 的 MULTI_TOKEN **1D 逐 token serial loop**（iter2 best 已验证的执行组织）+ **w/b 循环外提取**（相对 iter2 best 的唯一净增量）。
- **否决/回退的尝试**：iter3 被评版的 `F2 + (TILE, BLOCK_C) 2D tile` 路径。实测 avg_speedup 0.822（较 iter2 的 1.321 退步 -37.8%），21 个 MULTI_TOKEN case kernel 耗时 768-4524us（退步 22-153x），profiler kernel_details 证实根因是 triton-ascend 3.2.1 对 2D load/store/axis=1 归约生成灾难性低效代码（fp32 case 17 无 cast 仍退步 132x，TILE=2 也退步 79-153x，排除 cast 与 TILE 取值/UB 溢出）。详见 `attempts[0]`（develop/iter3/fusion_library.json）与 knowledge/regression_patterns.md iter3 条目。

## 二、为什么不更换融合方法（保留旧方案的理由）

1. **融合数据流本身已被三轮实测验证有效**：iter0-iter2 中，小 grid 组（case 22-30，F2 单 kernel vs torch 多算子）speedup 稳定 1.98-3.71，大 C 组（16-18/8/13）在 MULTI_TOKEN serial loop 下 1.52-1.73。iter3 退步与融合结构无关——MULTI_TOKEN=False 的 case 22-30（同一份融合 kernel 的另一分支）在 iter3 被评版中保持 1.9-3.7 正常，证明问题严格限于 2D 执行组织。
2. **Tech Lead 裁定明确**：direction 指出"立即回退 MULTI_TOKEN=True 分支至 iter2 best 的 1D 逐 token serial loop，保留 w/b 循环外提取……不叠加新优化"。回退即恢复到已验证 avg_speedup=1.321 的形态，风险最低。
3. **不选 F5（0.58）/F8（0.44）**：按 shape 路由或多路径流水属于新增方案变量，与"零新增风险回退"的最高优先级冲突；且 tokens>64 条件路由（F5.schedule_routing 的一个受限形态）已在 iter1-iter2 验证并保留，本轮无需扩展。
4. **2D tile 方向永久记为失败教训**：knowledge/regression_patterns.md 已收录"triton-ascend 3.2.1 上 2D tensor 操作（2D load/store、axis=1 归约、2D broadcast mask）退步 22-153x"。未来重试需先确认后端修复。

## 三、参考的概率与已测证据

- 初始概率仅作先验：F2=0.79 为最可行的初始方向，与 iter0 首版选择一致；实际取舍完全以实测为准。
- 已测证据链：
  - iter2 best（回退目标）：`selection/records/iter2-844fc6b40708c857f98b/reports/perf_result.json`，avg_speedup=1.321，达标 14/30；
  - iter3 被评版（被回退对象）：`eval/iter3/perf_result.json`，avg_speedup=0.822，达标 9/30；
  - 本轮回退版自测：30/30 精度 PASS、18/18 连续调用 PASS（develop/iter3/self_test.log）；
  - 本轮 TC6 热缓存 A/B（当前实现 vs iter2 快照，同口径同进程条件）：MULTI_TOKEN 组 median 19.2-25.0us vs 22.8-28.4us，case 8 -15.8%、case 12 -16.1%（w/b 提取收益最大的 case：tokens=2304，每 program 从 58 次循环×8 次 masked load 降为 8 次），其余 case 差异在热缓存噪声（±14%，iter2 冒烟同量级）内；single 路径 case 22 -3.3%（噪声，无回归）。

## 四、针对哪些 case、预期影响

- **主目标**：worst_6_cases（eval/iter3）：case 15/18/16/12/17/13（speedup 0.009-0.013，kernel 3891-4524us）。回退后恢复 iter2 量级（iter2 HAP：case 15=54.66us、18=29.56us、16=30.26us、12=202.98us、17=30.52us、13=49.14us），w/b 提取在冷缓存口径下预期再贡献 5-15%（Tech Lead 估计，**待正式评测确认**）。
- **覆盖目标**：全部 21 个 MULTI_TOKEN case（tokens>=392）。
- **保护目标**：9 个 single 路径 case 22-30（tokens<=40）零回归（自测已验证：case 22/25 逐位一致，28-30 与 iter0-iter2 基线同量级）。

## 五、预期收益与已测收益的区分

| 项 | 已测（本轮自测口径） | 预期（待正式评测） |
|---|---|---|
| 精度 | 30/30 PASS + 18/18 连续调用 PASS | cann-bench 正式精度待评测 |
| kernel 耗时量级恢复 | 热缓存 median 19-25us ≈ iter2 快照（远低于被评版 768-4524us） | CacheClean 冷缓存下恢复 24.9-202.98us 区间（iter2 HAP 口径） |
| w/b 提取独立收益 | case 8/12 热缓存 median -15.8%/-16.1% | 冷缓存下 5-15% 量级（Tech Lead 估计） |
| avg_speedup | 未评测 | 目标恢复 ≥1.32（iter2 水平） |

> ⚠️ 正式性能结论以 cann-bench 同口径评测（CacheClean 冷缓存）为准；本轮所有本地计时均为热缓存 profiler 冒烟，只用于验证"回退生效、量级恢复"，不作为收益结论。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/state.json`；相对工作目录：`selection/state.json`；用途：有效评测索引与语义窗口配置；怎么看：以下 window 和 case_trends 由程序按同口径历史快照计算，不将失败轮计入窗口
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/impl`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/impl`；迭代模板：`selection/records/<iter>-<指纹>/impl`；用途：最佳已验证实现的独立代码快照目录；怎么看：读取该目录的实现，勿把当前工作代码当作历史最佳版本
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/manifest.json`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/manifest.json`；迭代模板：`selection/records/<iter>-<指纹>/manifest.json`；用途：最佳实现快照清单；怎么看：核对 iteration、融合方案、avg_speedup、hap 及代码和评测证据路径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/reports/performance_source.json`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/reports/performance_source.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/performance_source.json`；用途：最佳实现对应的原始性能报告；怎么看：按 case 看 speedup、耗时和 HAP，确认使用同一评测口径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/reports/perf_result.json`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/reports/perf_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/perf_result.json`；用途：最佳实现的结构化性能结果；怎么看：看 avg_speedup、cases 和最慢用例，配合窗口和趋势判断
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/reports/precision_result.json`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/reports/precision_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/precision_result.json`；用途：最佳实现对应的精度结果；怎么看：核对 precision_overall、通过数量和原始精度报告定位
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/0_decision_rationale.md`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/evidence/0_decision_rationale.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/0_decision_rationale.md`；用途：最佳实现的融合选择依据快照；怎么看：看当时的选择理由，与当前方案区别及实测结果核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/1_fusion_library.json`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/evidence/1_fusion_library.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/1_fusion_library.json`；用途：最佳实现的当轮融合方案库快照；怎么看：看 selection 中实际方法和实现方案，不把初始概率当性能
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/2_self_test_report.md`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/evidence/2_self_test_report.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/2_self_test_report.md`；用途：最佳实现的自测报告快照；怎么看：核对给定 case 及连续调用的执行范围与结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/3_self_test_result.json`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/evidence/3_self_test_result.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/3_self_test_result.json`；用途：最佳实现的自测资格 JSON 快照；怎么看：核对两类自测状态；其中历史日志路径以本清单中的归档日志为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/4_selftest_log_continuous.log`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/evidence/4_selftest_log_continuous.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/4_selftest_log_continuous.log`；用途：最佳实现的连续调用原始日志快照；怎么看：核对同 shape 更换参数后逐次比较参考实现的结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/5_selftest_log_provided.log`；相对工作目录：`selection/records/iter4-60dea9e34578bd49ed55/evidence/5_selftest_log_provided.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/5_selftest_log_provided.log`；用途：最佳实现的给定 case 原始自测日志快照；怎么看：核对原始执行输出是否支持报告结论
{
  "eligible": true,
  "reason": "Valid measured implementation recorded",
  "should_exit": false,
  "review_fusion": false,
  "best": {
    "iteration": 4,
    "selection_label": "best_available",
    "all_cases_pass": false,
    "avg_speedup": 1.5389794201417895,
    "hap": {
      "performance_score": 29.397139156733665,
      "cases": [
        {
          "case_id": "level3/fused_patch_concat_layernorm_19",
          "perf_score": 0.4242040011270781,
          "t_hw_us": 3.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 44.22
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_10",
          "perf_score": 0.4251346499102334,
          "t_hw_us": 5.26,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 69.3
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_1",
          "perf_score": 0.42566191446028506,
          "t_hw_us": 3.48,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 45.78
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_12",
          "perf_score": 0.42681197944355836,
          "t_hw_us": 5.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 70.04
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_3",
          "perf_score": 0.4346855717474071,
          "t_hw_us": 3.68,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 46.74
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_11",
          "perf_score": 0.44066270484422837,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 67.56
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_21",
          "perf_score": 0.4426674282450379,
          "t_hw_us": 3.45,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 42.48
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_4",
          "perf_score": 0.44733082218469933,
          "t_hw_us": 3.83,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 46.38
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_6",
          "perf_score": 0.4719015603196753,
          "t_hw_us": 4.13,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 45.76
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_20",
          "perf_score": 0.5103276353276354,
          "t_hw_us": 3.18,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 30.68
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_2",
          "perf_score": 0.5113795518207283,
          "t_hw_us": 3.25,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 31.16
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_15",
          "perf_score": 0.5125366764470526,
          "t_hw_us": 4.27,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 40.82
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_7",
          "perf_score": 0.5591884641180416,
          "t_hw_us": 3.71,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 30.0
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_13",
          "perf_score": 0.5620582765034098,
          "t_hw_us": 5.04,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 40.36
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_9",
          "perf_score": 0.577039757304806,
          "t_hw_us": 4.01,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 30.5
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_14",
          "perf_score": 0.5941712204007287,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 38.86
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_5",
          "perf_score": 0.6016768292682927,
          "t_hw_us": 4.39,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 30.52
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_24",
          "perf_score": 0.628731343283582,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.98
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_22",
          "perf_score": 0.6402214022140222,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.9
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_16",
          "perf_score": 0.645302485540097,
          "t_hw_us": 4.59,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 27.28
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_18",
          "perf_score": 0.659841075794621,
          "t_hw_us": 4.8,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 27.06
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_30",
          "perf_score": 0.6760204081632654,
          "t_hw_us": 2.65,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 14.08
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_17",
          "perf_score": 0.6764357608052102,
          "t_hw_us": 5.08,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 26.94
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_8",
          "perf_score": 0.6979466858789626,
          "t_hw_us": 4.31,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 21.08
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_28",
          "perf_score": 0.7050750906266182,
          "t_hw_us": 3.03,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 14.42
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_29",
          "perf_score": 0.7197242206235012,
          "t_hw_us": 2.67,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 12.02
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_26",
          "perf_score": 0.7947368421052632,
          "t_hw_us": 1.34,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.46
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
          "case_id": "level3/fused_patch_concat_layernorm_27",
          "perf_score": 0.7995391705069124,
          "t_hw_us": 1.54,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 5.02
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_25",
          "perf_score": 0.8189910979228486,
          "t_hw_us": 1.53,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.58
            }
          }
        }
      ]
    },
    "implementation_dir": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/impl",
    "manifest_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/manifest.json",
    "performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/reports/performance_source.json",
    "precision_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/reports/precision_result.json",
    "fusion_scheme": {
      "method_ids": [
        "F2"
      ],
      "implementation_plan": "保持 iter0-iter2 已验证的 F2 垂直融合单 kernel 骨架（Slice×4 strided gather → UB 内 Concat → LayerNorm，fp32 统计 + corr 修正轮）与 tokens>64 → grid=min(tokens,40) 的 MULTI_TOKEN 路由。本轮按 Tech Lead 任务单 T1 执行零新增风险回退：(1) 删除 MULTI_TOKEN=True 分支的整个 (TILE, BLOCK_C) 2D tile 实现（lanes/hw/tmask/tile_mask/2D load/axis=1 归约/2D store 全部移除），恢复 iter2 best 的 1D 逐 token serial loop（token_lo=pid, token_hi=tokens, token_step=tl.num_programs(0)，1D load/store，axis=0 归约）；(2) kernel 签名删除 TILE: tl.constexpr；(3) host 删除 tile=min(16, max(1, 1024//next_power_of_2(C))) 计算与 TILE 实参，grid 与 MULTI_TOKEN=(tokens>64) 路由不动；(4) 保留 w/b 循环外提取（w0..b3 8 段 masked load 在循环前一次加载驻留 UB，store 引用 w0-b3），该改动独立于 2D tile，是相对 iter2 best 的唯一净增量；offs/channel_mask/offs_f 等循环不变量按 FIX_DIRECTIVE 骨架一并保留在分支外。数值路径（fp32 升精度、两遍法 + corr 修正、tl.maximum(var,0)、1/sqrt(var+eps)、回写原 dtype）与融合结构不变，不叠加任何新优化。",
      "reason": "维持初始库 rank-1 方法 F2（probability 0.79）不变：iter0-iter2 三轮实测证明融合数据流本身有效（小 grid 组 speedup 1.98-3.71，大 C 组 1.52-1.73），iter3 的灾难退步源于执行组织（2D tile 在 triton-ascend 3.2.1 编译质量极差）而非融合方案，Tech Lead 裁定与 FIX_DIRECTIVE 均要求回退而非更换 method。不选 F5（0.58）/F8（0.44）：本轮以恢复 iter2 水平为最高优先级，禁止叠加新优化或方案变体；2D tile 尝试已作为失败教训记录于 attempts[0]，本轮避免重复。初始 Jev 概率仅作先验参考，实测与 profiler 证据优先。",
      "target_cases": [
        "T1 主目标：6 个最慢 case level3/fused_patch_concat_layernorm_15/18/16/12/17/13（eval/iter3 speedup 0.009-0.013，kernel 3891-4524us）：回退 1D 后恢复 iter2 量级（30-203us），w/b 提取额外收益",
        "T1 覆盖目标：其余 15 个 MULTI_TOKEN case 1-11/14/19/20/21（tokens>=392）",
        "保护目标：9 个 MULTI_TOKEN=False case 22-30（tokens<=40，single 路径语义与 iter2 编译等价，零回归要求）",
        "精度回归目标：全部 30 case（含 C=3/65 尾块、零方差 25-27、近常数 28-30、grid=1 的 case 22）"
      ],
      "actual_changes": [
        "impl/cann_bench/fused_patch_concat_layernorm.py MULTI_TOKEN=True 分支：删除 2D tile 全部代码，替换为 1D 逐 token serial loop（token_lo=pid, token_hi=tokens, token_step=tl.num_programs(0)）；4 段 x 1D load、tl.sum axis=0、1D store",
        "kernel 签名删除 TILE: tl.constexpr；签名恢复为 (..., eps, BLOCK_C: tl.constexpr, MULTI_TOKEN: tl.constexpr)",
        "host 函数删除 tile=min(16, max(1, 1024//triton.next_power_of_2(channels))) 与 TILE=tile 实参；grid=(min(tokens,40),) if tokens>64 else (tokens,) 与 MULTI_TOKEN=(tokens>64) 原样保留",
        "w/b 循环外 8 段 masked load（w0/b0/.../w3/b3，段偏移 0/C/2C/3C）保留在循环前；store 表达式引用 w0-b3（未恢复 iter2 快照的循环内逐 token w/b 加载）",
        "offs/channel_mask/offs_f 循环不变量按 FIX_DIRECTIVE 骨架保留在分支外定义；输入校验/contiguous 守护/torch.empty/tokens==0 早退/函数签名与导出均未改动"
      ],
      "expected_benefits": [
        "21 个 MULTI_TOKEN case（tokens 392-2304）kernel 耗时从 iter3 被评版的 768-4524us 恢复到 iter2 量级（iter2 HAP 24.9-202.98us）；TC6 热缓存冒烟 median 19.2-25.0us 与 iter2 快照同水平（22.8-28.4us），证明回退生效（正式幅度待 cann-bench 冷缓存评测确认）",
        "w/b 循环外提取的独立收益：case 12（tokens=2304）每 program 从 58 次循环×8 次 masked load 降为 8 次，冒烟 median -16.1%；case 8 -15.8%；其余 case 在热缓存噪声（±14%）内；冷缓存（CacheClean）正式口径下收益预期更大（5-15% 量级，Tech Lead 估计，待评测）",
        "9 个小 grid case 22-30 零回归：MULTI_TOKEN=False 路径与 iter2 编译等价（TC4 误差与 iter0-iter2 基线同量级，case 22 逐位一致）",
        "中间 concat 张量仍不在 HBM 物化（融合程度不变）；HBM 读写维持每 token 一次 4 段读 + 一次 4 段写，w/b 从每 token GM 重载改为每 program 一次"
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
    "valid_samples": 4,
    "completed_improvements": 3,
    "enough_samples": true,
    "start_iteration": 1,
    "end_iteration": 4,
    "start_best_avg_speedup": 1.208110829133419,
    "end_best_avg_speedup": 1.5389794201417895,
    "cumulative_improvement": 0.2738727135205827,
    "threshold": 0.05,
    "stagnated": false
  },
  "case_trends": [
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "current_speedup": 0.7608125819134993,
      "currently_underperforming": true,
      "gap_to_one": 0.23918741808650068,
      "change_from_previous": 0.7272880280412835,
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
        },
        {
          "iteration": 4,
          "speedup": 0.7608125819134993,
          "gap_to_one": 0.23918741808650068
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "current_speedup": 0.7593073593073593,
      "currently_underperforming": true,
      "gap_to_one": 0.24069264069264074,
      "change_from_previous": 0.7459211118831053,
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
        },
        {
          "iteration": 4,
          "speedup": 0.7593073593073593,
          "gap_to_one": 0.24069264069264074
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "current_speedup": 0.8049141503848432,
      "currently_underperforming": true,
      "gap_to_one": 0.19508584961515685,
      "change_from_previous": 0.7909183095019573,
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
        },
        {
          "iteration": 4,
          "speedup": 0.8049141503848432,
          "gap_to_one": 0.19508584961515685
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "current_speedup": 0.7641347801256425,
      "currently_underperforming": true,
      "gap_to_one": 0.2358652198743575,
      "change_from_previous": 0.7523046459977392,
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
        },
        {
          "iteration": 4,
          "speedup": 0.7641347801256425,
          "gap_to_one": 0.2358652198743575
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "current_speedup": 1.248017839444995,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.2350713167827971,
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
        },
        {
          "iteration": 4,
          "speedup": 1.248017839444995,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "current_speedup": 1.399125064333505,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.3852272999385702,
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
        },
        {
          "iteration": 4,
          "speedup": 1.399125064333505,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "current_speedup": 1.0460558549730525,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.0366112507838066,
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
        },
        {
          "iteration": 4,
          "speedup": 1.0460558549730525,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "current_speedup": 1.6814516129032255,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.6700874697848869,
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
        },
        {
          "iteration": 4,
          "speedup": 1.6814516129032255,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "current_speedup": 1.8849294729027468,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.8722876947886073,
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
        },
        {
          "iteration": 4,
          "speedup": 1.8849294729027468,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "current_speedup": 1.7730968218773095,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.7624757151393928,
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
        },
        {
          "iteration": 4,
          "speedup": 1.7730968218773095,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "current_speedup": 0.7566711895070104,
      "currently_underperforming": true,
      "gap_to_one": 0.2433288104929896,
      "change_from_previous": 0.713096672655419,
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
        },
        {
          "iteration": 4,
          "speedup": 0.7566711895070104,
          "gap_to_one": 0.2433288104929896
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "current_speedup": 1.0417201540436458,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.0101086079410397,
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
        },
        {
          "iteration": 4,
          "speedup": 1.0417201540436458,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "current_speedup": 1.0378096479791394,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.9966439768390676,
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
        },
        {
          "iteration": 4,
          "speedup": 1.0378096479791394,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "current_speedup": 0.8109698681732581,
      "currently_underperforming": true,
      "gap_to_one": 0.1890301318267419,
      "change_from_previous": 0.7792093489417797,
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
        },
        {
          "iteration": 4,
          "speedup": 0.8109698681732581,
          "gap_to_one": 0.1890301318267419
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "current_speedup": 1.620408163265306,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.5140004388852315,
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
        },
        {
          "iteration": 4,
          "speedup": 1.620408163265306,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "current_speedup": 3.0507246376811596,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.2990906507530551,
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
        },
        {
          "iteration": 4,
          "speedup": 3.0507246376811596,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "current_speedup": 1.5542168674698795,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.40032858707557506,
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
        },
        {
          "iteration": 4,
          "speedup": 1.5542168674698795,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "current_speedup": 3.3471615720524017,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.3737122143553653,
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
        },
        {
          "iteration": 4,
          "speedup": 3.3471615720524017,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "current_speedup": 3.008968609865471,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.5985582718549591,
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
        },
        {
          "iteration": 4,
          "speedup": 3.008968609865471,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "current_speedup": 3.071713147410359,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.37025113830392664,
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
        },
        {
          "iteration": 4,
          "speedup": 3.071713147410359,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "current_speedup": 2.0984743411927878,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.2729990130391875,
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
        },
        {
          "iteration": 4,
          "speedup": 2.0984743411927878,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "current_speedup": 2.219633943427621,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.12483002845287094,
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
        },
        {
          "iteration": 4,
          "speedup": 2.219633943427621,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "current_speedup": 0.7871202396234488,
      "currently_underperforming": true,
      "gap_to_one": 0.2128797603765512,
      "change_from_previous": 0.7583811954564046,
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
        },
        {
          "iteration": 4,
          "speedup": 0.7871202396234488,
          "gap_to_one": 0.2128797603765512
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "current_speedup": 1.8821022727272727,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.0243725474166121,
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
        },
        {
          "iteration": 4,
          "speedup": 1.8821022727272727,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "current_speedup": 0.8251401466149203,
      "currently_underperforming": true,
      "gap_to_one": 0.1748598533850797,
      "change_from_previous": 0.8063399776246373,
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
        },
        {
          "iteration": 4,
          "speedup": 0.8251401466149203,
          "gap_to_one": 0.1748598533850797
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "current_speedup": 1.4370904325032765,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.4151378213137513,
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
        },
        {
          "iteration": 4,
          "speedup": 1.4370904325032765,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "current_speedup": 0.9031905594405595,
      "currently_underperforming": true,
      "gap_to_one": 0.09680944055944052,
      "change_from_previous": 0.8839329085664401,
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
        },
        {
          "iteration": 4,
          "speedup": 0.9031905594405595,
          "gap_to_one": 0.09680944055944052
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "current_speedup": 1.2353333333333334,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.2169514909826797,
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
        },
        {
          "iteration": 4,
          "speedup": 1.2353333333333334,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "current_speedup": 2.0426944971537004,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 2.0212532638595384,
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
        },
        {
          "iteration": 4,
          "speedup": 2.0426944971537004,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "current_speedup": 1.3163934426229509,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 1.2978120175688033,
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
        },
        {
          "iteration": 4,
          "speedup": 1.3163934426229509,
          "gap_to_one": 0.0
        }
      ]
    }
  ]
}
程序定义的填写格式（先读后填）：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision_schema.json`；相对工作目录：`knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision_schema.json`；迭代模板：`knowledge/stage9/<iter>/0f505cd0c7c74b8781c319af2b697690/decision_schema.json`；用途：程序定义的 Stage9 JSON 格式（只读）；怎么看：按本轮字段、类型及层级填写；未列出的经验字段必须省略；case_analysis 按本轮要求填写；修改文件由 changes 自动汇总
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision_template.json`；相对工作目录：`knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision_template.json`；迭代模板：`knowledge/stage9/<iter>/0f505cd0c7c74b8781c319af2b697690/decision_template.json`；用途：本请求的决策填写模板（只读）；怎么看：复制 iteration/request_id，填写真实结论、任务和已列出的性能经验；经验 case 按实测增补受益及受损项；空白及 null 必须按 schema 填写
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/case_catalog.json`；相对工作目录：`knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/case_catalog.json`；迭代模板：`knowledge/stage9/<iter>/0f505cd0c7c74b8781c319af2b697690/case_catalog.json`；用途：本轮可核对的完整 case ID 清单（只读）；怎么看：已列出 ID 时按原名选择；没有清单时回查 task，不把最慢六例当成全部用例
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/impl`；相对工作目录：`impl`；用途：本轮建议实际实施的代码基底（只读查阅）；怎么看：在此核对 case 路由与待改函数；changes 仍填写实施时的 impl/... 相对路径。若本轮需回退最佳版本，不按退步代码臆造修改位置
