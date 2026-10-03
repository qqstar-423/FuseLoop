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

  ✅ iter4: speedup 0.8224094689497551 → 1.5389794201417895 (+87.1%)
     改了什么: 回退 MULTI_TOKEN=True 分支的 (TILE,BLOCK_C) 2D tile 实现为 iter2 的 1D 逐 token serial loop，同时保留 w/b 8 段 masked load 循环外提取（每 program 加载 1 次驻留 UB，不再每 token 重载）。具体改动：(1) 删除 2D tile 全部代码（lanes/hw/tmask/tile_mask/2D load/axis=1 归约/2D store），替换为 1D serial loop（token_lo=pid, token_step=tl.num_programs(0)）+ 4 段 1D load + axis=0 归约 + 1D store；(2) 删除 TILE constexpr 参数和 host tile 计算；(3) w/b 8 段 masked load 保持循环前（lines 37-44），store 引用 w0-b3。代码：impl/cann_bench/fused_patch_concat_layernorm.py。选择依据：develop/iter3/融合方案选择决策依据.md。
     框架=Triton；后端=triton-ascend；芯片=Ascend910B3；SoC=Ascend910B3；编程模型=Triton block program (SPMD)
     为什么有效: 两个独立收益来源叠加：(1) 消除 2D tile 灾难——triton-ascend 3.2.1 后端对 2D tensor 操作（2D strided gather、axis=1 归约、2D broadcast mask/store）生成极低效编译代码，导致 iter3 全部 21 个 MULTI_TOKEN case 退步 22-153x（kernel 768-4524us）。回退到 1D 后，同样的数据流用 1D load/store/axis=0 归约实现，后端编译质量正常（21-70us），恢复 iter2 量级。(2) w/b 循环外提取的独立净收益——对比 iter2（w/b 循环内每 token 8 次 HBM masked load），iter4 每 program 仅 8 次，case 12（tokens=2304）从 464 次降为 8 次。正式冷缓存评测证实：大 C + fp32 组收益最显著（case 8 C=384 fp32 speedup 1.73→2.04 +18%，case 17 C=512 fp32 1.66→1.89 +13%，case 14 C=256 fp32 0.99→1.40 +41%），因 per-token w/b 重载的 HBM 延迟在冷缓存下被放大。小 C + cast case 收益被残余 cast/索引开销掩盖。
     适用条件: 适用于 Ascend910B3 + triton-ascend 3.2.1，F2 单 kernel 融合算子的 MULTI_TOKEN serial loop。1D 回退部分适用所有 case；w/b 循环外提取在循环次数多（tokens/40>10）且 C>=256 或 fp32 时收益最大（10-41%）。小 C(80-128) + fp16/bf16 的 w/b 提取净收益被 cast + 索引解码开销掩盖（iter4 这些 case 仍未达标，0.757-0.903）。小 grid 组（MULTI_TOKEN=False, tokens<=64）零影响（编译等价）。2D tile 在当前后端版本确认灾难性低效，未来重试需确认后端修复。未验证范围：w/b 提取在更大 C(>512) 或更长循环下的边际收益趋势。

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
  avg_speedup: 1.5389794201417895 → 1.701061769561243 (+10.5%)
  对比轮次: iter4 → iter5
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/design_rationale.md`；相对工作目录：`develop/iter4/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：上轮评测附近的历史设计参考；怎么看：用于回顾改动意图；实际被评测版本以代码绑定的选择依据和快照为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/design_rationale.md`；相对工作目录：`develop/iter4/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：本轮评测附近的历史设计参考；怎么看：与上轮参考比较；先用代码绑定依据或快照确认版本，再归因成功经验
  逐 case 对比（提升最大的在前）:
    case_level3/fused_patch_concat_layernorm_22: 1.620408163265306 → 2.100529100529101 (+29.6%)
    case_level3/fused_patch_concat_layernorm_3: 0.7871202396234488 → 1.0107142857142857 (+28.4%)
    case_level3/fused_patch_concat_layernorm_24: 1.5542168674698795 → 1.9744897959183674 (+27.0%)
    case_level3/fused_patch_concat_layernorm_19: 0.7566711895070104 → 0.943066516347238 (+24.6%)
    case_level3/fused_patch_concat_layernorm_1: 0.7608125819134993 → 0.918512658227848 (+20.7%)
    case_level3/fused_patch_concat_layernorm_9: 1.3163934426229509 → 1.5844514601420678 (+20.4%)

⚡ 本轮性能大幅提升！请对照上面两轮的 design_rationale，总结这次改动为什么有效，填写 proven_pattern 字段。- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/fusion/fusion_library.json`；相对工作目录：`fusion/fusion_library.json`；用途：JSON 融合算子库；怎么看：比较候选数据流、前提与初始概率，以实测证据决定保留或更换
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter5/perf_result.json`；相对工作目录：`eval/iter5/perf_result.json`；迭代模板：`eval/<iter>/perf_result.json`；用途：本轮性能结果；怎么看：看 avg_speedup、各 case 的 speedup 和最慢用例，不只看平均值
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter5/prof_data`；相对工作目录：`eval/iter5/prof_data`；迭代模板：`eval/<iter>/prof_data`；用途：本轮 profiler 数据（按需回查）；怎么看：仅在结论矛盾或缺少证据时按 case 回查
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter5/perf_reports`；相对工作目录：`eval/iter5/perf_reports`；迭代模板：`eval/<iter>/perf_reports`；用途：本轮性能报告；怎么看：回查原始计时与评分，核实性能汇总及反作弊信号
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/profile/iter5/bottleneck_analysis.md`；相对工作目录：`profile/iter5/bottleneck_analysis.md`；迭代模板：`profile/<iter>/bottleneck_analysis.md`；用途：本轮瓶颈分析；怎么看：优先读 Stage7 结论，区分已证实根因和推测
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter5/FIX_DIRECTIVE.md`；相对工作目录：`search/iter5/FIX_DIRECTIVE.md`；迭代模板：`search/<iter>/FIX_DIRECTIVE.md`；用途：本轮修改指令；怎么看：优先读 Stage8 结论，转成原有 P0/P1/P2 优先级
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/search/iter5/SEARCH_REPORT.md`；相对工作目录：`search/iter5/SEARCH_REPORT.md`；迭代模板：`search/<iter>/SEARCH_REPORT.md`；用途：本轮搜索报告；怎么看：按需检查候选方法、依据及适用条件
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/build/iter5/build.log`；相对工作目录：`build/iter5/build.log`；迭代模板：`build/<iter>/build.log`；用途：本轮编译日志；怎么看：看 STATUS 和首个真实错误，定位构建/接口问题
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter5/precision_result.json`；相对工作目录：`eval/iter5/precision_result.json`；迭代模板：`eval/<iter>/precision_result.json`；用途：本轮精度结果；怎么看：核对总数、通过数及失败 case，优先处理正确性
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter5/precision_reports`；相对工作目录：`eval/iter5/precision_reports`；迭代模板：`eval/<iter>/precision_reports`；用途：本轮精度报告详情；怎么看：按失败 case 查误差、输入和原始报错
当前场景缺失材料（不能当作已完成结论）：
历轮设计思路：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter0/design_rationale.md`；相对工作目录：`develop/iter0/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter0 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter1/design_rationale.md`；相对工作目录：`develop/iter1/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter1 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter2/design_rationale.md`；相对工作目录：`develop/iter2/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter2 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter3/design_rationale.md`；相对工作目录：`develop/iter3/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter3 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/design_rationale.md`；相对工作目录：`develop/iter4/design_rationale.md`；迭代模板：`develop/<iter>/design_rationale.md`；用途：iter4 的实现设计思路；怎么看：按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证
请分析本轮结果，回顾历轮设计思路；只读 /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/history.json，不要覆盖它。
Stage9 输出文件：/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/decision.json
Stage9 请求编号：c8760c429ed246679212092d7367fd85
Stage9 当前轮次：5
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/decision.json`；相对工作目录：`knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/decision.json`；迭代模板：`knowledge/stage9/<iter>/c8760c429ed246679212092d7367fd85/decision.json`；用途：本次 Stage9 决策输出（待你生成）；怎么看：按 role 写 iteration、request_id、ledger_entry 和模型结论；经验使用 proven_pattern/regression_pattern/pitfall 公共字段

本次 ledger_entry.case_analysis 必须逐一覆盖以下 case_id（原样复制，不能缩写；每例写 observation、explanation、evidence、next_action；根因未证实须注明待验证）：
["level3/fused_patch_concat_layernorm_10", "level3/fused_patch_concat_layernorm_12", "level3/fused_patch_concat_layernorm_11", "level3/fused_patch_concat_layernorm_1", "level3/fused_patch_concat_layernorm_19", "level3/fused_patch_concat_layernorm_21"]


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
实现 SHA256：a356695fea4bcab4349d3327534244b47f580a627d77e94e5460256c07d4ede6
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/fusion_library.json`；相对工作目录：`develop/iter4/fusion_library.json`；迭代模板：`develop/<iter>/fusion_library.json`；用途：当前轮实际融合选择与方案库；怎么看：看 selection 和新增方法；原始 Jev probability 只是先验参考
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/融合方案选择决策依据.md`；相对工作目录：`develop/iter4/融合方案选择决策依据.md`；迭代模板：`develop/<iter>/融合方案选择决策依据.md`；用途：当前代码的融合选择决策依据；怎么看：先看选定方案、未选其他方案的原因及目标 case，再与实测核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/self_test_report.md`；相对工作目录：`develop/iter4/self_test_report.md`；迭代模板：`develop/<iter>/self_test_report.md`；用途：当前代码的自测报告；怎么看：看给定 case 与连续调用是否实际执行，区分失败、未执行和通过
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/self_test_result.json`；相对工作目录：`develop/iter4/self_test_result.json`；迭代模板：`develop/<iter>/self_test_result.json`；用途：程序校验使用的自测 JSON；怎么看：看 provided_cases、continuous_calls 的通过状态及 evidence_path
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/self_test.log`；相对工作目录：`develop/iter4/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：给定 case 的原始自测日志；怎么看：核对实际命令、case 数量及结果是否支持自测报告
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/develop/iter4/self_test.log`；相对工作目录：`develop/iter4/self_test.log`；迭代模板：`develop/<iter>/self_test.log`；用途：连续调用的原始自测日志；怎么看：核对相同 shape 更换参数后逐次对照参考实现的结果
最佳实现入库资格：True；Source, fusion decision and self-tests are bound.
本轮库与只读初始 Jev 库区分；真实评测优先于初始概率。
{
  "method_ids": [
    "F2"
  ],
  "implementation_plan": "维持 iter0-iter4 已验证的 F2 垂直融合单 kernel 骨架（Slice×4 strided gather → UB 内 Concat → LayerNorm，fp32 统计 + corr 修正轮）与 tokens>64 → grid=min(tokens,40) 的 MULTI_TOKEN 路由。本轮按 Tech Lead 任务单 T1+T2 实施 MULTI_TOKEN 分支的配对段向量化 + int32 索引优化 + 软件流水：(1) 配对段采用 2 个全宽 affine load（x_ptr+base+offs2 产出 [TL,TR]、x_ptr+base+wc+offs2 产出 [BL,BR]，与 FIX_DIRECTIVE 的 load 形式逐字一致），归约改 tl.sum(seg_at+seg_bb, axis=0)；(2) store 侧因输入行配对（TL-TR 相邻）与输出通道配对（TL-BL 相邻）存在结构性转置，指令字面的 2 store 映射在数学上不成立（会把 TR 写进 out[C:2C] 的 BL 槽位），改为 4 个半掩码 affine store 将各段精确放置到 golden 通道块（out[0:C]=TL、out[C:2C]=BL、out[2C:3C]=TR、out[3C:4C]=BR），w/b 配对参数按半掩码加载（w_tl/w_bl/w_tr/w_br）；(3) 删除 .to(tl.int64)（base/out_base 纯 int32）并将 wc、n=4C、解码分母 hw 外提循环外；(4) T2：token 循环改 tl.range(token_lo, token_hi, token_step, num_stages=2 if MULTI_TOKEN else 1) 软件流水，single 路径保持 num_stages=1；(5) kernel 签名新增 BLOCK_2C、PAIRED_SEG constexpr，PAIRED_SEG=(tokens>64)，else 分支保留原 8 段 w/b 与 4 段循环体（case 22-30 语义等价，循环头 num_stages=1）；(6) 数值路径（fp32 统计、两遍法 + corr 修正、tl.maximum(var,0)、1/sqrt(var+eps)、回写原 dtype）与融合结构不变。",
  "reason": "维持初始库 rank-1 方法 F2（probability 0.79）不变：iter0-iter4 五轮实测证明融合数据流有效且 iter4 为新最佳（avg_speedup 1.539），本轮只优化 MULTI_TOKEN 分支的执行组织，不更换 method、不引入 F5/F8。Tech Lead 任务单的配对方向经 Level1 正式口径复现实验验证有效，但两点修正：(a) 指令字面的 'x_ptr+base+offs2 配对 load + out_base+offs2 配对 store' 组合在本算子上数学不成立（输入行主序下 TL/TR 内存相邻、输出通道序下 TL/BL 相邻，2 affine load + 2 affine store 无法同时满足），A/B 实测证实 piecewise 两段式 load 无法合成单次 DMA（无收益且 p90 尖峰 540-630us）、半掩码 load 配对（变体 B）在 C>=128 case 读放大退步 +8~+25%；(b) 全宽 affine load 配对（变体 E，load 形式与指令逐字一致）+ 半掩码 store 在全部测量 case 无超噪声退步，叠加 tl.range 软件流水（T2）后 9 个目标 case 提升 10-21%。初始 Jev 概率仅作先验参考，实测与 profiler 证据优先。",
  "target_cases": [
    "T1 主目标：9 个未达标 case level3/fused_patch_concat_layernorm_19/10/1/12/3/11/21/4/6（eval/iter4 speedup 0.757-0.903）：配对段 + int32 + 软件流水后 TC6 正式口径实测 kernel 提升 -10.4%~-21.5%（case 3/6 预计跨过 speedup>=1.0，其余大幅收窄差距）",
    "保护目标：已达标 21 case 零回归要求 —— TC6 实测 case 8/13/16/17 -1.2%~-5.1%（改善），case 22（single 路径）+2.2%（3.6us 内核上 0.08us，噪声）",
    "single 路径保护：case 22-30 走 PAIRED_SEG=False 分支（8 段 w/b 与 4 段循环体逐行保留；循环头由 range 改为 tl.range(num_stages=1)，语义等价；TC6 实测 case 22 +2.2% 属噪声，TC4 精度与 iter0-iter4 基线同量级）",
    "精度回归目标：全部 30 case（含 C=3/65 尾块、零方差 25-27、近常数 28-30、grid=1 的 case 22）——已全量通过"
  ],
  "actual_changes": [
    "impl/cann_bench/fused_patch_concat_layernorm.py kernel 签名：新增 BLOCK_2C: tl.constexpr 与 PAIRED_SEG: tl.constexpr（位于 BLOCK_C 与 MULTI_TOKEN 之间）",
    "kernel 循环前：hw=height_half*width_half、wc=width*channels、n=4.0*channels 外提；PAIRED_SEG 分支新增 offs2/mask2/offs2_f、mask_lo/mask_hi 与 8 个半掩码配对 w/b 参数（w_tl/b_tl、w_tr/b_tr、w_bl/b_bl、w_br/b_br），else 分支保留原 8 段 w0-b3",
    "kernel MULTI_TOKEN=True 循环体：PAIRED_SEG 分支 2 次全宽 affine load（x_ptr+base+offs2=[TL,TR]、x_ptr+base+wc+offs2=[BL,BR]）+ tl.sum(seg_at+seg_bb,axis=0) 归约 + 2 处 tl.where(offs2_f<2C,...) + corr/sq_sum 同构 + 4 次半掩码 affine store（golden 通道序）；else 分支保留原 4 段代码",
    "kernel 循环体标量化：删除 base 与 out_base 的 .to(tl.int64)（纯 int32 运算，全部 case 张量 <=1.18M 元素 << 2^31）",
    "kernel 循环头：for token in tl.range(token_lo, token_hi, token_step, num_stages=2 if MULTI_TOKEN else 1)（T2 软件流水，single 路径 num_stages=1）",
    "host 函数：kernel 调用新增 BLOCK_2C=triton.next_power_of_2(2*channels) 与 PAIRED_SEG=(tokens>64)；grid 与 MULTI_TOKEN=(tokens>64) 路由、输入校验、contiguous 守护、torch.empty、tokens==0 早退、函数签名与导出均未改动"
  ],
  "expected_benefits": [
    "9 个未达标 case（MULTI_TOKEN + 小 C 80-128 或 tokens=2304）kernel 耗时显著下降：TC6 正式口径（Level1+CacheClean，25 轮中位数）实测 vs iter4 最佳快照：case 1 -21.5%、case 3 -21.3%、case 19 -17.8%、case 21 -17.5%、case 4 -16.3%、case 6 -15.3%、case 10 -11.0%、case 12 -10.9%、case 11 -10.4%；按 eval/iter4 speedup 折算，case 3 约 1.00、case 6 约 1.07、case 4 约 0.99、case 21 约 0.98、case 1 约 0.97 跨过或逼近达标线，case 19/10/12/11 差距从 -24~-32% 收窄到 -8~-15%（正式幅度待 cann-bench 评测确认）",
    "已达标 case 无回归：case 8 -3.1%、case 13 -1.3%、case 16 -1.2%、case 17 -5.1%（全部改善或噪声内）；single 路径 case 22 +2.2%（3.68us vs 3.60us，绝对差 0.08us 属噪声），case 23-30 编译路径与 iter4 等价（PAIRED_SEG=False + num_stages=1）",
    "per-token 指令与事务数下降：x load 每 token 4 条 C 宽指令 → 2 条 2C 宽指令；store 4 条 C 宽 → 4 条半掩码 2C 宽（读侧指令减半、无读放大）；base/out_base int64→int32；wc/n/hw 每 token 计算外提；58 次迭代 case（10/11/12）的 load 延迟经 num_stages=2 与计算重叠",
    "中间 concat 张量仍不在 HBM 物化（融合程度不变）；HBM 读写量不变（每 token 读 4C 元素写 4C 元素），数值路径 fp32 统计 + corr 修正不变（精度 30/30 与 iter0-iter4 基线同量级）"
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

# 融合方案选择决策依据 — fused_patch_concat_layernorm（develop/iter4）

> 本轮任务类型 perf_optimize，输入为 eval/iter4 结果（avg_speedup=1.539，达标 21/30，9 个未达标 case speedup 0.757-0.903）。Tech Lead 任务单：T1（P1）MULTI_TOKEN 分支配对段向量化 + int32 索引优化；T2（P2）tl.range 软件流水独立 A/B。本文只陈述方案选择，与自测报告（`self_test_report.md`）分开。

## 一、本轮实际选择

- **选中方法**：`F2 Vertical Fusion（垂直融合／单 kernel 全融合）`，初始 Jev probability=0.79（初始库 rank-1），无新增方法、无新概率。
- **实现形态**：F2 单 kernel 骨架不变，MULTI_TOKEN 分支执行组织升级为：
  1. **配对段 load（与 FIX_DIRECTIVE load 形式逐字一致）**：每 token 2 个全宽 affine load——`x_ptr+base+offs2` 产出 `[TL,TR]`、`x_ptr+base+wc+offs2` 产出 `[BL,BR]`（输入行主序下同行相邻列内存连续），归约 `tl.sum(seg_at+seg_bb, axis=0)`；
  2. **半掩码 affine store 修正 lane 映射**：4 个 store 将各段精确写入 golden 通道块 `out[0:C]=TL、out[C:2C]=BL、out[2C:3C]=TR、out[3C:4C]=BR`，w/b 配对参数按半掩码加载（`w_tl=weight[offs2](lo)`、`w_tr=weight[C+offs2](hi)`、`w_bl=weight[C+offs2](lo)`、`w_br=weight[2C+offs2](hi)`）；
  3. **int32 + 不变量外提**：删除 `base`/`out_base` 的 `.to(tl.int64)`，`wc`、`n=4C`、解码分母 `hw` 外提循环外；
  4. **T2 软件流水**：`for token in tl.range(..., num_stages=2 if MULTI_TOKEN else 1)`；
  5. **签名**：新增 `BLOCK_2C`、`PAIRED_SEG` constexpr，`PAIRED_SEG=(tokens>64)`；else 分支保留原 8 段 w/b + 4 段循环体（case 22-30 语义等价）。

## 二、为什么不能按指令字面实现 2 load + 2 store（结构性转置，数学证明）

FIX_DIRECTIVE/decision.json 的字面组合是「2 次配对 load（`x_ptr+base+offs2`、`x_ptr+base+wc+offs2`）+ 2 次配对 store（`y_ptr+out_base+offs2` 配 `w_a=weight[0:2C]`、`y_ptr+out_base+2C+offs2` 配 `w_b=weight[2C:4C]`）」。在本算子的数据布局下该组合**数学上不成立**：

- 输入 `x[B,H*W,C]` 行主序连续：`TL=x[base+c]`、`TR=x[base+C+c]`（同行相邻列，**内存连续**）、`BL=x[base+wc+c]`、`BR=x[base+wc+C+c]`。故 `x[base..base+2C) = [TL,TR]`、`x[base+wc..base+wc+2C) = [BL,BR]`。
- 输出通道序由 golden `torch.cat((TL, BL, TR, BR), dim=-1)` 固定为 `[TL,BL,TR,BR]`：`out[0:C]=TL`、`out[C:2C]=BL`、`out[2C:3C]=TR`、`out[3C:4C]=BR`。
- 因此按字面把 load1=[TL,TR] store 到 `out[0:2C]` 会把 **TR 写进 out[C:2C] 的 BL 槽位**（load2 同理把 BL 写进 TR 槽位）——输出变成 `[TL,TR,BL,BR]`，中段两段置换，精度必挂。
- 「TL/TR 在输入输出中均连续」的验证前提对输出不成立（输出中 TL 与 TR 相隔 2C）。2 个 affine load + 2 个 affine store 同时满足两侧配对，在本布局下不可能（输入按行配对、输出按列配对，属结构性转置）。

本轮以 A/B 实验在三种正确实现间选择（全部 1D、无 2D 算子，遵守 iter3 失败教训）：

| 变体 | load 侧 | store 侧 | 正式口径 A/B 结论 | 判定 |
|---|---|---|---|---|
| A | 2 次 piecewise 两段式地址 load（lane_jump） | 2 全宽 store（=指令 store 结构） | 中位数无收益（-0.8%~+7.8% 噪声）且 p90 尖峰 540-630us（tokens=2304 case）——两段式地址无法合成单次 GM→UB DMA，后端仍拆 2 事务并增加地址运算 | 淘汰 |
| B | 4 次半掩码 affine load 合成 [TL,BL]/[TR,BR] | 2 全宽 store（=指令 store 结构） | 小 C 组 -14.6~-20.3%，但 C≥128 组 **+7.9~+24.9% 退步**（case 13 +24.9% 会使已达标 case 落榜）——半掩码 load 在该后端存在读放大 | 淘汰 |
| E | **2 次全宽 affine load（=指令 load 形式逐字一致）** | 4 次半掩码 affine store（修正 lane 映射） | 全部测量 case 无超噪声退步（最差 case 17 +7.3%，仍 1.79x），小 C 组 -5.9~-22.6% | 采纳为 T1 基础 |
| H=E+T2 | 同 E | 同 E | 叠加 `tl.range(num_stages=2)`：tokens=2304 组再降 -2.4~-5.0%（case 19 从 -5.9% 扩大到 -17.6%），12/14 测量 case 改善 | **采纳（最终交付）** |

A/B 口径说明：本地用正式评测同款 `cann_bench_utils.cann_bench_cache_clean`（每测量步清 L2）+ `ProfilerLevel.Level1` + 25-30 轮中位数复现正式口径，并验证其绝对值与 eval/iter4 正式 kernel 耗时一致（cur: case 19=42.9/43.1 vs 正式 44.22；case 10=72.3 vs 69.30；case 12=71.0/74.9 vs 70.04；case 11=66.7/70.3 vs 67.56；case 22=3.69/3.52 vs 4.9），故变体间相对结论可直接外推到正式评测。变体 A/B/E/G/H 全部先过精度哨兵（case 1/3/19/16/22/25/28/30 vs golden + vs iter4 实现）再测性能。

## 三、为什么不更换融合方法（保留 F2 的理由）

1. **融合数据流五轮实测有效**：iter0-iter4 中小 grid 组 speedup 1.55-3.71、大 C 组 1.68-2.04；iter4 为历史最佳（1.539）。本轮 9 个未达标 case 的瓶颈是 MULTI_TOKEN 串行循环的 per-token 固定开销，属执行组织问题而非融合结构问题。
2. **Tech Lead 任务单明确**：T1 配对段向量化 + int32 优化、T2 软件流水独立 A/B，均不更换 method、不引入 F5（0.58）/F8（0.44）等新方案变量。
3. **2D tile 方向维持禁用**：iter3 实测 2D tensor 操作退步 22-153x（regression_patterns 已收录）。本轮全部候选均为 1D affine/掩码操作，配对收益不依赖 2D。
4. **tokens>64 条件路由保留**：iter1-iter4 验证的 `MULTI_TOKEN=(tokens>64)` 路由与 `grid=min(tokens,40)` 原样保留；PAIRED_SEG 与 MULTI_TOKEN 同条件（`tokens>64`），single 路径（case 22-30）走 PAIRED_SEG=False 分支（8 段 w/b 与 4 段循环体逐行保留，循环头 num_stages=1 语义等价），TC6 实测 case 22 +2.2%（0.08us）为噪声、TC4 精度与基线同量级。

## 四、参考的概率与已测证据

- 初始概率仅作先验：F2=0.79 与 iter0 首版以来五轮选择一致；实际取舍完全以上述正式口径 A/B 为准。
- 已测证据链：
  - 变体筛选：`/tmp/opencode/fpcl_iter5_l1.json`（正式口径交错 A/B，B/E/G/H × 14 case）、`/tmp/opencode/fpcl_iter5_inter.json`（非 Level1 交错 A/B）、`/tmp/opencode/fpcl_iter5_ab.json`（热缓存 A/B，变体 A p90 证据）；
  - 最终代码 vs iter4 最佳快照（`selection/records/iter4-60dea9e34578bd49ed55/impl`，同一份被评代码）：`develop/iter4/self_test.log` TC6 节（Level1+CacheClean，25 轮中位数，14 case）；
  - 精度：`develop/iter4/self_test.log` TC4（30/30）+ TC5（18/18）+ TC3（chrome trace 12 事件证实真实 NPU kernel）。

## 五、针对哪些 case、预期影响（正式口径 TC6 实测 → 折算）

| case 组 | TC6 实测（cur vs iter4 快照） | 折算 speedup（eval/iter4 ÷ 实测降幅因子） |
|---|---|---|
| case 1（C=96 fp16） | **-21.5%**（42.08→33.04us） | 0.761 → ~0.97 |
| case 3（C=96 bf16） | **-21.3%**（42.08→33.10us） | 0.787 → ~1.00 |
| case 19（C=80 fp16，worst_6 之首） | **-17.8%**（40.78→33.52us） | 0.757 → ~0.92 |
| case 21（C=80 bf16） | **-17.5%**（40.74→33.60us） | 0.811 → ~0.98 |
| case 4（C=192 fp16） | **-16.3%**（41.62→34.84us） | 0.825 → ~0.99 |
| case 6（C=192 bf16） | **-15.3%**（42.50→35.98us） | 0.903 → ~1.07 ✓ |
| case 10（C=128 fp16，58 迭代） | **-11.0%**（63.64→56.64us） | 0.759 → ~0.85 |
| case 12（C=128 bf16，58 迭代） | **-10.9%**（63.80→56.84us） | 0.764 → ~0.86 |
| case 11（C=128 fp32，58 迭代） | **-10.4%**（63.08→56.54us） | 0.805 → ~0.90 |
| case 8/13/16/17（已达标保护组） | -3.1%/-1.3%/-1.2%/-5.1%（改善） | 2.04/1.25/1.68/1.89 → 不降反升 |
| case 22（single 路径对照） | +2.2%（3.68 vs 3.60us，0.08us 噪声） | 1.62 → ~1.59（噪声内） |

## 六、预期收益与已测收益的区分

| 项 | 已测（本轮正式口径自测） | 预期（待 cann-bench 正式评测） |
|---|---|---|
| 精度 | TC4 30/30 PASS + TC5 18/18 连续调用 PASS + TC3 真实 NPU kernel trace | 正式精度评测（预期通过，数值路径未变） |
| 9 个目标 case kernel 耗时 | TC6 实测 -10.4%~-21.5%（vs iter4 快照） | speedup 按 1/0.79-1/0.90 因子折算提升，case 3/6 预计跨过 1.0，其余收窄但多数可能仍 <1.0 |
| 已达标 21 case | TC6 实测 8/13/16/17 改善 -1.2~-5.1%，case 22 噪声内 | 零回归（预期） |
| avg_speedup | 未评测 | 从 1.539 提升（量级待评测确认；9 个小 C case 的残余瓶颈：BLOCK_C 利用率与 cast 开销本轮未动） |

> ⚠️ 指令预期的「-45% 循环体指令数 → 9 case 全部达标」未完全兑现：实测最优组合（E+T2）在正式口径下提升 10-21%，不足以让全部 9 个 case 跨过 1.0。原因：(a) piecewise 2-load 与半掩码 load 两条「指令数减半」路径分别被后端 DMA 拆分与读放大抵消（变体 A/B 实测证据）；(b) where/reduce 向量链与 4 次整数除法/取模解码仍在循环体内。剩余方向（BLOCK_C 利用率、cast 消除、索引解码位运算化）记录给下一轮，本轮不越界实施。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/state.json`；相对工作目录：`selection/state.json`；用途：有效评测索引与语义窗口配置；怎么看：以下 window 和 case_trends 由程序按同口径历史快照计算，不将失败轮计入窗口
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/impl`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/impl`；迭代模板：`selection/records/<iter>-<指纹>/impl`；用途：最佳已验证实现的独立代码快照目录；怎么看：读取该目录的实现，勿把当前工作代码当作历史最佳版本
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/manifest.json`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/manifest.json`；迭代模板：`selection/records/<iter>-<指纹>/manifest.json`；用途：最佳实现快照清单；怎么看：核对 iteration、融合方案、avg_speedup、hap 及代码和评测证据路径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/reports/performance_source.json`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/reports/performance_source.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/performance_source.json`；用途：最佳实现对应的原始性能报告；怎么看：按 case 看 speedup、耗时和 HAP，确认使用同一评测口径
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/reports/perf_result.json`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/reports/perf_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/perf_result.json`；用途：最佳实现的结构化性能结果；怎么看：看 avg_speedup、cases 和最慢用例，配合窗口和趋势判断
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/reports/precision_result.json`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/reports/precision_result.json`；迭代模板：`selection/records/<iter>-<指纹>/reports/precision_result.json`；用途：最佳实现对应的精度结果；怎么看：核对 precision_overall、通过数量和原始精度报告定位
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/evidence/0_decision_rationale.md`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/evidence/0_decision_rationale.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/0_decision_rationale.md`；用途：最佳实现的融合选择依据快照；怎么看：看当时的选择理由，与当前方案区别及实测结果核对
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/evidence/1_fusion_library.json`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/evidence/1_fusion_library.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/1_fusion_library.json`；用途：最佳实现的当轮融合方案库快照；怎么看：看 selection 中实际方法和实现方案，不把初始概率当性能
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/evidence/2_self_test_report.md`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/evidence/2_self_test_report.md`；迭代模板：`selection/records/<iter>-<指纹>/evidence/2_self_test_report.md`；用途：最佳实现的自测报告快照；怎么看：核对给定 case 及连续调用的执行范围与结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/evidence/3_self_test_result.json`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/evidence/3_self_test_result.json`；迭代模板：`selection/records/<iter>-<指纹>/evidence/3_self_test_result.json`；用途：最佳实现的自测资格 JSON 快照；怎么看：核对两类自测状态；其中历史日志路径以本清单中的归档日志为准
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/evidence/4_selftest_log_continuous.log`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/evidence/4_selftest_log_continuous.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/4_selftest_log_continuous.log`；用途：最佳实现的连续调用原始日志快照；怎么看：核对同 shape 更换参数后逐次比较参考实现的结果
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/evidence/5_selftest_log_provided.log`；相对工作目录：`selection/records/iter5-09860e675aa178f9e0bf/evidence/5_selftest_log_provided.log`；迭代模板：`selection/records/<iter>-<指纹>/evidence/5_selftest_log_provided.log`；用途：最佳实现的给定 case 原始自测日志快照；怎么看：核对原始执行输出是否支持报告结论
{
  "eligible": true,
  "reason": "Valid measured implementation recorded",
  "should_exit": false,
  "review_fusion": false,
  "best": {
    "iteration": 5,
    "selection_label": "best_available",
    "all_cases_pass": false,
    "avg_speedup": 1.701061769561243,
    "hap": {
      "performance_score": 30.830526225930587,
      "cases": [
        {
          "case_id": "level3/fused_patch_concat_layernorm_10",
          "perf_score": 0.4466239155035836,
          "t_hw_us": 5.26,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 63.94
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_12",
          "perf_score": 0.45179140874132434,
          "t_hw_us": 5.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 63.8
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_11",
          "perf_score": 0.4676094018727308,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 61.16
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_1",
          "perf_score": 0.47651618787049693,
          "t_hw_us": 3.48,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 37.92
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_19",
          "perf_score": 0.4837724935732648,
          "t_hw_us": 3.35,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 35.48
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_21",
          "perf_score": 0.49229791964427505,
          "t_hw_us": 3.45,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 35.42
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_4",
          "perf_score": 0.49235167977126515,
          "t_hw_us": 3.83,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 39.34
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_3",
          "perf_score": 0.5029621753000152,
          "t_hw_us": 3.68,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 36.4
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_6",
          "perf_score": 0.5131742309284039,
          "t_hw_us": 4.13,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 39.42
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_15",
          "perf_score": 0.5196051919956733,
          "t_hw_us": 4.27,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 39.8
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_20",
          "perf_score": 0.5325157933853586,
          "t_hw_us": 3.18,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 28.34
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_2",
          "perf_score": 0.5451661067562523,
          "t_hw_us": 3.25,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 27.62
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_13",
          "perf_score": 0.5665541807274092,
          "t_hw_us": 5.04,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 39.72
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_14",
          "perf_score": 0.5923011741919865,
          "t_hw_us": 5.44,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 39.12
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_5",
          "perf_score": 0.6011270179713677,
          "t_hw_us": 4.39,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 30.58
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_7",
          "perf_score": 0.6081327498176513,
          "t_hw_us": 3.71,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 25.2
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_9",
          "perf_score": 0.6288498346963634,
          "t_hw_us": 4.01,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 25.34
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_16",
          "perf_score": 0.6512068149550403,
          "t_hw_us": 4.59,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 26.7
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_18",
          "perf_score": 0.6628799508750384,
          "t_hw_us": 4.8,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 26.76
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
          "case_id": "level3/fused_patch_concat_layernorm_24",
          "perf_score": 0.6977225672877847,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.92
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_17",
          "perf_score": 0.6909585727245238,
          "t_hw_us": 5.08,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 25.52
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
          "case_id": "level3/fused_patch_concat_layernorm_8",
          "perf_score": 0.7102272727272727,
          "t_hw_us": 4.31,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 20.12
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_29",
          "perf_score": 0.7360515021459227,
          "t_hw_us": 2.67,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 11.28
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_28",
          "perf_score": 0.7399456521739131,
          "t_hw_us": 3.03,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 12.6
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_23",
          "perf_score": 0.7843551797040168,
          "t_hw_us": 1.0,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.04
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_26",
          "perf_score": 0.8262653898768809,
          "t_hw_us": 1.34,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 3.88
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_27",
          "perf_score": 0.8401937046004843,
          "t_hw_us": 1.54,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.18
            }
          }
        },
        {
          "case_id": "level3/fused_patch_concat_layernorm_25",
          "perf_score": 0.847145488029466,
          "t_hw_us": 1.53,
          "op_times": {
            "device_kernels": {
              "_fused_patch_concat_layernorm_kernel": 4.02
            }
          }
        }
      ]
    },
    "implementation_dir": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/impl",
    "manifest_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/manifest.json",
    "performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/reports/performance_source.json",
    "precision_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter5-09860e675aa178f9e0bf/reports/precision_result.json",
    "fusion_scheme": {
      "method_ids": [
        "F2"
      ],
      "implementation_plan": "维持 iter0-iter4 已验证的 F2 垂直融合单 kernel 骨架（Slice×4 strided gather → UB 内 Concat → LayerNorm，fp32 统计 + corr 修正轮）与 tokens>64 → grid=min(tokens,40) 的 MULTI_TOKEN 路由。本轮按 Tech Lead 任务单 T1+T2 实施 MULTI_TOKEN 分支的配对段向量化 + int32 索引优化 + 软件流水：(1) 配对段采用 2 个全宽 affine load（x_ptr+base+offs2 产出 [TL,TR]、x_ptr+base+wc+offs2 产出 [BL,BR]，与 FIX_DIRECTIVE 的 load 形式逐字一致），归约改 tl.sum(seg_at+seg_bb, axis=0)；(2) store 侧因输入行配对（TL-TR 相邻）与输出通道配对（TL-BL 相邻）存在结构性转置，指令字面的 2 store 映射在数学上不成立（会把 TR 写进 out[C:2C] 的 BL 槽位），改为 4 个半掩码 affine store 将各段精确放置到 golden 通道块（out[0:C]=TL、out[C:2C]=BL、out[2C:3C]=TR、out[3C:4C]=BR），w/b 配对参数按半掩码加载（w_tl/w_bl/w_tr/w_br）；(3) 删除 .to(tl.int64)（base/out_base 纯 int32）并将 wc、n=4C、解码分母 hw 外提循环外；(4) T2：token 循环改 tl.range(token_lo, token_hi, token_step, num_stages=2 if MULTI_TOKEN else 1) 软件流水，single 路径保持 num_stages=1；(5) kernel 签名新增 BLOCK_2C、PAIRED_SEG constexpr，PAIRED_SEG=(tokens>64)，else 分支保留原 8 段 w/b 与 4 段循环体（case 22-30 语义等价，循环头 num_stages=1）；(6) 数值路径（fp32 统计、两遍法 + corr 修正、tl.maximum(var,0)、1/sqrt(var+eps)、回写原 dtype）与融合结构不变。",
      "reason": "维持初始库 rank-1 方法 F2（probability 0.79）不变：iter0-iter4 五轮实测证明融合数据流有效且 iter4 为新最佳（avg_speedup 1.539），本轮只优化 MULTI_TOKEN 分支的执行组织，不更换 method、不引入 F5/F8。Tech Lead 任务单的配对方向经 Level1 正式口径复现实验验证有效，但两点修正：(a) 指令字面的 'x_ptr+base+offs2 配对 load + out_base+offs2 配对 store' 组合在本算子上数学不成立（输入行主序下 TL/TR 内存相邻、输出通道序下 TL/BL 相邻，2 affine load + 2 affine store 无法同时满足），A/B 实测证实 piecewise 两段式 load 无法合成单次 DMA（无收益且 p90 尖峰 540-630us）、半掩码 load 配对（变体 B）在 C>=128 case 读放大退步 +8~+25%；(b) 全宽 affine load 配对（变体 E，load 形式与指令逐字一致）+ 半掩码 store 在全部测量 case 无超噪声退步，叠加 tl.range 软件流水（T2）后 9 个目标 case 提升 10-21%。初始 Jev 概率仅作先验参考，实测与 profiler 证据优先。",
      "target_cases": [
        "T1 主目标：9 个未达标 case level3/fused_patch_concat_layernorm_19/10/1/12/3/11/21/4/6（eval/iter4 speedup 0.757-0.903）：配对段 + int32 + 软件流水后 TC6 正式口径实测 kernel 提升 -10.4%~-21.5%（case 3/6 预计跨过 speedup>=1.0，其余大幅收窄差距）",
        "保护目标：已达标 21 case 零回归要求 —— TC6 实测 case 8/13/16/17 -1.2%~-5.1%（改善），case 22（single 路径）+2.2%（3.6us 内核上 0.08us，噪声）",
        "single 路径保护：case 22-30 走 PAIRED_SEG=False 分支（8 段 w/b 与 4 段循环体逐行保留；循环头由 range 改为 tl.range(num_stages=1)，语义等价；TC6 实测 case 22 +2.2% 属噪声，TC4 精度与 iter0-iter4 基线同量级）",
        "精度回归目标：全部 30 case（含 C=3/65 尾块、零方差 25-27、近常数 28-30、grid=1 的 case 22）——已全量通过"
      ],
      "actual_changes": [
        "impl/cann_bench/fused_patch_concat_layernorm.py kernel 签名：新增 BLOCK_2C: tl.constexpr 与 PAIRED_SEG: tl.constexpr（位于 BLOCK_C 与 MULTI_TOKEN 之间）",
        "kernel 循环前：hw=height_half*width_half、wc=width*channels、n=4.0*channels 外提；PAIRED_SEG 分支新增 offs2/mask2/offs2_f、mask_lo/mask_hi 与 8 个半掩码配对 w/b 参数（w_tl/b_tl、w_tr/b_tr、w_bl/b_bl、w_br/b_br），else 分支保留原 8 段 w0-b3",
        "kernel MULTI_TOKEN=True 循环体：PAIRED_SEG 分支 2 次全宽 affine load（x_ptr+base+offs2=[TL,TR]、x_ptr+base+wc+offs2=[BL,BR]）+ tl.sum(seg_at+seg_bb,axis=0) 归约 + 2 处 tl.where(offs2_f<2C,...) + corr/sq_sum 同构 + 4 次半掩码 affine store（golden 通道序）；else 分支保留原 4 段代码",
        "kernel 循环体标量化：删除 base 与 out_base 的 .to(tl.int64)（纯 int32 运算，全部 case 张量 <=1.18M 元素 << 2^31）",
        "kernel 循环头：for token in tl.range(token_lo, token_hi, token_step, num_stages=2 if MULTI_TOKEN else 1)（T2 软件流水，single 路径 num_stages=1）",
        "host 函数：kernel 调用新增 BLOCK_2C=triton.next_power_of_2(2*channels) 与 PAIRED_SEG=(tokens>64)；grid 与 MULTI_TOKEN=(tokens>64) 路由、输入校验、contiguous 守护、torch.empty、tokens==0 早退、函数签名与导出均未改动"
      ],
      "expected_benefits": [
        "9 个未达标 case（MULTI_TOKEN + 小 C 80-128 或 tokens=2304）kernel 耗时显著下降：TC6 正式口径（Level1+CacheClean，25 轮中位数）实测 vs iter4 最佳快照：case 1 -21.5%、case 3 -21.3%、case 19 -17.8%、case 21 -17.5%、case 4 -16.3%、case 6 -15.3%、case 10 -11.0%、case 12 -10.9%、case 11 -10.4%；按 eval/iter4 speedup 折算，case 3 约 1.00、case 6 约 1.07、case 4 约 0.99、case 21 约 0.98、case 1 约 0.97 跨过或逼近达标线，case 19/10/12/11 差距从 -24~-32% 收窄到 -8~-15%（正式幅度待 cann-bench 评测确认）",
        "已达标 case 无回归：case 8 -3.1%、case 13 -1.3%、case 16 -1.2%、case 17 -5.1%（全部改善或噪声内）；single 路径 case 22 +2.2%（3.68us vs 3.60us，绝对差 0.08us 属噪声），case 23-30 编译路径与 iter4 等价（PAIRED_SEG=False + num_stages=1）",
        "per-token 指令与事务数下降：x load 每 token 4 条 C 宽指令 → 2 条 2C 宽指令；store 4 条 C 宽 → 4 条半掩码 2C 宽（读侧指令减半、无读放大）；base/out_base int64→int32；wc/n/hw 每 token 计算外提；58 次迭代 case（10/11/12）的 load 延迟经 num_stages=2 与计算重叠",
        "中间 concat 张量仍不在 HBM 物化（融合程度不变）；HBM 读写量不变（每 token 读 4C 元素写 4C 元素），数值路径 fp32 统计 + corr 修正不变（精度 30/30 与 iter0-iter4 基线同量级）"
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
    "valid_samples": 5,
    "completed_improvements": 3,
    "enough_samples": true,
    "start_iteration": 2,
    "end_iteration": 5,
    "start_best_avg_speedup": 1.3213246705259825,
    "end_best_avg_speedup": 1.701061769561243,
    "cumulative_improvement": 0.2873912123990675,
    "threshold": 0.05,
    "stagnated": false
  },
  "case_trends": [
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "current_speedup": 0.918512658227848,
      "currently_underperforming": true,
      "gap_to_one": 0.081487341772152,
      "change_from_previous": 0.15770007631434868,
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
        },
        {
          "iteration": 5,
          "speedup": 0.918512658227848,
          "gap_to_one": 0.081487341772152
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "current_speedup": 0.8229590240850797,
      "currently_underperforming": true,
      "gap_to_one": 0.1770409759149203,
      "change_from_previous": 0.06365166477772044,
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
        },
        {
          "iteration": 5,
          "speedup": 0.8229590240850797,
          "gap_to_one": 0.1770409759149203
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "current_speedup": 0.8891432308698497,
      "currently_underperforming": true,
      "gap_to_one": 0.1108567691301503,
      "change_from_previous": 0.08422908048500655,
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
        },
        {
          "iteration": 5,
          "speedup": 0.8891432308698497,
          "gap_to_one": 0.1108567691301503
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "current_speedup": 0.8388714733542321,
      "currently_underperforming": true,
      "gap_to_one": 0.16112852664576793,
      "change_from_previous": 0.07473669322858956,
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
        },
        {
          "iteration": 5,
          "speedup": 0.8388714733542321,
          "gap_to_one": 0.16112852664576793
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "current_speedup": 1.2681268882175227,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.020109048772527727,
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
        },
        {
          "iteration": 5,
          "speedup": 1.2681268882175227,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "current_speedup": 1.3898261758691206,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.009298888464384314,
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
        },
        {
          "iteration": 5,
          "speedup": 1.3898261758691206,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "current_speedup": 1.0728643216080402,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.026808466634987704,
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
        },
        {
          "iteration": 5,
          "speedup": 1.0728643216080402,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "current_speedup": 1.7179775280898877,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.03652591518666215,
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
        },
        {
          "iteration": 5,
          "speedup": 1.7179775280898877,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "current_speedup": 1.9898119122257054,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.1048824393229586,
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
        },
        {
          "iteration": 5,
          "speedup": 1.9898119122257054,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "current_speedup": 1.7929745889387143,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.01987776706140476,
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
        },
        {
          "iteration": 5,
          "speedup": 1.7929745889387143,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "current_speedup": 0.943066516347238,
      "currently_underperforming": true,
      "gap_to_one": 0.05693348365276196,
      "change_from_previous": 0.18639532684022764,
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
        },
        {
          "iteration": 5,
          "speedup": 0.943066516347238,
          "gap_to_one": 0.05693348365276196
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "current_speedup": 1.1752353367125272,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.1335151826688814,
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
        },
        {
          "iteration": 5,
          "speedup": 1.1752353367125272,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "current_speedup": 1.123500352858151,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.08569070487901165,
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
        },
        {
          "iteration": 5,
          "speedup": 1.123500352858151,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "current_speedup": 0.9726143421795596,
      "currently_underperforming": true,
      "gap_to_one": 0.027385657820440445,
      "change_from_previous": 0.16164447400630144,
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
        },
        {
          "iteration": 5,
          "speedup": 0.9726143421795596,
          "gap_to_one": 0.027385657820440445
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "current_speedup": 2.100529100529101,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.4801209372637949,
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
        },
        {
          "iteration": 5,
          "speedup": 2.100529100529101,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "current_speedup": 2.7697368421052633,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.28098779557589637,
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
        },
        {
          "iteration": 5,
          "speedup": 2.7697368421052633,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "current_speedup": 1.9744897959183674,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.42027292844848785,
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
        },
        {
          "iteration": 5,
          "speedup": 1.9744897959183674,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "current_speedup": 3.8134328358208958,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.46627126376849404,
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
        },
        {
          "iteration": 5,
          "speedup": 3.8134328358208958,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "current_speedup": 3.4587628865979383,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.4497942767324674,
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
        },
        {
          "iteration": 5,
          "speedup": 3.4587628865979383,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "current_speedup": 3.688995215311005,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.6172820679006459,
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
        },
        {
          "iteration": 5,
          "speedup": 3.688995215311005,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "current_speedup": 2.401587301587302,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.3031129603945142,
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
        },
        {
          "iteration": 5,
          "speedup": 2.401587301587302,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "current_speedup": 2.3652482269503547,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.14561428352273387,
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
        },
        {
          "iteration": 5,
          "speedup": 2.3652482269503547,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "current_speedup": 1.0107142857142857,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.22359404609083688,
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
        },
        {
          "iteration": 5,
          "speedup": 1.0107142857142857,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "current_speedup": 1.8821022727272727,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.0,
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
        },
        {
          "iteration": 5,
          "speedup": 1.8821022727272727,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "current_speedup": 0.972801220132181,
      "currently_underperforming": true,
      "gap_to_one": 0.027198779867819045,
      "change_from_previous": 0.14766107351726065,
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
        },
        {
          "iteration": 5,
          "speedup": 0.972801220132181,
          "gap_to_one": 0.027198779867819045
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "current_speedup": 1.4342707652060172,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": -0.002819667297259354,
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
        },
        {
          "iteration": 5,
          "speedup": 1.4342707652060172,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "current_speedup": 1.0484525621511922,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.14526200271063272,
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
        },
        {
          "iteration": 5,
          "speedup": 1.0484525621511922,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "current_speedup": 1.4706349206349207,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.23530158730158734,
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
        },
        {
          "iteration": 5,
          "speedup": 1.4706349206349207,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "current_speedup": 2.140159045725646,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.09746454857194564,
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
        },
        {
          "iteration": 5,
          "speedup": 2.140159045725646,
          "gap_to_one": 0.0
        }
      ]
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "current_speedup": 1.5844514601420678,
      "currently_underperforming": false,
      "gap_to_one": 0.0,
      "change_from_previous": 0.26805801751911695,
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
        },
        {
          "iteration": 5,
          "speedup": 1.5844514601420678,
          "gap_to_one": 0.0
        }
      ]
    }
  ]
}
程序定义的填写格式（先读后填）：
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/decision_schema.json`；相对工作目录：`knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/decision_schema.json`；迭代模板：`knowledge/stage9/<iter>/c8760c429ed246679212092d7367fd85/decision_schema.json`；用途：程序定义的 Stage9 JSON 格式（只读）；怎么看：按本轮字段、类型及层级填写；未列出的经验字段必须省略；case_analysis 按本轮要求填写；修改文件由 changes 自动汇总
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/decision_template.json`；相对工作目录：`knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/decision_template.json`；迭代模板：`knowledge/stage9/<iter>/c8760c429ed246679212092d7367fd85/decision_template.json`；用途：本请求的决策填写模板（只读）；怎么看：复制 iteration/request_id，填写真实结论、任务和已列出的性能经验；经验 case 按实测增补受益及受损项；空白及 null 必须按 schema 填写
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/case_catalog.json`；相对工作目录：`knowledge/stage9/iter5/c8760c429ed246679212092d7367fd85/case_catalog.json`；迭代模板：`knowledge/stage9/<iter>/c8760c429ed246679212092d7367fd85/case_catalog.json`；用途：本轮可核对的完整 case ID 清单（只读）；怎么看：已列出 ID 时按原名选择；没有清单时回查 task，不把最慢六例当成全部用例
- 文件：`/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/impl`；相对工作目录：`impl`；用途：本轮建议实际实施的代码基底（只读查阅）；怎么看：在此核对 case 路由与待改函数；changes 仍填写实施时的 impl/... 相对路径。若本轮需回退最佳版本，不按退步代码臆造修改位置
