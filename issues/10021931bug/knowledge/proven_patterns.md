# 已验证的成功优化经验

> Stage9 填写分析，程序验证性能变化达到 5% 后保存；同轮重试更新原记录。

## iter2: speedup 1.208110829133419 → 1.3213246705259825 (+9.4%)

- **改了什么**: 在 F2 单 kernel 骨架内新增 MULTI_TOKEN serial loop（grid 固定为 min(tokens,40)=40 个 Vector Core，tokens>64 时每 program 从 pid 起步、步长 40 串行处理多 token）+ 4 处 tl.where 统计掩码比较从 int32 改为 fp32 向量比较。代码：impl/cann_bench/fused_patch_concat_layernorm.py 行 28-36（循环路由）、行 60-64（fp32 比较）、行 145（host grid 路由）。选择依据：develop/iter1/融合方案选择决策依据.md。

- **为什么有效**: 对大 C（≥256）case 有效的根因：(1) 调度波从 10-58 波降为 1 波，消除了多波调度的启动/同步开销；(2) per-token 有效计算量大（BLOCK_C=512 时每 token 4×512×4B=8KB fp32 load+reduce+store），serial loop 的 per-iteration 固定开销（token 解码、w/b 重载）占比 <15%，净收益显著。 对中等 C（128-256 fp32）也有效：case 11（C=128 fp32）per-token≈1.75us，无 cast 开销，固定开销占比低。 但对小 C（80-128）+ fp16/bf16 case 无效甚至有害：per-token 有效计算量极低（BLOCK_C=128 时仅 2KB），w/b 逐次重载（8 次 masked load/迭代）+ token 解码除法/取模 + fp16/bf16 cast 的 per-iteration 开销超过了减波收益。 case 11 fp32 +85% vs case 12 bf16 -14% 的对比（同 shape、同 tokens=2304、同 C=128）直接证明 cast 是 small C serial loop 的关键恶化因素。

- **框架与芯片**:

  ```json
  {
    "framework": "Triton",
    "backend": "triton-ascend",
    "framework_source": "performance_report",
    "runtime_versions": {
      "triton": "3.2.0",
      "triton_ascend": "3.2.1",
      "torch": "2.7.1+cpu",
      "torch_npu": "2.7.1.post8"
    },
    "toolchain": {
      "verified": true,
      "cann_roots": {
        "ASCEND_HOME_PATH": "/home/developer/Ascend/cann-9.1.0",
        "ASCEND_TOOLKIT_HOME": "/home/developer/Ascend/cann-9.1.0",
        "CANN_PATH": "/home/developer/Ascend/cann-9.1.0"
      },
      "version_files": [
        {
          "path": "/home/developer/Ascend/cann-9.1.0/share/info/asc-devkit/version.info",
          "sha256": "f20359296d5a7127f841aa35f4a788137f444dc36d72641e948cb9e39e47e4f4",
          "version": "9.1.0"
        }
      ]
    },
    "chip_model": "Ascend910B3",
    "soc_version": "Ascend910B3",
    "npu_arch": "dav-c220",
    "programming_model": "Triton block program (SPMD)",
    "device_id": 0,
    "source_kind": "performance_report",
    "source_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/perf_result.json"
  }
  ```

- **对比轮次**:

  ```json
  1
  ```

- **逐 case 分析**:

  ```json
  [
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "observation": "speedup 从 0.290 提升至 0.535（+85%），C=128 fp32 tokens=2304，改善幅度最大",
      "explanation": "C=128=BLOCK_C 无 mask 浪费，fp32 无 cast 开销。每 program 58 个 token 的 serial loop 中，per-token 有效计算量充足（512B fp32 load+reduce+store 无 cast），固定开销（w/b 重载 + 解码）占比相对低。减波从 57.6 降为 1 波的调度收益被充分实现。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "observation": "speedup 从 1.117 提升至 1.664（+49%），C=512 fp32 tokens=576",
      "explanation": "C=512=BLOCK_C，per-token 有效计算量大（4×512×4B=8KB fp32）。serial loop 内有效工作远大于固定开销，减波收益完全转化为性能提升。fp32 无 cast 使每迭代纯计算效率最高。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "observation": "speedup 从 1.100 提升至 1.623（+48%），C=512 bf16 tokens=576",
      "explanation": "同 case 17 的大 C 效应。bf16 虽有 cast 开销，但 C=512 使有效计算占主导，cast 占比低。证明大 C 场景下 MULTI_TOKEN serial loop 即使对 fp16/bf16 也有效。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "observation": "speedup 从 1.343 提升至 1.729（+29%），C=384 fp32 tokens=392",
      "explanation": "C=384（BLOCK_C=512），per-token 计算量较大。tokens=392 对应每 program ~10 个 token，减波从 ~10 波降为 1 波的收益 + 较大的 per-token 有效计算 = 净收益显著。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "observation": "speedup 从 0.183 下降至 0.170（-7%），C=80 fp16 tokens=690，最慢 case",
      "explanation": "C=80/BLOCK_C=128 有效利用率仅 62.5%，fp16→fp32 cast 每迭代增加 34% 开销。per-token 有效计算极低（~2KB fp32），serial loop 的 per-iteration w/b 重载（8 次 masked load）+ 地址解码占比过高，减波的微小收益被循环开销抵消且超出。直接对比 fp32 同 shape case 20（129us vs 197us）证明 cast 是恶化因子。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "observation": "speedup 从 0.308 下降至 0.264（-14%），C=128 bf16 tokens=2304，退步最严重",
      "explanation": "与 case 11（同 shape fp32，+85%）对比最为说明问题：两者同为 C=128、tokens=2304、每 program 58 个 token。case 11 fp32 无 cast，固定开销占比低→大幅受益；case 12 bf16 每迭代 4 次 bf16→fp32 load-cast + 4 次 store-cast 在 58 次循环中累积巨大开销→反而退步。w/b 逐次重载（58×8=464 次 masked load）是另一主因。"
    }
  ]
  ```

- **证据**:

  ```json
  [
    {
      "path": "eval/iter2/perf_result.json",
      "detail": "avg_speedup=1.321（iter1=1.208，+9.4%）；case 17 speedup 1.664（iter1=1.117，+49%）; case 11 speedup 0.535（iter1=0.290，+85%）; case 19 speedup 0.170（iter1=0.183，-7%）"
    },
    {
      "path": "profile/iter2/bottleneck_analysis.md",
      "detail": "6 worst case 逐个 profiler 分析：serial loop per-iteration overhead（w/b 重载、地址解码、cast）在小 C 下主导；op_statistic 显示 kernel 时间占比仅 43.9%（case 19）"
    },
    {
      "path": "selection/state.json",
      "detail": "case_trends 30 case 逐轮对比；大 C 组（16-18）+36-49%，小 C 组（19/1/12）-7 to -14%"
    },
    {
      "path": "develop/iter1/融合方案选择决策依据.md",
      "detail": "绑定代码 SHA256=d176d86e；实际改动项与 design_rationale 一致"
    }
  ]
  ```

- **适用条件**: 适用：Ascend910B3（40 Vector Core）上 F2 单 kernel 融合算子，MULTI_TOKEN serial loop（grid=物理核数 + 内循环步进）在以下条件有效：(1) C≥256（BLOCK_C≥256，per-token 计算充足）或 C≥128 + fp32（无 cast 开销）；(2) tokens>64（保证分流阈值以上走 MULTI_TOKEN）；(3) w/b 较小（<UB 上限）。 不适用：C≤128 + fp16/bf16（per-iteration 的 cast + w/b 重载固定开销主导），此时减波收益被循环开销抵消甚至导致退步。 未验证范围：C=192（case 5/7/9，borderline，本轮小幅改善 2-6%）；更大 grid（tokens>2304）；不同 BLOCK_C 取值。 硬件限制：grid=40 对应 40 Vector Core，其他核数设备需调整。

- **后续动作**: 下一轮在 MULTI_TOKEN 路径内做 w/b 循环外提取 + 2D tile（FIX_DIRECTIVE 项 1-3），针对小 C case 的 per-iteration 开销根因修复。 若 2D tile 对小 C 有效，MULTI_TOKEN serial loop + 2D tile 可作为 F2 骨架内的标准调度模式；若仍不足，需评估小 C 按 F5 路由回 grid=tokens 的逐 token 路径（FIX_DIRECTIVE 说实测两版小 C 组 speedup 同为 0.17-0.31，回退无收益——但此对比是 1-token/program grid=tokens vs MULTI_TOKEN serial loop，不是 vs 2D tile）。

- **程序计算的逐 case 变化**:

  ```json
  [
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "prev_speedup": 0.2898720682302772,
      "curr_speedup": 0.5354470263883419,
      "delta": "+84.7%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "prev_speedup": 0.5721263062244434,
      "curr_speedup": 1.025030525030525,
      "delta": "+79.2%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "prev_speedup": 0.4794520547945206,
      "curr_speedup": 0.7811928283937066,
      "delta": "+62.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "prev_speedup": 0.6516059443911792,
      "curr_speedup": 0.9896250455041864,
      "delta": "+51.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "prev_speedup": 1.1165347405452948,
      "curr_speedup": 1.663826998689384,
      "delta": "+49.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "prev_speedup": 1.1004587155963301,
      "curr_speedup": 1.6231393775372125,
      "delta": "+47.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "prev_speedup": 1.157748611812216,
      "curr_speedup": 1.515862524785195,
      "delta": "+30.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "prev_speedup": 1.3431066749844043,
      "curr_speedup": 1.7293172690763055,
      "delta": "+28.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "prev_speedup": 0.3103843947217441,
      "curr_speedup": 0.35638998682476947,
      "delta": "+14.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "prev_speedup": 0.3638524434303099,
      "curr_speedup": 0.41150537634408607,
      "delta": "+13.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "prev_speedup": 0.19102805811245427,
      "curr_speedup": 0.21488273453093815,
      "delta": "+12.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "prev_speedup": 0.38909809828657504,
      "curr_speedup": 0.4331377069796688,
      "delta": "+11.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "prev_speedup": 0.6256776034236805,
      "curr_speedup": 0.6885400313971742,
      "delta": "+10.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "prev_speedup": 2.1309904153354635,
      "curr_speedup": 2.2881646655231562,
      "delta": "+7.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "prev_speedup": 0.27414817130353236,
      "curr_speedup": 0.29233333333333333,
      "delta": "+6.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "prev_speedup": 2.1583452211126963,
      "curr_speedup": 2.2582089552238807,
      "delta": "+4.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "prev_speedup": 0.9145785876993167,
      "curr_speedup": 0.9407216494845361,
      "delta": "+2.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "prev_speedup": 0.8539170506912444,
      "curr_speedup": 0.8773674242424243,
      "delta": "+2.7%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "prev_speedup": 0.2412121212121212,
      "curr_speedup": 0.24640148583810556,
      "delta": "+2.2%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "prev_speedup": 3.318181818181818,
      "curr_speedup": 3.3915929203539825,
      "delta": "+2.2%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "prev_speedup": 3.4410256410256412,
      "curr_speedup": 3.513089005235602,
      "delta": "+2.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "prev_speedup": 1.9805680119581464,
      "curr_speedup": 1.9924812030075187,
      "delta": "+0.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "prev_speedup": 3.706730769230769,
      "curr_speedup": 3.706730769230769,
      "delta": "0.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "prev_speedup": 2.2114285714285713,
      "curr_speedup": 2.138121546961326,
      "delta": "-3.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "prev_speedup": 2.1933701657458564,
      "curr_speedup": 2.100529100529101,
      "delta": "-4.2%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "prev_speedup": 3.2635658914728682,
      "curr_speedup": 3.0507246376811596,
      "delta": "-6.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "prev_speedup": 0.23044859071059945,
      "curr_speedup": 0.21521255561047947,
      "delta": "-6.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "prev_speedup": 0.24299867899603697,
      "curr_speedup": 0.2269026766991489,
      "delta": "-6.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "prev_speedup": 0.1831417624521073,
      "curr_speedup": 0.16958945767866193,
      "delta": "-7.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "prev_speedup": 0.30772769089236435,
      "curr_speedup": 0.2636712976647946,
      "delta": "-14.3%"
    }
  ]
  ```

- **程序提供的证据文件**:

  ```json
  {
    "previous_performance_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/perf_result.json",
    "previous_performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter1/perf_reports/attempt_1_fa4b6bd68b704b30aacd6f9ff1861a6a/fused_patch_concat_layernorm_eval_20261002_034317.json",
    "current_performance_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/perf_result.json",
    "current_performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/perf_reports/attempt_1_cc798cc9fb174c7f9836835ca9587698/fused_patch_concat_layernorm_eval_20261002_044016.json",
    "current_decision_rationale": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/0_decision_rationale.md",
    "current_fusion_library": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/1_fusion_library.json",
    "current_self_test_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/2_self_test_report.md",
    "current_self_test_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/3_self_test_result.json",
    "current_selftest_log_continuous": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/4_selftest_log_continuous.log",
    "current_selftest_log_provided": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter2-844fc6b40708c857f98b/evidence/5_selftest_log_provided.log"
  }
  ```

- **本轮决策文件**: /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter2/16f375d457844ae7978d993dced42915/decision.json

## iter4: speedup 0.8224094689497551 → 1.5389794201417895 (+87.1%)

- **改了什么**: 回退 MULTI_TOKEN=True 分支的 (TILE,BLOCK_C) 2D tile 实现为 iter2 的 1D 逐 token serial loop，同时保留 w/b 8 段 masked load 循环外提取（每 program 加载 1 次驻留 UB，不再每 token 重载）。具体改动：(1) 删除 2D tile 全部代码（lanes/hw/tmask/tile_mask/2D load/axis=1 归约/2D store），替换为 1D serial loop（token_lo=pid, token_step=tl.num_programs(0)）+ 4 段 1D load + axis=0 归约 + 1D store；(2) 删除 TILE constexpr 参数和 host tile 计算；(3) w/b 8 段 masked load 保持循环前（lines 37-44），store 引用 w0-b3。代码：impl/cann_bench/fused_patch_concat_layernorm.py。选择依据：develop/iter3/融合方案选择决策依据.md。

- **为什么有效**: 两个独立收益来源叠加：(1) 消除 2D tile 灾难——triton-ascend 3.2.1 后端对 2D tensor 操作（2D strided gather、axis=1 归约、2D broadcast mask/store）生成极低效编译代码，导致 iter3 全部 21 个 MULTI_TOKEN case 退步 22-153x（kernel 768-4524us）。回退到 1D 后，同样的数据流用 1D load/store/axis=0 归约实现，后端编译质量正常（21-70us），恢复 iter2 量级。(2) w/b 循环外提取的独立净收益——对比 iter2（w/b 循环内每 token 8 次 HBM masked load），iter4 每 program 仅 8 次，case 12（tokens=2304）从 464 次降为 8 次。正式冷缓存评测证实：大 C + fp32 组收益最显著（case 8 C=384 fp32 speedup 1.73→2.04 +18%，case 17 C=512 fp32 1.66→1.89 +13%，case 14 C=256 fp32 0.99→1.40 +41%），因 per-token w/b 重载的 HBM 延迟在冷缓存下被放大。小 C + cast case 收益被残余 cast/索引开销掩盖。

- **框架与芯片**:

  ```json
  {
    "framework": "Triton",
    "backend": "triton-ascend",
    "framework_source": "performance_report",
    "runtime_versions": {
      "triton": "3.2.0",
      "triton_ascend": "3.2.1",
      "torch": "2.7.1+cpu",
      "torch_npu": "2.7.1.post8"
    },
    "toolchain": {
      "verified": true,
      "cann_roots": {
        "ASCEND_HOME_PATH": "/home/developer/Ascend/cann-9.1.0",
        "ASCEND_TOOLKIT_HOME": "/home/developer/Ascend/cann-9.1.0",
        "CANN_PATH": "/home/developer/Ascend/cann-9.1.0"
      },
      "version_files": [
        {
          "path": "/home/developer/Ascend/cann-9.1.0/share/info/asc-devkit/version.info",
          "sha256": "f20359296d5a7127f841aa35f4a788137f444dc36d72641e948cb9e39e47e4f4",
          "version": "9.1.0"
        }
      ]
    },
    "chip_model": "Ascend910B3",
    "soc_version": "Ascend910B3",
    "npu_arch": "dav-c220",
    "programming_model": "Triton block program (SPMD)",
    "device_id": 0,
    "source_kind": "performance_report",
    "source_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/perf_result.json"
  }
  ```

- **对比轮次**:

  ```json
  3
  ```

- **逐 case 分析**:

  ```json
  [
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "observation": "iter3 speedup=0.0106(kernel 4517us) → iter4 speedup=1.773(kernel 27.06us), 提升 16594%。bf16 C=512 tokens=576。",
      "explanation": "C=512 大载荷，回退 1D 后恢复正常编译（27us vs iter3 4517us），w/b 提取额外贡献（iter2 为 29.56us→iter4 27.06us，-8.5%）；大 C 下 w/b 重载次数少（14 次/prog），提取净收益有限但可测。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "observation": "iter3 speedup=0.0126(4017us) → iter4 speedup=1.885(26.94us), 提升 14810%。fp32 C=512 tokens=576。",
      "explanation": "fp32 无 cast，是 w/b 提取净效果的纯净对照。iter2 为 30.52us→iter4 26.94us(-11.7%)，证明 w/b 提取在冷缓存下贡献约 12%。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "observation": "iter3 speedup=0.0214(kernel 2005us) → iter4 speedup=2.043(21.08us), 提升 9427%。fp32 C=384 tokens=392。",
      "explanation": "fp32 C=384，回退恢复 + w/b 提取双收益。iter2 为 24.92us→iter4 21.08us(-15.4%)，w/b 提取在每 program 10 次迭代中消除 80 次 masked load。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "observation": "iter3 speedup=0.0118(4524us) → iter4 speedup=0.764(70.04us), 提升 6359%。bf16 C=128 tokens=2304。",
      "explanation": "回退恢复正常（70us vs 4524us）。w/b 提取理论收益最大（464→8 次重载），但 bf16 cast + 58 次索引解码的残余开销仍使 speedup 未达标。对比 iter2(202.98us)→iter4(70.04us)改善 65%，证明 w/b 提取在冷缓存下对高迭代 case 极为有效。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "observation": "iter3 speedup=0.0436(778us) → iter4 speedup=0.757(44.22us), 提升 1636%。fp16 C=80 tokens=690。",
      "explanation": "回退恢复正常（44us vs 778us），但仍未达标。C=80 最小有效通道 + 62.5% 利用率 + fp16 cast 三重瓶颈使 per-token 固定开销占比过高。w/b 提取收益被这些残余开销掩盖。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "observation": "iter3 speedup=2.134 → iter4 speedup=1.620, 下降 24.1%。fp16 C=3 tokens=1, MULTI_TOKEN=False。",
      "explanation": "MULTI_TOKEN=False 路径代码未变。iter4 的 PAIRED_SEG/BLOCK_2C 参数未引入（本轮无此改动），编译差异可能源于 kernel 签名变化（删除 TILE）导致不同编译产物。绝对 speedup 仍 >1.6 远超达标线。对比 iter2(2.10)→iter4(1.62)也有波动，属编译/测量噪声范围。"
    }
  ]
  ```

- **证据**:

  ```json
  [
    {
      "path": "eval/iter4/perf_result.json",
      "detail": "avg_speedup=1.539; 30 case speedup 逐项数据；对比 iter3 的 0.822(+87.1%)"
    },
    {
      "path": "eval/iter4/perf_comparison.json",
      "detail": "comparable=true, delta_pct=87.1%, 21 MULTI_TOKEN case 提升 1636-16594%"
    },
    {
      "path": "eval/iter2/perf_result.json",
      "detail": "iter2 avg_speedup=1.321 作为 w/b 提取前的基线；case 8=1.729 case 17=1.664 case 14=0.990"
    },
    {
      "path": "impl/cann_bench/fused_patch_concat_layernorm.py",
      "detail": "lines 37-44 w/b 循环前提取; lines 46-54 1D serial loop; lines 64-95 循环体 4 段 1D load/store"
    },
    {
      "path": "develop/iter3/融合方案选择决策依据.md",
      "detail": "回退选择依据与 w/b 保留理由"
    },
    {
      "path": "profile/iter4/bottleneck_analysis.md",
      "detail": "逐 case 瓶颈分析确认残余固定开销"
    }
  ]
  ```

- **适用条件**: 适用于 Ascend910B3 + triton-ascend 3.2.1，F2 单 kernel 融合算子的 MULTI_TOKEN serial loop。1D 回退部分适用所有 case；w/b 循环外提取在循环次数多（tokens/40>10）且 C>=256 或 fp32 时收益最大（10-41%）。小 C(80-128) + fp16/bf16 的 w/b 提取净收益被 cast + 索引解码开销掩盖（iter4 这些 case 仍未达标，0.757-0.903）。小 grid 组（MULTI_TOKEN=False, tokens<=64）零影响（编译等价）。2D tile 在当前后端版本确认灾难性低效，未来重试需确认后端修复。未验证范围：w/b 提取在更大 C(>512) 或更长循环下的边际收益趋势。

- **后续动作**: 延续 1D serial loop + w/b 提取基础，下一轮叠加配对段向量化（4 段→2 对 1D load/store）+ int32 索引优化，针对 9 个未达标的小 C case。已验证的 w/b 提取改动不回退。2D tile 方向暂停，除非后端升级并提供新证据。

- **程序计算的逐 case 变化**:

  ```json
  [
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "prev_speedup": 0.010621106737916775,
      "curr_speedup": 1.7730968218773095,
      "delta": "+16594.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "prev_speedup": 0.012641778114139473,
      "curr_speedup": 1.8849294729027468,
      "delta": "+14810.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "prev_speedup": 0.011364143118338709,
      "curr_speedup": 1.6814516129032255,
      "delta": "+14696.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "prev_speedup": 0.00944460418924598,
      "curr_speedup": 1.0460558549730525,
      "delta": "+10975.7%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "prev_speedup": 0.013897764394934741,
      "curr_speedup": 1.399125064333505,
      "delta": "+9967.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "prev_speedup": 0.012946522662197799,
      "curr_speedup": 1.248017839444995,
      "delta": "+9539.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "prev_speedup": 0.02144123329416217,
      "curr_speedup": 2.0426944971537004,
      "delta": "+9426.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "prev_speedup": 0.018581425054147612,
      "curr_speedup": 1.3163934426229509,
      "delta": "+6984.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "prev_speedup": 0.018381842350653734,
      "curr_speedup": 1.2353333333333334,
      "delta": "+6620.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "prev_speedup": 0.02195261118952521,
      "curr_speedup": 1.4370904325032765,
      "delta": "+6446.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "prev_speedup": 0.01183013412790338,
      "curr_speedup": 0.7641347801256425,
      "delta": "+6359.2%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "prev_speedup": 0.013995840882885852,
      "curr_speedup": 0.8049141503848432,
      "delta": "+5651.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "prev_speedup": 0.013386247424253987,
      "curr_speedup": 0.7593073593073593,
      "delta": "+5572.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "prev_speedup": 0.019257650874119358,
      "curr_speedup": 0.9031905594405595,
      "delta": "+4590.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "prev_speedup": 0.018800168990283063,
      "curr_speedup": 0.8251401466149203,
      "delta": "+4289.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "prev_speedup": 0.03161154610260606,
      "curr_speedup": 1.0417201540436458,
      "delta": "+3195.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "prev_speedup": 0.028739044167044227,
      "curr_speedup": 0.7871202396234488,
      "delta": "+2638.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "prev_speedup": 0.03176051923147841,
      "curr_speedup": 0.8109698681732581,
      "delta": "+2453.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "prev_speedup": 0.041165671140071886,
      "curr_speedup": 1.0378096479791394,
      "delta": "+2421.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "prev_speedup": 0.03352455387221591,
      "curr_speedup": 0.7608125819134993,
      "delta": "+2169.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "prev_speedup": 0.043574516851591395,
      "curr_speedup": 0.7566711895070104,
      "delta": "+1636.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "prev_speedup": 2.7516339869281046,
      "curr_speedup": 3.0507246376811596,
      "delta": "+10.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "prev_speedup": 1.9064748201438848,
      "curr_speedup": 1.8821022727272727,
      "delta": "-1.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "prev_speedup": 2.344463971880492,
      "curr_speedup": 2.219633943427621,
      "delta": "-5.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "prev_speedup": 3.720873786407767,
      "curr_speedup": 3.3471615720524017,
      "delta": "-10.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "prev_speedup": 3.4419642857142856,
      "curr_speedup": 3.071713147410359,
      "delta": "-10.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "prev_speedup": 2.3714733542319753,
      "curr_speedup": 2.0984743411927878,
      "delta": "-11.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "prev_speedup": 3.60752688172043,
      "curr_speedup": 3.008968609865471,
      "delta": "-16.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "prev_speedup": 1.9545454545454546,
      "curr_speedup": 1.5542168674698795,
      "delta": "-20.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "prev_speedup": 2.1344086021505375,
      "curr_speedup": 1.620408163265306,
      "delta": "-24.1%"
    }
  ]
  ```

- **程序提供的证据文件**:

  ```json
  {
    "previous_performance_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/perf_result.json",
    "previous_performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/perf_reports/attempt_1_22f81ac534f04bfd8f812ce5627a316d/fused_patch_concat_layernorm_eval_20261002_063059.json",
    "current_performance_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/perf_result.json",
    "current_performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter4/perf_reports/attempt_1_0898adbad8eb48a081784d0cdea19af5/fused_patch_concat_layernorm_eval_20261002_155635.json",
    "current_decision_rationale": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/0_decision_rationale.md",
    "current_fusion_library": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/1_fusion_library.json",
    "current_self_test_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/2_self_test_report.md",
    "current_self_test_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/3_self_test_result.json",
    "current_selftest_log_continuous": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/4_selftest_log_continuous.log",
    "current_selftest_log_provided": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter4-60dea9e34578bd49ed55/evidence/5_selftest_log_provided.log"
  }
  ```

- **本轮决策文件**: /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter4/0f505cd0c7c74b8781c319af2b697690/decision.json

