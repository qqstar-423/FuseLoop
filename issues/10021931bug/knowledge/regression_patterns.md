# 已验证的失败教训（退步记录）

> Stage9 填写分析，程序验证性能变化达到 5% 后保存；同轮重试更新原记录。

## iter3: speedup 1.3213246705259825 → 0.8224094689497551 (-37.8%)

- **改了什么**: 在 F2 单 kernel MULTI_TOKEN=True 分支内，将 1D 逐 token serial loop 改为 (TILE, BLOCK_C) 2D tile 路径：新增 TILE: tl.constexpr，token 解码改为 (TILE,) 向量化，4 段 x 读改为 2D masked strided gather，tl.sum 从 axis=0 改为 axis=1，store 改为 2D masked store，新增 tmask 尾块钳位。w/b 8 段 masked load 移出循环。Host 新增 tile 计算。代码：impl/cann_bench/fused_patch_concat_layernorm.py 行 47-106（2D 分支）、行 27（TILE 签名）、行 38-45（w/b 提取）、行 207（tile 计算）。依据：develop/iter2/融合方案选择决策依据.md。

- **为什么退步**: triton-ascend 3.2.1 后端对 (TILE, BLOCK_C) 2D tensor 操作生成灾难性低效编译代码。证据：(1) 全部 21 个 MULTI_TOKEN=True case 退步 22-153x，kernel 768-4524us vs iter2 24-203us；(2) MULTI_TOKEN=False case 22-30 不受影响（speedup 1.9-3.7）；(3) fp32 case 17 退步 132x 排除 cast；(4) TILE=2 退步 79-153x 排除 TILE 取值/UB 溢出；(5) kernel_details Duration 证实退步在 kernel 内部。2D strided gather/axis=1 reduction/2D broadcast mask 的具体低效点为推测，需编译产物验证。

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
    "source_path": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/perf_result.json"
  }
  ```

- **对比轮次**:

  ```json
  2
  ```

- **逐 case 分析**:

  ```json
  [
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "observation": "speedup 0.781->0.009 (退步 98.8%), kernel 54.66->4521us",
      "explanation": "TILE=4 BLOCK_C=256 bf16 2D tile 编译产物极低效，单 tile 开销远超 1D 单 token。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "observation": "speedup 1.623->0.011 (退步 99.3%), kernel 29.56->4517us",
      "explanation": "TILE=2 BLOCK_C=512 bf16，最小 TILE 仍退步 153x，排除 TILE 取值原因。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "observation": "speedup 1.664->0.013 (退步 99.2%), kernel 30.52->4017us",
      "explanation": "fp32 无 cast 仍退步 132x，排除 cast 为主因，确证 2D 编译质量问题。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "observation": "speedup 1.729->0.021 (退步 98.8%), kernel 24.9->2008us",
      "explanation": "C=384 fp32 TILE=2，iter2 最佳 case，2D 路径 80x 退步。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "observation": "speedup 0.264->0.012 (退步 95.5%), kernel 202.98->4524us",
      "explanation": "TILE=8 BLOCK_C=128 bf16，per-tile 565us vs 1D per-token 3.5us。w/b 提取收益被 2D 退步掩盖。"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "observation": "speedup 2.10->2.13 (稳定)",
      "explanation": "MULTI_TOKEN=False 1D 路径不受影响，证明退步严格限于 2D 代码路径。"
    }
  ]
  ```

- **证据**:

  ```json
  "eval/iter3/perf_result.json 全 30 case; profile/iter3/bottleneck_analysis.md 逐 case Duration 对比; selection/records/iter2-844fc6b40708c857f98b/reports/perf_result.json iter2 对照; develop/iter2/融合方案选择决策依据.md 记录改动。"
  ```

- **适用条件**: 适用于 Ascend910B3 + triton-ascend 3.2.1。2D tensor 操作（tl.load/tl.store 2D 索引、tl.sum axis=1、2D broadcast mask）在当前后端生成极低效代码，退步 22-153x。覆盖所有测试的 BLOCK_C(128/256/512)、dtype(fp16/bf16/fp32)、TILE(2/4/8)。未验证：未来后端版本；reshape+1D 变通。

- **后续动作**: 立即回退 1D serial loop + 保留 w/b 提取。永久禁止在 triton-ascend 3.2.1 上使用 2D tensor pattern；后续若需重试须确认后端修复。

- **程序计算的逐 case 变化**:

  ```json
  [
    {
      "case_id": "level3/fused_patch_concat_layernorm_25",
      "prev_speedup": 3.3915929203539825,
      "curr_speedup": 3.720873786407767,
      "delta": "+9.7%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_28",
      "prev_speedup": 2.2582089552238807,
      "curr_speedup": 2.3714733542319753,
      "delta": "+5.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_26",
      "prev_speedup": 3.513089005235602,
      "curr_speedup": 3.60752688172043,
      "delta": "+2.7%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_29",
      "prev_speedup": 2.2881646655231562,
      "curr_speedup": 2.344463971880492,
      "delta": "+2.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_22",
      "prev_speedup": 2.100529100529101,
      "curr_speedup": 2.1344086021505375,
      "delta": "+1.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_30",
      "prev_speedup": 1.9924812030075187,
      "curr_speedup": 1.9064748201438848,
      "delta": "-4.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_27",
      "prev_speedup": 3.706730769230769,
      "curr_speedup": 3.4419642857142856,
      "delta": "-7.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_24",
      "prev_speedup": 2.138121546961326,
      "curr_speedup": 1.9545454545454546,
      "delta": "-8.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_23",
      "prev_speedup": 3.0507246376811596,
      "curr_speedup": 2.7516339869281046,
      "delta": "-9.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_19",
      "prev_speedup": 0.16958945767866193,
      "curr_speedup": 0.043574516851591395,
      "delta": "-74.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_20",
      "prev_speedup": 0.24640148583810556,
      "curr_speedup": 0.041165671140071886,
      "delta": "-83.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_1",
      "prev_speedup": 0.21521255561047947,
      "curr_speedup": 0.03352455387221591,
      "delta": "-84.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_21",
      "prev_speedup": 0.21488273453093815,
      "curr_speedup": 0.03176051923147841,
      "delta": "-85.2%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_3",
      "prev_speedup": 0.2269026766991489,
      "curr_speedup": 0.028739044167044227,
      "delta": "-87.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_2",
      "prev_speedup": 0.35638998682476947,
      "curr_speedup": 0.03161154610260606,
      "delta": "-91.1%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_10",
      "prev_speedup": 0.29233333333333333,
      "curr_speedup": 0.013386247424253987,
      "delta": "-95.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_4",
      "prev_speedup": 0.41150537634408607,
      "curr_speedup": 0.018800168990283063,
      "delta": "-95.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_12",
      "prev_speedup": 0.2636712976647946,
      "curr_speedup": 0.01183013412790338,
      "delta": "-95.5%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_6",
      "prev_speedup": 0.4331377069796688,
      "curr_speedup": 0.019257650874119358,
      "delta": "-95.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_5",
      "prev_speedup": 0.6885400313971742,
      "curr_speedup": 0.02195261118952521,
      "delta": "-96.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_11",
      "prev_speedup": 0.5354470263883419,
      "curr_speedup": 0.013995840882885852,
      "delta": "-97.4%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_7",
      "prev_speedup": 0.8773674242424243,
      "curr_speedup": 0.018381842350653734,
      "delta": "-97.9%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_9",
      "prev_speedup": 0.9407216494845361,
      "curr_speedup": 0.018581425054147612,
      "delta": "-98.0%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_14",
      "prev_speedup": 0.9896250455041864,
      "curr_speedup": 0.013897764394934741,
      "delta": "-98.6%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_13",
      "prev_speedup": 1.025030525030525,
      "curr_speedup": 0.012946522662197799,
      "delta": "-98.7%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_15",
      "prev_speedup": 0.7811928283937066,
      "curr_speedup": 0.00944460418924598,
      "delta": "-98.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_8",
      "prev_speedup": 1.7293172690763055,
      "curr_speedup": 0.02144123329416217,
      "delta": "-98.8%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_17",
      "prev_speedup": 1.663826998689384,
      "curr_speedup": 0.012641778114139473,
      "delta": "-99.2%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_18",
      "prev_speedup": 1.6231393775372125,
      "curr_speedup": 0.010621106737916775,
      "delta": "-99.3%"
    },
    {
      "case_id": "level3/fused_patch_concat_layernorm_16",
      "prev_speedup": 1.515862524785195,
      "curr_speedup": 0.011364143118338709,
      "delta": "-99.3%"
    }
  ]
  ```

- **程序提供的证据文件**:

  ```json
  {
    "previous_performance_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/perf_result.json",
    "previous_performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter2/perf_reports/attempt_1_cc798cc9fb174c7f9836835ca9587698/fused_patch_concat_layernorm_eval_20261002_044016.json",
    "current_performance_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/perf_result.json",
    "current_performance_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/eval/iter3/perf_reports/attempt_1_22f81ac534f04bfd8f812ce5627a316d/fused_patch_concat_layernorm_eval_20261002_063059.json",
    "current_decision_rationale": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter3-d117e98094ba7f244076/evidence/0_decision_rationale.md",
    "current_fusion_library": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter3-d117e98094ba7f244076/evidence/1_fusion_library.json",
    "current_self_test_report": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter3-d117e98094ba7f244076/evidence/2_self_test_report.md",
    "current_self_test_result": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter3-d117e98094ba7f244076/evidence/3_self_test_result.json",
    "current_selftest_log_continuous": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter3-d117e98094ba7f244076/evidence/4_selftest_log_continuous.log",
    "current_selftest_log_provided": "/mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/selection/records/iter3-d117e98094ba7f244076/evidence/5_selftest_log_provided.log"
  }
  ```

- **本轮决策文件**: /mnt/workspace/pypto-pro-workflow/work/fused_patch_concat_layernorm_20261002_025914/knowledge/stage9/iter3/4b3764bf69c14a94a4be8cf959e4391a/decision.json

