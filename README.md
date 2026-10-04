# Triton Ascend Workflow

> A multi-agent collaborative framework for automated Triton Ascend operator development and iterative optimization

For migration notes, see the [Triton Ascend Migration Plan](docs/triton_ascend_migration_plan.md); for environment setup, see the [Environment Migration Guide](docs/environment_migration_guide.md). The current generation target is fixed to Triton Ascend; `pypto-pro-workflow` on disk is just a retained directory name — it does not determine the generated framework, nor does it require installing PyPTO.

## 📖 Project Overview

Triton Ascend Workflow is a **multi-agent automated operator development framework** for Ascend NPU. Based on the Triton Ascend programming paradigm, it works through the collaboration of 3 CLI Agents (CANNBot writes code, Kerminal compiles and evaluates, Hermes searches and optimizes), the Jev fusion scheme selection node, and 1 deterministic control plane (orchestrator), achieving **full-pipeline automated iteration** for operators: requirements analysis → fusion scheme selection → code writing → build and deployment → precision verification → performance optimization → report generation.

| Dimension | Description |
|------|------|
| Target framework | Triton Ascend (triton-ascend, Ascend NPU) |
| Evaluation tool | cann-bench (precision + performance evaluation) |
| Agents | CANNBot (GLM-5.3-Flash), Kerminal (kernelcat1.0), Hermes (GLM-5.3-Flash), Jev (fusion scheme probability evaluation) |
| Iteration stages | Original 10 stages + Stage1.5, up to 20 automatic iterations |
| Core features | Cross-iteration memory (history), Tech Lead directional guidance, anti-cheating knowledge base, code version traceability |

## ✨ Core Features

- **Fusion scheme selection**: Stage1.5 is added after requirements analysis; Jev evaluates probabilities for fusion methods based on the requirements and hardware, and the program takes the top n to form a JSON fusion scheme library, passed to Stage2/3/7/8/9
- **Stage pipeline**: requirements analysis → fusion scheme selection → first implementation → build and deployment → precision evaluation → performance evaluation → Profiling → search optimization → Tech Lead summary → code modification → final report
- **Cross-iteration memory (history mechanism)**: the Tech Lead distills insights/ledger each iteration, tracks the slowest cases, guides the next iteration's direction, and avoids repeated failures
- **Fusion direction tracking**: the `fusion_kernel_strategy` append-only field records each iteration's fusion scheme attempts (true fusion / fake fusion / on-chip direct transfer vs HBM intermediate read/write), filled in stage9 and read by stage3
- **Semantic exit and best implementation**: after valid evaluation, snapshots of the code, fusion scheme, and reports are saved; when all cases pass and x consecutive valid improvements accumulate less than 5%, the best passing implementation is delivered
- **Emergency re-run with existing implementation**: `--init-impl` copies the specified code and its corresponding development materials, skipping Stage1/1.5/2 and starting directly from Stage4 build and evaluation; it does not affect the normal from-scratch flow
- **Anti-cheating knowledge base**: a real-world error handling table and development self-check checklist, combined with source code, kernel CSV, and formal report analysis
- **Automatic chip detection**: reads real UB/L1/L0A/L0B/L0C/core count values directly from the CANN compiler tbe API and injects them into all stages; hardcoding tiling parameters is forbidden
- **Code version traceability**: impl is automatically backed up to `operator_iter/` before each iteration
- **Direct execution of performance evaluation code**: in stage6, the orchestrator directly invokes cann-bench via subprocess, without going through an agent, with automatic retry + old report filtering

stage6 explicitly selects cann-bench's `kernel_details` performance strategy, directly reusing the baseline, HAP, and scores produced by the tool without measuring the baseline separately. This strategy aggregates kernel execution time and does not include synchronization and scheduling intervals between kernels, so it does not represent full call time; old `trace_view` scores cannot be directly compared with this strategy's scores.

## Stage1.5: Jev Fusion Scheme Selection

**Stage order: [Stage1 Requirements Analysis](roles/n1_stage1_requirements_analysis.md) → [Stage1.5 Jev Fusion Scheme Selection](roles/n1_stage1.5_jev_fusion_selection.md) → [Stage2 First Implementation](roles/n1_stage2_first_impl.md).** The Stage1.5 role file describes inputs, responsibilities, and outputs; the actual English request is assembled by the program, which calls Jev.

[Jev Console](https://console.typesafe.ai/home)

Jev has been connected between Stage1 and Stage2. Inputs include the original text of `knowledge/fusion_method.md`, the `knowledge/fusion_options.json` decomposed from that document (F1–F10, with 24 variants), the `<work>/fusion_requirements.en.json` newly added in Stage1, and current hardware information. The requirements JSON uses compact English to preserve operator semantics, interface, case characteristics, precision, and fusion constraints; the program first estimates the available byte budget for this run and passes it to Stage1 (absolute cap of 6000 bytes), while the full requirements analysis remains in ANALYSIS.md. Jev independently gives an applicability probability for each of the ten method categories, and the probabilities are not required to sum to 1; the program sorts by probability, takes the top n, and saves them as the **JSON fusion operator library** for this run.

For each run, `<work>/example` points to `examples/triton_ascend_cann_example/` under `paths.cannbench_repo`; if missing, execution stops with a prompt to install the full example. This repository's `examples/triton_ascend_example/` is an additional minimal fusion demo and NPU self-test entry, not the source of that default symlink. The task's `proto.yaml`, cases, golden files, and evaluation entry are retained; what needs to change is the candidate implementation and runtime environment. Stage4 builds the wheel, and Stage3 self-testing and Stage5 real invocation must also cover first-time JIT.

The Stage1.5 call sequence is: **read materials → reuse Kerminal for translation → assemble English JSON by program and validate → Jev scoring → select top n schemes**. The original fusion method text, the scheme library's deep descriptions, and any non-English content in the requirements and hardware information all enter the translation check; already-English content is kept as-is. JSON structure, numbers, and scheme IDs are preserved by the program, and numbers and technical identifiers in the translation are also validated. The actually sent request must pass the English check and the post-translation size check; on translation failure, residual Chinese, or exceeding the size limit, execution stops — no mixed-language requests are sent, and old probabilities are never passed off as new results. The underlying Jev client likewise checks English input on all call entry points.

Modify n via `fusion_selection.top_n` in `config.yaml`, default `3`; translation reuses `agents.kerminal.cli`, with the timeout configured by `fusion_selection.translation_timeout_seconds` (default 240 seconds); Jev connection parameters still use the `jev` section. Standalone invocation and offline test entry points are retained.

| Receiving node | Use of the scheme library |
|---|---|
| Stage2 First Implementation | Design the first version based on the highest-probability scheme; if the hardware/framework objectively does not support it, provide evidence and select an implementable follow-up candidate |
| Stage3 Modification and Optimization | Understand the existing scheme and alternative directions, and execute existing modification instructions combined with evaluation evidence |
| Stage7 Performance Analysis | Analyze bottlenecks against candidate data flows and resource conditions |
| Stage8 Search Optimization | Search for implementation methods of candidate schemes around bottlenecks, verifying applicability on the current hardware |
| Stage9 Tech Lead | Review the fusion direction combining candidate schemes, historical attempts, and measured evidence |

All five nodes of a new task read `<work>/fusion/fusion_library.json`. Probabilities are an initial selection reference; measured evidence takes priority. Capabilities such as cross-core storage and synchronization must be verified; a shared L2 Cache cannot be directly treated as DSM. The initial Jev library is maintained by the program; Stage2/3 record the iteration's actual choices, changes, and new unscored schemes in `develop/iterN/fusion_library.json`, without rewriting the initial probabilities. Stage3 also writes `fusion_scheme_rationale.md`, separate from the self-test report, and passes it to Stage7/8/9.

Run artifacts are located in `<work>/fusion/`: `fusion_library.json` saves the top n complete candidates and their probabilities, `ranking.json` saves probabilities for all methods, and `jev_request.json` and `jev_response.json` save the request and response for traceability.

`<work>/fusion/translation/` saves `translation_input.json` (materials before translation), `english_inputs.json` (English materials), `translation_manifest.json` (fingerprints of inputs and translations), and Kerminal translation records. When inputs are unchanged and translations pass validation, the translation cache is reused; the original scheme library and the readable scheme descriptions used by the subsequent five nodes retain the original text.

Checkpoint recovery reuses probability results with identical inputs; when inputs change, re-evaluation occurs, and if only n is modified, re-selection happens directly. The hash of the complete ANALYSIS.md also participates in the cache check. If an old working directory has already entered iteration, and there is neither a new requirements JSON nor a `fusion/` directory, the existing iteration is retained with a warning logged, without retroactively running Stage1.5; if `fusion/` exists but requirements are missing, execution stops with an error to avoid misusing incomplete old results.

| File | Function |
|---|---|
| `lib/fusion_selection.py` | Stage1.5 request construction, probability validation and sorting, top-n candidate persistence, cache reuse, and downstream prompt injection |
| `lib/jev_translation.py` | Reuses Kerminal to translate input text, validates English, structure, and numbers, and caches English materials |
| `lib/jev_client.py` | Reads file contents, assembles JSON, checks request size, calls Jev, and returns native results such as option probabilities |
| `lib/kerminal_rpc.py` | Translates text fields via the local Kerminal CLI |
| `tools/jev_smoke_test.py` | Standalone invocation entry supporting custom source code, English evidence, and question JSON |
| `tools/kerminal_translate_smoke.py` | Example of translation and numeric fidelity checks |

Connection parameters and local secrets are placed uniformly in the `jev` section of `config.yaml`; the secret reading order is environment variable → `jev.api_key`. `config.local.yaml` is no longer needed; the old file is only read for compatibility if an old configuration explicitly retains `credentials_file`. For offline verification, run `.venv-jev\Scripts\python.exe tools/run_jev_offline_tests.py`; results are written to `output/jev_tests/results.json` and real secret configurations are not read.

## 🧭 Applicable Scenarios

- Development and performance optimization of Triton Ascend fused operators on Ascend NPU
- cann-bench evaluation-driven iterative operator optimization
- Research on automating multi-agent collaborative operator development workflows

## 🔧 Automatic Chip Detection

At startup, the orchestrator automatically detects the real hardware parameters of the current NPU chip, writes them to `<work>/device_info.json`, and injects them into the prompts of all stages, ensuring agents design tiling parameters according to actual hardware specs. **No hardcoded mapping tables are used** — all parameters are read from the device, and if unavailable, execution exits.

Detection process:

1. Via `torch_npu.npu.get_device_properties(device_id)`, obtain the SoC model (e.g. `Ascend950PR_9579`), Cube/Vector core counts, and L2 Cache size of the logical device configured
2. Via the CANN compiler internal interface `tbe.common.platform.get_soc_spec`, read the real on-chip memory values (UB/L1/L0A/L0B/L0C/CORE_NUM) — the same data the compiler depends on when compiling kernels
3. Retain the chip name reported by the device and supplement the architecture codename (e.g. `dav-3510`; pure name mapping, does not involve hardware parameters)
4. Import Triton, verify the actual runtime backend is `npu`, and record the versions of torch, torch_npu, triton, and triton-ascend; merely importing a same-named upstream package is not sufficient to pass
5. Validate that key parameters (soc_version/UB/L1/CORE_NUM/L0A) are all obtained; if any is missing, print troubleshooting steps and exit with `sys.exit(1)`

Manual verification commands:
```bash
# View the SoC model
python3 -c "import torch,torch_npu; print(torch_npu.npu.get_device_properties(0).name)"

# View on-chip memory parameters (requires CANN Toolkit)
python3 -c "
from tbe.common.platform import get_soc_spec, set_current_compile_soc_info
set_current_compile_soc_info('Ascend950PR_9579')  # replace with your SoC model
for k in ['UB_SIZE','L1_SIZE','L0A_SIZE','L0B_SIZE','L0C_SIZE','CORE_NUM']:
    v = get_soc_spec(k)
    print(f'{k} = {v} ({v//1024} KB)' if isinstance(v,int) and v>1024 else f'{k} = {v}')
"
```

If automatic detection fails (e.g. CANN is not fully installed), you can manually fill in all parameters in the `hardware` section of `config.yaml`.

## Running Guide

### Step 1: Configure config.yaml

**The repository directly provides the single `config.yaml`, tracked by Git; day-to-day you only modify this one run configuration.** After pulling the repository on a new machine, just adjust the actual paths in it and `jev.api_key`; you can still set the `TYPESAFE_API_KEY` environment variable, which takes priority over the key in the file.

```bash
# One-shot retrieval of all paths on the current machine
echo "cannbot:      $(which cannbot)"
echo "kerminal:     $(which kerminal)"
echo "hermes:       $(which hermes)"
echo "cann toolkit: $(ls -d ~/Ascend/cann-*)"
echo "node bin:     $(dirname $(which node))"
echo "arch:         $(uname -m | sed 's/x86_64/x86_64-linux/;s/aarch64/aarch64-linux/')"
```

Fields to change in config.yaml (7 places in total):

```yaml
agents:
  cannbot:
    cli: "<which cannbot>"
  kerminal:
    cli: "<which kerminal>"
  hermes:
    cli: "<which hermes>"

paths:
  cannbench_repo: "<absolute path of the cann-bench repository>"

cann:
  toolkit_path: "<ls ~/Ascend/cann-*>"
  arch: "<aarch64-linux or x86_64-linux>"
  node_bin: "<dirname $(which node)>"
```

`cann.driver_path` usually needs no change (default `/usr/local/Ascend/driver`). The `hardware` section needs no change (auto-detected by the orchestrator).

Also confirm the service connection and key configuration in the `jev` section; use `fusion_selection.top_n` to control the number of fusion schemes retained by Stage1.5, default 3.

### Step 2: Verify the Environment

```bash
# NPU available
npu-smi info
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"

# CANN Toolkit complete
python3 -c "from tbe.common.platform import get_soc_spec; print('tbe OK')"

# All three Agents available
cannbot --version
kerminal --version
hermes --version
```

### Step 3: Run

**Write an operator from scratch**

```bash
cd pypto-pro-workflow
python3 orchestrator.py \
  --task-dir /path/to/cann-bench/bench_lab/<bench>/<level>/<operator>
```

**With a fusion direction hint (recommended)**

`--optimize-hint` tells the agent which fusion strategy to use; it is injected into stage1 analysis and stage2 implementation:

```bash
python3 orchestrator.py \
  --task-dir /path/to/cann-bench/bench_lab/multimodal-fusion-bench/level3/fused_rmsnorm_pos_qkv_qknorm \
  --optimize-hint "Implement RMSNorm, positional encoding, QKV projection, and Q/K normalization with Triton Ascend; first verify each case's shape and precision requirements, then choose an implementable fusion approach. Reduce intermediate data movement, and choose a single-kernel or multi-kernel scheme based on measured results."
```

How to write the hint: state the computation steps, precision constraints, and optimization goals clearly. Whether to use a single kernel, and how to tile and schedule, is decided by the agent based on current Triton Ascend capabilities, chip resources, and evaluation evidence.

When the direction description is long, save it as a UTF-8 file and use `--optimize-hint-file <direction>.md` instead of `--optimize-hint "<text>"`; the two are mutually exclusive, with the same injection point and effect.

**After modifying the workflow, emergency re-run with an existing operator**

```bash
python3 orchestrator.py \
  --task-dir /path/to/task \
  --init-impl /path/to/old_work/impl
```

`--init-impl` takes the code you specify as authoritative; it does not require `selection/best.json` to exist and will not automatically re-select the historical best. It is intended for re-running with an already-generated operator after modifying the workflow engineering. On the first pass it goes directly through **Stage4 → Stage5 → Stage6 → the existing subsequent flow**, without calling Stage1, Jev, or Stage2, and without letting cannbot modify the code before the first evaluation.

- Copies `ANALYSIS.md`, `fusion_requirements.en.json`, `device_info.json`, the entire `fusion/`, plus the development materials (scheme, fusion selection, self-test report, structured results, and logs) corresponding to the specified `impl/` for that iteration, into the new `develop/iter0/`. If the code has already been optimized by Stage3, take the development iteration whose code and evidence hashes match; the first-version report cannot substitute for it.
- Does not copy old `eval/`, `build/`, `profile/`, `search/`, history, success/failure experiences, best records, exit counters, or human comments. The new task's performance and history are recorded fresh from this formal evaluation.
- At startup it still checks the actual hardware and runtime environment, but only locally verifies the imported scores, **without calling Jev again**. If required materials are missing or the task or hardware does not match, it stops explicitly and does not silently re-develop. Candidate probabilities and Top N follow the imported library.
- Matched development self-test evidence may be inherited; when code or documentation does not match, it is treated as reference only, not passed off as a passed self-test, and the original admission criteria for the best implementation are not relaxed.
- `init_impl_manifest.json` saves the source, copied files, and hashes; the log lists copied directories and skipped stages. The source directory is not modified, and even same-second starts create distinct new directories.

Cannot be used together with `--work-dir`; after an interruption, resume with `--work-dir <new work dir>`, which keeps emergency mode. An optional `--optimize-hint` is only recorded as a reference for subsequent optimization; the first pass still evaluates the specified implementation first.

**Resuming from a Checkpoint**

If the workflow is interrupted midway (Ctrl+C / timeout / container restart), use `--work-dir` to point to the existing work directory and continue:

```bash
python3 orchestrator.py \
  --task-dir /path/to/task \
  --work-dir work/fused_rmsnorm_pos_qkv_qknorm_20260916_131832
```

The orchestrator continues from the checkpoint recorded in `.state.json`. Existing data such as impl / eval / history is preserved. New tasks record `workflow_target.json` identifying Triton Ascend; work directories with old-framework implementations or states are not directly resumed. Please create a new task to re-implement and re-evaluate; you cannot import old results by only modifying the identity file.

**Controlling the Number of Iterations**

Default is at most 20 iterations (`workflow.max_iterations` in `config.yaml`). You can override with `--max-iter`:

```bash
# First run 1 iteration to verify the environment is fine
python3 orchestrator.py --task-dir /path/to/task --max-iter 1

# Full run of 20 iterations
python3 orchestrator.py --task-dir /path/to/task
```

### How --optimize-hint Is Injected in Different Scenarios

| Scenario | Command | Handling |
|------|------|---------|
| Writing an operator from scratch | `--optimize-hint "fusion direction..."` | Stage1 analyzes feasibility, Stage2 develops along that direction |
| Emergency re-run with existing implementation | `--init-impl ... --optimize-hint "subsequent direction..."` | Skips Stage1/1.5/2; saves the hint for later analysis and optimization, evaluates the specified code first |
| No hint passed | — | No additional direction |

### Exit Conditions

After performance targets are met (all cases speedup ≥ 1.0), the program **does not exit immediately**:

1. The valid score becomes the baseline, then goes through x more valid improvements; when the historical best avg_speedup across the two ends of the window has cumulatively improved by less than 5%, the program exits and hands the best passing snapshot to Stage10.
2. While any case has not passed, it keeps the Stage6 → 7 → 8 → 9 → 3 loop; when y consecutive valid improvements cumulatively total less than 5%, Stage9 is prompted to focus on reviewing candidate schemes. When human consultation is enabled, after the 3rd new valid iteration triggers this condition, the human is consulted first before forming the final modification opinion. If underperforming cases keep improving, the scheme may be retained.
3. x/y are configured by `workflow.semantic_exit.passed_window` / `underperforming_window`, both defaulting to 3; a baseline plus 3 valid scores are required. Switching pass status resets the window; failed iterations do not count, recovery after a regression is not treated as a new improvement, and exactly 5% does not count as stagnation.
4. P0/P1/P2 routing, build/precision/anti-cheating failure routing, and the maximum iteration cap are retained. Stage9's `exit_decision` no longer controls exit, nor does it force a scheme change because of multi-kernel or HBM intermediate read/write.

Stage9 still records, per its original logic, success experiences with improvement ≥5% and failure lessons with regression ≥5%, including the iteration that satisfies the semantic exit.

In the configuration, **x = `passed_window` (scenario 1, all cases passing)**, **y = `underperforming_window` (scenario 2, at least one case not passing)**. They control the valid iteration window; the "nth time entering the scenario" in logs is a trigger count, and the "3rd time" for human consultation is a separate counter — do not confuse the three.

For the differences among the three judgments, the current code state, and follow-up suggestions, see [Performance Decision Rules](docs/workflow_performance_decision_rules.md).

A single-iteration regression still compares this iteration's result with the most recent iteration's `avg_speedup > 0` result, triggering at `≤ -5%` after rounding the percentage to one decimal place. **When all cases in this iteration are ≥1**, Stage9 first writes the regression lesson, then the program restores the best passing code of the same measurement standard, and Stage3 continues modifying from that baseline; **when some case in this iteration is still <1**, only the knowledge is recorded and the code is not automatically restored. Restoration adds no evaluations, does not change original scores or the stagnation window. If the current evaluation fails version binding or snapshot verification, no automatic restoration occurs.

The two stagnation types and regression handling are prominently marked with `=====` in `log/workflow.log` and `log/state_transitions.log`. Stagnation records the window, cumulative improvement, and "which time entering this scenario"; the two scenarios accumulate separately, recovery in the same iteration is not double-counted, and counts are independent of the human consultation counter. Counts are saved in `selection/semantic_events.json`; regression handling is saved in `selection/rollbacks/iterN/record.json`, including before/after performance, best source, Stage9 decision, and knowledge location. The code before actual restoration is additionally saved in `failed_impl/`, and the old evidence binding is saved in `failed_binding.json`; the Stage9/3 prompts make explicit the difference between the regression report and the restored baseline.

### Human Participation and Stage9 Scenario Trimming

The common configuration already enables `workflow.human_review.proactive_enabled` and `consultation_enabled`, which can be disabled separately; when an old custom configuration lacks this section, unattended behavior is preserved. Submit from **another terminal** in the project root directory, without interrupting the current node:

```bash
python tools/human_review.py --work-dir "current work dir" --message "Keep the fusion scheme for now, focus on optimizing the two slow cases"
python tools/human_review.py --work-dir "current work dir" --kind question --message "Why is this scheme recommended?"
python tools/human_review.py --work-dir "current work dir" --status
```

The 3rd stagnation event prompts the location of the `questions_document.md` in the log, containing 2–3 options and recommendation rationale. Reply to the options directly through the same entry; replying "please wait" only extends the original 2-minute deadline by one 10-minute increment, while a formal opinion ends the wait immediately. On timeout, execution continues in the recommended direction and cannot be recorded as human consent. Resumption uses the original deadline, and the same iteration is not double-counted.

Stage9 first processes the human's original words; substantive directions must become human P0 items with opinion numbers; only after program validation are they handed to Stage3. Stage3 records implementation status in the fusion selection rationale and in `develop/<iter>/human_feedback.json`; processed, executed, and not-executed-due-to-exit are recorded separately. Pure questions are answered first and are not automatically treated as instructions.

The original words and status are in `<work>/human_review/inbox/` and `human_review/state.json`; each question/feedback package/evidence version is in `human_review/<iter>/<request number>/`. The role, prompt, and final decision actually received by Stage9 are in `knowledge/stage9/<iter>/<request number>/`, for verifying whether human opinions entered P0.

Stage9 loads tasks and default inputs in six categories: build failure, precision failure, evaluation anomaly, normal optimization, stagnation review, and all-pass. Consultation only generates a question; the final ledger is submitted only after a reply. The all-pass branch still skips Stage7/8, and best records, exit windows, and the ±5% experience rules are retained. See [Human Participation Design](docs/workflow_human_review_design.md) for details.

### Where the Best Fusion Implementation Lives and When It Is Used

It is only recorded after Stage6 completes a valid full performance evaluation. It also requires a formal precision pass, passing self-tests for the given cases and for consecutive invocations, and confirmation that self-test, precision, and performance correspond to the same code version. If materials are missing or the code has changed, it is not admitted, and the original flow of repair and supplementary testing continues.

If an old task never generated the Stage1.5 requirements and scheme directory, the original recovery flow still applies; Stage3 may backfill the actual scheme marked `probability=null`, and it is admitted after self-testing and a new evaluation. A task that has already started Stage1.5 but lacks a scheme library is not admitted under the old-task rules. Re-running in the same iteration must update the selection rationale, self-test, and logs; the program will not re-bind old materials to new code.

| Location (relative to the work directory) | Content |
|---|---|
| `selection/best.json` | Best record under the current evaluation standard: implementation scheme, fusion scheme, raw HAP metrics, avg_speedup/avg_speed, report and code snapshot paths |
| `selection/records/iterN-<fingerprint>/impl/` | Independent copy of that evaluated implementation; later code writing does not overwrite it |
| `selection/records/iterN-<fingerprint>/reports/` | Copies of raw performance reports, performance/precision summaries, and available profiler data |
| `selection/records/iterN-<fingerprint>/evidence/` | Corresponding fusion selection rationale, this iteration's scheme library, self-test report, and logs |
| `selection/state.json` | Valid evaluation index, comparison standard, and window configuration; repeated restoration of the same iteration is not counted twice |
| `selection/current_implementation.json` | Hash binding of the current development artifact and code; not itself a performance record |

**How the best is chosen: after precision and self-test pass, all-cases-passing takes priority; within the same category, compare avg_speedup, and on ties keep the earlier version.** HAP's `performance_score` and per-case `perf_score/t_hw_us/op_times` are saved as-is from the tool, without fabricating new composite values. When no all-passing implementation exists yet, the best available candidate is retained and explicitly marked `best_available`, and it cannot be claimed as passing.

Only scores with the same framework/backend and runtime versions, CANN toolchain fingerprint, hardware, task/case set, evaluation tool, and baseline/timing strategy are compared; the actually executed device_id is recorded, and changes to the fixed `metadata/*.json` baseline files also open a new comparison group, while normal fluctuation of reported values under the same standard does not create a new group. Historical results under different standards must be re-measured and are not imported automatically. Stage3/7/8/9 read the best record and per-case historical trends; on semantic exit or reaching the iteration cap, Stage10 reads the code and report of the best snapshot, and the current `impl/` may keep the last not-yet-evaluated modification.

## Project Structure

```
pypto-pro-workflow/
├── orchestrator.py          ← Main control plane (single entry point)
├── config.yaml              ← Single run configuration (paths, iteration parameters, Jev connection and secrets, tracked by Git)
├── test_cases.csv           ← Test case records
├── roles/                   ← Role definitions for each stage (agent prompt or program execution instructions)
│   ├── n1_stage1_requirements_analysis.md   ← Stage1 requirements analysis (cannbot)
│   ├── n1_stage1.5_jev_fusion_selection.md   ← Stage1.5 fusion scheme selection (Jev, invoked by the program)
│   ├── n1_stage2_first_impl.md              ← Stage2 first implementation (cannbot)
│   ├── n1_stage3_fix_and_optimize.md        ← Stage3 modification and optimization (cannbot)
│   ├── n2_stage4_build.md                   ← Stage4 build and deployment (kerminal)
│   ├── n2_stage5_precision_eval.md          ← Stage5 precision evaluation (kerminal)
│   ├── n2_stage6_perf_eval.md               ← Stage6 performance evaluation instructions (executed directly by the program)
│   ├── n2_stage7_kerminal_profile.md        ← Stage7 profiling (kerminal)
│   ├── n3_stage8_search.md                  ← Stage8 search optimization (hermes)
│   ├── n4_stage9_tech_lead_guide.md         ← Stage9 experience distillation (kerminal/tech_lead)
│   └── n5_stage10_kerminal_report.md        ← Stage10 final report (kerminal)
├── lib/                     ← Utility library
│   ├── framework_target.py  ← Triton Ascend target, backend, and work directory validation
│   ├── fusion_selection.py  ← Stage1.5 Jev scoring, candidate sorting and persistence, scheme library injection
│   ├── fusion_evidence.py   ← Binding of fusion selection, self-test results, and code versions
│   ├── semantic_exit.py     ← Best implementation snapshot, valid performance windows, and per-case trends
│   ├── jev_translation.py   ← English material preparation, fidelity validation, and translation cache
│   ├── cann_env.py          ← CANN environment variable construction (read from config.yaml + source set_env.sh)
│   ├── agent_runner.py      ← Agent CLI invocation + environment injection
│   ├── bench_parser.py      ← cann-bench evaluation result parsing + anti-cheating kernel_csv automatic analysis
│   ├── history_manager.py   ← Cross-iteration memory management (history/ledger/fix_plan/rounds protection)
│   ├── state.py             ← .state.json state management
│   ├── logger.py            ← Logging system (8 logs total: workflow/state/history/N1-N5)
│   └── handoff.py           ← File read/write utilities
├── knowledge/               ← Knowledge base (not shown directly to agents; injected via the orchestrator)
│   ├── anti_cheat_reference.md  ← Real-world error handling table + development self-check checklist
│   ├── fusion_method.md         ← Original description of fusion methods
│   ├── fusion_options.json      ← Detailed candidate methods for Stage1.5 (F1–F10)
│   ├── arch_programming_guide.md ← Triton Ascend programming and hardware boundaries
│   └── profiling_guide.md       ← Profiling data interpretation guide
├── docs/                    ← Design, migration plan, and environment configuration notes
├── examples/
│   ├── triton_ascend_example/   ← Minimal fused operator, build scripts, and NPU self-test
│   └── jev_smoke/               ← Standalone Jev input example
└── skills/triton-profiling-analysis/ ← Ascend Triton performance analysis
```

## Runtime Work Directory

Each run automatically creates `work/<op>_<timestamp>/`:

```
work/exp_20260828_232950/
├── task/                  ← Symlink → cann-bench task directory (read-only)
├── example/               ← Symlink → triton_ascend_cann_example (read-only)
├── workflow_target.json   ← Program-recorded Triton Ascend target identity
├── device_info.json       ← Current chip, runtime, and CANN information
├── impl/                  ← Global: current latest operator code
│   ├── cann_bench/         ← Triton implementation and __init__.py export interface
│   ├── setup.py           ← Package installation configuration
│   └── build.sh           ← Build entry point
├── ANALYSIS.md            ← Global: Stage1 requirements analysis
├── fusion_requirements.en.json ← Stage1's compact English requirements for Jev (≤6000 bytes)
├── WORK_RECORD.md         ← Global: work audit trail
├── FINAL_REPORT.md        ← Global: Stage10 final report
├── .state.json            ← Global: state machine (iteration/stage/history)
├── fusion/                ← Stage1.5 artifacts; Stage2/3/7/8/9 read the scheme library
│   ├── fusion_library.json    ← Top n complete fusion candidates by probability
│   ├── ranking.json           ← All fusion methods and their probabilities
│   ├── jev_request.json       ← Jev request record
│   ├── jev_response.json      ← Jev response record
│   └── translation/          ← Original text, English materials, cache fingerprints, and Kerminal translation records
├── knowledge/
│   ├── history.json           ← Cross-iteration memory (insights/ledger/rounds/suggest_next/fusion_kernel_strategy)
│   ├── proven_patterns.md     ← Success experiences (auto-recorded when performance improves ≥5%; stage9 fills content + program fills numbers)
│   ├── regression_patterns.md ← Failure lessons (auto-recorded when performance regresses ≥5%; same as above)
│   ├── tech_lead_pitfalls.md  ← Decision mistake log (cannbot feedback → tech_lead ruling, ✅misjudged/❌rejected)
│   ├── stage9/iterN/<request number>/ ← Independent handoff directory for each Stage9 call
│   │   ├── request.json       ← Program-recorded iteration, failure reason, performance comparison, and output paths
│   │   ├── history_before.json ← Complete history snapshot before the call
│   │   ├── decision.json      ← Stage9 only submits this iteration's ledger, model conclusions, and experience analysis
│   │   └── commit.json        ← Record after the program completes merging and knowledge persistence
│   └── _rounds_snapshot.json  ← Stage6 score backup (for compatibility with old work directories)
├── selection/                ← Program-maintained development evidence and valid performance records
│   ├── current_implementation.json ← Version binding of code, scheme selection, and self-test
│   ├── best.json             ← Best record, created only after the first valid evaluation
│   ├── state.json            ← Valid evaluation index and window configuration
│   └── records/iterN-<fingerprint>/   ← impl/, reports/, evidence/, and manifest.json
├── operator_iter/             ← impl backup for each iteration (for code traceability, not shown to agents)
│   ├── iter1/
│   └── iter{N}/
├── develop/
│   ├── iter0/
│   │   ├── design_rationale.md     ← Stage2 first-version design rationale
│   │   ├── fusion_library.json     ← This iteration's actual selection and original Jev probabilities
│   │   ├── self_test_report.md     ← Self-test report (including same-shape consecutive invocations with changed data)
│   │   └── self_test_result.json   ← Verifiable result summary, referencing actual test logs
│   ├── iter1/
│   │   ├── design_rationale.md     ← Stage3 iteration-1 optimization design rationale
│   │   ├── fusion_scheme_rationale.md ← Separate explanation of the fusion scheme selection and actual changes
│   │   ├── fusion_library.json     ← This iteration's selection, changes, and new unscored schemes
│   │   ├── self_test_report.md     ← Stage3 iteration-1 self-test report
│   │   └── self_test_result.json   ← Given-case and consecutive-invocation test results and log locations
│   └── iter{N}/...
├── build/
│   ├── iter1/build.log
│   └── iter2/build.log
├── eval/
│   ├── iter1/
│   │   ├── precision_result.json   ← Precision verdict
│   │   ├── precision_binding.json  ← Binding of precision results, code version, and evaluation standard
│   │   ├── precision_reports/      ← Symlinks to cann-bench precision reports
│   │   ├── perf_result.json        ← Performance verdict
│   │   ├── perf_reports/           ← Symlinks to cann-bench performance reports
│   │   └── prof_data/              ← Deep copy: profiler data for each case of this task
│   │       ├── 1/...kernel_details.csv
│   │       ├── 2/...
│   │       └── <case_id>/...
│   └── iter2/
│       ├── ...(same as above)
│       └── prof_data/              ← Independent snapshot per iteration, not overwritten by the next
├── profile/
│   ├── iter1/bottleneck_analysis.md ← Single report per iteration, body marked Stage7 or Stage9
│   └── iter2/bottleneck_analysis.md
├── search/
│   ├── iter1/
│   │   ├── SEARCH_REPORT.md
│   │   └── FIX_DIRECTIVE.md
│   └── iter2/...
└── log/
    ├── workflow.log             ← Global log
    ├── state_transitions.log   ← Dedicated to state transitions
    ├── history.log             ← history change log (complete JSON snapshot per iteration)
    ├── N1.log                  ← cannbot node log
    ├── N2.log                  ← kerminal node log
    ├── N3.log                  ← hermes node log
    ├── N4.log                  ← tech_lead node log
    └── N5.log                  ← Report node log
```

Notes:
- `profile/iterN/bottleneck_analysis.md` keeps a unified filename. When some cases still have not passed, Stage7 writes the report for Stage8/9/3 to read; when all cases pass, the program organizes the overall conclusion and per-case analysis from the JSON that Stage9 has accepted into a report with the same name for Stage3 to read, and it is also saved in the exit iteration. Source, iteration, and Stage9 decision paths are written inside the report. A failed retry is not published; a successful human re-review overwrites the same file. Stage9's direct rewrite of an existing report is reverted. Before actually re-running, Stage7 removes the previous iteration's old report and requires regeneration; when resuming from Stage8/9 the report is retained. If an extra Markdown report for this iteration is discovered, delivery stops rather than silently deleting it. Build, precision, or evaluation anomaly branches do not fabricate performance analysis.
- `develop/iter0/` holds Stage2's first-implementation design rationale and self-test report; `iter{N}/` holds the design rationale and self-test report after each Stage3 optimization iteration
- `eval/iter{N}/prof_data/` is the complete profiling copy corresponding to the report; the program saves it first, then generates `perf_result.json`. `source_csv_dir` and all valid `kernel_csv` entries point to files in this iteration's work directory, so overwriting the original collection directory does not change these references.
- Each evaluation and retry uses an independent `eval/iter{N}/perf_reports/attempt_<count>_<number>/`; the log records the source directory, save directory, and number of associated cases. Batch collections `_batched/` are saved the same way; CSVs that cannot be uniquely mapped to a case are left empty with an explanation, never associated with old files or other cases.
- The design_rationale read by Stage7 is from the previous iteration (fallback searches for the most recent existing one, handling iterations with no output due to build_fail)
- `build_fail`/`precision_fail` only trigger bug fixes and produce no design_rationale (the design did not change)
- Each Stage3 run (including fix iterations) updates its own fusion selection rationale, scheme library, and self-test materials, noting which schemes were retained or actually changed.

## One Iteration Flow (using the exp operator as an example)

```
Start → Stage1 (cannbot analyzes task) → Stage1.5 (Jev evaluates fusion schemes, program selects top n)
     → Stage2 (cannbot writes Triton Ascend code per the highest-probability scheme)
     → Enter the iteration loop:
       Stage4: kerminal packages, installs the wheel, and verifies import outside the source directory
              → FAILED → stage9 (tech_lead summary) → Stage3 (cannbot fixes) → back to Stage4
              → SUCCESS ↓
       Stage5: kerminal runs cann-bench --no-perf precision evaluation
              → precision_fail → stage9 (tech_lead summary) → Stage3 (cannbot fixes precision) → back to Stage4
              → true ↓
       Stage6: orchestrator directly executes cann-bench performance evaluation (not via an agent)
              → anti-cheating zero score → stage9 (tech_lead anti-cheating analysis) → Stage3 → back to Stage4
              → all cases speedup≥1.0 → stage9 (evidence review and existing knowledge accumulation)
                   → x window satisfied → Stage10 (best evaluated snapshot) → end
                   → not yet satisfied → Stage3 (continue optimizing per bottleneck) → back to Stage4
              → some case <1.0 ↓
       Stage7: kerminal analyzes profiling data and locates bottlenecks
       Stage8: hermes searches for optimization schemes, outputs FIX_DIRECTIVE.md
       Stage9: kerminal (tech_lead) submits this iteration's decision.json; after program validation, history.json is updated
       Stage3: cannbot modifies code per FIX_DIRECTIVE + insights + fusion_kernel_strategy
     → Back to Stage4, next iteration
```

The above describes the process, not measured Triton results. Full JIT, precision, and performance results on real NPU hardware require configuring the environment and running a new task to verify.

## Cross-Iteration Memory (history mechanism)

After each iteration ends, the Stage9 tech_lead reads this iteration's results and previous iterations' designs, and writes only to this iteration's `knowledge/stage9/<iter>/<request number>/decision.json`. After the program verifies the request number, iteration, and fields, it merges the `ledger_entry` into this iteration's ledger, fully preserving other iterations' entries and program-filled scores; Stage9 does not directly overwrite `history.json`. In the next iteration, Stage3's cannbot still reads the existing history summary and P0/P1/P2.

Each call retains `request.json`, the pre-call `history_before.json`, the raw `decision.json`, and the `commit.json` after a successful submission. When output is missing, invalid, belongs to another iteration, or omits this iteration's required experiences/rulings, the program aggregates independently checkable errors and lets Stage9 make at most two corrections within the same iteration, saved in `retry1/` and `retry2/` respectively; the first attempt plus corrections total three attempts, and only if the third still fails does it stop — without re-evaluating, without adding iterations, and without passing old suggestions to Stage3. The current ledger's `stage9_decision_path` points to the actually adopted decision.

**Main fields of history.json**:

| Field | Who writes | How updated | Who reads |
|------|------|---------|--------|
| `rounds` | Program (automatically after stage6) | Appended each iteration | tech_lead |
| `ledger` | Program writes hard data + merges tech_lead's directions/file scope for this iteration | Appended or updated per iteration, other iterations preserved | tech_lead + cannbot |
| `insights` | tech_lead (Stage9) | Fully rewritten each iteration | cannbot (most critical) |
| `bottleneck_now` | tech_lead (Stage9) | Replaced each iteration | cannbot |
| `suggest_next` | tech_lead (Stage9) | Replaced each iteration (**P0/P1/P2 priority list**) | cannbot (highest execution priority) |
| `worst_cases_tracker` | Program merges Stage9's case analysis | Updated per case, other history preserved | cannbot |
| `fusion_kernel_strategy` | tech_lead (Stage9) | **Appended cumulatively each iteration**, old entries never overwritten | cannbot + tech_lead |

**suggest_next is a structured task list**: each entry specifies the goal, evidence, target cases and their actual implementation mapping, per-file modification steps, and acceptance criteria; files forbidden to modify are recorded uniformly in this iteration's ledger, and modifiable files are aggregated by the program from the steps. See [Stage9 Task Plan Description](docs/workflow_stage9_plan_v2.md) for a complete example.

P0 covers correctness issues, key directions, or human guidance that must be addressed; P1 covers evidence-backed improvements; P2 covers optimizations worth trying. Fusion changes are judged by evidence and are not fixed ahead of all other items. The rendered result cannbot receives: `🔴[P0]` > `🟡[P1]` > `🟢[P2]`.

**fusion_kernel_strategy example** (cumulative tracking of fusion scheme evolution):
```json
"fusion_kernel_strategy": [
  {"iter": 1, "direction": "Two-stage kernels communicating via HBM", "evidence": "Benefit must be judged against the full evaluation", "status": "🔄pending verification"},
  {"iter": 3, "direction": "On-chip direct transfer", "evidence": "Precision passed and all cases measured at target", "status": "✅effective"}
]
```

**Historical experience cannbot receives in Stage3 (ordered by priority)**:
1. `★ This iteration's modification instructions (P0/P1/P2)` — prioritize correctness and underperforming cases, then overall performance; fusion changes are ordered by actual evidence
2. `Historical experience knowledge base (insights)` — verify the hardware, shape, and implementation conditions of already-rejected directions to avoid repeated failures
3. `Current performance bottleneck` — optimizations must target this bottleneck
4. `Fusion operator strategy tracking` — reuse effective schemes within their applicable conditions; when new evidence conflicts with history, check back
5. `Verified success experiences` — continue effective directions (knowledge/proven_patterns.md)
6. `Verified failure lessons` — avoid repeating failures under the same conditions (knowledge/regression_patterns.md)
7. `Decision mistake log` — tech_lead's past misjudgments; avoid them when writing code (knowledge/tech_lead_pitfalls.md)
8. `Slowest case tracking` — view per-case trends; hardware limits must be backed by evidence from the current environment
9. `Hypothesis tracking ledger (ledger)` — do not repeat directions already marked regression/no_change
10. `Performance trend (rounds)` — avg_speedup changes per iteration

## Knowledge Accumulation (proven_patterns / regression_patterns)

After stage6, the program automatically computes the performance diff (this iteration vs the previous iteration's valid data) and triggers knowledge accumulation based on the magnitude of change:

| Condition | Trigger | File | Who writes content | Who writes numbers |
|------|------|------|---------|---------|
| avg_speedup improves ≥5% | Record success experience | `knowledge/proven_patterns.md` | stage9 tech_lead fills what_changed / why_it_worked | Program fills speedup_before/after/delta_pct |
| avg_speedup regresses ≤-5% | Record failure lesson | `knowledge/regression_patterns.md` | stage9 tech_lead fills what_changed / why_it_failed | Program fills speedup_before/after/delta_pct |
| -5% < delta < +5% | Not recorded | — | — | — |

Flow: the program determines improvement/regression → injects the diff and evidence paths → Stage9 fills `proven_pattern` or `regression_pattern` in this iteration's `decision.json` → the program reads the same-named field directly, adds the real numbers, and writes the corresponding Markdown. Repeated review in the same iteration updates the original entry instead of appending a duplicate. When the threshold is reached but the analysis is missing, Stage9 stops rather than writing a placeholder experience that says "not filled in".

Detailed records include: actual changes, reasons for success/failure, affected cases and analysis, code/profiling evidence, applicable hardware and shape conditions, conclusion limitations, and follow-up reuse or correction suggestions. The program additionally attaches the before/after iterations, average speedup and percentage change, score changes for alignable cases, report/scheme evidence paths, and the original decision location. Unproven attributions should be marked as pending verification.

`history.json` holds cross-iteration conclusions and the ledger; the two Markdown files above hold detailed improvement/regression experiences; each raw analysis remains in its own `decision.json`. Old history formats remain readable, and old placeholder content is never automatically rewritten as verified experience.

These two knowledge files are injected into the prompts of **stage2/3/7/8/9**, so every role writing code or doing analysis can see them.

## Subordinate Feedback + Mistake Log (question.md / tech_lead_pitfalls.md)

A "subordinate feedback → superior ruling → accumulation → feedback into prompts" closed loop that prevents tech_lead from repeatedly giving wrong advice.

**Trigger scenario**: while implementing a tech_lead suggestion, stage3's cannbot discovers that a suggestion is genuinely infeasible at the hardware/framework level (with hard evidence).

**Complete closed loop**:
```
iter N:  stage9 (gives suggestion) → stage3 (implements it, finds the suggestion is wrong)
                              ↓ strict judgment + hard evidence
                            write develop/iterN/question.md
iter N+1: stage9 reads iterN/question.md + corresponding history opinion
            ↓ ruling
            ├─ ✅confirmed (misjudgment confirmed) → admit the mistake, give the correct approach
            └─ ❌rejected (rejected)      → cannbot misunderstood; explain the correct understanding
          Program: ruling written back to question.md + appended to the mistake log
          Afterwards stage9 must read the mistake log before giving advice, so mistakes are not repeated
```

**Strict gatekeeping** (preventing cannbot from using question.md to shirk work):
- Must have genuinely attempted it + have hard evidence (compile errors / logs / profiler / official documentation limits)
- Must be objectively infeasible, not "I don't think it's necessary"
- Before raising a question, check the mistake log first; already-recorded items are not raised again
- Regardless of whether the suggestion was right or wrong, **the current iteration's task must still be completed with an alternative scheme** — question.md is accompanying feedback, not a refusal to execute

**Mistake log file** `knowledge/tech_lead_pitfalls.md`:
| verdict | Meaning | Recorded content |
|---------|------|---------|
| ✅confirmed | tech_lead indeed gave wrong advice | Misjudgment reason + correct approach |
| ❌rejected | cannbot misunderstood | Rejection reason + correct understanding |

- Who writes: after tech_lead's ruling it outputs a `pitfall` field → the program appends (cumulative, never overwritten)
- Injection scope: **stage2/3/7/9** (code writers, analyzers, and decision makers can all see it); if the file does not exist, empty content is returned without affecting the flow
- Fallback: if early exit leaves a question.md never ruled on, stage10 scans for it and mentions it in the report

## Post-stage6 Routing Decisions

After stage6 performance evaluation completes, the program makes 3 decisions in the following order (**the order cannot be swapped**):

```
stage6 done → program writes rounds/ledger → program computes perf_diff
  ↓
  ① Anti-cheating? (score_error_code exists or avg_speedup=0)
  │  YES → stage9 (anti-cheating analysis) → stage3 → back to stage4
  │         Log: [Decision] Anti-cheating triggered: no_npu_kernel_detected
  │
  ② Performance passing? (all cases speedup ≥ 1.0)
  │  YES → record valid implementation, compute x window → stage9 (review and knowledge accumulation)
  │         ├─ cumulative improvement <5% and window satisfied → stage10 (best snapshot) → end
  │         └─ not yet satisfied → stage3 (perf_pass_optimize) → back to stage4
  │
  ③ Normal case: performance not passing
     → record valid implementation, compute y window (only prompts re-review, does not force a scheme change)
     → stage7 (profiling) → stage8 (search) → stage9 (perf_optimize) → stage3 → back to stage4
       Log: [Decision] Performance not passing: avg_speedup=0.5, perf_pass=False
```

perf_diff data (computed once by the program, shared by stage7/8/9):
- Improvement ≥5%: injects `📊 Performance improvement detected` + per-case comparison + "please summarize the success experience"
- Regression ≤-5%: injects `⚠️ Performance regression detected` + per-case comparison + "please analyze the regression cause"
- Within ±5%: no diff text is injected

## state_transitions.log Log Format

Each state transition carries a reason; key decision points carry `[Decision]` / `[Knowledge accumulation]` tags:

```
iter1_stage4 → iter1_stage5 | Build succeeded, starting precision evaluation
[Decision] Precision failed: passed=18/24
iter1_stage5 → iter1_stage9 | tech_lead experience summary (precision_fail)
iter1_stage9 → iter1_stage3 | Fall back to code modification (precision_fail)

After iter2_stage6:
[Decision] Performance improvement: avg_speedup 0.5→1.5 (+200%), iter1→iter2
[Decision] Performance not passing: avg_speedup=1.5, perf_pass=False → stage7→8→9→3
[Knowledge accumulation] Success experience written to proven_patterns.md: +200%

After iter5_stage6:
[Decision] Performance regression: avg_speedup 3.0→2.5 (-16.7%), iter4→iter5
[Knowledge accumulation] Regression lesson written to regression_patterns.md: -16.7%

After iter9_stage6:
iter9_stage9 → iter9_stage10 | Program semantic exit, delivering the best passing implementation
```
```bash
tail -10 work/exp_xxx/log/N4.log   # tech_lead
```

## The Three Agents

| Agent | Node | Model | CLI path | Config file |
|-------|------|------|---------|---------|
| cannbot | N1 (writes code) | GLM-5.3-Flash | `config.yaml → agents.cannbot.cli` | `~/.config/opencode/opencode.jsonc` |
| kerminal | N2 (build and evaluation) | kernelcat1.0 | `config.yaml → agents.kerminal.cli` | `~/.kerminal/config.toml` |
| hermes | N3 (search) | GLM-5.3-Flash | `config.yaml → agents.hermes.cli` | `~/.hermes/config.yaml` + `~/.hermes/.env` |

### Long Prompt Delivery and Startup Failure Recovery

On Linux, `Argument list too long` can come from a single argument, the total argument size, or environment variables — it cannot be judged only by the ~2 MB total. All three Agents no longer pass the full prompt as a startup argument; the stage order, role body, history, and human P0 items remain unchanged.

| Agent | How the full prompt is passed in | Interface basis |
|-------|------------------|----------|
| CANNBot (Stage1/2/3) | `run` reads the full text from stdin fed from a file | [Official Linux 1.1.2 release package](https://registry.npmjs.org/cannbot-linux-x64/1.1.2) |
| Kerminal (Stage4/5/7/9/10) | `-a never exec --skip-git-repo-check -C <cwd> -` reads the full text from stdin and returns the actual exit code | Verified with the local 0.8.12 `help exec` and argument parsing of the full command; no terminal emulation needed |
| Hermes (Stage8) | The same Python environment first starts `lib/hermes_prompt.py`, which reads the file and hands the full text to the original `-z` entry inside the process | [Official 0.20.4 entry point](https://github.com/NousResearch/hermes-agent/blob/e624e9fde561e1add9388384012b295fde669ade/pyproject.toml#L372); that version's `-z` does not read stdin |

Each call retains `work/<task>/log/prompts/<agent>_<role filename>_<unique number>.txt`, containing the complete role, task, and common constraints. The log prints the delivery method, file path, and UTF-8 byte count; retries in the same iteration keep independent files, without truncation or compression. If the system still reports E2BIG, the log additionally reports the total byte count of arguments and environment and the largest single item, without printing environment values. The model's own context capacity remains a separate limit.

Hermes must use the official pip Python entry pointed to by `agents.hermes.cli`, supporting direct entry points or symlinks; keep the original venv interpreter and the `-t web,file --yolo --in <cwd>` arguments. Unknown binaries or custom shell launchers produce a clear error, with no guessing and no fallback to long arguments. If Hermes itself enables a host-managed container or a secondary launch and tries to put the same long prompt into the system command line again, the bridge stops explicitly; this project uses a Python CLI installed directly in the target environment. CLI interfaces have been verified; the actual model, tool permissions, and NPU still need joint testing in the deployment environment.

Other entry points have been verified: Jev translation goes through stdin JSON-RPC, Jev scoring goes through the SDK request body, and evaluation and hardware probing only pass paths, fixed options, or short scripts. When providing long text manually, use `--optimize-hint-file <direction>.md` instead of `--optimize-hint "<text>"` (mutually exclusive, same subsequent effect); human suggestions use `python3 tools/human_review.py --work-dir <work> --file <opinion>.md`.

If an older version was interrupted at Stage3 startup for this reason, after syncing this iteration's runner and bridge files, just start with the original `--task-dir` and original `--work-dir` to return to the same Stage3; if the original command had `--config`, keep it too. No new work directory is needed, and completed evaluations or already-written experiences are not redone or duplicated. Before startup, the environment and recovery materials are still checked under the existing rules.

## New Environment Installation Guide

### 1. cannbot (Node.js)

```bash
# Prerequisite: Node.js >= 18
npm install -g cannbot @cannbot-ai/install-helper

# Verify
cannbot --version   # should show 1.1.2+

# Configure the API key
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/opencode.jsonc << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM (ZhipuAI)",
      "options": {
        "baseURL": "https://open.bigmodel.cn/api/paas/v4",
        "apiKey": "<your ZhipuAI API key>"
      },
      "models": {
        "glm-5.3-flash": { "name": "GLM-5.3-Flash" }
      }
    }
  },
  "model": "glm/glm-5.3-flash"
}
EOF
```

### 2. kerminal (binary)

```bash
# Install (from the official install script; downloads to ~/.local/bin/kerminal)
curl -fsSL https://kerminal.cn/install.sh | bash

# Verify
kerminal --version

# Configure
mkdir -p ~/.kerminal
cat > ~/.kerminal/config.toml << 'EOF'
model = "kernelcat1.0"
model_provider = "autokernel"
thinking_enabled = true
thinking_budget_tokens = 10000
show_thinking = true
show_raw_agent_reasoning = true

[features]
skills = true

[model_providers.autokernel]
name = "autokernel"
wire_api = "messages"
experimental_bearer_token = "<your autokernel token>"
EOF
```

### 3. hermes (Python venv)

```bash
# Install (official one-click script, includes a Python 3.11 venv)
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# Or install from source (if hermes-agent-main/ is available)
cd hermes-agent-main && pip install -e .

# Verify
hermes status

# If /usr/local/bin/hermes reports ModuleNotFoundError, create a symlink:
ln -sf ~/.hermes/venvs/hermes-dev/bin/hermes /usr/local/bin/hermes

# Configure the model
cat > ~/.hermes/config.yaml << 'EOF'
model:
  default: glm/glm-5.3-flash
EOF

# Configure the API key + search engine
cat > ~/.hermes/.env << 'EOF'
GLM_API_KEY=<your ZhipuAI API key>
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
TAVILY_API_KEY=<your tavily key, optional; if absent, use the free ddgs>
EOF
```

### 4. Workflow Configuration

After installing the three Agents, modify the paths per "Step 1: Configure config.yaml" in the Running Guide, then confirm all components work per "Step 2: Verify the Environment".
