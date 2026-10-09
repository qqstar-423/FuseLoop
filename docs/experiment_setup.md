# Experimental setup and reproduction

This guide configures the concrete runtime, agent, and evaluator integrations used by this release. The paper's experiments run on an **Ascend 910B3 NPU** with **CANN 9.1.0, Triton 3.2.0, and triton-ascend 3.2.1**. Generation and search use GLM5.3-Flash; analysis and review use kernelCAT. The method and its three modules are described in the [README](../README.md).

## Runtime and evaluation environment

Prepare a Linux machine with:

- Python 3.10 or later, a compatible NPU driver, CANN Toolkit, `torch`, `torch_npu`, and Triton Ascend. Select matching Python and runtime builds for the device; use the versions above when reproducing the paper's toolchain.
- A configured cann-bench checkout with the task's correctness references, cases, baseline metadata, and `examples/triton_ascend_cann_example/`.
- Node.js 20 or later, the agent CLIs below, and their configured model providers.
- Jev service credentials for fusion planning.

Install the workflow dependencies into the Python environment used to launch `orchestrator.py`:

```bash
python3 -m pip install "typesafe-sdk==0.7.0" "PyYAML==6.0.3"
```

A task directory provides `desc.md`, `proto.yaml`, `cases.yaml`, and `golden.py`. Preserve these evaluation inputs. The generated implementation exports the task interface through a `cann_bench` package; its `build.sh` produces the wheel installed for evaluation. The first device invocation triggers Triton JIT compilation.

The run's `task/` links to the supplied task directory. Its `example/` links to `examples/triton_ascend_cann_example/` in the evaluator checkout. The repository also provides a [minimal fused operator implementation](../examples/triton_ascend_example/README.md) with build and NPU self-test commands.

## Agent integrations

Install the recorded CLI versions and the remaining agent distributions:

```bash
npm install -g cannbot@1.1.2 @cannbot-ai/install-helper@1.2.0
curl -fsSL https://kerminal.cn/install.sh | bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

The current generation integration invokes `cannbot run` and uses the OpenCode-compatible configuration layout. Set the model and provider in the corresponding agent configuration:

| Role | Executable | Configuration | Experimental model |
|---|---|---|---|
| Generation and revision | `cannbot` | `~/.config/opencode/opencode.jsonc` | GLM5.3-Flash |
| Build, precision checks, analysis, review, and reporting | `kerminal` | `~/.kerminal/config.toml` | kernelCAT |
| Optimization research | `hermes` | `~/.hermes/config.yaml`, `~/.hermes/.env` | GLM5.3-Flash |

The configured Hermes executable must resolve to its Python entry point in its own environment. FuseLoop retains full prompts under each run's `log/prompts/` and delivers them through files and stdin or the Hermes Python bridge.

### Triton development resources

Run `install-helper` and select **Install Plugin → OpenCode → global → Triton operator development → Confirm**. This installs the eight Triton skills and the plugin's `AGENTS.md` under `~/.config/opencode/`. The OpenCode choice selects the installation layout; the configured generation executable remains `cannbot`.

Verify discovery using the workflow's user, environment, and working directory. Set `CANNBOT_BIN` to the value of `agents.cannbot.cli`, and capture the complete JSON in a regular file before parsing:

```bash
CANNBOT_BIN=cannbot
CANNBOT_CHECK=$(mktemp)
"$CANNBOT_BIN" debug skill > "$CANNBOT_CHECK"
python3 - "$CANNBOT_CHECK" <<'PY'
import json, sys
from pathlib import Path
available = {s["name"]: s for s in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))}
expected = """triton-task-extractor triton-op-designer triton-op-coding
triton-op-verifier triton-latency-optimizer triton-precision-debug
npu-arch triton-simulator-optimizer""".split()
missing = [n for n in expected if not Path(available.get(n, {}).get("location", "")).is_file()]
print(f"{8 - len(missing)}/8 Triton skills readable", missing)
raise SystemExit(bool(missing))
PY
```

Check instructions and templates separately. For the global installation, add the template link if its entry is absent, then verify both resources:

```bash
CANNBOT_PLUGIN="$HOME/.cannbot/repo/plugins-official/triton-op-generator"
CANNBOT_CONFIG="$HOME/.config/opencode"
test -r "$CANNBOT_PLUGIN/template/convolution.md"
if [ ! -e "$CANNBOT_CONFIG/template" ] && [ ! -L "$CANNBOT_CONFIG/template" ]; then
  ln -s "$CANNBOT_PLUGIN/template" "$CANNBOT_CONFIG/template"
fi
readlink -f "$CANNBOT_CONFIG/AGENTS.md"
test -r "$CANNBOT_CONFIG/AGENTS.md" && test -r "$CANNBOT_CONFIG/template/convolution.md"
```

Confirm `AGENTS.md` resolves to the installed Triton plugin. Resolve template references inside its skills, including `.claude/template/...` paths, to readable files in the installed template directory. Apply these resources within FuseLoop's stage instructions and the task's package interface. The harness controls iteration, the reviewer directs changes, and formal evaluation supplies the results.

## Configuration

Edit [config.yaml](../config.yaml) for the target machine:

| Setting | Required value or purpose |
|---|---|
| `agents.cannbot.cli`, `agents.kerminal.cli`, `agents.hermes.cli` | Commands on `PATH` or absolute executable paths; Hermes resolves to its virtual environment's Python entry point |
| `paths.cannbench_repo` | Absolute path to the configured cann-bench checkout |
| `cann.toolkit_path` | Installed Toolkit, defaulting to `/usr/local/Ascend/ascend-toolkit/latest` |
| `cann.driver_path` | Installed driver, defaulting to `/usr/local/Ascend/driver` |
| `cann.arch` | Host architecture, such as `aarch64-linux` or `x86_64-linux` |
| `cann.node_bin` | Optional Node.js binary directory; an empty value uses `PATH` |
| `hardware.device_id` | Logical device index in the visible NPU mapping |
| `jev` | Service endpoint, model, credential lookup, timeouts, and request-size limits |
| `fusion_selection.top_n` | Candidate count, from 1 to 10; default `3` |
| `workflow.max_iterations` | Maximum iteration count; default `20` |
| `workflow.semantic_exit` | Valid-result windows for stopping and fusion-plan review |

Jev reads `TYPESAFE_API_KEY` first, then `jev.api_key`. Keep credentials in the local environment when sharing configuration. Planning requirements, catalog text, and submitted evidence must be English. New runs archive validated inputs in `fusion/english_inputs.json`. Implementation import verifies the archived inputs against the scoring request and source fingerprints; earlier English archives remain readable after the same consistency checks. Use `--config /path/to/config.yaml` to select a machine-specific configuration.

Startup reads the device model and compute resources through `torch_npu`, queries on-chip capacities through CANN's TBE interface, verifies Triton's active `npu` backend, and writes runtime information to `device_info.json`. The current integration interprets the compute units and memory hierarchy of this runtime family. Porting to a different NPU architecture requires corresponding runtime, detection, evaluation, profiling, and programming-knowledge integrations.

Verify the installed components:

```bash
npu-smi info
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"
python3 -c "from tbe.common.platform import get_soc_spec; print('CANN TBE ready')"
python3 -c "from typesafe_sdk import TypeSafeClient; import yaml; print('Planning dependencies ready')"
cannbot --version
kerminal --version
hermes status
```

## Execution and measurement

Run from the repository root after selecting the task and machine configuration:

```bash
python3 orchestrator.py --task-dir /path/to/task --config /path/to/config.yaml
```

[Run, resume, and implementation import commands](../README.md#run-and-resume) describe the remaining entry points.

The harness requests `--perf-metric-strategy kernel_details`, two warmup calls, and three repeats. It preserves the evaluator's reference time, candidate time, speedups, and HAP values. The paper's timing procedure sums repeated calls to each kernel within a repetition, takes the median across repetitions for each kernel name, and sums those medians. The timing scope covers the device kernels producing the outputs; compilation, initialization, host overhead, and gaps between kernels are outside that metric.

For comparable measurements, use the same task cases, reference implementation, baseline files, runtime, hardware, and timing strategy, and keep the device free of other workloads. The paper alternates evaluation between systems. Its reported task-level speedups use a geometric mean; the workflow's iteration and selection logic consumes the evaluator-provided `avg_speedup` field.

Retrieve the final evaluated implementation through `selection/best.json` in the work directory, together with its archived correctness, performance, and development evidence. The active `impl/` directory holds the current development code.
