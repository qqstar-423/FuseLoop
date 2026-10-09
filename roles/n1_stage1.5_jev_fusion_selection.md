# Stage1.5: Jev Fusion Scheme Selection

**Stage order: [Stage1 Requirements Analysis](n1_stage1_requirements_analysis.md) → Stage1.5 Fusion Scheme Selection → [Stage2 First Implementation](n1_stage2_first_impl.md).**

The current implementation target is Triton / triton-ascend (Ascend NPU); scoring must take into account the actual expressive power of this backend, and must not treat other frameworks or GPU-specific capabilities as already available.

This file describes the responsibilities, inputs, and outputs of this stage for human readers; it is not directly sent to Jev as a request. Actual execution is performed by `orchestrator.py` calling `lib/fusion_selection.py`; everything sent to Jev is in English.

## Inputs

Path bases are noted in the table; the program reads the files and assembles the English content — Jev does not directly access these local paths.

| Relative path | Relative directory | Purpose and how to read |
|---|---|---|
| `knowledge/fusion_method.md` | project root | Original text of fusion methods; understand each method's data flow, applicability conditions, and limitations. |
| `knowledge/fusion_options.json` | project root | Detailed candidate catalog, containing 10 categories F1–F10 with 24 variants; compare methods and their constraints by fixed ID, and do not treat variants as separate scoring items. |
| `fusion_requirements.en.json` | `<work>` | Fusion requirements summary from Stage1; judge applicability by combining semantics, case characteristics, precision, and implementation constraints. |
| `device_info.json` | `<work>` | Real parameters of the current hardware; use architecture, core count, and storage capacity to verify method prerequisites. |
| `ANALYSIS.md` | `<work>` | Full requirements analysis; the program reads its hash for input consistency checking, the body is not sent directly to Jev, and scoring uses the requirements summary above. |

## Responsibilities

1. The program validates the supplied English material and JSON values without rewriting scheme IDs, numbers, text, or structure, then checks the complete request size. Non-English inputs are rejected before scoring.
2. The program assembles the `model / state / questions` request; Jev gives an independent `noul` applicability probability for each major category. The 24 variants participate in the judgment as scheme details and are not scored separately.
3. The program verifies that all schemes have valid probabilities and sorts them in descending order of probability; ties keep the catalog order, and the top n are taken to form the **JSON fusion operator library**.

`n` is configured by `fusion_selection.top_n` in `config.yaml`, default 3. The probability expresses an initial judgment that "this direction is feasible and worth exploring"; it is not required to sum to 1, and does not represent measured performance or a unique optimal scheme. This stage does not generate operator code or self-test reports.

## Outputs and Routing

| Output | Content and purpose |
|---|---|
| `<work>/fusion/english_inputs.json` | Unchanged English source, catalog, requirements, and hardware used to build the request |
| `<work>/fusion/jev_request.json`, `jev_response.json` | Full English request and Jev's raw response, for traceability |
| `<work>/fusion/ranking.json` | Probabilities and ranking of all 10 major categories |
| `<work>/fusion/fusion_library.json` | The top n complete candidates, each containing `rank`, `probability`, `method`, passed to Stage2, 3, 7, 8, 9 |

After Stage1.5 succeeds, proceed to Stage2, using the highest-probability implementable scheme as the direction for the first version; Stage3, 7, 8, 9 refer to the scheme library combined with their own responsibilities and actual evaluation evidence. The output library retains the scheme details from the supplied English catalog.

Stop on input validation, request size, or scoring failure — do not enter Stage2; when inputs and scoring protocol are consistent, validated results can be reused, and if only n changes, reselect directly. A normal scoring run refreshes an older protocol's cache. Import and recovery can retain verified English v2 archives without contacting Jev, after checking the original request, response, and source fingerprints. This stage reuses the existing downstream routing, P0/P1/P2, and exit logic.
