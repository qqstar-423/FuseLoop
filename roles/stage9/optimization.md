## Current Scenario: Not All Cases Meet Target, Ordinary Performance Optimization

First read `profile/<iter>/bottleneck_analysis.md` (Stage7's bottleneck conclusions) and `search/<iter>/FIX_DIRECTIVE.md` (Stage8's changes and their basis), then check against the actual scheme, last round's suggestions, and relevant history. Only when a specific attribution is in doubt should you check the raw data in `eval/<iter>/prof_data/`; do not redo the whole profiling analysis.

Use `eval/<iter>/perf_result.json` to check the average, HAP, and the at most 6 slowest valid cases actually listed, combined with the window and case trends in `selection/state.json` and the version manifest in `selection/best.json`, to see which changes were effective. The best snapshot is in `selection/records/<iter>-<fingerprint>/manifest.json`; read code and reports per the manifest to avoid mixing versions.

For the few slow cases, first look at local causes such as shape tiling, tail blocks, and fixed overhead; only with evidence of a structural bottleneck should you compare the candidate conditions and probabilities in `fusion/fusion_library.json`. Decide whether to keep, locally optimize, or change direction next round, stating the target cases and verification method. Do not force a change of fusion scheme merely because it is not a single kernel.

Maintain this round's actual fusion attempts and cross-round conclusions; record experience per the program's single-round up/down conditions.
