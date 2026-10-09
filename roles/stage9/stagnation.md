## Current Scenario: Not All Cases Meet Target, Stagnation Review

The program has confirmed that the best average speedup across y valid performance iterations has cumulatively improved by less than 5%. First read Stage7's `profile/<iter>/bottleneck_analysis.md`, Stage8's `search/<iter>/FIX_DIRECTIVE.md`, the bound scheme rationale, and relevant history, then compare the candidate conditions and probabilities in `fusion/fusion_library.json`.

Combined with the average, HAP, and failing cases in `eval/<iter>/perf_result.json`, the window and case trends in `selection/state.json`, and `selection/best.json` with its `selection/records/<iter>-<fingerprint>/manifest.json` snapshot, judge whether it is a local problem or a structural limitation. When needed, check that version's code, performance sources, and `eval/<iter>/prof_data/`; do not explain old scores with the current code.

When the average improvement is small but slow cases keep approaching 1, the current direction may continue. Keeping, locally optimizing, or changing all require evidence, target cases, and next-step verification; stagnation is not the same as a single-round regression of ≥5%, and the two knowledge record types must not be confused.

The program determines the stagnation window and routing. Submit a review decision with evidence-backed next-step tasks using the decision contract below.
