## Conditional Task: Record This Round's Performance Experience in Detail

Base it solely on this round's comparable scores injected by the program: a single-round average speedup gain of ≥5% requires filling `proven_pattern`; a regression of ≥5% requires filling `regression_pattern`. Omit untriggered fields; y-window stagnation does not substitute for the single-round condition.

The object contains:

- `what_changed`: the specific code changes, citing the bound selection rationale and checking the code.
- For successes `why_it_worked`, for regressions `why_it_failed`: explain the cause with the corresponding computation, movement, or scheduling evidence; clearly mark assumptions.
- `fusion_related`: boolean, whether it relates to the fusion scheme.
- `case_analysis`: non-empty list; each item fills non-empty text `case_id`, `observation`, `explanation`, covering both benefited and harmed cases. `explanation` separately states why that case changed; do not write only the symptom, and do not substitute the overall `why_it_worked/why_it_failed` or `ledger_entry.case_analysis`. When the cause is uncertain, state it as speculated or to-be-verified with a verification method; do not fabricate conclusions.
- `evidence`: evidence paths and that path's case/field/actual measurement; replace `<iter>` with the real bound round.
- `applicability`: a non-empty paragraph stating the applicable hardware, shapes, parameters, unverified scope, and limitations; do not write it as an object or array.
- `next_action`: non-empty text stating later reuse, avoidance, or comparative verification methods.

This round's `decision_schema.json` and `decision_template.json` already list the triggered experience objects and their complete fields; the template's empty values must be filled. Before submitting, also check both the ledger's and the experience's `case_analysis` — they serve different purposes and cannot substitute for each other. When correcting in the same round, process each entry of `validation_error.json`'s `errors` list, not just the first, and do not bypass checks by deleting affected cases.

The program directly consumes the common fields, supplements the before/after averages, differences, rounds, full-case numeric diffs, and performance sources, and writes to `knowledge/proven_patterns.md` or `knowledge/regression_patterns.md`. Do not output `_pending_*` or placeholder reasons; missing this round's required experience stops Stage9. Same-round follow-ups update the original experience, and the original decisions are each archived.

The framework and chip `environment` are filled automatically by the program from this round's performance report; you need not fill it. `applicability` still requires analyzing hardware, shape, and other applicability conditions; before reusing old experience, verify its environment, and do not treat content not noted in old records as valid for the current environment.
