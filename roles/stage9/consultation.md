## Current Stage: Generate Consultation Questions Only

The program has confirmed the 3rd stagnation trigger, requiring a human for directional judgment. This call does not submit a final decision, does not fill in the final ledger or P0, does not write knowledge experience, does not dispatch Stage3, and does not wait on its own.

The sole output is the prompt-specified `human_review/<iter>/<request number>/question.json`. After the program validates it, it renders the "Question Document.md" in the same directory, notifies, and handles the 2-minute wait; extension, deadline, and replies are handled by the program. Write the questions clearly, short and readable; the human may propose other directions.

JSON fields:

Except for `options`, which is a list of objects, all fields are non-empty strings (including `attempts`, `evidence`).

- `request_id`: copy this consultation's request number verbatim.
- `question`: the clear question you want the human to judge.
- `difficulty`: the current difficulty, which cases fail to meet the target, and why the trade-off is currently hard.
- `current_scheme`: the actual current fusion scheme and its targets.
- `attempts`: directions already tried and their results; do not propose options already falsified under the same conditions.
- `evidence`: the 3 triggering rounds, the best avg_speedup/HAP, the slow-case improvement trend, and the actual version paths and reading instructions for the implementation/performance reports/scheme rationale; cite program materials directly; do not fabricate numbers.
- `options`: 2–3 objects, each with a unique `id` (e.g. A/B/C), `title`, `benefit`, `cost`, `risk`, proposing genuinely feasible different directions.
- `recommended_option`: the id of one real option above.
- `recommendation_reason`: why this option is recommended first, based on current evidence.

Candidates may include local optimization, switching fusion for specific shapes, or running additional experiments first, but they must suit the current hardware and evidence; do not copy templates. Prior human opinions are included as question background; the final call after the wait completes handles the full feedback and the original scenario's decision.
