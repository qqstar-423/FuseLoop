## Conditional Task: Adjudicate the Developer Node's question.md

The prompt-provided `develop/<iter>/question.md` is a technical objection not yet adjudicated. Against the original advice, the developer feedback's hard evidence, and known constraints, judge whether to confirm the misjudgment, reject it, or partially uphold it; do not change direction based on the objection alone.

You must output a `pitfall` object in this decision: `verdict` is `confirmed` or `rejected`, and the remaining required fields `topic`, `target_advice`, `feedback`, `root_cause`, `correct_approach` are non-empty strings. For partially upheld, use `confirmed`, stating separately in the reasons and correct approach which parts hold and which do not.

The program writes it into `knowledge/tech_lead_pitfalls.md` and writes back to the original question; you do not modify it directly. Check new suggestions against the existing mistake log to avoid repeating confirmed misjudgments. Missing this field stops delivery, and the old plan will not be dispatched.

The program also records the framework, chip, and decision file sources of this review; environment facts are filled in by the program, not guessed by the model. When reusing old adjudications, verify the applicable hardware, framework, and backend versions.
