## Human Opinion Handling: Significant Relevance, Forming a Traceable P0

First read the prompt's full original wording, opinion number, and type; if it is a consultation follow-up, also read all options, recommendation reasons, and the feedback package's version evidence in the original `human_review/<iter>/<request number>/Question Document.md`. Do not take only one sentence of the human's while discarding the original question.

Respond to each item in the final decision's `human_responses` list: each entry contains `message_id`, `kind` (`direction`/`question`/`wait`/`conflict`), and a non-empty `answer`.

- Substantive direction: `kind=direction`, explain how this round handles it; list a P0 in `suggest_next` with `source="human"` and `human_message_id` pointing to that opinion number. Human suggestions also get the complete v2 task order: task_id, target cases/engineering reason, routing evidence between case and implementation, per-file changes methods, and acceptance_checks; do not downgrade to P1/P2 or leave it as background only. The allowed modification scope is aggregated automatically from changes; do not skip the global read-only constraints because of a human P0.
- Pure question: `kind=question`; answer it first; do not turn it into an execution directive on your own.
- Waiting expression: `kind=wait`; confirm it is only a wait instruction, not a human P0. Wait duration is handled by the program.
- Conflicts with correctness/hardware: `kind=conflict`; explain the basis and fill in a non-empty `alternative`; for human P0s associated with the same number, adopt an alternative satisfying the constraints; do not directly execute infeasible demands or silently drop them.

Stage3 will record how each human direction was implemented; your recording "handled" does not mean it was executed. When the program determines it cannot continue, state the limitations and unexecuted items and hand them to Stage10; do not exceed the exit or iteration limits.

When a substantive reply is not received before the timeout, decide per the original question's recommended direction, explicitly noting "no human opinion received"; do not pass off the default direction as human agreement or a human P0. All received messages must be preserved. The final decision after consultation must still satisfy this round's conditional experience and technical adjudication; do not create another evaluation.
