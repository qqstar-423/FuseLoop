## Current Scenario: Precision Failure

Read the failing cases in `eval/<iter>/precision_result.json`, then check the raw errors in `eval/<iter>/precision_reports/`, `task/cases.yaml`, and `task/golden.py`, combined with the build log, the code, and the bound self-test results.

Explain the discrepancy between the self-test and the formal checks, focusing on distinguishing computation/boundary-input errors, stale data reuse, uncovered consecutive calls, and environment problems. Check whether last round's correctness fix goals were implemented, and state the fix direction, file scope, and next round's verification cases. With no formal performance results this round, no performance optimization, benefit attribution, or up/down experience is required.
