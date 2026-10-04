## Current Scenario: Build Failure

Read the first real error and the final STATUS line in `build/<iter>/build.log`, combined with the bound code, self-test, last round's changes, and relevant failure experience, to distinguish installation/import/environment problems from code compilation problems. A passing self-test does not exempt code inspection, and build cache problems must not be attributed to the algorithm.

Determine this round's root cause to fix, the file scope, and the verification method, and form the next round's P0/P1/P2. Review only history related to this failure; you are not required to redo performance analysis round by round. When this round's formal precision and performance results do not yet exist, do not fabricate data, do not update the best implementation, and do not record this round's performance up/down experience.
