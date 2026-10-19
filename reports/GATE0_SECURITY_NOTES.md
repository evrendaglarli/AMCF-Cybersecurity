# Gate 0 — provisional fail-closed safety hardening (NOT certified SOC safety)

This draft replaces the prior *zero failure implies safe* assumption with a one-sided exact binomial upper bound. With 0 failures out of 200 observations the 95% upper bound is ~1.49%, far above the manuscript target 0.1% (1e-3); the reference gate correctly denies automated action. The existing 200-event window cannot establish this stringent bound, even if no recorded mistakes occur. Consequently, an external validation mechanism and governed human-confirmed execution pathway would be needed for any future conditional automation.

The module also rejects malformed detector input and denies action when no empirically measured advisor-correlation estimate is available. These narrow changes **do not** certify an automated-response system, authenticate ground truth, or establish the independence assumption of the binomial model. No physical actuation or SOAR connector is enabled by this PR.

All downstream controller integration and test-suite compatibility need independent review before merging this draft.

## Integration prerequisite

PR #2 has now been merged into `main` (commit `6afa0c5fdebdc4f6e70e362a58a24a33a3b6b438`), resolving the repository-wide `run_experiments.py` syntax error that previously blocked Gate 0's `compileall` job. This branch still requires passing merged-base CI before acceptance; a failed/absent run must not be treated as approval.
