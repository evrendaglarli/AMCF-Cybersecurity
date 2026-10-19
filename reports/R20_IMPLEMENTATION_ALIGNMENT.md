# IEEE Access R20 — AMCF implementation consistency (2026-10-20)

**Scope:** Source-level reconciliation of the public reference implementation with the *method definitions* in manuscript v17 R20. This document **does not** claim to reproduce manuscript experiments or historical operational outcomes.

## Code and paper correspondence

| R20 element | Reference-code behavior after this change | Evidence limit |
|---|---|---|
| Eq. (2) | Advisor weights update using gradient of the *fused-posterior* Brier loss; effective learning rate (1 - λ)η and nonnegative normalized weights | Label must be confirmed and causally available; no independent enterprise validation |
| Eq. (3) | Existing competence-weighted logit fusion unchanged | Probability inputs must be correctly calibrated; fused posterior not independently calibrated |
| Eq. (4)-(5) | Bounded, de-duplicated entity-centred hypotheses; explicitly ATT&CK-aligned stage tags, weighted path consistency, coverage, temporal plausibility, normalized belief values | Narrow reference subset; does **not** discover true ATT&CK stages, fit attack templates or validate long-dwell claims from UNSW |
| Eq. (6)-(10) | Bayes-risk cost terms retained. Conditional VoI uses an explicitly supplied validated outcome distribution; no fake random posterior changes. If absent, zero **recommendation**, not a claim that enrichment has no information | Discrete-outcome expectation of Eq. (10), not manuscript's calibrated Monte Carlo production pipeline |
| Algorithm 5 / Eq. (15) | Select ONE positive, budget-feasible maximal-VoI enrichment per round; conditional recomputation required after an actual outcome | No EDR/sandbox connector is installed, so no real query is executed |
| Drift monitoring | If a validation-frozen fused-posterior reference is supplied, compute PSI/KS and optional ECE with confirmed labels; conservative escalation on detected shift | Missing reference => unknown; online recalibration/retuning and 3–5 day drift response **not reproduced** |
| Algorithm 4 | Fail-closed safety inputs, empirical bound, validated dependence requirement; no SOAR action connector | Not an operational safety or rare-event certification |
| Statistical methods | True scipy BCa for independent flow-level recall/precision, tied-score paired DeLong, Holm step-down, centred HAC estimate | Manuscript temporal block BCa/bootstrap and actual Table IX significance cannot be re-created without original labels/blocks |
| Long-dwell recall/TTC | Require real incident IDs, timestamped stage sequence and verified containment time; return **NaN** when insufficient | No positive flow-classification output is silently treated as containment |
| TCN | Reject 2D tabular features that lack chronological sequences; keep batch dimension on single input | Temporal sequence construction required for full TCN comparison |
| UNSW-NB15 pilot | Measured real designated train/test files for logistic, HGB, **AMCF-lite**; multiple seeds | AMCF-lite is **not** the full AMCF or a replacement for manuscript Table V–IX |

## Deliberately excluded from completion claims

1. CIC-IDS2017 and CSE-CIC-IDS2018 experimental results, full R20 baseline settings and original ablation results.
2. 60-day, ~470 million-event SOC evaluation, incident containment logs, provenance and operator study. No original private corpus or original human-factors records are in this repository.
3. Operational SOAR/EDR/NDR connectivity and real automated containment.
4. Independently verified calibration/causal feedback pipelines, historical enrichment-outcome models, complete online retuning, and production-level attack-chain inference.

The public repository supports a research-prototype subset and a real UNSW *static flow* pilot. Its automated tests are **method-contract tests**, not evidence that the empirical R20 values were observed.

## Appendix D contradiction requiring manuscript attention

The present R20 Appendix D says the repository includes "materials required to reproduce the public-dataset experiments" and that public reproducibility is "complete." This assertion is **not supported by the available code and data**. Because the original benchmark protocols, enterprise data and results are missing, *no source-code change can truthfully make the previous numerical tables reproducible*.

**Suggested scientifically accurate replacement (only if revising R20; manuscript not modified by this PR):**

> Reference implementation components, configuration examples, and a separately documented UNSW-NB15 feasibility pilot are available in the public repository. The pilot evaluates a limited AMCF-lite inference proxy and does not reproduce the numerical results reported for the complete framework. The original public-benchmark preprocessing and experiment artifacts required to regenerate the manuscript tables have not been independently verified as available in the repository. Original enterprise SOC telemetry, containment outcomes, and human-factors records are unavailable for public reproduction. Consequently, independent reproduction of those reported results cannot presently be claimed.

**Important:** This is not a cosmetic documentation change. Before publication, the paper's experimental statements must be supported by authentic evidence or reported as appropriately scoped/unverified. Do not silently fabricate matching data, calibration records, CI bounds or attack-chain labels.

## Acceptance / audit

- `tests/test_r20_method_contracts.py` tests evidence de-duplication, normalized bounded beliefs, conditional VoI, one-step budget selection, fused Brier gradient, BCa/Holm/DeLong, incident-only metrics and honest drift availability.
- `.github/workflows/r20-methods-ci.yml` compiles Python modules and executes these regression checks.
- R20 PDF remains unchanged; the original reported numerical values are neither regenerated nor verified by this alignment patch.
