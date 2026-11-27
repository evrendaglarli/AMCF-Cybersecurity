# Manuscript v45 R3 — code alignment and reproducibility boundary

**Basis:** *Adaptive Metacognitive Framework for Proactive Network Security Operations*, uploaded 43-page manuscript **v45 R3** (2026-11-27). This document supersedes manuscript-version descriptions from `reports/R20_IMPLEMENTATION_ALIGNMENT.md`; it does **not** claim that the historical results were reproduced.

## Important distinction

A matching equation or passing unit test establishes **method-level consistency only**. It does not establish reproduction of Table 4–12, Figures 3–5, the restricted SOC evaluation, the human-factors pilot, or the online policy's actual production behavior. Neither arbitrary prepared pickles nor `--synthetic-demo` are original benchmark evidence.

## Source mapping and findings

| Manuscript | Public reference-code location | Verified method correspondence / limit |
|---|---|---|
| Eq. (1), Algorithm 1 | `amcf/core/calibration.py::TemperatureScaler`; `metacognitive_controller.py::_sense` | Sigmoid on logit-scale scores; positive temperature; update_frequency=200. Reference code uses a bounded buffer and a floor of 0.1; it does **not** independently demonstrate manuscript's delayed-label class-enriched rolling experiment. |
| Eq. (2), Algorithm 6 | `calibration.py::CompetenceWeightManager`; `metacognitive_controller.py::_update_from_feedback` | Fused-posterior Brier gradient and nonnegative normalized weights. Causal feedback depends on external confirmation and `allow_inline_confirmed_feedback`; no original label-availability ledger or versioned replay evidence is bundled. |
| Eq. (3), Algorithm 1 | `fusion.py::LogitWeightedFusion` | Weighted logit fusion with bias argument; current controller uses neutral bias `0.0` without fitted context-prior model. |
| Eq. (4)–(5), Algorithm 2 | `amcf/core/hypotheses.py` | Stage-based, deduplicated bounded hypotheses; reference subset is not a verified full ATT&CK-conditioned production pipeline. |
| Eq. (6)–(8), Table 2, Algorithm 2 | `amcf/core/bayes_risk.py` | Ignore/automate/escalate cost algebra and severity hazard `κ_s=0.1 min^-1` align with reference values (CFP=1, low/med/high=5/15/50, Cauto=2.5, CSE=1.2, triage=3, delay=0.5, queue=0.3, residual=0.1). The controller currently supplies placeholder asset weight, queue, and delay; those are not production estimates. `gather=inf` is intentional until a validated conditional enrichment model exists. |
| Eq. (9)–(10), Algorithm 2 | `ValueOfInformationEstimator` | One enrichment cost subtraction and validated conditional outcomes. The controller now uses `amcf/core/monte_carlo_voi.py`: 64–4096 conditional samples, BCa 95% half-width, absolute tolerance 0.02, and 150 ms budget with fail-closed timeout. The older `ValueOfInformationEstimator` retains exact discrete summation for backwards-compatible method tests. No historically validated query-outcome distributions are bundled, so production VoI performance and reported paper numbers remain unverified. |
| Eq. (11)–(12) | No corresponding verified bounded queue/microservice replay package | Mathematical statements are in manuscript; the public controller does not demonstrate the reported enterprise throughput, p95 latencies or SOAR execution. |
| Eq. (13)–(14), Algorithm 4 | `amcf/safety/safety_gate.py` | The branch aligns the nominal endpoint-isolation posterior criterion `1-π_t <= 10^-3` and adds `check_endpoint_isolation`: explicit concrete action class, selected playbook identity, risk baseline `min(ignore,gather,escalate)`, and `ΔRisk >= τ_risk` with no invented numerical default. The reference controller passes the selected playbook class and rejects requests without a validation-defined `τ_risk`. The legacy binomial guard remains as an extra conservative admission requirement, although the paper describes its bootstrap statistic as non-certifying monitoring. Actual advisor-independence validation, dynamic risk inputs, and SOAR reserve/execute/rollback are still absent, so full Algorithm 4 equivalence is **not established**. |
| Eq. (15), Algorithm 5 | `amcf/safety/enrichment_scheduler.py` | One positive-VoI budget-feasible maximizer per round. The reference controller now shares one cost source with the scheduler, removing its previously inconsistent ED R/sandbox prices. Actual Eq. (9) measured byte/CPU/latency cost coefficients, an enrichment connector, outcome and rescheduling loop remain absent. |
| Algorithm 3 | `metacognitive_controller.py::_act` | Reference controller escalates instead of falsely claiming real SOAR containment or enrichment when connectors are absent; unlike the manuscript's production execution. |
| Appendix B | Manuscript v45 R3 pp.38–39 | Seven-argument Algorithm 2 call and snapshot outside the playbook `if/else` are now correct *in the paper*. Public controller does not yet provide equivalent SHA256 decision snapshots and immutable audit logs. |
| Table 13 | `configs/amcf_config.yaml`, method class defaults | Many reference constants exist; several algorithm parameters remain defaults/placeholders, not independently authenticated manuscript experiment configs. |

## Reviewer R2.1: what is missing before claiming full reproduction

1. **CIC-IDS2017 / CSE-CIC-IDS2018 / UNSW-NB15**: authenticated source manifests and licenses; actual acquisition/preprocessing scripts; feature specifications and stage tags; chronological train/validation/test splits with event- and label-availability timestamps; leakage checks; **real data hashes** and seeds.
2. **Evaluation harness**: identical replay ordering across models, real 250+ rule SIEM configuration (or documented restricted boundary), LightGBM/TCN/Isolation Forest + stacking and validation isotonic calibration, controlled matched-context/equal-information and random-enrichment variants, and a full AMCF controller with causal enrichment/feedback. Code currently does not replicate this full design.
3. **Statistics**: temporal-session/block bootstrap, per-dataset matched Wald tests with 7-day HAC standard errors and Holm over six comparisons, complete ablation variants and output files. The generic flow-level routines in `amcf/utils/statistics.py` cannot substitute for the manuscript's temporal block procedures.
4. **Tables/figures provenance**: machine-readable result records and per-run metadata that regenerate paper Table 4 (Recall@B = 0.5, 1, 2), Table 7 calibration summaries and the relevant public ablations, with scripts to regenerate confidence intervals and plots. No hand-entered “matching” numbers.
5. **Restricted enterprise corpus**: descriptive schema and auditable aggregation/incident/labeling definitions can be public; underlying telemetry, 41 containment logs, the 12-analyst assessment and private Table 5–12 results cannot be independently reconstructed without authorized data access. Synthetic examples are clearly labeled illustrations.
6. **Figure 5**: enterprise probability-bin occupation is listed in Supplementary S1, but binwise *mean posterior* and *observed malicious frequency* used to compute ECE are not exposed. Do not infer ECE from bin midpoints or invent raw outputs.

## Findings requiring careful interpretation

- The current `experiments/run_experiments.py` uses reference model predictions and a prepared-pickle interface; it does not implement the entire manuscript experiment. Its “reproduce Table 1/2” legacy labels refer to earlier paper numbering and should not be cited as evidence of **v45 R3** Table 1/2 (which now mean architecture comparison and the loss matrix). The script also does not implement the full evaluation at each false-positive budget for AMCF as described in Section 5.5.
- The reference safety-gate code retains a **more conservative historical-outcome binomial guard** in addition to the nominal posterior guard. This is not the same as the paper's explicitly non-certifying monitoring statistic. Do not claim Algorithm 4 fully implemented until the intended admission/monitor split and action-specific risk are specified and tested.
- Public code does not include a live SOAR/EDR enrichment connector; the automatic-escalation behavior is intentionally fail-closed and is **not** evidence of 41 verified automated containments or a production latency distribution.
- Temperature scaling is constrained to positive T, but the present reference update and confirmation flow are not a verified recreation of the manuscript's class-enriched chronological calibration protocol.

## Current branch changes and tests

- `amcf/safety/safety_gate.py`: nominal posterior threshold now enforces `1 - pi <= epsilon` (v45 R3 Eq.14) instead of a generic 0.80 threshold. Other gates remain, so no unsafe broadening of the reference prototype's permission path.
- `amcf/core/monte_carlo_voi.py`: sampled conditional VoI with a 95% BCa stopping half-width, bounded sample count and wall-clock budget, and explicit status on timeout/missing evidence.
- `amcf/core/metacognitive_controller.py`: uses Monte Carlo estimator when authentic conditional outcomes are registered, checks playbook class and ΔRisk with explicit validation-supplied threshold, and uses the enrichment scheduler's single cost source.
- `amcf/memory/procedural_memory.py`: explicit `endpoint_isolation` action class for the corresponding playbook only; unclassified playbooks are denied by this specialized gate.
- `tests/test_v45_r3_contracts.py`, `tests/test_v45_r3_monte_carlo.py`, `tests/test_v45_r3_controller.py`: algebraic and reference controller tests. **Passing tests will not constitute reproduction of paper results.**

## Artifact-provenance verification (additional guard, not reproduction)

The script `experiments/verify_v45_r3_artifacts.py` accepts a JSON manifest with `manuscript_version: "v45 R3"` and an entry for each of CIC-IDS2017, CSE-CIC-IDS2018, and UNSW-NB15. Each entry requires an HTTPS `official_source`, an integer `seeds` list, and a dictionary of artifacts with `path` and lowercase `sha256`. Required roles: `raw_data`, `preprocessing`, `split_manifest`, `configuration`, `run_log`, `recall_table4`, `calibration`, `ablation`, `statistics`.

Run `python experiments/verify_v45_r3_artifacts.py /path/to/authentic_manifest.json`. The command fails if files are missing or hashes differ and never unpickles or substitutes data. Even when all hashes pass, this is **artifact identity only**, not verification of model correctness, temporal leakage, CI computation, or manuscript result reproduction. Do not populate it with invented hashes or demo files.

## Acceptance rule

**Do not label R2.1 CLOSED** until real, reproducible public-benchmark artifacts exist and an independently executed command produces documented outputs that match the paper, with expected tolerances and uncertainty analysis. If source artifacts are absent, report the gap; do not fabricate class labels, time stamps, confidence intervals, containment outcomes, Figure 5 bins, or matching numbers. Preserve `main` until the review branch has passed validation.
