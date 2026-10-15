# R14 scientific-method validation: status and evidence requirements

This patch is **not** a reproduction of the paper's experimental results. It
implements correctly named statistical procedures, makes missing operational
annotations fail explicitly, and supplies a reconciliation tool. No reported
figure/table value is changed or regenerated.

## What was repaired in the proposed code

- `bootstrap_ci`: actual SciPy BCa bootstrap (event-level recall, positives
  sampled); the old routine incorrectly labelled percentile CI as BCa.
- `delong_test`: paired correlated ROC-AUC inference using DeLong placement
  covariance; the old routine permuted labels and both score vectors together.
- Holm correction: reject only after monotone adjusted p-values are computed.
- Operational metrics: long-dwell recall requires chain/impact timestamps;
  TTC requires observed containment actions and censoring; escalation
  precision requires validated actual escalations; FP/endpoint-day requires
  audited active endpoint-days. Event-level positive predictions cannot be
  substituted for these case-level quantities.
- `audit/enterprise_reconciliation.py`: verifies alarm/FP counts using actual
  per-day fleet activity. Its `--active-endpoints` argument is labelled a
  scenario assumption, **not** observed data.

## Remaining blockers before manuscript approval

1. Provide daily CSV with date, active_endpoints, false_positives, alerts
   from the **same** held-out 15-day stream. Reconcile against 338 alerts,
   published 0.061 FP/endpoint-day and daily active fleet. The 8,400-endpoint
   scenario yields ~7,686 FPs; this is a contradiction if all 8,400 were
   indeed active daily. Nothing in the PDF proves that assumption.
2. Load real CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15 benchmark splits and the
   restricted enterprise replay. The existing public fallback produces random
   synthetic data, not benchmark evaluation. Publish dataset splits, configs,
   evaluation hashes, and a working reproduction pipeline or qualify public
   reproduction claims honestly.
3. Recalculate reported BCa CIs, Holm-adjusted p-values, ROC-AUC DeLong
   comparisons, paired Wilcoxon outcomes, and effect sizes from actual event
   predictions/participants. Correct implementation alone does not validate
   the paper's already reported numerical values.
4. Supply incident IDs/timestamps/action outcomes, censoring indicators,
   escalation validations, and annotated attack-chain impact times required
   for verified TTC, precision and long-dwell recall.
5. Supply n=12 paired Human-factors participant scores and day/case matching
   to check Wilcoxon p-values/Cliff's delta and reported 95% mean confidence
   intervals. Consent and confidentiality remain protected; de-identified
   aggregates and reproducible analysis may suffice.
6. Ablation Table IX gives seven CIs crossing zero and non-significant Holm
   tests. Interpret component contributions **descriptively**, not as
   statistically established causal effects.

## Tested locally

The dedicated R14 statistical, operational and reconciliation tests passed.
These tests do not run the original paper experiments or establish any
numerical result in the manuscript.

## Suggested workflow

Start with real daily logs; stop if input data cannot be located. Then run the
statistical tools on matched real predictions and compare Table V-XII outputs.
Merge this patch only after integration review and existing test suite pass.