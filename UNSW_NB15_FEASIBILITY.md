# UNSW-NB15: real-data feasibility pilot (not a reproduction of R20)

This is a narrow, **decision-only AMCF-lite feasibility probe**. It **does not run the full AMCF** and does not validate/reproduce any table, figure, long-dwell/incident-level metric, drift result, safety claim, or human-factors result in IEEE Access v17 R20. The purpose is to validate a real-data loading and leakage-controlled evaluation path before undertaking a full reproduction project.

## Data: obtain the genuine release

Original dataset information, authors, and terms: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Obtain **UNSW_NB15_training-set.csv** (175,341 rows) and **UNSW_NB15_testing-set.csv** (82,332 rows) from the academic release. Store locally under `data/raw/unsw_nb15/`; **do not commit the data**. The UNSW dataset has an academic-use permission statement and requests scholarly citations; check license/redistribution before publishing copies. A third-party mirror is not the official provenance and should be verified by checksum before research use.

The release partitions are not guaranteed to be chronological. They do **not** support FP/hour, incident-level TTC, long-dwell detection, real containment, or SOC event-replay claims from this script. No random or synthetic dataset fallback exists.

## Run

```bash
python -m pip install numpy pandas scipy scikit-learn
python experiments/unsw_nb15_feasibility.py \
  --train data/raw/unsw_nb15/UNSW_NB15_training-set.csv \
  --test data/raw/unsw_nb15/UNSW_NB15_testing-set.csv \
  --max-train 20000 --max-test 10000 --target-fpr 0.05
```

`--max-train 0 --max-test 0` requests the full prepared train/test partitions (more memory/time). The 20k/10k defaults are **stratified, seeded, partial pilot evaluations**, not full-scale benchmark results. `--allow-partial` exists **only** for small-format smoke tests and makes a dataset noncanonical by definition.

## Protocol

- **Source separation**: official training partition for training, probability calibration, and operating-point selection; official testing partition used exactly once for final evaluation.
- **Within training partition**: seeded stratified 70% model fit, 15% isotonic probability calibration, and 15% operating-point tuning. Test labels never influence preprocessing, training, calibration, fusion weights, or thresholds.
- **Feature hygiene**: exclude `label`, `attack_cat`, and `id` from all model inputs. Handle `proto`, `service`, and `state` with training-only one-hot encoder; numeric missingness with training-only imputation, then scaling.
- **Comparators**: logistic regression and histogram gradient boosting; both trained only on fit split, and calibrated only with calibration split.
- **AMCF-lite**: equal-weight logit fusion of their separately calibrated outputs plus a simple Bayes-risk ignore/escalate threshold. **Not** a faithful implementation of full AMCF: no working/episodic/procedural memory, coherence, VoI scheduling, drift adaptation, learned competence, real automation, or independent safety gate.
- **Fairness**: threshold separately selected from tuning negatives to target *flow-level false-positive rate*; actual test FPR can differ and is reported unchanged. AMCF-lite risk threshold also applies, so its operating point may be more conservative than the baselines.
- **Metrics**: test ROC-AUC, average precision (PR-AUC convention here), Brier score, 10-bin ECE, precision, recall, FPR, confusion matrix; all **flow-level**. No unsupported FP/hour or incident-level metrics are output.
- **Outputs**: `results/feasibility/unsw_nb15/pilot_metrics.csv` and `provenance.json`, containing input SHA256 hashes, row counts, package versions, random seed and explicit limitations.

## Tests

```bash
python -m unittest discover -s tests -p 'test_unsw_nb15_feasibility.py' -v
```

The unit tests use **generated fixtures explicitly marked as tests**, never as a reported benchmark. A successful unit/smoke test only validates pipeline mechanics; a real-data performance report requires both genuine CSV files and a full run on them.

## Before citing in an article

Inspect source provenance and hashes; fix experimental questions and quantitative claims; run the real release; verify class distribution and out-of-distribution performance; repeat with multiple random seeds and confidence intervals; and independently review whether AMCF is actually implemented to the revised paper's specification. **Do not substitute the AMCF-lite metrics for R20's existing numbers.**