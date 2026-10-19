# UNSW-NB15 pilot — genuine CSV data evaluated (2026-10-19)

**Scope:** Flow-level feasibility experiment on user-supplied UNSW-NB15 designated train/test CSVs. **NOT a reproduction of IEEE Access R20, NOT a test of full AMCF, and NOT real-world SOC evaluation.** The uploaded files have release-consistent columns and expected record counts; independent comparison against authoritative original-file checksums has **not** been completed.

## Input checks and provenance

| Item | Train | Test |
|---|---:|---:|
| Records | 175,341 | 82,332 |
| Columns | 45 | 45 |
| Positive labels | 119,341 | 45,332 |
| Negative labels | 56,000 | 37,000 |
| Missing values | 0 | 0 |

- Train SHA-256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`
- Test SHA-256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`
- Runtime: Python 3.13.5, scikit-learn 1.8.0, pandas 2.2.3, NumPy 2.3.5.
- Random seed: 42; validation-tuning target flow FPR: 5%.

## Full-release pilot outcome

**175,341 training records, 82,332 test records**, with training partition internally split into 122,738 model fitting / 26,301 isotonic calibration / 26,302 tuning records. Original test labels were not used for threshold selection.

| Model | ROC-AUC | Average precision | Recall | Precision | Test FPR | Brier | ECE | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.95484 | 0.96438 | 0.87375 | 0.89207 | 0.12951 | 0.10755 | 0.09025 | 39,609 | 32,208 | 4,792 | 5,723 |
| HistGradientBoosting | **0.98410** | **0.98729** | **0.96027** | **0.89544** | 0.13738 | 0.07953 | 0.09007 | 43,531 | 31,917 | 5,083 | 1,801 |
| AMCF-lite proxy | 0.97884 | 0.98432 | 0.94408 | 0.88290 | 0.15341 | 0.08963 | 0.09635 | 42,797 | 31,324 | 5,676 | 2,535 |

**Observed:** HistGradientBoosting outperformed AMCF-lite on both recall and FPR in this single full-release pilot. All models exceeded the validation-target FPR on the separate test partition. In particular, 5% is a **tuning target**, not a validated or guaranteed test bound.

## Supporting runs (same seed)

| Mode | Train records | Test records | LogReg AUC | HGB AUC | AMCF-lite AUC | Test FPR LogReg/HGB/AMCF-lite |
|---|---:|---:|---:|---:|---:|---|
| Partial pilot | 20,000 | 10,000 | 0.95738 | 0.98213 | 0.98086 | 0.11215 / 0.06920 / 0.12105 |
| Full test, partial training | 20,000 | 82,332 | 0.95663 | 0.98279 | 0.98111 | 0.11116 / 0.06519 / 0.11924 |
| Full train + full test | 175,341 | 82,332 | 0.95484 | 0.98410 | 0.97884 | 0.12951 / 0.13738 / 0.15341 |

These are **single-seed descriptive** results. No confidence intervals, significance claims, chronological incident labels, or claims of paper-table reproduction are justified.

## Reproduction command

With user-obtained original files under `data/raw/unsw_nb15/`, execute at repository root:

```bash
python -m pip install numpy pandas scipy scikit-learn
python -m unittest discover -s tests -p 'test_unsw_nb15_feasibility.py' -v
python experiments/unsw_nb15_feasibility.py \
  --train data/raw/unsw_nb15/UNSW_NB15_training-set.csv \
  --test data/raw/unsw_nb15/UNSW_NB15_testing-set.csv \
  --max-train 0 --max-test 0 --target-fpr 0.05
```

Raw CSVs intentionally **not committed**. The original input files' SHA-256 hashes above allow later identity checks.

## Limitations and decision

- AMCF-lite is **equal-weight fusion of two calibrated conventional classifiers plus a virtual ignore/escalate cost gate**. It does **not** implement AMCF's episodic/working/procedural memory, coherence, VoI, online drift adaptation, automated containment, or operational safety controls.
- This dataset/runner does not support FP/hour, multi-stage long-dwell recall, incident time-to-containment, enterprise SOC claims, or human factors.
- The release is not a verified chronological SOC split; 55% of test records are positive, unlike a typical low-prevalence operational stream.
- Full AMCF evaluation is a **new empirical research/development task**. These outputs do not establish validity of numerical findings in IEEE Access v17 R20.
- All tests in the pilot's provided `test_unsw_nb15_feasibility.py` suite passed locally (4/4); no claims made that the original AMCF repository's broader tests have passed.

**Decision:** This is a technically successful real-data feasibility pilot. It is **not evidence that AMCF outperforms its baselines**. The draft PR should receive scientific/code review before merging.
