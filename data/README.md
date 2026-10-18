# Data provenance and evaluation inputs

## What this repository includes

- `data/synthetic/generate_synthetic_data.py` generates **random demonstration samples**, not the original CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, or enterprise SOC records.
- The main experiment runner can also generate temporary synthetic demonstration samples only when `--synthetic-demo` is explicitly supplied.
- Neither of these random-data generators reconstructs the temporal attack chains, true labels, feature correlations, split protocol, or security operations data described in the manuscript.

## What is not included

The named public benchmark datasets and verified preprocessed versions are **not currently committed** to this repository. The original enterprise SOC telemetry is confidential and is also **not committed**. No verified automatic download/preprocessing pipeline for those datasets is included here.

Therefore, **this repository alone cannot currently reproduce the published paper's dataset-based numerical findings.** Do not mistake demonstration output for the manuscript's empirical evidence.

## Inputs expected by the reference runner

Verified, locally prepared datasets are expected under:

```text
data/processed/cic_ids_2017.pkl
data/processed/cse_ids_2018.pkl
data/processed/unsw_nb15.pkl
data/processed/enterprise.pkl
```

The legacy runner expects a trusted Python pickle of the form:

```python
(X_train, y_train, X_val, y_val, X_test, y_test, events_test)
```

`X_*` denotes feature matrices; `y_*` binary labels; and `events_test` a sequence of event records. This is a code-level input contract, **not** evidence that a supplied file follows the manuscript's validated experiment protocol.

Only load pickle files from trusted sources. Proper reproduction requires documented data origin, preprocessing, split boundaries, class counts, event-to-incident mapping, replay-clock rules, and causally available labels.

## Running

By default, missing prepared data cause a clear error:

```bash
python experiments/run_experiments.py
```

To *explicitly* exercise the reference runner with randomly generated examples:

```bash
python experiments/run_experiments.py --synthetic-demo
```

Synthetic runs use `results/demo/tables/` for outputs. They are illustrations only and must not be reported as benchmark or enterprise SOC results.

The separate `data/synthetic/generate_synthetic_data.py` program uses a small 50,000-event random dataset (12,500 test events). The runner's internal random generator uses other small sample sizes. These numbers do **not** represent the full 60-day enterprise corpus or a real sample extracted from it.
