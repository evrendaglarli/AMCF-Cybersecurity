# Adaptive Metacognitive Framework (AMCF)

Reference code for the research manuscript **"Adaptive Metacognitive Framework for Proactive Network Security Operations: Threat Detection and Response Orchestration Inspired by Human Reasoning Mechanisms"**.

AMCF uses a Sense–Reason–Act control loop combining detector calibration and fusion, working/episodic/procedural memory, risk-aware decisions, evidence enrichment, and safety-gated escalation/response.

## Important: scope and reproducibility

**This public repository is a reference-code and demonstration package. It is not currently a verified end-to-end reproduction of the results reported in manuscript v17 R20.**

- Original CIC-IDS2017, CSE-CIC-IDS2018, and UNSW-NB15 benchmark records are **not bundled** here. The repository does **not** currently implement an authenticated, verified end-to-end download and preprocessing pipeline for them.
- The enterprise SOC records described in the manuscript are **not included** because of data-governance and confidentiality constraints. Public access to the raw enterprise stream is not offered.
- If a prepared dataset is missing, the experiment runner **fails by default**, rather than silently substituting random data for real benchmark traffic.
- An explicit `--synthetic-demo` switch produces *random illustration-only* data. These events are **not** derived from the named intrusion-detection datasets or genuine SOC telemetry, and their numerical outputs must **not** be represented as manuscript results.
- The public runner covers three reference evaluation routines (detection comparison, operational metrics, and calibration). It does **not** establish independent replication of the manuscript's full ablations, drift/safety trials, human-factors assessment, significance testing, or all results tables.

The empirical claims and numerical tables in the manuscript should be supported by the **original experiment scripts, datasets or documented access arrangements, configurations, run logs, and output artifacts**, not by the random-data demonstration.

## Setup

From the repository root, using a Python environment with dependencies installed:

```bash
git clone https://github.com/evrendaglarli/AMCF-Cybersecurity.git
cd AMCF-Cybersecurity
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Running the reference evaluation

### Prepared dataset mode (default)

Provide the verified files below before running:

```text
data/processed/cic_ids_2017.pkl
data/processed/cse_ids_2018.pkl
data/processed/unsw_nb15.pkl
data/processed/enterprise.pkl
```

Then run:

```bash
python experiments/run_experiments.py
```

The runner requires these files and raises `FileNotFoundError` if they are missing. It **does not** fetch the original data automatically. The currently supported pickle tuple is `(X_train, y_train, X_val, y_val, X_test, y_test, events_test)`. Input data must be prepared with documented label provenance, split chronology, replay clock, and preprocessing, and loaded only from trusted pickle files.

**Providing arbitrary files with these names does not establish reproduction of the paper.** The preprocessing and experimental protocol still need independent verification against the manuscript.

### Optional synthetic demonstration

```bash
python experiments/run_experiments.py --synthetic-demo
```

This mode **always** generates random synthetic illustrations (even if prepared input files exist). Its output tables are written separately under `results/demo/tables/` rather than `results/tables/`. The two modes cannot silently mix.

The synthetic generator assigns random features, labels, timestamps, and detector scores. It is suitable only for development and illustrative execution, **not** for measuring real-world detection performance, safety, or calibration. No claims about research reproducibility or actual performance follow from successful execution of this command.

## Code organization

| Path | Role |
|---|---|
| `amcf/core/` | Metacognitive controller, calibration, fusion, and Bayes-risk utilities |
| `amcf/memory/` | Working, episodic, and procedural memory components |
| `amcf/safety/` | Safety gate and enrichment scheduling |
| `amcf/baselines/` | Reference SIEM/ML baselines |
| `experiments/run_experiments.py` | Reference runner with explicit dataset provenance guard |
| `data/synthetic/generate_synthetic_data.py` | Small synthetic test-data generator, not original SOC data |
| `configs/` | Reference configuration files |
| `tests/` | Included unit/integration tests |

## Validation and limitations

This repository includes research prototype code. Availability of source files, demonstration runs, or unit tests does **not** certify the manuscript's empirical results, enterprise operational deployment, or the claimed rare-event safety target. Model calibration, timing, class imbalance, label availability, correlation effects, and statistical significance require the corresponding original experiment artifacts.

The public repository is being aligned with the revised manuscript. If the original evaluation scripts or shareable benchmark preprocessing materials become available, they can be added with provenance, tested outputs, exact dependency versions, and an explicit mapping to manuscript tables.

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT License; see [LICENSE](LICENSE).
