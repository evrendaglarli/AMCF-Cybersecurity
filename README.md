# AMCF: Adaptive Metacognitive Framework for Proactive Cyber Security

Official implementation of **"Adaptive Meta-Cognitive Framework for Proactive Cyber Security System: Mimicking Human-Level Cyber Threat Detection and Elimination Behaviors"** submitted to Big Data and Cognitive Computing.

## Overview

AMCF is a bio-inspired cybersecurity architecture that elevates control over detection to a first-class design objective. Rather than treating detection as a binary decision, AMCF frames cyber defense as a closed-loop **Sense→Reason→Act** cycle with **metacognitive control**.

### Key Features

- **Metacognitive Control**: Monitors calibration, uncertainty, and value-of-information to adapt thresholds, sampling, and playbook selection online
- **Human-Mimetic Memory**: Working, episodic, and procedural memories for evidence correlation and attack narrative building
- **Formal Safety Gates**: Dual corroboration with independence constraints for automation authorization
- **Confidence-Weighted Decisions**: Bayes-risk minimization under explicit safety constraints
- **Online Adaptation**: Learns from drift and analyst feedback without retraining base detectors

Evaluation Metrics
Primary Metrics (Section 5.3)

Recall@B: Detection rate under fixed false-positive budget
Long-dwell recall: Detection of multi-stage attacks before exfiltration
Alerts/hour: Volume of generated alerts at fixed recall
FP/endpoint-day: False positives normalized by fleet size
TTC: Median time-to-containment (minutes)
Escalation precision: Fraction of escalations that are true positives

Statistical Analysis (Section 5.4)

95% BCa Bootstrap CI: Bias-corrected and accelerated confidence intervals
Holm-Bonferroni correction: Multiple comparison correction for significance tests
DeLong test: Comparing ROC-AUC between models
Block bootstrap: Temporal autocorrelation adjustment (7-day lag)

Reproducibility
Data Availability

Public datasets: Automatically downloaded/generated
Enterprise corpus: Synthetic 60-day corpus with 4.7M events
Ground truth: Combination of confirmed incidents, red-team injects, analyst labels

## Results

### Table 1: Detection Performance (Recall@B)

| Dataset | Budget | SIEM | ML Ensemble | **AMCF** |
|---------|--------|------|-------------|----------|
| CIC-IDS2017 | 1.0 FP/hr | 0.791 | 0.823 | **0.847** |
| CSE-CIC-IDS2018 | 1.0 FP/hr | 0.763 | 0.804 | **0.822** |
| UNSW-NB15 | 1.0 FP/hr | 0.719 | 0.754 | **0.781** |
| Enterprise | 1.0 FP/hr | 0.774 | 0.807 | **0.836** |

### Table 2: Operational Metrics

| Metric | SIEM | ML | **AMCF** |
|--------|------|----|---------:|
| Long-dwell recall (%) | 58% | 64% | **73%** |
| Alerts/hour @ 0.80 recall | 1.42 | 1.21 | **0.94** |
| Median TTC (minutes) | 49 | 45 | **23** |
| Escalation precision | 0.61 | 0.72 | **0.82** |


### Table 3: Calibration
| Metric | SIEM | ML | AMCF |
|--------|------|----|----|
| ROC-AUC | 0.928 | 0.941 | **0.952** |
| PR-AUC | 0.824 | 0.851 | **0.868** |
| ECE | 0.058 | 0.041 | **0.024** |


## Quick Start
Run Complete Evaluation
bash# Reproduce all results (Tables 1, 2, 3 and ablation study)

# Setup
git clone https://github.com/evrendaglarli/framework.git
cd amcf-framework
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run all experiments
python experiments/run_experiments.py

# Run tests
pytest tests/ -v
```



## Key Algorithms
Algorithm	Location			Purpose	
Algorithm 1	metacognitive_controller.py	Evidence fusion and memory update
Algorithm 2	metacognitive_controller.py	Metacognitive evaluation
Algorithm 3	metacognitive_controller.py	Action execution and learning
Algorithm 4	safety_gate.py			Safety gate and actuation handler
Algorithm 5	enrichment_scheduler.py		Budgeted enrichment selection
Algorithm 6	metacognitive_controller.py	Feedback ingestion and recalibration



## Reproducing Results
```python
from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# Table 1: Detection Performance
table1 = runner.run_fixed_fp_budget_experiment(
    datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise'],
    budgets=[0.5, 1.0, 2.0]
)

# Table 2: Operational Metrics
table2 = runner.run_operational_metrics_experiment(
    datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise']
)

# Table 3: Calibration
table3 = runner.run_calibration_study(
    datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise']
)
```

## Architecture

**Sense→Reason→Act Loop:**
1. **SENSE** (Algorithm 1): Calibrate detectors, fuse probabilities, update memory
2. **REASON** (Algorithm 2): Compute risks, estimate VoI, make decisions
3. **ACT** (Algorithm 3): Execute actions, update memories

**Memory Systems:**
- **Working Memory**: Short-horizon event buffer (adaptive window)
- **Episodic Memory**: HNSW-indexed incident retrieval
- **Procedural Memory**: Parameterized playbooks

**Safety Mechanisms:**
- Dual corroboration with independence checking
- False isolation probability bounds
- Budgeted enrichment selection (Algorithm 5)

## Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=amcf --cov-report=html

# Run specific test
pytest tests/test_calibration.py::TestTemperatureScaler -v
```

## Configuration

Edit `configs/amcf_config.yaml` to adjust:
- Memory sizes
- Safety thresholds
- Cost model parameters
- VoI estimation settings

## Data

**Datasets:**
- CIC-IDS2017
- CSE-CIC-IDS2018
- UNSW-NB15
- 60-day enterprise corpus (synthetic)

All datasets are automatically generated/loaded. No external downloads required.

Data Availability Statement:
All source code, configuration files, and scripts required to reproduce the reported experiments are publicly available at: https://github.com/evrendaglarli/AMCF-Cybersecurity
. The repository also provides preprocessed versions or loaders for the public intrusion detection datasets used in this work (CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15), as well as a 60-day synthetic enterprise telemetry corpus that matches the schema and approximate statistics of the original Security Operations Center (SOC) data.
The original enterprise SOC logs (≈4.7×10^8 events) cannot be shared due to contractual data-governance and confidentiality agreements with the collaborating organization. These constraints only affect the raw enterprise stream; all analysis logic, evaluation code, and synthetic approximations are fully available in the public repository.

## Citation
```bibtex
@article{daglarli2025amcf,
  title={Adaptive Meta-Cognitive Framework for Proactive Cyber Security System},
  author={Daglarli, Evren},
  journal={},
  year={2025}
}
```

## License

MIT License - See LICENSE file

## Contact

**Evren Daglarli**
- Email: daglarli@itu.edu.tr
- Affiliation: Istanbul Technical University


## Key Files

| File | Purpose |
|------|---------|
| `amcf/core/metacognitive_controller.py` | Main controller (Algorithms 1-3) |
| `amcf/core/bayes_risk.py` | Risk computation and VoI (Equations 6, 10) |
| `amcf/safety/safety_gate.py` | Safety gate (Algorithm 4, Equations 13-14) |
| `amcf/memory/working_memory.py` | Short-horizon buffer |
| `amcf/memory/episodic_memory.py` | Incident retrieval (HNSW) |
| `experiments/run_experiments.py` | Main evaluation |


## Directory Structure

framework/
|
├── amcf/                           # Main package
│   ├── core/                       # Core algorithms
│   │   ├── calibration.py          # Temperature scaling, competence weights (Eq. 1-2)
│   │

│   │   ├── fusion.py                # Logit-weighted aggregation (Eq. 3)
│   │   ├── bayes_risk.py            # Bayes risk, VoI estimation (Eq. 6, 10)
│   │   └── metacognitive_controller.py  # Main controller (Alg. 1-3)
│   │
│   ├── memory/                     # Memory systems
│   │   ├── working_memory.py       # Short-horizon buffer (Sec. 3.1)
│   │   ├── episodic_memory.py      # HNSW incident retrieval (Sec. 3.1)
│   │   └── procedural_memory.py    # Playbook repository (Sec. 3.1)
│   │
│   ├── detectors/                  # Detector implementations
│   │   ├── base_detector.py
│   │   ├── lightgbm_detector.py
│   │   ├── tcn_detector.py
│   │   └── isolation_forest_detector.py
│   │
│   ├── baselines/                  # Baseline implementations
│   │   ├── siem_baseline.py        # Calibrated SIEM (Sec. 5.2)
│   │   └── ml_ensemble_baseline.py # ML ensemble (Sec. 5.2)
│   │
│   ├── safety/                     # Safety mechanisms
│   │   ├── safety_gate.py          # Dual corroboration (Alg. 4, Eq. 13-14)
│   │   └── enrichment_scheduler.py # VoI-driven selection (Alg. 5)
│   │
│   └── utils/                      # Utilities
│       ├── metrics.py              # Recall@B, TTC, precision (Sec. 5.3)
│       ├── statistics.py           # Bootstrap CI, Holm-Bonferroni (Sec. 5.4)
│       └── visualization.py        # Plotting functions
│
├── data/                           # Datasets
│   ├── raw/                        # Raw datasets (placeholder)
│   ├── processed/                  # Preprocessed features
│   └── synthetic/
│       ├── generate_synthetic_data.py
│       ├── enterprise_60day.pkl    # 60-day enterprise corpus
│       ├── cic_ids_2017_sample.pkl
│       └── metadata.json
│
├── experiments/                    # Evaluation scripts
│   ├── run_experiments.py          # Main experiment runner
│   ├── ablation_studies.py         # Component ablation
│   └── human_factors_analysis.py   # Human factors evaluation
│
├── configs/                        # Configuration files
│   ├── amcf_config.yaml
│   ├── siem_baseline_config.yaml
│   └── ml_ensemble_config.yaml
│
├── tests/                          # Unit tests
│   ├── test_calibration.py
│   ├── test_memory_systems.py
│   ├── test_safety_gate.py
│   └── test_end_to_end.py
│
├── results/                        # Output results
│   ├── table1_detection_performance.csv
│   ├── table2_operational_metrics.csv
│   └── ablation_study.csv
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_reproduce_table1.ipynb
│   ├── 03_reproduce_table2.ipynb
│   └── 04_reproduce_table3.ipynb
│
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
