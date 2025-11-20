# AMCF-Cybersecurity
Adaptive Meta-Cognitive Framework for Proactive Cyber Security System - Complete Implementation

Project structure :

amcf-framework/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── environment.yml
│
├── amcf/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── calibration.py          # Temperature scaling, competence weights
│   │   ├── fusion.py                # Logit-weighted aggregation
│   │   ├── bayes_risk.py            # Risk computation, VoI estimation
│   │   └── metacognitive_controller.py  # Main controller (Algorithms 1-3)
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working_memory.py        # Short-horizon buffer
│   │   ├── episodic_memory.py       # HNSW-based retrieval
│   │   └── procedural_memory.py     # Playbook repository
│   │
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base_detector.py
│   │   ├── lightgbm_detector.py
│   │   ├── tcn_detector.py          # Temporal Convolutional Network
│   │   └── isolation_forest_detector.py
│   │
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── safety_gate.py           # Dual corroboration, independence checks
│   │   └── enrichment_scheduler.py  # Algorithm 5: Budgeted selection
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── siem_baseline.py         # Calibrated SIEM correlation
│   │   └── ml_ensemble_baseline.py  # LightGBM + TCN + IsolationForest
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py               # Recall@B, TTC, escalation precision
│       ├── statistics.py            # BCa bootstrap, DeLong test, Holm-Bonferroni
│       └── visualization.py         # Plotting functions
│
├── data/
│   ├── README.md
│   ├── raw/                         # Placeholder for CIC-IDS2017, etc.
│   ├── processed/                   # Preprocessed features
│   ├── synthetic/                   # Synthetic test data
│   │   ├── enterprise_60day.pkl
│   │   ├── cic_ids_2017_sample.pkl
│   │   └── metadata.json
│   └── ground_truth/
│       └── labels.csv
│
├── configs/
│   ├── amcf_config.yaml
│   ├── siem_baseline_config.yaml
│   └── ml_ensemble_config.yaml
│
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py           # Main evaluation script
│   ├── ablation_studies.py
│   └── human_factors_analysis.py
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_reproduce_table1.ipynb   # Detection performance
│   ├── 03_reproduce_table2.ipynb   # Operational metrics
│   └── 04_reproduce_table3.ipynb   # ROC-AUC/PR-AUC
│
├── tests/
│   ├── __init__.py
│   ├── test_calibration.py
│   ├── test_memory_systems.py
│   ├── test_safety_gate.py
│   └── test_end_to_end.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── ablations/
│
└── docs/
    ├── architecture.md
    ├── algorithms.md
    └── reproducibility.md
