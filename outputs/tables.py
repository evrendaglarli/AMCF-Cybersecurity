from experiments.run_experiments import ExperimentRunner

runner = ExperimentRunner('configs/amcf_config.yaml')

# Table 1: Detection Performance
table1 = runner.run_fixed_fp_budget_experiment(
    datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise'],
    budgets=[0.5, 1.0, 2.0]
)
print(table1)

# Table 2: Operational Metrics
table2 = runner.run_operational_metrics_experiment(
    datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise']
)
print(table2)

# Ablation Study
ablation = runner.run_ablation_study(dataset_name='enterprise')
print(ablation)
```

## Architecture Overview

### Sense→Reason→Act Loop
```
Event Stream (EDR, SIEM, NDR)
        ↓
    ┌─────────────────────────────────────┐
    │  SENSE (Algorithm 1)                │
    │  - Calibrate detector outputs       │
    │  - Fuse via logit aggregation (Eq.3)│
    │  - Update working memory            │
    │  - Retrieve episodic analogues      │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │  REASON (Algorithm 2)               │
    │  - Compute hypothesis coherence     │
    │  - Evaluate Bayes risks (Eq.6)      │
    │  - Estimate VoI (Eq.10)             │
    │  - Metacognitive decision policy    │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │  ACT (Algorithm 3)                  │
    │  - Safety gate check (Alg.4)        │
    │  - Execute playbook or escalate     │
    │  - Update memories & weights        │
    └─────────────────────────────────────┘
        ↓
   {ignore, gather, automate, escalate}