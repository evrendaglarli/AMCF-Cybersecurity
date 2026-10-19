"""
AMCF reference evaluation runner.

Prepared datasets are required by default. --synthetic-demo uses randomly
simulated data for illustration only; it cannot reproduce manuscript results.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import yaml
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from amcf.core.metacognitive_controller import MetacognitiveController, Event, CostModel
from amcf.baselines.siem_baseline import SIEMBaseline
from amcf.baselines.ml_ensemble_baseline import MLEnsembleBaseline
from amcf.utils.metrics import (
    compute_recall_at_budget, compute_ttc, compute_escalation_precision,
    compute_long_dwell_recall, compute_alerts_per_hour, compute_fp_per_endpoint_day,
    compute_roc_auc, compute_pr_auc, compute_expected_calibration_error
)
from amcf.utils.statistics import bootstrap_ci, bootstrap_ci_precision, delong_test
from amcf.utils.visualization import (
    plot_recall_comparison, plot_operational_metrics, plot_ablation_results
)


class ExperimentRunner:
    """Main experiment orchestration"""
    
    def __init__(self, config_file: str = 'configs/amcf_config.yaml',
                 allow_synthetic_demo: bool = False):
        """Initialize experiment runner"""
        self.config_file = config_file
        self.allow_synthetic_demo = allow_synthetic_demo
        self.tables_dir = Path('results/demo/tables' if allow_synthetic_demo else 'results/tables')
        
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.results = {}
        
        # Create output directories
        Path('results').mkdir(exist_ok=True)
        Path('results/figures').mkdir(exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple:
        """
        Load preprocessed dataset
        
        Args:
            dataset_name: 'cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise'
            
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test, events_test)
        """
        data_path = Path('data/processed') / f'{dataset_name}.pkl'
        
        if self.allow_synthetic_demo:
            print(f"  [SYNTHETIC DEMO, NOT PAPER RESULTS] {dataset_name}")
            return self._generate_synthetic_data(dataset_name)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Missing prepared dataset: {data_path}. "
                "The repository does not download or bundle the public datasets. "
                "Supply verified processed files, or use --synthetic-demo "
                "for random demonstration data (NOT paper reproduction)."
            )
        
        print(f"  Loading {dataset_name}...")
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    
    def run_fixed_fp_budget_experiment(self, datasets: List[str],
                                       budgets: List[float]) -> pd.DataFrame:
        """
        Reproduce Table 1: Detection Performance Under Fixed False-Positive Budgets
        
        Args:
            datasets: List of dataset names
            budgets: False-positive budgets B ∈ {0.5, 1.0, 2.0}
            
        Returns:
            DataFrame with Recall@B for each model/dataset/budget combination
        """
        print("\n" + "="*80)
        print("TABLE 1: DETECTION PERFORMANCE UNDER FIXED FALSE-POSITIVE BUDGETS")
        print("="*80)
        
        results = []
        
        for dataset_name in datasets:
            print(f"\nDataset: {dataset_name}")
            print("-" * 60)
            
            X_train, y_train, X_val, y_val, X_test, y_test, events_test = \
                self.load_dataset(dataset_name)
            
            # Initialize models
            print("  Initializing models...")
            siem = SIEMBaseline(self.config.get('siem_config', {}))
            ml_ensemble = MLEnsembleBaseline(self.config.get('ml_config', {}))
            
            cost_model = CostModel()
            amcf = MetacognitiveController(
                num_detectors=3,
                detector_ids=['lightgbm', 'tcn', 'isolation_forest'],
                cost_model=cost_model,
                config=self.config.get('amcf_config', {})
            )
            
            # Train models
            print("  Training models...")
            siem.train(X_train, y_train, X_val, y_val)
            ml_ensemble.train(X_train, y_train, X_val, y_val)
            
            # Evaluate at each budget
            for budget in budgets:
                print(f"\n  Budget B = {budget} FP/hour")
                
                # Get predictions
                siem_scores = siem.predict(X_test)
                siem_threshold = siem.select_threshold(X_val, y_val, target_budget=budget)
                siem_preds = (siem_scores >= siem_threshold).astype(int)
                
                ml_scores = ml_ensemble.predict_proba(X_test)
                ml_threshold = ml_ensemble.select_threshold(X_val, y_val, target_budget=budget)
                ml_preds = (ml_scores >= ml_threshold).astype(int)
                
                # AMCF predictions
                amcf_preds = []
                amcf_scores = []
                for event in events_test:
                    action, metadata = amcf.process_event(event)
                    pred = 1 if action in ['automate', 'escalate'] else 0
                    amcf_preds.append(pred)
                    amcf_scores.append(metadata['posterior'])
                amcf_preds = np.array(amcf_preds)
                amcf_scores = np.array(amcf_scores)
                
                # Compute Recall@B
                siem_recall = compute_recall_at_budget(y_test, siem_preds)
                ml_recall = compute_recall_at_budget(y_test, ml_preds)
                amcf_recall = compute_recall_at_budget(y_test, amcf_preds)
                
                # Bootstrap CI
                siem_ci = bootstrap_ci(y_test, siem_preds, n_bootstrap=10000)
                ml_ci = bootstrap_ci(y_test, ml_preds, n_bootstrap=10000)
                amcf_ci = bootstrap_ci(y_test, amcf_preds, n_bootstrap=10000)
                
                results.append({
                    'dataset': dataset_name,
                    'budget': budget,
                    'siem_recall': siem_recall,
                    'siem_ci_lower': siem_ci[0],
                    'siem_ci_upper': siem_ci[1],
                    'ml_recall': ml_recall,
                    'ml_ci_lower': ml_ci[0],
                    'ml_ci_upper': ml_ci[1],
                    'amcf_recall': amcf_recall,
                    'amcf_ci_lower': amcf_ci[0],
                    'amcf_ci_upper': amcf_ci[1]
                })
                
                print(f"    SIEM:  {siem_recall:.4f} [{siem_ci[0]:.4f}, {siem_ci[1]:.4f}]")
                print(f"    ML:    {ml_recall:.4f} [{ml_ci[0]:.4f}, {ml_ci[1]:.4f}]")
                print(f"    AMCF:  {amcf_recall:.4f} [{amcf_ci[0]:.4f}, {amcf_ci[1]:.4f}]")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.tables_dir / 'table1_detection_performance.csv', index=False)
        
        print("\n" + "="*80)
        print(f"Table 1 saved to {self.tables_dir / 'table1_detection_performance.csv'}")
        print("="*80)
        
        return results_df
    
    def run_operational_metrics_experiment(self, datasets: List[str]) -> pd.DataFrame:
        """
        Reproduce Table 2: Operational Impact Metrics
        """
        print("\n" + "="*80)
        print("TABLE 2: OPERATIONAL IMPACT METRICS")
        print("="*80)
        
        results = []
        
        for dataset_name in datasets:
            print(f"\nDataset: {dataset_name}")
            print("-" * 60)
            
            X_train, y_train, X_val, y_val, X_test, y_test, events_test = \
                self.load_dataset(dataset_name)
            
            # Train models
            print("  Training models...")
            siem = SIEMBaseline()
            ml_ensemble = MLEnsembleBaseline()
            siem.train(X_train, y_train, X_val, y_val)
            ml_ensemble.train(X_train, y_train, X_val, y_val)
            
            # AMCF
            amcf = MetacognitiveController(
                num_detectors=3,
                detector_ids=['lightgbm', 'tcn', 'isolation_forest'],
                cost_model=CostModel(),
                config=self.config.get('amcf_config', {})
            )
            
            # Get predictions
            siem_preds = siem.predict(X_test)
            ml_preds = ml_ensemble.predict(X_test)
            
            amcf_preds = []
            for event in events_test:
                action, _ = amcf.process_event(event)
                pred = 1 if action in ['automate', 'escalate'] else 0
                amcf_preds.append(pred)
            amcf_preds = np.array(amcf_preds)
            
            # Compute metrics
            print("  Computing metrics...")
            
            # Long-dwell recall
            siem_dwell = compute_long_dwell_recall(y_test, siem_preds, events=events_test)
            ml_dwell = compute_long_dwell_recall(y_test, ml_preds, events=events_test)
            amcf_dwell = compute_long_dwell_recall(y_test, amcf_preds, events=events_test)
            
            # Alerts per hour
            siem_alerts = compute_alerts_per_hour(events_test, siem_preds)
            ml_alerts = compute_alerts_per_hour(events_test, ml_preds)
            amcf_alerts = compute_alerts_per_hour(events_test, amcf_preds)
            
            # FP per endpoint-day
            siem_fp_rate = compute_fp_per_endpoint_day(events_test, siem_preds, y_test)
            ml_fp_rate = compute_fp_per_endpoint_day(events_test, ml_preds, y_test)
            amcf_fp_rate = compute_fp_per_endpoint_day(events_test, amcf_preds, y_test)
            
            # TTC
            siem_ttc = compute_ttc(events_test, y_test, siem_preds)
            ml_ttc = compute_ttc(events_test, y_test, ml_preds)
            amcf_ttc = compute_ttc(events_test, y_test, amcf_preds)
            
            # Escalation precision
            siem_prec = compute_escalation_precision(y_test, siem_preds)
            ml_prec = compute_escalation_precision(y_test, ml_preds)
            amcf_prec = compute_escalation_precision(y_test, amcf_preds)
            
            results.append({
                'dataset': dataset_name,
                'siem_dwell_recall': siem_dwell,
                'ml_dwell_recall': ml_dwell,
                'amcf_dwell_recall': amcf_dwell,
                'siem_alerts_hr': siem_alerts,
                'ml_alerts_hr': ml_alerts,
                'amcf_alerts_hr': amcf_alerts,
                'siem_fp_ep_day': siem_fp_rate,
                'ml_fp_ep_day': ml_fp_rate,
                'amcf_fp_ep_day': amcf_fp_rate,
                'siem_ttc_min': siem_ttc,
                'ml_ttc_min': ml_ttc,
                'amcf_ttc_min': amcf_ttc,
                'siem_esc_prec': siem_prec,
                'ml_esc_prec': ml_prec,
                'amcf_esc_prec': amcf_prec
            })
            
            print(f"  Long-dwell recall: SIEM={siem_dwell:.3f}, ML={ml_dwell:.3f}, AMCF={amcf_dwell:.3f}")
            print(f"  TTC (min): SIEM={siem_ttc:.1f}, ML={ml_ttc:.1f}, AMCF={amcf_ttc:.1f}")
            print(f"  Escalation precision: SIEM={siem_prec:.3f}, ML={ml_prec:.3f}, AMCF={amcf_prec:.3f}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.tables_dir / 'table2_operational_metrics.csv', index=False)
        
        print("\n" + "="*80)
        print(f"Table 2 saved to {self.tables_dir / 'table2_operational_metrics.csv'}")
        print("="*80)
        
        return results_df
    
    def run_calibration_study(self, datasets: List[str]) -> pd.DataFrame:
        """
        Reproduce Table 3: Calibration Performance (ROC-AUC, PR-AUC, ECE)
        """
        print("\n" + "="*80)
        print("TABLE 3: CALIBRATION AND DISCRIMINATION METRICS")
        print("="*80)
        
        results = []
        
        for dataset_name in datasets:
            print(f"\nDataset: {dataset_name}")
            print("-" * 60)
            
            X_train, y_train, X_val, y_val, X_test, y_test, events_test = \
                self.load_dataset(dataset_name)
            
            # Train models
            print("  Training models...")
            siem = SIEMBaseline()
            ml_ensemble = MLEnsembleBaseline()
            siem.train(X_train, y_train, X_val, y_val)
            ml_ensemble.train(X_train, y_train, X_val, y_val)
            
            # Get scores
            siem_scores = siem.predict(X_test)
            ml_scores = ml_ensemble.predict_proba(X_test)
            
            amcf_scores = []
            amcf = MetacognitiveController(
                num_detectors=3,
                detector_ids=['lightgbm', 'tcn', 'isolation_forest'],
                cost_model=CostModel(),
                config=self.config.get('amcf_config', {})
            )
            for event in events_test:
                _, metadata = amcf.process_event(event)
                amcf_scores.append(metadata['posterior'])
            amcf_scores = np.array(amcf_scores)
            
            # Compute metrics
            print("  Computing calibration metrics...")
            
            siem_roc = compute_roc_auc(y_test, siem_scores)
            ml_roc = compute_roc_auc(y_test, ml_scores)
            amcf_roc = compute_roc_auc(y_test, amcf_scores)
            
            siem_pr = compute_pr_auc(y_test, siem_scores)
            ml_pr = compute_pr_auc(y_test, ml_scores)
            amcf_pr = compute_pr_auc(y_test, amcf_scores)
            
            siem_ece = compute_expected_calibration_error(y_test, siem_scores)
            ml_ece = compute_expected_calibration_error(y_test, ml_scores)
            amcf_ece = compute_expected_calibration_error(y_test, amcf_scores)
            
            results.append({
                'dataset': dataset_name,
                'siem_roc_auc': siem_roc,
                'ml_roc_auc': ml_roc,
                'amcf_roc_auc': amcf_roc,
                'siem_pr_auc': siem_pr,
                'ml_pr_auc': ml_pr,
                'amcf_pr_auc': amcf_pr,
                'siem_ece': siem_ece,
                'ml_ece': ml_ece,
                'amcf_ece': amcf_ece
            })
            
            print(f"  ROC-AUC: SIEM={siem_roc:.4f}, ML={ml_roc:.4f}, AMCF={amcf_roc:.4f}")
            print(f"  PR-AUC: SIEM={siem_pr:.4f}, ML={ml_pr:.4f}, AMCF={amcf_pr:.4f}")
            print(f"  ECE: SIEM={siem_ece:.4f}, ML={ml_ece:.4f}, AMCF={amcf_ece:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.tables_dir / 'table3_calibration.csv', index=False)
        
        print("\n" + "="*80)
        print(f"Table 3 saved to {self.tables_dir / 'table3_calibration.csv'}")
        print("="*80)
        
        return results_df
    
    def _generate_synthetic_data(self, dataset_name: str) -> Tuple:
        """Generate random illustration-only events, not real benchmark/SOC data."""
        
        # Dataset sizes
        if dataset_name == 'cic_ids_2017':
            n_samples = 10_000
            n_endpoints = 50
            pos_rate = 0.05
        elif dataset_name == 'cse_ids_2018':
            n_samples = 12_000
            n_endpoints = 60
            pos_rate = 0.04
        elif dataset_name == 'unsw_nb15':
            n_samples = 8_000
            n_endpoints = 40
            pos_rate = 0.06
        else:  # enterprise
            n_samples = 25_000
            n_endpoints = 100
            pos_rate = 0.05
        
        # Split
        n_train = int(n_samples * 0.5)
        n_val = int(n_samples * 0.25)
        n_test = n_samples - n_train - n_val
        
        # Generate features
        X = np.random.randn(n_samples, 50)
        y = np.random.binomial(1, pos_rate, n_samples)
        
        X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
        y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
        
        # Generate events
        base_ts = 1718190000000
        events_test = []
        
        for i in range(n_test):
            ts = base_ts + i * 1000
            event = Event(
                ts=ts,
                src=f"host_{i % n_endpoints}",
                dst=f"host_{(i+1) % n_endpoints}" if np.random.rand() > 0.4 else None,
                kind=np.random.choice(['process', 'netflow', 'auth', 'dns', 'http', 'edr']),
                attrs={'bytes_sent': np.random.randint(100, 10000)},
                detector_scores={
                    'lightgbm': np.random.randn(),
                    'tcn': np.random.randn(),
                    'isolation_forest': np.random.randn()
                },
                label=y_test[i]
            )
            events_test.append(event)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, events_test


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--synthetic-demo', action='store_true',
        help='Use random synthetic data for missing inputs. Results are demonstrations only.'
    )
    args = parser.parse_args()
    print("\n" + "="*80)
    print("AMCF: ADAPTIVE METACOGNITIVE FRAMEWORK")
    print("SYNTHETIC DATA DEMONSTRATION - NOT PAPER REPRODUCTION" if args.synthetic_demo
          else "PREPARED DATASET EVALUATION - NOT INDEPENDENTLY VERIFIED")
    print("="*80)
    
    runner = ExperimentRunner('configs/amcf_config.yaml',
                              allow_synthetic_demo=args.synthetic_demo)
    
    # Run the three reference evaluation routines.
    start_time = time.time()
    
    # Table 1
    table1 = runner.run_fixed_fp_budget_experiment(
        datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise'],
        budgets=[0.5, 1.0, 2.0]
    )
    
    # Table 2
    table2 = runner.run_operational_metrics_experiment(
        datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise']
    )
    
    # Table 3
    table3 = runner.run_calibration_study(
        datasets=['cic_ids_2017', 'cse_ids_2018', 'unsw_nb15', 'enterprise']
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print("\nResults saved to:")
    print(f"  - {runner.tables_dir / 'table1_detection_performance.csv'}")
    print(f"  - {runner.tables_dir / 'table2_operational_metrics.csv'}")
    print(f"  - {runner.tables_dir / 'table3_calibration.csv'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()