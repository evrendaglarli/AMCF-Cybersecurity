"""
Calibrated SIEM Correlation Baseline (Section 5.2)
Production SIEM with >250 correlation rules
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional
from scipy.special import expit


class SIEMBaseline:
    """
    Production SIEM baseline with static correlation rules
    Represents vendor-grade SIEM (Splunk Enterprise Security)
    """
    
    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        self.config = config or {}
        self.random_state = random_state
        
        # Rule-based scoring (simulated)
        self.rules = self._initialize_correlation_rules()
        
        # Calibration model
        self.calibrator = None
        self.temperature = 1.0
        self.is_calibrated = False
        
    def _initialize_correlation_rules(self) -> Dict:
        """Initialize >250 simulated correlation rules"""
        rules = {}
        
        # Rule categories (simplified representation of 250+ rules)
        rule_categories = {
            'authentication': {'weight': 0.15, 'count': 40},
            'network': {'weight': 0.20, 'count': 60},
            'process': {'weight': 0.25, 'count': 70},
            'dns': {'weight': 0.10, 'count': 25},
            'http': {'weight': 0.12, 'count': 30},
            'file_access': {'weight': 0.08, 'count': 25}
        }
        
        rule_id = 0
        for category, info in rule_categories.items():
            for i in range(info['count']):
                rules[f"{category}_rule_{i}"] = {
                    'category': category,
                    'weight': info['weight'],
                    'threshold': np.random.uniform(0.3, 0.7),
                    'rule_id': rule_id
                }
                rule_id += 1
        
        return rules
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        """
        Train SIEM baseline with calibration
        """
        print("Training SIEM baseline...")
        
        # Generate rule-based scores
        train_scores = self._compute_rule_scores(X_train)
        val_scores = self._compute_rule_scores(X_val)
        
        # Temperature scaling calibration
        self._calibrate_temperature(train_scores, y_train, val_scores, y_val)
        
        self.is_calibrated = True
        print(f"  Temperature scaling factor: {self.temperature:.4f}")
    
    def _compute_rule_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute scores by evaluating correlation rules
        Simplified: aggregate feature-based rule scores
        """
        n_samples = len(X)
        scores = np.zeros(n_samples)
        
        for rule_id, rule in self.rules.items():
            # Simulate rule evaluation: features match threshold
            feature_idx = int(rule['rule_id']) % X.shape[1]
            matches = X[:, feature_idx] > rule['threshold']
            scores += rule['weight'] * matches.astype(float)
        
        # Normalize to [0, 1]
        scores = np.clip(scores / len(self.rules), 0, 1)
        
        return scores
    
    def _calibrate_temperature(self, train_scores: np.ndarray, y_train: np.ndarray,
                               val_scores: np.ndarray, y_val: np.ndarray):
        """Temperature scaling calibration (Equation 1)"""
        # Find temperature that minimizes NLL on validation set
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in np.linspace(0.5, 2.0, 30):
            calibrated = expit(np.log(np.clip(val_scores, 1e-7, 1-1e-7)) / temp)
            
            # NLL
            calibrated = np.clip(calibrated, 1e-7, 1-1e-7)
            nll = -np.mean(y_val * np.log(calibrated) + 
                          (1 - y_val) * np.log(1 - calibrated))
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict raw scores"""
        if not self.is_calibrated:
            raise ValueError("SIEM not calibrated")
        
        scores = self._compute_rule_scores(X)
        return scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities"""
        scores = self.predict(X)
        
        # Apply temperature scaling
        calibrated = expit(np.log(np.clip(scores, 1e-7, 1-1e-7)) / self.temperature)
        
        return calibrated
    
    def select_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                        target_budget: float) -> float:
        """
        Select threshold to achieve target FP budget
        target_budget: false positives per hour
        """
        scores = self.predict(X_val)
        
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        
        # Find threshold that achieves target budget
        # Simplified: select top target_budget fraction
        n_target = max(1, int(len(y_val) * target_budget / 2.0))
        
        if n_target >= len(sorted_indices):
            threshold = -np.inf
        else:
            threshold = scores[sorted_indices[n_target]]
        
        return threshold