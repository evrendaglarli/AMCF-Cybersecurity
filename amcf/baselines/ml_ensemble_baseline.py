"""
ML Ensemble Baseline (Section 5.2)
Combines LightGBM + TCN + Isolation Forest with stacking
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..detectors.lightgbm_detector import LightGBMDetector
from ..detectors.tcn_detector import TCNDetector
from ..detectors.isolation_forest_detector import IsolationForestDetector


class MLEnsembleBaseline:
    """
    ML Ensemble: LightGBM + TCN + IsolationForest
    Meta-learner: Stacking with Logistic Regression
    """
    
    def __init__(self, config: Optional[Dict] = None, random_state: int = 42):
        self.config = config or {}
        self.random_state = random_state
        
        # Individual detectors
        self.lightgbm = LightGBMDetector()
        self.tcn = TCNDetector()
        self.isolation_forest = IsolationForestDetector()
        
        self.detectors = [self.lightgbm, self.tcn, self.isolation_forest]
        self.detector_ids = ['lightgbm', 'tcn', 'isolation_forest']
        
        # Meta-learner
        self.meta_learner = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            verbose=0
        )
        
        self.is_trained = False
        self.threshold = 0.5
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        """
        Train ensemble with stacking
        1. Train individual detectors on train set
        2. Generate meta-features on validation set
        3. Train meta-learner on meta-features
        """
        print("Training ML Ensemble baseline...")
        
        # Step 1: Train individual detectors
        print("  Training LightGBM...")
        self.lightgbm.train(X_train, y_train, X_val, y_val)
        
        print("  Training TCN...")
        self.tcn.train(X_train, y_train, X_val, y_val, epochs=50)
        
        print("  Training Isolation Forest...")
        self.isolation_forest.train(X_train, y_train, X_val, y_val)
        
        # Step 2: Generate meta-features on validation set
        print("  Generating meta-features...")
        lgb_preds = self.lightgbm.predict_proba(X_val).reshape(-1, 1)
        tcn_preds = self.tcn.predict_proba(X_val).reshape(-1, 1)
        if_preds = self.isolation_forest.predict_proba(X_val).reshape(-1, 1)
        
        meta_features = np.hstack([lgb_preds, tcn_preds, if_preds])
        
        # Step 3: Train meta-learner
        print("  Training meta-learner...")
        self.meta_learner.fit(meta_features, y_val)
        
        self.is_trained = True
        
        # Validation AUC
        ensemble_preds = self.meta_learner.predict_proba(meta_features)[:, 1]
        auc = roc_auc_score(y_val, ensemble_preds)
        print(f"  Ensemble Validation AUC: {auc:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ensemble probabilities
        1. Get predictions from individual detectors
        2. Stack them
        3. Apply meta-learner
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        # Get individual predictions
        lgb_preds = self.lightgbm.predict_proba(X).reshape(-1, 1)
        tcn_preds = self.tcn.predict_proba(X).reshape(-1, 1)
        if_preds = self.isolation_forest.predict_proba(X).reshape(-1, 1)
        
        # Stack
        meta_features = np.hstack([lgb_preds, tcn_preds, if_preds])
        
        # Meta-learner prediction
        ensemble_scores = self.meta_learner.predict_proba(meta_features)[:, 1]
        
        return ensemble_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions"""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
    
    def select_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                        target_budget: float) -> float:
        """Select threshold for target FP budget"""
        scores = self.predict_proba(X_val)
        
        sorted_indices = np.argsort(-scores)
        n_target = max(1, int(len(y_val) * target_budget / 2.0))
        
        if n_target >= len(sorted_indices):
            threshold = -np.inf
        else:
            threshold = scores[sorted_indices[n_target]]
        
        self.threshold = threshold
        return threshold
    
    def get_detectors(self) -> List:
        """Get trained detectors for AMCF"""
        return self.detectors