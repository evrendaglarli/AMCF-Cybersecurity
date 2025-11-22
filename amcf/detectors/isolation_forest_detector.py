"""
Isolation Forest Detector for unsupervised anomaly scoring
Part of ML ensemble baseline (Section 5.2)
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any
from .base_detector import BaseDetector


class IsolationForestDetector(BaseDetector):
    """Unsupervised anomaly detector using Isolation Forest"""
    
    def __init__(self, detector_id: str = 'isolation_forest',
                 contamination: float = 0.05,
                 random_state: int = 42):
        super().__init__(detector_id)
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        self.contamination = contamination
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train Isolation Forest
        Note: unsupervised, so y_train is ignored for training
        but used for validation
        """
        print(f"Training {self.detector_id}...")
        self.model.fit(X_train)
        self.is_trained = True
        
        # Validate on validation set if provided
        if X_val is not None and y_val is not None:
            val_scores = self.predict_proba(X_val)
            val_auc = self._compute_auc(y_val, val_scores)
            print(f"  Validation AUC: {val_auc:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores
        Isolation Forest returns decision scores, convert to [0,1]
        Lower (more negative) = more anomalous
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get decision scores (negative = anomalous)
        decision_scores = self.model.decision_function(X)
        
        # Convert to [0, 1]: higher score = more anomalous
        # Use sigmoid-like transformation
        scores = 1.0 / (1.0 + np.exp(decision_scores))
        
        return scores
    
    def extract_features(self, event: Any) -> np.ndarray:
        """Extract features from event"""
        features = []
        
        # Hash-based features
        features.append(hash(event.src) % 10000)
        features.append(hash(event.dst or '') % 10000)
        features.append(len(event.attrs) if event.attrs else 0)
        
        # Attribute-based features (simplified)
        if event.attrs:
            if 'bytes_sent' in event.attrs:
                features.append(min(event.attrs['bytes_sent'], 100000))
            if 'bytes_recv' in event.attrs:
                features.append(min(event.attrs['bytes_recv'], 100000))
        
        return np.array(features, dtype=float)
    
    @staticmethod
    def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute AUC-ROC"""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_score)
        except:
            return 0.5