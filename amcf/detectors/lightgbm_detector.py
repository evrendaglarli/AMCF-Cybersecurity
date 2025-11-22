"""
LightGBM Detector for event classification
Part of ML ensemble baseline (Section 5.2)
"""

import numpy as np
import lightgbm as lgb
from typing import Dict
from .base_detector import BaseDetector

class LightGBMDetector(BaseDetector):
    """Gradient Boosted Decision Tree detector"""
    
    def __init__(self, detector_id: str = 'lightgbm'):
        super().__init__(detector_id)
        self.model = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        """
        Train LightGBM with class-weighted loss and early stopping
        """
        # Compute class weights
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'scale_pos_weight': pos_weight
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=['validation'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores (to be calibrated later)"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        scores = self.model.predict(X)
        return scores
    
    def extract_features(self, event: 'Event') -> np.ndarray:
        """Extract features from event"""
        # Placeholder: extract relevant features
        # In full implementation, engineer features from event attributes
        features = []
        
        # Example features
        features.append(hash(event.src) % 10000)  # Source hash
        features.append(hash(event.dst or '') % 10000)  # Dest hash
        features.append(len(event.attrs) if event.attrs else 0)
        
        return np.array(features)