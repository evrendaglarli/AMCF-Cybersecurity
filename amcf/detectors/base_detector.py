"""
Base detector interface
All detectors inherit from this class
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, detector_id: str):
        self.detector_id = detector_id
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        """Train detector"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores (not probabilities yet)"""
        pass
    
    @abstractmethod
    def extract_features(self, event: Any) -> np.ndarray:
        """Extract features from event"""
        pass
    
    def get_detector_id(self) -> str:
        """Get detector identifier"""
        return self.detector_id
    
    def is_ready(self) -> bool:
        """Check if detector is trained"""
        return self.is_trained