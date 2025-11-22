"""
Logit-Weighted Fusion for Multi-Detector Aggregation
Implements Equation (3) from the paper
"""

import numpy as np
from typing import Optional

class LogitWeightedFusion:
    """
    Fuse multiple detector outputs using competence-weighted logit aggregation
    Equation (3): logit(π_t) = Σ w_{k,t} * logit(p_{k,t}) + b_t
    """
    
    def __init__(self, num_detectors: int):
        self.num_detectors = num_detectors
        
    def fuse(self, detector_probs: np.ndarray, weights: np.ndarray,
             bias: float = 0.0) -> np.ndarray:
        """
        Compute fused posterior via logit aggregation
        
        Args:
            detector_probs: (N, K) calibrated probabilities from K detectors
            weights: (K,) competence weights
            bias: Contextual prior (asset criticality, user role)
            
        Returns:
            π_t: (N,) fused posterior probabilities
        """
        # Convert to logits
        detector_probs = np.clip(detector_probs, 1e-7, 1 - 1e-7)
        logits = self._logit(detector_probs)  # (N, K)
        
        # Weighted sum
        fused_logit = np.dot(logits, weights) + bias  # (N,)
        
        # Back to probability
        posterior = self._sigmoid(fused_logit)
        
        return posterior
    
    @staticmethod
    def _logit(p):
        return np.log(p / (1 - p))
    
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))