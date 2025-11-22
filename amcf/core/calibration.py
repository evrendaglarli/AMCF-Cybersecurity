"""
Temperature Scaling and Competence Weight Management
Implements Equations (1) and (2) from the paper
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple

class TemperatureScaler:
    """Online temperature scaling for detector calibration"""
    
    def __init__(self, detector_id: str, update_frequency: int = 200):
        self.detector_id = detector_id
        self.update_frequency = update_frequency
        self.temperature = 1.0
        self.sample_buffer = []
        self.label_buffer = []
        
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling: p = σ(s/T)
        
        Args:
            scores: Raw detector scores (N,)
            
        Returns:
            Calibrated probabilities (N,)
        """
        return self._sigmoid(scores / self.temperature)
    
    def update(self, scores: np.ndarray, labels: np.ndarray):
        """
        Update temperature parameter using NLL minimization
        Equation (1): T_new = argmin_T Σ[y*log(σ(s/T)) + (1-y)*log(1-σ(s/T))]
        """
        self.sample_buffer.extend(scores)
        self.label_buffer.extend(labels)
        
        if len(self.sample_buffer) >= self.update_frequency:
            # Take last 200 samples
            recent_scores = np.array(self.sample_buffer[-self.update_frequency:])
            recent_labels = np.array(self.label_buffer[-self.update_frequency:])
            
            # Optimize temperature
            def nll(T):
                T = T[0]
                if T <= 0:
                    return 1e10
                probs = self._sigmoid(recent_scores / T)
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                return -np.mean(
                    recent_labels * np.log(probs) + 
                    (1 - recent_labels) * np.log(1 - probs)
                )
            
            result = minimize(nll, x0=[self.temperature], method='Nelder-Mead')
            self.temperature = max(0.1, result.x[0])  # Prevent collapse
            
            # Keep buffer bounded
            self.sample_buffer = self.sample_buffer[-self.update_frequency:]
            self.label_buffer = self.label_buffer[-self.update_frequency:]
    
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class CompetenceWeightManager:
    """
    Manage detector competence weights with EMA updates
    Equation (2): w_{k,t+1} ← (1-λ)(w_{k,t} - η∇L_Brier) + λw_{k,t}
    """
    
    def __init__(self, num_detectors: int, lambda_ema: float = 0.96, 
                 learning_rate: float = 0.01):
        self.num_detectors = num_detectors
        self.lambda_ema = lambda_ema
        self.learning_rate = learning_rate
        
        # Initialize weights uniformly
        self.weights = np.ones(num_detectors) / num_detectors
        
    def update(self, detector_probs: np.ndarray, labels: np.ndarray):
        """
        Update weights using Brier score gradient
        
        Args:
            detector_probs: (N, K) array of calibrated probabilities
            labels: (N,) array of binary labels
        """
        # Compute Brier score gradient for each detector
        N, K = detector_probs.shape
        gradients = np.zeros(K)
        
        for k in range(K):
            # Brier score: E[(p_k - y)^2]
            brier_grad = 2 * np.mean(
                (detector_probs[:, k] - labels) * detector_probs[:, k]
            )
            gradients[k] = brier_grad
        
        # EMA update with gradient step
        new_weights = self.weights - self.learning_rate * gradients
        self.weights = (1 - self.lambda_ema) * new_weights + self.lambda_ema * self.weights
        
        # Normalize and ensure positivity
        self.weights = np.maximum(self.weights, 0.01)
        self.weights /= self.weights.sum()
    
    def get_weights(self) -> np.ndarray:
        return self.weights.copy()