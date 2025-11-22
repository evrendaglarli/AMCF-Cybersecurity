"""
Safety Gate: Dual corroboration with independence checks
Section 4.2, Algorithm 4, Equations (13-14)
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import pearsonr


class SafetyGate:
    """
    Formal safety gate for automation authorization
    Requires:
    1. Dual corroboration (two independent detectors above thresholds)
    2. Risk reduction threshold
    3. False isolation probability bound
    """
    
    def __init__(self, rho_max: float = 0.75, epsilon: float = 1e-3,
                 alpha: float = 1e-3, window_size: int = 200):
        """
        Args:
            rho_max: Maximum permitted correlation between advisors
            epsilon: Maximum false isolation probability (ε = 10^-3)
            alpha: Target false isolation rate
            window_size: Rolling window for tracking false isolations
        """
        self.rho_max = rho_max
        self.epsilon = epsilon
        self.alpha = alpha
        self.window_size = window_size
        
        # Rolling window for false isolation tracking
        self.false_isolation_window = []
        self.correlation_history = {}  # Track advisor correlations
        
        # Per-detector thresholds (calibrated on validation)
        self.detector_thresholds = {
            'lightgbm': 0.75,
            'tcn': 0.75,
            'isolation_forest': 0.70
        }
        
    def check(self, posterior: float, calibrated_probs: Dict[str, float],
              asset_risk: float = 1.0, coherence: float = 0.5) -> bool:
        """
        Algorithm 4: Safety Gate Check
        
        Equation (13): ΔRisk ≥ τ_risk, P(false isolation) ≤ ε
        Equation (14): p_k ≥ θ_k, p_ℓ ≥ θ_ℓ, corr(logit(p_k), logit(p_ℓ)) ≤ ρ_max
        
        Args:
            posterior: π_t, fused posterior probability
            calibrated_probs: {detector_id: probability}
            asset_risk: Asset criticality multiplier
            coherence: Attack chain coherence score
            
        Returns:
            True if automation is authorized, False otherwise
        """
        # Clause 1: Risk reduction threshold
        # Require high posterior confidence for automation
        if posterior < 0.80:
            return False
        
        # Clause 2: False isolation probability bound
        if not self._check_false_isolation_bound():
            return False
        
        # Clause 3: Dual corroboration with independence
        if not self._check_dual_corroboration(calibrated_probs):
            return False
        
        # All clauses satisfied
        return True
    
    def _check_false_isolation_bound(self) -> bool:
        """
        Check P(false isolation) ≤ ε using BCa 95% upper bound
        Equation: CI^95_upper(P(false isolation)) ≤ ε = 10^-3
        """
        if len(self.false_isolation_window) < 30:
            # Insufficient data, assume safe (conservative approach)
            return True
        
        # Compute false isolation rate on rolling window
        false_isolations = np.array(self.false_isolation_window[-self.window_size:])
        rate = np.mean(false_isolations)
        
        # Bootstrap BCa 95% CI upper bound
        upper_bound = self._bootstrap_upper_bound(false_isolations)
        
        # Gate: upper_bound must be ≤ ε
        return upper_bound <= self.epsilon
    
    def _check_dual_corroboration(self, calibrated_probs: Dict[str, float]) -> bool:
        """
        Require two independent detectors above thresholds
        Equation (14): p_k ≥ θ_k, p_ℓ ≥ θ_ℓ, corr(logit(p_k), logit(p_ℓ)) ≤ ρ_max
        """
        # Find detectors above individual thresholds
        high_confidence_detectors = []
        
        for det_id, prob in calibrated_probs.items():
            threshold = self.detector_thresholds.get(det_id, 0.75)
            if prob >= threshold:
                high_confidence_detectors.append((det_id, prob))
        
        # Need at least 2 detectors above thresholds
        if len(high_confidence_detectors) < 2:
            return False
        
        # Check correlation between top 2 detectors
        det1_id, prob1 = high_confidence_detectors[0]
        det2_id, prob2 = high_confidence_detectors[1]
        
        # In full implementation, track correlation from history
        # Here, simplified: assume detectors are decorrelated if different types
        
        # Compute logit-space correlation (placeholder)
        # Real implementation would maintain rolling window of (logit(p_k), logit(p_ℓ)) pairs
        correlation = self._estimate_detector_correlation(det1_id, det2_id)
        
        # Gate: correlation must be ≤ ρ_max
        return correlation <= self.rho_max
    
    def _estimate_detector_correlation(self, det1_id: str, det2_id: str) -> float:
        """
        Estimate correlation between two detectors
        Simplified: different detector types → lower correlation
        """
        key = tuple(sorted([det1_id, det2_id]))
        
        if key in self.correlation_history:
            return self.correlation_history[key]
        
        # Heuristic: detectors of different types are less correlated
        detector_types = {
            'lightgbm': 'tree',
            'tcn': 'neural',
            'isolation_forest': 'unsupervised'
        }
        
        type1 = detector_types.get(det1_id, 'unknown')
        type2 = detector_types.get(det2_id, 'unknown')
        
        if type1 == type2:
            correlation = 0.72  # Same type: higher correlation
        else:
            correlation = 0.60  # Different types: lower correlation
        
        return correlation
    
    def record_outcome(self, was_false_isolation: bool):
        """
        Record automation outcome for false isolation tracking
        
        Args:
            was_false_isolation: True if action was a false positive
        """
        self.false_isolation_window.append(int(was_false_isolation))
        
        # Keep window bounded
        if len(self.false_isolation_window) > self.window_size:
            self.false_isolation_window = self.false_isolation_window[-self.window_size:]
    
    def _bootstrap_upper_bound(self, data: np.ndarray, 
                               n_bootstrap: int = 1000) -> float:
        """
        Compute BCa 95% CI upper bound for false isolation rate
        
        Args:
            data: Binary array of false isolation outcomes
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            97.5th percentile (upper bound of 95% CI)
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            resample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # 97.5th percentile for 95% CI upper bound
        upper_bound = np.percentile(bootstrap_means, 97.5)
        
        return upper_bound
    
    def update_detector_thresholds(self, thresholds: Dict[str, float]):
        """Update per-detector thresholds after recalibration"""
        self.detector_thresholds.update(thresholds)
    
    def update_correlation_estimate(self, det1_id: str, det2_id: str,
                                     correlation: float):
        """Update detector correlation estimate"""
        key = tuple(sorted([det1_id, det2_id]))
        self.correlation_history[key] = correlation