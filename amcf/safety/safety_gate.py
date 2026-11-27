"""
Safety Gate: Dual corroboration with independence checks
Section 4.2, Algorithm 4, Equations (13-14)
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
from .rare_event_risk import one_sided_binomial_upper


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
        # Fail closed on malformed evidence; only a detector->probability map is valid.
        if not isinstance(calibrated_probs, dict) or not calibrated_probs:
            return False
        # Reject NaN, infinities, strings, and out-of-range scores before the
        # comparator; invalid evidence must never crash the safety decision.
        if not isinstance(posterior, (int, float, np.floating)):
            return False
        if not np.isfinite(posterior) or not 0.0 <= posterior <= 1.0:
            return False
        for detector_id, probability in calibrated_probs.items():
            if not isinstance(detector_id, str) or not detector_id:
                return False
            if not isinstance(probability, (int, float, np.floating)):
                return False
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                return False

        # Equation (14), Section 4.2: nominal posterior criterion for
        # candidate endpoint isolation (1 - pi_t <= epsilon). This is a
        # model-based design gate, not an empirically certified failure rate.
        # Independent corroboration and the additional conservative
        # retrospective monitoring gate below remain unchanged.
        if (1.0 - posterior) > self.epsilon:
            return False
        
        # Clause 2: False isolation probability bound
        if not self._check_false_isolation_bound():
            return False
        
        # Clause 3: Dual corroboration with independence
        if not self._check_dual_corroboration(calibrated_probs):
            return False
        
        # All clauses satisfied
        return True
    
    def check_endpoint_isolation(self, *, posterior: float,
                                 calibrated_probs: Dict[str, float],
                                 risk_by_action: Dict[str, float],
                                 tau_risk: float,
                                 playbook_id: str,
                                 action_class: str) -> bool:
        """Candidate-specific Eq. (13)-(14) gate for *endpoint isolation* only.

        Algorithm 2's high-level 'automate' is not a concrete playbook class.
        The caller must explicitly identify an endpoint-isolation playbook,
        provide the current four action risks, and supply the validation-frozen
        risk threshold. There is intentionally NO default tau_risk, since the
        manuscript does not publish a numerical value for it.

        Does not execute a SOAR playbook or certify a 1e-3 error frequency.
        """
        if action_class != "endpoint_isolation" or not playbook_id:
            return False
        if not isinstance(risk_by_action, dict):
            return False
        try:
            values = {name: float(risk_by_action[name]) for name in
                      ("ignore", "gather", "automate", "escalate")}
            threshold = float(tau_risk)
        except (KeyError, TypeError, ValueError):
            return False
        # Gather risk may be +inf when no validated enrichment outcome model
        # is available. This must NOT be treated as a free candidate action.
        if (not np.isfinite(threshold) or threshold < 0 or
                any(np.isnan(v) or v < 0 for v in values.values()) or
                any(not np.isfinite(values[k]) for k in
                    ("ignore", "automate", "escalate"))):
            return False
        baseline = min(values[k] for k in ("ignore", "gather", "escalate"))
        risk_reduction = baseline - values["automate"]
        if not np.isfinite(risk_reduction) or risk_reduction < threshold:
            return False
        # Use the existing corroboration/dependence and conservative
        # retrospectively monitored outcome gate. This remains stricter
        # than the manuscript's explicitly *non-certifying* monitoring rule.
        return self.check(posterior, calibrated_probs)

    def _check_false_isolation_bound(self) -> bool:
        """
        Check a conservative one-sided binomial upper confidence bound.
        Small samples / no failures do NOT prove the 1e-3 risk target.
        Assumes independently observed, accurately labeled outcomes.
        """
        # Rare-event validation requires actual labeled outcomes. No outcomes or
        # too few outcomes cannot establish a stringent 1e-3 upper-risk bound.
        outcomes = self.false_isolation_window[-self.window_size:]
        upper_bound = one_sided_binomial_upper(
            failures=sum(outcomes), trials=len(outcomes), confidence=0.95
        )
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
        
        # Correlation must already be measured on independent validation data.
        # Missing estimates fail closed; assigning type-based priors is unsafe.
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
        
        # An unmeasured correlation is NOT evidence of independence.
        # Unknown dependence must deny automated action.
        return 1.0
    
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
    
    def update_detector_thresholds(self, thresholds: Dict[str, float]):
        """Update per-detector thresholds after recalibration"""
        self.detector_thresholds.update(thresholds)
    
    def update_correlation_estimate(self, det1_id: str, det2_id: str,
                                     correlation: float):
        """Update detector correlation estimate"""
        key = tuple(sorted([det1_id, det2_id]))
        self.correlation_history[key] = correlation