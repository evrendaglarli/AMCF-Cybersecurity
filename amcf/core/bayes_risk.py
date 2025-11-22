"""
Bayes Risk Computation and Value of Information Estimation
Implements Equations (6), (10) and Algorithm 2 from the paper
"""

import numpy as np
from typing import Dict, Tuple, Callable
from dataclasses import dataclass

@dataclass
class CostModel:
    """Cost parameters for Bayes risk calculation"""
    C_FP: float = 1.0           # False positive cost
    C_FN_low: float = 5.0       # Miss cost (low severity)
    C_FN_med: float = 15.0      # Miss cost (medium severity)
    C_FN_high: float = 50.0     # Miss cost (high severity)
    C_auto: float = 2.5         # Automation execution cost
    C_SE: float = 1.2           # Side-effect cost
    c_triage: float = 3.0       # Analyst triage cost
    lambda_delay: float = 0.5   # Delay penalty weight
    lambda_queue: float = 0.3   # Queue congestion weight


class BayesRiskCalculator:
    """
    Compute Bayes risk for actions {ignore, gather, automate, escalate}
    Equation (6): R(a) = π_t*C_FN*1{a=ignore} + (1-π_t)*C_FP*1{a=automate} + C(a)
    """
    
    def __init__(self, cost_model: CostModel):
        self.costs = cost_model
        
    def compute_risks(self, posterior: float, severity: str = 'medium',
                      asset_weight: float = 1.0, queue_length: int = 0,
                      expected_delay: float = 5.0,
                      residual_risk_factor: float = 0.1) -> Dict[str, float]:
        """
        Compute Bayes risk for all actions
        
        Args:
            posterior: π_t, fused probability of maliciousness
            severity: 'low', 'medium', or 'high'
            asset_weight: Criticality multiplier
            queue_length: Current escalation queue size
            expected_delay: δ, expected analyst response time (minutes)
            residual_risk_factor: ρ(τ,s), automation imperfection
            
        Returns:
            Dictionary of risks {action: risk_value}
        """
        # Select severity-specific false negative cost
        C_FN = {
            'low': self.costs.C_FN_low,
            'medium': self.costs.C_FN_med,
            'high': self.costs.C_FN_high
        }[severity]
        
        # Weighted miss loss
        L_FN = asset_weight * C_FN
        
        # IGNORE risk (Equation 6, first term)
        R_ignore = posterior * L_FN
        
        # AUTOMATE risk (Equation 6, second term + execution costs)
        R_automate = (
            (1 - posterior) * self.costs.C_FP +  # False alarm loss
            posterior * residual_risk_factor * L_FN +  # Residual miss risk
            self.costs.C_SE +  # Side effect cost
            self.costs.C_auto  # Execution cost
        )
        
        # ESCALATE risk (Equation 8: delay-aware cost)
        delta_L = asset_weight * C_FN * (1 - np.exp(-0.1 * expected_delay))
        C_escalate = (
            self.costs.c_triage +
            self.costs.lambda_delay * posterior * delta_L +
            self.costs.lambda_queue * queue_length
        )
        R_escalate = C_escalate
        
        # GATHER risk (placeholder, will be computed via VoI)
        R_gather = 0.0
        
        return {
            'ignore': R_ignore,
            'automate': R_automate,
            'escalate': R_escalate,
            'gather': R_gather
        }
    
    def select_action(self, risks: Dict[str, float], voi_positive: bool = False,
                      high_uncertainty: bool = False) -> str:
        """
        Metacognitive decision policy (Algorithm 2, step 6)
        """
        if voi_positive and high_uncertainty:
            return 'gather'
        
        # Minimum risk action among {ignore, automate, escalate}
        primary_actions = {k: v for k, v in risks.items() if k != 'gather'}
        return min(primary_actions, key=primary_actions.get)


class ValueOfInformationEstimator:
    """
    Estimate VoI for enrichment queries using Monte Carlo sampling
    Equation (10): VoI(e) = min_a R(a) - E[min_a R^(e)(a)] - C_gather(e)
    """
    
    def __init__(self, risk_calculator: BayesRiskCalculator,
                 N_min: int = 64, N_max: int = 4096,
                 tolerance: float = 0.02, wall_clock_budget_ms: float = 150):
        self.risk_calc = risk_calculator
        self.N_min = N_min
        self.N_max = N_max
        self.tolerance = tolerance
        self.budget_ms = wall_clock_budget_ms
        
    def estimate_voi(self, posterior_before: float, enrichment_cost: float,
                     enrichment_type: str, severity: str = 'medium',
                     **risk_kwargs) -> Tuple[float, int]:
        """
        Monte Carlo VoI estimation with adaptive early termination
        
        Args:
            posterior_before: Current π_t before enrichment
            enrichment_cost: C_gather(e)
            enrichment_type: 'edr_deep', 'sandbox', 'pcap', etc.
            severity: Incident severity
            **risk_kwargs: Additional args for risk computation
            
        Returns:
            (VoI, num_samples_used)
        """
        # Baseline risk
        risks_base = self.risk_calc.compute_risks(posterior_before, severity, **risk_kwargs)
        R_base = min(risks_base['ignore'], risks_base['automate'], risks_base['escalate'])
        
        # Monte Carlo sampling
        samples = []
        n = 0
        
        while n < self.N_max:
            # Sample enrichment outcome (simplified: assume posterior shifts)
            posterior_after = self._sample_posterior_after_enrichment(
                posterior_before, enrichment_type
            )
            
            # Compute risk after enrichment
            risks_after = self.risk_calc.compute_risks(posterior_after, severity, **risk_kwargs)
            R_after = min(risks_after['ignore'], risks_after['automate'], risks_after['escalate'])
            
            samples.append(R_after)
            n += 1
            
            # Early termination check (BCa CI half-width < tolerance)
            if n >= self.N_min and n % 32 == 0:
                ci_half_width = self._bootstrap_ci_half_width(samples)
                if ci_half_width < self.tolerance:
                    break
        
        # Expected risk after enrichment
        E_R_after = np.mean(samples)
        
        # VoI = risk reduction - enrichment cost
        voi = R_base - E_R_after - enrichment_cost
        
        return voi, n
    
    def _sample_posterior_after_enrichment(self, posterior: float, 
                                            enrichment_type: str) -> float:
        """
        Simulate posterior update after enrichment
        Simplified model: add Gaussian noise with enrichment-specific variance
        """
        # Enrichment informativeness (higher = more informative)
        info_gain = {
            'edr_deep': 0.15,
            'sandbox': 0.20,
            'pcap': 0.10,
            'memory_dump': 0.25
        }.get(enrichment_type, 0.10)
        
        # Posterior shifts toward 0 or 1 based on ground truth (unknown in practice)
        # Here we simulate: if posterior > 0.5, shift up; else shift down
        direction = 1 if posterior > 0.5 else -1
        shift = direction * np.random.beta(2, 5) * info_gain
        
        posterior_after = np.clip(posterior + shift, 0.01, 0.99)
        return posterior_after
    
    def _bootstrap_ci_half_width(self, samples: list, n_bootstrap: int = 1000) -> float:
        """Compute BCa 95% CI half-width for early stopping"""
        samples = np.array(samples)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            resample = np.random.choice(samples, size=len(samples), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        return (ci_upper - ci_lower) / 2.0