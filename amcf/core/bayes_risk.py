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
        
        # GATHER has no standalone action loss before an enrichment model
        # is supplied; it must not be selected by a spurious zero cost.
        R_gather = float("inf")
        
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
    """Conditional expected value of information, R20 Eq. (10).

    In the absence of validated conditional enrichment outcomes the safe
    response is no information-gathering recommendation (VoI=0, samples=0).
    Synthetic posterior movement is never substituted for real evidence.
    """

    def __init__(self, risk_calculator: BayesRiskCalculator,
                 N_min: int = 64, N_max: int = 4096,
                 tolerance: float = 0.02, wall_clock_budget_ms: float = 150,
                 outcome_models=None):
        self.risk_calc = risk_calculator
        self.N_min, self.N_max = N_min, N_max
        self.tolerance, self.budget_ms = tolerance, wall_clock_budget_ms
        # Mapping from enrichment name to validated callable(context)->
        # iterable of (P(outcome|context), posterior_given_outcome).
        self.outcome_models = dict(outcome_models or {})

    def register_outcome_model(self, enrichment_type: str, model):
        if not callable(model):
            raise TypeError("An enrichment outcome model must be callable")
        self.outcome_models[enrichment_type] = model

    def estimate_voi(self, posterior_before: float, enrichment_cost: float,
                     enrichment_type: str, severity: str = "medium",
                     **risk_kwargs) -> Tuple[float, int]:
        if enrichment_type not in self.outcome_models:
            return 0.0, 0
        if not 0 <= posterior_before <= 1 or enrichment_cost < 0:
            raise ValueError("Invalid posterior or cost")
        options = tuple(self.outcome_models[enrichment_type](
            posterior_before=posterior_before, severity=severity,
            **risk_kwargs
        ))
        if not options:
            raise ValueError("Empty predictive outcome distribution")
        total = 0.0
        expected_risk = 0.0
        for probability, posterior_after in options:
            if not (np.isfinite(probability) and np.isfinite(posterior_after)
                    and probability >= 0 and 0 <= posterior_after <= 1):
                raise ValueError("Invalid conditional outcome distribution")
            total += probability
            risks = self.risk_calc.compute_risks(posterior_after, severity, **risk_kwargs)
            expected_risk += probability * min(risks[k] for k in
                                                ("ignore", "automate", "escalate"))
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Predictive outcomes must have unit probability mass")
        risks = self.risk_calc.compute_risks(posterior_before, severity, **risk_kwargs)
        baseline = min(risks[k] for k in ("ignore", "automate", "escalate"))
        # Exact summation over supplied discrete outcomes; zero Monte Carlo samples.
        return float(baseline - expected_risk - enrichment_cost), 0
