"""Adaptive Monte Carlo conditional Value of Information (v45 R3, Eq. 10).

This module samples ONLY from a validated user-supplied conditional enrichment
outcome distribution. It does not invent incident labels, posterior shifts, or
benchmark results. Runtime budget expiration and missing evidence fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Optional

import numpy as np
from scipy.stats import bootstrap

from .bayes_risk import BayesRiskCalculator


@dataclass(frozen=True)
class MonteCarloVoIResult:
    voi: float
    samples: int
    ci_halfwidth: Optional[float]
    elapsed_ms: float
    status: str


class MonteCarloVoIEstimator:
    """Eq. (10) / Section 3.1: 64..4096 outcome draws, BCa 95% CI, 150 ms.

    An outcome model callable(context) must return a finite probability
    distribution as (probability, posterior_given_outcome) pairs.
    The controller must supply this model from historical *causally available*
    enrichment evidence; no surrogate posterior movements are constructed here.

    On budget timeout or insufficient confidence evidence, the returned VoI is
    0.0 (no gather recommendation), not a fabricated estimate of true VoI.
    """

    def __init__(self, risk_calculator: BayesRiskCalculator, *,
                 N_min: int = 64, N_max: int = 4096, tolerance: float = 0.02,
                 wall_clock_budget_ms: float = 150.0, seed: int = 42,
                 ci_resamples: int = 999, outcome_models=None):
        if (not isinstance(N_min, int) or not isinstance(N_max, int)
                or N_min < 2 or N_max < N_min or
                not np.isfinite(tolerance) or tolerance <= 0 or
                not np.isfinite(wall_clock_budget_ms) or wall_clock_budget_ms <= 0 or
                ci_resamples < 100):
            raise ValueError("Invalid Monte Carlo budget, CI, or tolerance")
        self.risk_calc = risk_calculator
        self.N_min, self.N_max = N_min, N_max
        self.tolerance, self.budget_ms = float(tolerance), float(wall_clock_budget_ms)
        self.seed, self.ci_resamples = int(seed), int(ci_resamples)
        self.outcome_models: Dict[str, Callable] = dict(outcome_models or {})
        self.last_result = MonteCarloVoIResult(0., 0, None, 0., "not_evaluated")

    def register_outcome_model(self, enrichment_type: str, model: Callable) -> None:
        if not isinstance(enrichment_type, str) or not enrichment_type or not callable(model):
            raise TypeError("Named callable conditional enrichment model required")
        self.outcome_models[enrichment_type] = model

    @staticmethod
    def _validated_distribution(entries):
        options = tuple(entries)
        if not options:
            raise ValueError("Empty conditional outcome distribution")
        try:
            values = np.asarray(options, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Malformed conditional outcome distribution") from exc
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("Expected finite (probability, posterior) pairs")
        probabilities, posteriors = values[:, 0], values[:, 1]
        if (np.any(probabilities < 0) or
                not np.isclose(probabilities.sum(), 1., atol=1e-6, rtol=0.) or
                np.any((posteriors < 0) | (posteriors > 1))):
            raise ValueError("Invalid probabilities or posterior domain")
        return probabilities / probabilities.sum(), posteriors

    def _min_primary_risk(self, posterior: float, severity: str, risk_kwargs) -> float:
        risks = self.risk_calc.compute_risks(posterior, severity, **risk_kwargs)
        return float(min(risks[a] for a in ("ignore", "automate", "escalate")))

    def estimate_detailed(self, posterior_before: float, enrichment_cost: float,
                          enrichment_type: str, severity: str = "medium",
                          **risk_kwargs) -> MonteCarloVoIResult:
        started = perf_counter()
        if (not np.isfinite(posterior_before) or not 0 <= posterior_before <= 1
                or not np.isfinite(enrichment_cost) or enrichment_cost < 0):
            raise ValueError("Invalid posterior or enrichment cost")
        if enrichment_type not in self.outcome_models:
            result = MonteCarloVoIResult(0., 0, None, 0., "missing_outcome_model")
            self.last_result = result
            return result
        model = self.outcome_models[enrichment_type]
        distribution = model(posterior_before=posterior_before,
                             severity=severity, **risk_kwargs)
        probabilities, posteriors = self._validated_distribution(distribution)
        if (perf_counter() - started) * 1000 >= self.budget_ms:
            result = MonteCarloVoIResult(0., 0, None,
                      (perf_counter() - started) * 1000, "time_budget_exhausted")
            self.last_result = result
            return result

        baseline = self._min_primary_risk(posterior_before, severity, risk_kwargs)
        outcome_risks = np.asarray([
            self._min_primary_risk(float(p), severity, risk_kwargs)
            for p in posteriors
        ], dtype=float)
        # Independent per-estimate deterministic stream; does not claim the
        # identical draw stream as the original enterprise experiment.
        rng = np.random.default_rng(self.seed)
        observations = []
        target = self.N_min
        while True:
            n_add = target - len(observations)
            draws = rng.choice(len(probabilities), size=n_add, p=probabilities)
            observations.extend(outcome_risks[draws].tolist())
            elapsed_ms = (perf_counter() - started) * 1000
            if elapsed_ms >= self.budget_ms:
                result = MonteCarloVoIResult(0., len(observations), None,
                                             elapsed_ms, "time_budget_exhausted")
                self.last_result = result
                return result
            arr = np.asarray(observations, dtype=float)
            halfwidth = None
            if np.all(arr == arr[0]):
                halfwidth = 0.0
            else:
                # BCa of the post-enrichment minimum Bayes risk (paper Section 3.1).
                # Bootstrap is not cost-free: explicitly account for its wall time.
                ci = bootstrap(
                    (arr,), np.mean, confidence_level=0.95,
                    n_resamples=self.ci_resamples, method="BCa",
                    random_state=np.random.default_rng(self.seed + len(arr))
                ).confidence_interval
                halfwidth = float((ci.high - ci.low) / 2)
            elapsed_ms = (perf_counter() - started) * 1000
            if elapsed_ms >= self.budget_ms:
                result = MonteCarloVoIResult(0., len(observations), halfwidth,
                                             elapsed_ms, "time_budget_exhausted")
                self.last_result = result
                return result
            if halfwidth <= self.tolerance:
                status = "tolerance_reached"
            elif len(observations) >= self.N_max:
                status = "maximum_samples"
            else:
                target = min(2 * target, self.N_max)
                continue
            value = baseline - float(arr.mean()) - enrichment_cost
            result = MonteCarloVoIResult(value, len(observations), halfwidth,
                                         elapsed_ms, status)
            self.last_result = result
            return result

    def estimate_voi(self, posterior_before: float, enrichment_cost: float,
                     enrichment_type: str, severity: str = "medium", **risk_kwargs):
        result = self.estimate_detailed(posterior_before, enrichment_cost,
                                        enrichment_type, severity, **risk_kwargs)
        return result.voi, result.samples
