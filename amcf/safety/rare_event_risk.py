"""Conservative one-sided false-isolation bound for safety admission.

An empirical zero error rate is not proof of rare-event reliability.
This helper intentionally says nothing about external validity, dependence,
label correctness or SOC operation; it only bounds a binomial rate under an
independent-trials model with observed outcomes.
"""
from __future__ import annotations

import math
from scipy.stats import beta


def one_sided_binomial_upper(failures: int, trials: int, *, confidence: float = 0.95) -> float:
    """Exact Clopper-Pearson one-sided binomial probability upper bound.

    The bound assumes i.i.d. Bernoulli trials with complete, accurately
    labeled outcomes. If no observations exist, return 1.0 (fail closed).
    """
    if not isinstance(failures, int) or not isinstance(trials, int):
        raise TypeError('failures/trials must be integers')
    if trials < 0 or failures < 0 or failures > trials:
        raise ValueError('require 0 <= failures <= trials')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between 0 and 1')
    if trials == 0 or failures == trials:
        return 1.0
    upper = float(beta.ppf(confidence, failures + 1, trials - failures))
    if not math.isfinite(upper):
        raise ArithmeticError('binomial upper bound not finite')
    return min(1.0, max(0.0, upper))