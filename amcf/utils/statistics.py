"""AMCF statistical routines with explicit estimands and resampling limits.

BCa confidence limits require i.i.d. observational units. Incident/temporal
claims must instead supply appropriate independent blocks, not flat events.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Tuple


def _bca_mean(sample, n_bootstrap, ci, seed=42):
    data = np.asarray(sample, dtype=float)
    if data.size < 2:
        raise ValueError("At least two observations are required")
    if np.all(data == data[0]):
        return float(data[0]), float(data[0])
    bounds = stats.bootstrap(
        (data,), np.mean, n_resamples=n_bootstrap, confidence_level=ci,
        method="BCa", random_state=np.random.default_rng(seed)
    ).confidence_interval
    return float(bounds.low), float(bounds.high)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """95% BCa CI for *flow-level* recall, sampling labelled positives."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    if y.shape != p.shape:
        raise ValueError("Label/prediction length mismatch")
    return _bca_mean((p[y == 1] == 1).astype(float), n_bootstrap, ci)


def bootstrap_ci_precision(y_true: np.ndarray, y_pred: np.ndarray,
                           n_bootstrap: int = 10000) -> Tuple[float, float]:
    """95% BCa CI for precision of predicted positive actions."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    if y.shape != p.shape:
        raise ValueError("Label/prediction length mismatch")
    return _bca_mean((y[p == 1] == 1).astype(float), n_bootstrap, 0.95)


def bootstrap_ci_median(data: np.ndarray, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """95% percentile CI for an observed median, e.g. verified incident TTC."""
    data = np.asarray(data, dtype=float)
    if data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError("At least two finite observations required")
    rng = np.random.default_rng(42)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        estimates[i] = np.median(rng.choice(data, data.size, replace=True))
    return tuple(map(float, np.quantile(estimates, [0.025, 0.975])))


def _delong_influences(y, score):
    """Exact empirical AUC placements using ranks, not pairwise dense matrices."""
    pos = np.asarray(score, dtype=float)[y == 1]
    neg = np.asarray(score, dtype=float)[y == 0]
    if not len(pos) or not len(neg):
        raise ValueError("Both classes required for DeLong")
    sorted_neg, sorted_pos = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sorted_neg, pos, "left") +
           np.searchsorted(sorted_neg, pos, "right")) / (2 * len(neg))
    v01 = (len(pos) - (np.searchsorted(sorted_pos, neg, "left") +
                          np.searchsorted(sorted_pos, neg, "right")) / 2) / len(pos)
    return float(v10.mean()), v10, v01


def delong_test(y_true: np.ndarray, y_score1: np.ndarray,
                y_score2: np.ndarray) -> float:
    """Two-sided *paired DeLong* ROC-AUC comparison, including tied scores.

    Requires independent observations; do not apply directly to correlated
    telemetry events or incident-time blocks.
    """
    y = np.asarray(y_true)
    a, b = np.asarray(y_score1, float), np.asarray(y_score2, float)
    if not (y.ndim == a.ndim == b.ndim == 1 and len(y) == len(a) == len(b)):
        raise ValueError("Expected three equally-sized one-dimensional arrays")
    if not (set(np.unique(y)).issubset({0, 1}) and np.all(np.isfinite(a))
            and np.all(np.isfinite(b))):
        raise ValueError("Binary labels and finite scores required")
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        raise ValueError("At least two samples of each class required")
    auc1, v10a, v01a = _delong_influences(y, a)
    auc2, v10b, v01b = _delong_influences(y, b)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    cov10 = np.cov(np.stack((v10a, v10b)), ddof=1) / n1
    cov01 = np.cov(np.stack((v01a, v01b)), ddof=1) / n0
    total = cov10 + cov01
    variance = float(total[0, 0] + total[1, 1] - 2 * total[0, 1])
    difference = auc1 - auc2
    if variance <= 1e-15:
        return 1.0 if abs(difference) < 1e-12 else 0.0
    return float(2 * stats.norm.sf(abs(difference) / np.sqrt(variance)))


def holm_bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05):
    """Holm step-down multiplicity adjustment with monotone adjusted p-values."""
    p = np.asarray(p_values, float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite within [0,1]")
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(p[order] * np.arange(n, 0, -1)))
    return adjusted, adjusted <= alpha


def newey_west_se(residuals: np.ndarray, lag: int = 7) -> float:
    """HAC standard error of the *mean*, based on centred residuals."""
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 1 or len(r) < 2 or lag < 0 or not np.all(np.isfinite(r)):
        raise ValueError("Nonempty finite univariate residual series required")
    x = r - r.mean()
    n = len(x)
    gamma0 = float(np.dot(x, x) / n)
    estimate = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        cov = float(np.dot(x[k:], x[:-k]) / n)
        estimate += 2 * (1 - k / (lag + 1)) * cov
    return float(np.sqrt(max(0.0, estimate) / n))


def compute_confidence_interval(values: np.ndarray, ci: float = 0.95):
    data = np.asarray(values, dtype=float)
    if not len(data) or np.any(~np.isfinite(data)) or not (0 < ci < 1):
        raise ValueError("Invalid confidence interval inputs")
    q = (1-ci)/2
    return tuple(map(float, np.quantile(data, [q, 1-q])))
