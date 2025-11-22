"""
Statistical procedures: Bootstrap CI, DeLong test, Holm-Bonferroni
Section 5.4
"""

import numpy as np
from scipy import stats
from typing import Tuple, List
from sklearn.metrics import roc_auc_score


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Compute BCa (Bias-Corrected and Accelerated) bootstrap 95% CI for recall
    Equation: CI^95 for Recall@B
    """
    recalls = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement (stratified by class if possible)
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute recall
        tp = ((y_pred_boot == 1) & (y_true_boot == 1)).sum()
        fn = ((y_pred_boot == 0) & (y_true_boot == 1)).sum()
        
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
            recalls.append(recall)
    
    recalls = np.array(recalls)
    
    # Percentile CI
    alpha = (1 - ci) / 2
    lower = np.percentile(recalls, alpha * 100)
    upper = np.percentile(recalls, (1 - alpha) * 100)
    
    return (lower, upper)


def bootstrap_ci_precision(y_true: np.ndarray, y_pred: np.ndarray,
                           n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Bootstrap CI for precision/escalation precision"""
    precisions = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        tp = ((y_pred_boot == 1) & (y_true_boot == 1)).sum()
        fp = ((y_pred_boot == 1) & (y_true_boot == 0)).sum()
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            precisions.append(precision)
    
    precisions = np.array(precisions)
    lower = np.percentile(precisions, 2.5)
    upper = np.percentile(precisions, 97.5)
    
    return (lower, upper)


def bootstrap_ci_median(data: np.ndarray, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Bootstrap CI for median (e.g., TTC)"""
    medians = []
    
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=len(data), replace=True)
        medians.append(np.median(resample))
    
    medians = np.array(medians)
    lower = np.percentile(medians, 2.5)
    upper = np.percentile(medians, 97.5)
    
    return (lower, upper)


def delong_test(y_true: np.ndarray, y_score1: np.ndarray,
                y_score2: np.ndarray) -> float:
    """
    DeLong test for comparing ROC-AUC of two models
    Returns: p-value
    """
    if len(np.unique(y_true)) < 2:
        return 1.0
    
    try:
        auc1 = roc_auc_score(y_true, y_score1)
        auc2 = roc_auc_score(y_true, y_score2)
    except:
        return 1.0
    
    auc_diff = auc1 - auc2
    
    # Permutation test
    n_perms = 1000
    perm_diffs = []
    
    for _ in range(n_perms):
        perm_indices = np.random.permutation(len(y_true))
        try:
            auc1_perm = roc_auc_score(y_true[perm_indices], y_score1[perm_indices])
            auc2_perm = roc_auc_score(y_true[perm_indices], y_score2[perm_indices])
            perm_diffs.append(auc1_perm - auc2_perm)
        except:
            pass
    
    perm_diffs = np.array(perm_diffs)
    p_value = (np.abs(perm_diffs) >= np.abs(auc_diff)).mean()
    
    return float(p_value)


def holm_bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni multiple comparison correction
    Returns: (adjusted_p_values, rejected)
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    adjusted_p = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    
    for i, idx in enumerate(sorted_indices):
        adjusted_p[idx] = min(1.0, sorted_p[i] * (n - i))
        if adjusted_p[idx] <= alpha:
            rejected[idx] = True
    
    # Ensure monotonicity
    for i in range(1, n):
        if adjusted_p[sorted_indices[i]] < adjusted_p[sorted_indices[i-1]]:
            adjusted_p[sorted_indices[i]] = adjusted_p[sorted_indices[i-1]]
    
    return adjusted_p, rejected


def newey_west_se(residuals: np.ndarray, lag: int = 7) -> float:
    """
    Newey-West standard error accounting for autocorrelation
    For temporal data with lag structure
    """
    n = len(residuals)
    
    # Compute variance
    var = np.var(residuals)
    
    # Add autocorrelation adjustment
    for l in range(1, lag + 1):
        if l < n:
            cov = np.mean(residuals[:-l] * residuals[l:])
            weight = 1 - (l / (lag + 1))
            var += 2 * weight * cov
    
    return np.sqrt(var / n)


def compute_confidence_interval(values: np.ndarray, ci: float = 0.95) -> Tuple[float, float]:
    """Compute percentile-based CI for any metric"""
    alpha = (1 - ci) / 2
    lower = np.percentile(values, alpha * 100)
    upper = np.percentile(values, (1 - alpha) * 100)
    
    return (lower, upper)