"""
Evaluation metrics: Recall@B, TTC, escalation precision, ROC-AUC, PR-AUC
Section 5.3
"""

import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from typing import Tuple, Dict


def compute_recall_at_budget(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Recall@B metric
    Recall = TP / (TP + FN)
    """
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    if (tp + fn) == 0:
        return 0.0
    
    return float(tp) / (tp + fn)


def compute_ttc(events, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute median Time-To-Containment (minutes)
    TTC = median time from first malicious event to containment action
    """
    ttcs = []
    
    incident_start = None
    for i, (event, label, pred) in enumerate(zip(events, y_true, y_pred)):
        if label == 1 and incident_start is None:
            incident_start = event.ts
        
        # Containment: prediction changes from negative to positive
        if incident_start is not None and pred == 1:
            ttc_ms = event.ts - incident_start
            ttc_min = ttc_ms / 1000 / 60  # Convert to minutes
            ttcs.append(ttc_min)
            incident_start = None
    
    if len(ttcs) == 0:
        return 0.0
    
    return float(np.median(ttcs))


def compute_escalation_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Escalation precision = fraction of escalations that are true positives
    """
    esc_mask = y_pred == 1
    if esc_mask.sum() == 0:
        return 0.0
    
    return float((y_true[esc_mask] == 1).sum()) / esc_mask.sum()


def compute_long_dwell_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute long-dwell (APT-chain) recall
    Percentage of multi-stage attacks detected
    """
    dwell_mask = y_true == 1
    if dwell_mask.sum() == 0:
        return 0.0
    
    return float((y_pred[dwell_mask] == 1).sum()) / dwell_mask.sum()


def compute_alerts_per_hour(events, y_pred: np.ndarray) -> float:
    """Compute alert volume (alerts per hour)"""
    n_alerts = (y_pred == 1).sum()
    
    if len(events) == 0:
        return 0.0
    
    time_span_ms = events[-1].ts - events[0].ts
    time_span_hours = time_span_ms / (1000 * 3600)
    
    if time_span_hours == 0:
        return float(n_alerts)
    
    return n_alerts / time_span_hours


def compute_fp_per_endpoint_day(events, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute FP/endpoint-day"""
    fp_mask = (y_pred == 1) & (y_true == 0)
    n_fp = fp_mask.sum()
    
    # Count unique endpoints
    endpoints = set(e.src for e in events)
    n_endpoints = len(endpoints)
    
    # Time span in days
    time_span_ms = events[-1].ts - events[0].ts
    time_span_days = time_span_ms / (1000 * 86400)
    
    if n_endpoints * time_span_days == 0:
        return 0.0
    
    return n_fp / (n_endpoints * time_span_days)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC"""
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall AUC"""
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    except:
        return 0.5


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                       n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE)
    """
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        
        avg_prob = y_prob[mask].mean()
        accuracy = (y_true[mask] == 1).mean()
        ece += (mask.sum() / len(y_true)) * np.abs(avg_prob - accuracy)
    
    return ece


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray,
                        y_score: np.ndarray, events=None) -> Dict[str, float]:
    """Compute all metrics and return as dictionary"""
    metrics = {
        'recall': compute_recall_at_budget(y_true, y_pred),
        'precision': compute_escalation_precision(y_true, y_pred),
        'long_dwell_recall': compute_long_dwell_recall(y_true, y_pred),
        'roc_auc': compute_roc_auc(y_true, y_score),
        'pr_auc': compute_pr_auc(y_true, y_score),
        'ece': compute_expected_calibration_error(y_true, y_score)
    }
    
    if events is not None:
        metrics['alerts_per_hour'] = compute_alerts_per_hour(events, y_pred)
        metrics['fp_per_endpoint_day'] = compute_fp_per_endpoint_day(events, y_pred, y_true)
        metrics['ttc'] = compute_ttc(events, y_true, y_pred)
    
    return metrics