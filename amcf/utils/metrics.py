"""AMCF evaluation metrics.

R14 scientific audit: operational metrics require case/action/time metadata.
Event-level positive predictions are NOT synonymous with containment,
escalation decisions, or pre-impact detection of an attack chain.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


def _validate_labels(y_true, y_pred):
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) == 0:
        raise ValueError('Expected nonempty, aligned 1-D y_true/y_pred.')
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Binary labels and predictions required.')
    return a, b


def compute_recall_at_budget(y_true, y_pred) -> float:
    """Event-level recall AFTER a caller has independently verified FP budget.

    Binary labels alone do not establish that the budget was met.
    """
    a, b = _validate_labels(y_true, y_pred)
    positives = a == 1
    if not positives.any():
        raise ValueError('Recall undefined: no positive ground-truth events.')
    return float(np.mean(b[positives] == 1))


def compute_ttc(events, y_true, y_pred, *, incident_records=None) -> float:
    """Kaplan–Meier median time to actual containment, in minutes.

    `incident_records` must contain one entry per confirmed incident:
      incident_id, first_malicious_event_ts_ms and EITHER
      containment_ts_ms (observed successful containment) OR
      censored_at_ts_ms (right-censored before containment).
    Passing detections as proxies for containment is intentionally forbidden.
    """
    if incident_records is None:
        raise ValueError('TTC requires audited incident_records with actual action/censor timestamps.')
    durations = []
    observed = []
    seen = set()
    for r in incident_records:
        key = r['incident_id']
        if key in seen:
            raise ValueError('Duplicate incident ID.')
        seen.add(key)
        start = float(r['first_malicious_event_ts_ms'])
        has_action = r.get('containment_ts_ms') is not None
        has_censor = r.get('censored_at_ts_ms') is not None
        if has_action == has_censor:
            raise ValueError('Exactly one containment timestamp or censor timestamp required.')
        end = float(r['containment_ts_ms'] if has_action else r['censored_at_ts_ms'])
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError('Containment/censor timestamp predates incident start.')
        durations.append((end - start) / 60000.0)
        observed.append(has_action)
    if not seen:
        raise ValueError('No confirmed incidents.')
    times = np.asarray(durations)
    status = np.asarray(observed, dtype=bool)
    survival = 1.0
    for t in np.unique(times):
        at_risk = int(np.sum(times >= t))
        failures = int(np.sum((times == t) & status))
        if failures:
            survival *= 1.0 - failures / at_risk
            if survival <= 0.5:
                return float(t)
    raise ValueError('Kaplan–Meier median is not identifiable due to censoring.')


def compute_escalation_precision(y_true, y_pred, *, escalation_records=None) -> float:
    """Validated true escalations / distinct analyst escalations.

    `escalation_records`: escalation_id, verified_true_positive (boolean).
    Does not count automated decisions as analyst escalations.
    """
    if escalation_records is None:
        raise ValueError('Escalation precision requires actual validated escalation records.')
    ids = set()
    labels = []
    for row in escalation_records:
        k = row['escalation_id']
        value = row['verified_true_positive']
        if k in ids or not isinstance(value, (bool, np.bool_)):
            raise ValueError('Unique escalation ID and verified Boolean outcome required.')
        ids.add(k)
        labels.append(bool(value))
    if not labels:
        raise ValueError('No verified analyst escalations.')
    return float(np.mean(labels))


def compute_long_dwell_recall(y_true, y_pred, *, chain_records=None) -> float:
    """Fraction of annotated long-dwell attack chains alerted BEFORE impact.

    `chain_records` requires chain_id, impact_ts_ms, and first_alert_ts_ms.
    `first_alert_ts_ms=None` records a missed chain.  Incident annotations
    must be collected independently of model predictions.
    """
    if chain_records is None:
        raise ValueError('Long-dwell recall requires annotated attack-chain / impact timestamps.')
    seen = set()
    detected = 0
    for row in chain_records:
        k = row['chain_id']
        if k in seen:
            raise ValueError('Duplicate chain ID.')
        seen.add(k)
        impact = float(row['impact_ts_ms'])
        first_alert = row['first_alert_ts_ms']
        if not np.isfinite(impact):
            raise ValueError('Invalid impact timestamp.')
        if first_alert is not None:
            if not np.isfinite(float(first_alert)):
                raise ValueError('Invalid alert timestamp.')
            detected += float(first_alert) < impact
    if not seen:
        raise ValueError('No annotated long-dwell attack chains.')
    return detected / len(seen)


def compute_alerts_per_hour(events, y_pred) -> float:
    """Positive operational decisions per recorded wall-clock hour."""
    if len(events) != len(y_pred) or len(events) < 2:
        raise ValueError('Aligned event timestamps, at least two, required.')
    elapsed_h = (max(e.ts for e in events) - min(e.ts for e in events)) / 3_600_000
    if elapsed_h <= 0:
        raise ValueError('Event observation window must span positive wall-clock time.')
    return float((np.asarray(y_pred) == 1).sum() / elapsed_h)


def compute_fp_per_endpoint_day(events, y_pred, y_true, *, active_endpoint_days=None) -> float:
    """False-positive decisions divided by audited active-endpoint-days.

    Do NOT infer the fleet denominator from distinct source hosts in event logs.
    """
    true, pred = _validate_labels(y_true, y_pred)
    if len(events) != len(true):
        raise ValueError('Event and prediction lengths do not match.')
    if active_endpoint_days is None or not float(active_endpoint_days).is_integer() or active_endpoint_days <= 0:
        raise ValueError('Audited active_endpoint_days denominator is required.')
    return float(np.sum((pred == 1) & (true == 0)) / float(active_endpoint_days))


def compute_roc_auc(y_true, y_score) -> float:
    """Area under ROC.  Invalid inputs must not silently become 0.5."""
    if len(np.unique(y_true)) < 2:
        raise ValueError('ROC-AUC undefined for single-class ground truth.')
    return float(roc_auc_score(y_true, y_score))


def compute_pr_auc(y_true, y_score) -> float:
    """Trapezoidal PR-AUC (not average precision)."""
    if len(np.unique(y_true)) < 2:
        raise ValueError('PR-AUC undefined for single-class ground truth.')
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def compute_expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """10-bin equal-width ECE for probability scores."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true)
    if len(y_prob) != len(y_true) or not len(y_prob) or n_bins < 1:
        raise ValueError('Aligned probabilities/labels and positive bin count required.')
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError('Finite probability scores must be in [0,1].')
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bins == i
        if mask.any():
            ece += (mask.sum() / len(y_true)) * abs(float(y_true[mask].mean() - y_prob[mask].mean()))
    return float(ece)


def compute_metrics_dict(y_true, y_pred, y_score, events=None,
                         *, active_endpoint_days=None, chain_records=None,
                         incident_records=None, escalation_records=None) -> Dict[str, float]:
    """Compute explicitly available metrics; never infer case-level outcomes.

    The caller must supply verified records for the extra operational metrics.
    """
    true, pred = _validate_labels(y_true, y_pred)
    metrics = {'event_recall': compute_recall_at_budget(true, pred),
               'event_precision': float(np.mean(true[pred == 1] == 1)) if (pred == 1).any() else float('nan'),
               'roc_auc': compute_roc_auc(true, y_score),
               'pr_auc': compute_pr_auc(true, y_score),
               'ece': compute_expected_calibration_error(true, y_score)}
    if events is not None:
        metrics['alerts_per_hour'] = compute_alerts_per_hour(events, pred)
        if active_endpoint_days is not None:
            metrics['fp_per_endpoint_day'] = compute_fp_per_endpoint_day(events, pred, true,
                active_endpoint_days=active_endpoint_days)
    if chain_records is not None:
        metrics['long_dwell_recall'] = compute_long_dwell_recall(true, pred,
            chain_records=chain_records)
    if incident_records is not None:
        metrics['ttc_km_median_min'] = compute_ttc(events, true, pred,
            incident_records=incident_records)
    if escalation_records is not None:
        metrics['escalation_precision'] = compute_escalation_precision(true, pred,
            escalation_records=escalation_records)
    return metrics