"""Validated-reference drift diagnostics for R20 monitoring thresholds.

This component reports observable shifts. It must not train on test labels or
pretend automatic recalibration was performed. Missing reference => unknown.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


class DriftMonitor:
    def __init__(self, reference=None, *, psi_threshold=0.25,
                 ks_threshold=0.12, ece_threshold=0.03, min_window=200):
        self.reference = None
        self.psi_threshold = float(psi_threshold)
        self.ks_threshold = float(ks_threshold)
        self.ece_threshold = float(ece_threshold)
        self.min_window = int(min_window)
        if reference is not None:
            self.fit_reference(reference)

    def fit_reference(self, reference):
        x = np.asarray(reference, dtype=float)
        if x.ndim != 1 or x.size < self.min_window or not np.all(np.isfinite(x)):
            raise ValueError("Reference requires sufficient finite scalar values")
        self.reference = x.copy()

    @staticmethod
    def calibration_error(scores, labels, bins=10):
        scores, labels = np.asarray(scores, float), np.asarray(labels)
        if (scores.ndim != 1 or scores.shape != labels.shape or
            len(scores) == 0 or not np.all(np.isfinite(scores)) or
            np.any((scores < 0) | (scores > 1)) or
            not set(np.unique(labels)).issubset({0, 1})):
            raise ValueError("Verified probability and binary label pairs required")
        cuts = np.minimum((scores * bins).astype(int), bins - 1)
        return float(sum((cuts == k).mean() *
                         abs(scores[cuts == k].mean() - labels[cuts == k].mean())
                         for k in range(bins) if (cuts == k).any()))

    def assess(self, observed, *, verified_labels=None):
        x = np.asarray(observed, dtype=float)
        if (self.reference is None or x.ndim != 1 or len(x) < self.min_window
            or not np.all(np.isfinite(x))):
            return {"status": "insufficient_evidence", "drift": None,
                    "psi": None, "ks": None, "ece": None}
        ref = self.reference
        # Fixed, validation-only equal-frequency bin edges (not test-fitted).
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            edges = np.array([-np.inf, np.median(ref), np.inf])
        edges[0], edges[-1] = -np.inf, np.inf
        expected = np.histogram(ref, bins=edges)[0] / len(ref)
        actual = np.histogram(x, bins=edges)[0] / len(x)
        e, a = np.clip(expected, 1e-6, 1), np.clip(actual, 1e-6, 1)
        psi = float(np.sum((a - e) * np.log(a / e)))
        ks = float(ks_2samp(ref, x).statistic)
        error = None
        if verified_labels is not None:
            error = self.calibration_error(x, verified_labels)
        triggered = (psi >= self.psi_threshold or ks >= self.ks_threshold or
                     (error is not None and error >= self.ece_threshold))
        return {"status": "shift_detected" if triggered else "stable",
                "drift": bool(triggered), "psi": psi, "ks": ks, "ece": error}
