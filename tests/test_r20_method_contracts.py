"""R20 method-contract regression tests; no claims of enterprise reproduction."""
import math
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name, relative in [
    ("amcf", "amcf"), ("amcf.core", "amcf/core"),
    ("amcf.safety", "amcf/safety"), ("amcf.utils", "amcf/utils"),
]:
    if name not in sys.modules:
        namespace = types.ModuleType(name)
        namespace.__path__ = [str(ROOT / relative)]
        sys.modules[name] = namespace

from amcf.core.hypotheses import HypothesisTracker
from amcf.core.bayes_risk import CostModel, BayesRiskCalculator, ValueOfInformationEstimator
from amcf.core.drift import DriftMonitor
from amcf.safety.enrichment_scheduler import EnrichmentScheduler
from amcf.utils.statistics import (bootstrap_ci, delong_test, holm_bonferroni_correction,
                                   newey_west_se)
from amcf.utils.metrics import compute_long_dwell_recall, compute_ttc


@dataclass
class Event:
    ts: int
    src: str = "host1"
    dst: str = "host2"
    kind: str = "edr"
    attrs: dict = None


class TestR20MethodContracts(unittest.TestCase):
    def test_hypothesis_evidence_once_and_normalised(self):
        tracker = HypothesisTracker(max_hypotheses=2)
        a = Event(1000, attrs={"attack_stage": "initial_access", "event_id": "a"})
        b = Event(1100, attrs={"attack_stage": "discovery", "event_id": "b"})
        c = Event(1200, src="host2", attrs={"attack_stage": "execution", "event_id": "c"})
        tracker.update(a, .8)
        first = tracker.active["host1"].log_evidence
        tracker.update(a, .8)
        self.assertAlmostEqual(tracker.active["host1"].log_evidence, first)
        tracker.update(b, .7)
        probs, scores = tracker.update(c, .9)
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertTrue(all(0 <= x <= 1 for x in scores.values()))
        tracker.update(Event(1300, attrs={"attack_stage": "unknown"}), .9)
        self.assertEqual(len(tracker.active), 2)

    def test_hypothesis_stale_pruning(self):
        tracker = HypothesisTracker(window_ms=20)
        tracker.update(Event(0, attrs={"attack_stage":"execution"}), .9)
        beliefs, _ = tracker.update(Event(100, attrs={}), .3)
        self.assertFalse(beliefs)

    def test_conditional_voi_without_model_is_not_invented(self):
        calc = BayesRiskCalculator(CostModel())
        model = ValueOfInformationEstimator(calc)
        self.assertEqual(model.estimate_voi(.6, 1., "sandbox"), (0.0, 0))
        model.register_outcome_model("sandbox", lambda **kw: [(0.5, .1), (0.5, .9)])
        voi, samples = model.estimate_voi(.5, .01, "sandbox")
        risks = [calc.compute_risks(x) for x in (.5,.1,.9)]
        expected = min(risks[0][a] for a in ("ignore","automate","escalate")) - (
            .5 * min(risks[1][a] for a in ("ignore","automate","escalate")) +
            .5 * min(risks[2][a] for a in ("ignore","automate","escalate"))) - .01
        self.assertAlmostEqual(voi, expected)
        self.assertEqual(samples, 0)
        with self.assertRaises(ValueError):
            model.register_outcome_model("broken", lambda **kw: [(0.8, 0.5)])
            model.estimate_voi(.5, 1.0, "broken")

    def test_one_step_argmax_voi_with_budget(self):
        scheduler = EnrichmentScheduler(budget=30.)
        values = {"expensive": {"voi": 99, "cost": 32},
                  "high": {"voi": 10, "cost": 25},
                  "ratio": {"voi": 8, "cost": 5}}
        self.assertEqual(scheduler.select_enrichments(values), ["high"])
        self.assertEqual(scheduler.select_enrichments(values, available_budget=5), ["ratio"])
        self.assertEqual(scheduler.select_enrichments(values, available_budget=0), [])

    def test_exact_delong_uses_paired_dependence(self):
        rng = np.random.default_rng(7)
        y = np.array([0, 1] * 300)
        x = y + rng.normal(0, .3, size=len(y))
        baseline = rng.normal(0, 1, size=len(y))
        self.assertAlmostEqual(delong_test(y, x, x), 1.0)
        self.assertLess(delong_test(y, x, baseline), 0.05)

    def test_bca_holm_and_hac(self):
        y = np.array([1] * 20 + [0] * 15)
        p = np.array([1] * 15 + [0] * 5 + [0] * 15)
        lo, hi = bootstrap_ci(y, p, n_bootstrap=200, ci=.95)
        self.assertLessEqual(lo, .75)
        self.assertGreaterEqual(hi, .75)
        q, rejected = holm_bonferroni_correction(np.array([.03, .01, .02]))
        self.assertTrue(np.allclose(q, [.04, .03, .04]))
        self.assertEqual(list(rejected), [True, True, True])
        self.assertAlmostEqual(newey_west_se(np.ones(100)), 0.)

    def test_incident_outcomes_do_not_equal_predictions(self):
        items = [Event(0, attrs={}), Event(7200_000, attrs={})]
        y = np.array([1, 1]); predictions = np.array([1, 1])
        self.assertTrue(math.isnan(compute_long_dwell_recall(y, predictions)))
        self.assertTrue(math.isnan(compute_ttc(items, y, predictions)))
        items = [
            Event(1000, attrs={"incident_id":"I1","attack_stage":"initial_access",
                               "containment_ts":3601000, "containment_verified":True}),
            Event(3601000, attrs={"incident_id":"I1","attack_stage":"lateral_movement"}),
        ]
        self.assertEqual(compute_long_dwell_recall(y, predictions, items, min_dwell_ms=3600_000), 1.)
        self.assertEqual(compute_ttc(items, y, predictions), 60.)

    def test_drift_does_not_claim_without_reference(self):
        obj = DriftMonitor(min_window=10)
        self.assertIsNone(obj.assess(np.ones(20))["drift"])
        obj.fit_reference(np.linspace(0.01, 0.9, 40))
        shifted = obj.assess(np.full(40, .99))
        self.assertTrue(shifted["drift"])
        self.assertIsNone(shifted["ece"])


if __name__ == "__main__":
    unittest.main()
