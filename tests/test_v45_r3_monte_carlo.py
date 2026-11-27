"""Independent v45 R3 Monte Carlo and candidate-safety contract checks."""
import sys
import types
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name, relative in [
    ("amcf", "amcf"), ("amcf.core", "amcf/core"),
    ("amcf.safety", "amcf/safety")
]:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / relative)]
        sys.modules[name] = module

from amcf.core.bayes_risk import CostModel, BayesRiskCalculator
from amcf.core.monte_carlo_voi import MonteCarloVoIEstimator
from amcf.safety.safety_gate import SafetyGate


class TestV45R3MonteCarlo(unittest.TestCase):
    def setUp(self):
        self.calc = BayesRiskCalculator(CostModel())

    def test_missing_outcome_distribution_fails_closed(self):
        est = MonteCarloVoIEstimator(self.calc)
        self.assertEqual(est.estimate_voi(.5, .1, "missing"), (0.0, 0))
        self.assertEqual(est.last_result.status, "missing_outcome_model")

    def test_constant_conditional_risk_stops_at_64(self):
        est = MonteCarloVoIEstimator(self.calc, wall_clock_budget_ms=3000)
        est.register_outcome_model("known", lambda **kw: [(1.0, 0.2)])
        result = est.estimate_detailed(.5, .1, "known")
        baseline = min(self.calc.compute_risks(.5)[a] for a in
                       ("ignore", "automate", "escalate"))
        after = min(self.calc.compute_risks(.2)[a] for a in
                    ("ignore", "automate", "escalate"))
        self.assertEqual(result.status, "tolerance_reached")
        self.assertEqual(result.samples, 64)
        self.assertEqual(result.ci_halfwidth, 0.0)
        self.assertAlmostEqual(result.voi, baseline-after-.1)

    def test_sampled_conditional_mean_and_bca(self):
        est = MonteCarloVoIEstimator(self.calc, N_min=64, N_max=256,
                                      tolerance=1e-5, wall_clock_budget_ms=3000,
                                      ci_resamples=250, seed=7)
        est.register_outcome_model("observed", lambda **kw: [(.5, .1), (.5, .9)])
        result = est.estimate_detailed(.5, .1, "observed")
        baseline = min(self.calc.compute_risks(.5)[a] for a in
                       ("ignore", "automate", "escalate"))
        expected = sum(.5*min(self.calc.compute_risks(p)[a] for a in
                                ("ignore", "automate", "escalate"))
                       for p in (.1, .9))
        self.assertEqual(result.status, "maximum_samples")
        self.assertEqual(result.samples, 256)
        self.assertIsNotNone(result.ci_halfwidth)
        self.assertGreater(result.ci_halfwidth, 0.0)
        self.assertAlmostEqual(result.voi, baseline-expected-.1, delta=.6)

    def test_rare_unobserved_outcome_not_false_convergence(self):
        est = MonteCarloVoIEstimator(
            self.calc, N_min=64, N_max=64,
            tolerance=0.02, wall_clock_budget_ms=3000,
            seed=42
        )
        est.register_outcome_model(
            "rare", lambda **kw: [(1.0 - 1e-12, 0.1), (1e-12, 0.9)]
        )
        result = est.estimate_detailed(.5, .1, "rare")
        self.assertEqual(result.samples, 64)
        self.assertEqual(result.status, "maximum_samples")
        self.assertEqual(result.ci_halfwidth, float("inf"))

    def test_malformed_distribution_rejected(self):
        est = MonteCarloVoIEstimator(self.calc)
        est.register_outcome_model("bad", lambda **kw: [(0.7, .5)])
        with self.assertRaises(ValueError):
            est.estimate_voi(.5, .1, "bad")


class TestV45R3SafetyMonitoring(unittest.TestCase):
    def test_monitor_requires_200_confirmed_outcomes(self):
        gate = SafetyGate(window_size=200)
        self.assertIsNone(gate.false_isolation_monitor())
        for _ in range(199):
            gate.record_outcome(False)
        self.assertIsNone(gate.false_isolation_monitor())
        self.assertFalse(gate._check_false_isolation_bound())

    def test_bootstrap_is_noncertifying_monitor(self):
        gate = SafetyGate(window_size=200, epsilon=1e-3)
        for _ in range(200):
            gate.record_outcome(False)
        self.assertEqual(gate.false_isolation_monitor(), 0.0)
        self.assertTrue(gate._check_false_isolation_bound())
        # A zero observed/bootstrap rate is NOT a confidence certificate
        # that the real false-isolation probability is below 1e-3.
        gate.record_outcome(True)
        self.assertGreater(gate.false_isolation_monitor(), 0.001)
        self.assertFalse(gate._check_false_isolation_bound())


class TestV45R3ActionRisk(unittest.TestCase):
    class VerifiedInputsOnlyGate(SafetyGate):
        """Isolate action-specific Eq. (13) test; NOT an actual admission gate."""
        def check(self, posterior, calibrated_probs, **kwargs):
            return posterior >= .999

    def test_missing_action_class_or_threshold_denied(self):
        g = self.VerifiedInputsOnlyGate()
        risks = {"ignore":12., "gather":float("inf"),
                 "automate":5., "escalate":7.}
        kw = dict(posterior=.9999, calibrated_probs={"a":.999,"b":.999},
                  risk_by_action=risks, tau_risk=1.0, playbook_id="quarantine")
        self.assertFalse(g.check_endpoint_isolation(action_class="automate", **kw))
        self.assertFalse(g.check_endpoint_isolation(
            action_class="endpoint_isolation", **{**kw,"tau_risk":None}))

    def test_rbase_excludes_automation_and_positive_delta(self):
        g = self.VerifiedInputsOnlyGate()
        risks = {"ignore":12., "gather":float("inf"),
                 "automate":5., "escalate":7.}
        kw = dict(posterior=.9999, calibrated_probs={"a":.999,"b":.999},
                  risk_by_action=risks, playbook_id="quarantine",
                  action_class="endpoint_isolation")
        self.assertTrue(g.check_endpoint_isolation(tau_risk=2., **kw))
        self.assertFalse(g.check_endpoint_isolation(tau_risk=2.01, **kw))


if __name__ == "__main__":
    unittest.main()
