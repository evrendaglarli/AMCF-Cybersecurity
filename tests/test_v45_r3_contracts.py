"""v45 R3 equation-contract checks; NOT empirical replication.

These test algebra/nominal guards, not the paper's results tables or actual SOAR.
"""
import math
import unittest
import numpy as np
from amcf.core.bayes_risk import BayesRiskCalculator, CostModel, ValueOfInformationEstimator
from amcf.core.fusion import LogitWeightedFusion
from amcf.core.calibration import CompetenceWeightManager
from amcf.safety.enrichment_scheduler import EnrichmentScheduler
from amcf.safety.safety_gate import SafetyGate


class TestV45R3Contracts(unittest.TestCase):
    def setUp(self):
        self.calc = BayesRiskCalculator(CostModel())

    def test_eq3_weighted_logit_fusion(self):
        p = np.array([[0.2, 0.8]])
        w = np.array([0.25, 0.75])
        bias = -0.15
        z = float(np.dot(np.log(p[0] / (1-p[0])), w) + bias)
        actual = LogitWeightedFusion(2).fuse(p, w, bias)[0]
        self.assertAlmostEqual(actual, 1/(1+math.exp(-z)), places=12)

    def test_eq2_fused_brier_weights_on_simplex(self):
        mgr = CompetenceWeightManager(2, lambda_ema=0.5, learning_rate=0.1)
        mgr.update(np.array([[0.9, 0.1], [0.8, 0.2]]), np.array([1, 1]))
        self.assertGreater(mgr.weights[0], mgr.weights[1])
        self.assertTrue(np.all(mgr.weights >= 0))
        self.assertAlmostEqual(mgr.weights.sum(), 1.0)

    def test_eq6_eq7_eq8_and_table2_costs(self):
        p, loss, delay = 0.6, 15.0, 5.0
        r = self.calc.compute_risks(p, severity="medium", asset_weight=1.0,
                                    queue_length=0, expected_delay=delay,
                                    residual_risk_factor=0.1)
        self.assertAlmostEqual(r["ignore"], p*loss)
        self.assertAlmostEqual(r["automate"], (1-p)*1+p*0.1*loss+1.2+2.5)
        self.assertAlmostEqual(r["escalate"], 3+0.5*p*loss*(1-math.exp(-0.1*delay)))
        self.assertTrue(math.isinf(r["gather"]))  # no enrichment outcome model

    def test_section7_pairwise_risk_boundaries(self):
        loss = 15.0
        delay_fraction = 1-math.exp(-0.5)
        b1 = 3.0/(loss-0.5*loss*delay_fraction)
        b2 = (1.0+1.2+2.5-3.0)/(1.0+0.5*loss*delay_fraction-0.1*loss)
        self.assertAlmostEqual(b1, 0.249, delta=0.001)
        self.assertAlmostEqual(b2, 0.694, delta=0.001)

    def test_eq10_exact_conditional_voi_counts_cost_once(self):
        est = ValueOfInformationEstimator(self.calc)
        self.assertEqual(est.estimate_voi(0.5, 0.25, "unknown"), (0.0, 0))
        est.register_outcome_model("specified",
                                   lambda **kw: [(0.5, 0.1), (0.5, 0.9)])
        low, n = est.estimate_voi(0.5, 0.25, "specified")
        high, _ = est.estimate_voi(0.5, 0.5, "specified")
        self.assertAlmostEqual(low-high, 0.25)
        self.assertEqual(n, 0)  # exact discrete sum, NOT manuscript Monte Carlo

    def test_eq15_one_feasible_positive_enrichment(self):
        scheduler = EnrichmentScheduler(budget=20)
        candidates = {"expensive": {"voi": 100, "cost": 21},
                      "best": {"voi": 8, "cost": 12},
                      "next": {"voi": 7, "cost": 5}}
        self.assertEqual(scheduler.select_enrichments(candidates), ["best"])
        self.assertEqual(scheduler.select_enrichments(
            {"nonpositive": {"voi": 0, "cost": 2}}), [])

    def test_eq14_nominal_isolation_probability(self):
        # Override the OTHER independent gates to isolate Eq (14).
        class IsolatedGate(SafetyGate):
            def _check_false_isolation_bound(self):
                return True
            def _check_dual_corroboration(self, calibrated_probs):
                return True
        gate = IsolatedGate(epsilon=1e-3)
        advisors = {"a": 0.9999, "b": 0.9999}
        self.assertFalse(gate.check(0.998, advisors))
        self.assertTrue(gate.check(0.9995, advisors))


if __name__ == "__main__":
    unittest.main()
