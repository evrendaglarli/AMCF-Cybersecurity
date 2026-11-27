"""Reference-controller v45 R3 call-path checks, without real EDR/SOAR."""
import unittest

# This existing test module injects lightweight in-memory adapters before
# importing the controller. No model downloads, SOAR calls, or telemetry.
from test_r20_controller_contracts import TestControllerConsistency


class TestV45R3Controller(unittest.TestCase):
    def test_voi_and_scheduler_share_one_cost_source(self):
        controller = TestControllerConsistency().new_controller()
        for name in ("edr_deep", "sandbox", "pcap", "memory_dump"):
            self.assertEqual(
                controller._get_enrichment_cost(name),
                controller.enrichment_scheduler.get_enrichment_cost(name)
            )

    def test_controller_uses_monte_carlo_and_does_not_invent_outcomes(self):
        controller = TestControllerConsistency().new_controller()
        self.assertEqual(controller.voi_estimator.N_min, 64)
        self.assertEqual(controller.voi_estimator.N_max, 4096)
        self.assertEqual(controller.voi_estimator.budget_ms, 150)
        value, count = controller.voi_estimator.estimate_voi(
            .5, controller._get_enrichment_cost("sandbox"), "sandbox")
        self.assertEqual((value, count), (0., 0))
        self.assertEqual(controller.voi_estimator.last_result.status,
                         "missing_outcome_model")

    def test_automation_without_validated_tau_fails_closed(self):
        controller = TestControllerConsistency().new_controller()
        from amcf.core.metacognitive_controller import Event
        event = Event(ts=1, src="host1", dst=None, kind="edr", attrs={},
                      detector_scores={"alpha":.9, "beta":.9})
        result = controller._act(
            event, "automate", .9999,
            {"ignore":15.0, "gather":float("inf"),
             "automate":5.0, "escalate":7.0},
            {"alpha":.9999,"beta":.9999}
        )
        self.assertEqual(result["effective_action"], "escalate")
        self.assertEqual(result["gate_reason"],
                         "missing_validation_risk_threshold")
        self.assertNotIn("containment_verified", result)


if __name__ == "__main__":
    unittest.main()
