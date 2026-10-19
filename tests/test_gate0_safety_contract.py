"""Isolated safety contract tests; NOT a certification of operational safety."""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, rel in [("amcf", "amcf"), ("amcf.safety", "amcf/safety")]:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / rel)]
        sys.modules[name] = module

from amcf.safety.safety_gate import SafetyGate


class TestSafetyGateFailClosed(unittest.TestCase):
    def test_missing_outcomes_deny(self):
        gate = SafetyGate()
        self.assertFalse(gate.check(.99, {'lightgbm': .99, 'tcn': .99}))

    def test_malformed_detector_evidence_denied(self):
        gate = SafetyGate()
        self.assertFalse(gate.check(.99, ['lightgbm', 'tcn']))
        self.assertFalse(gate.check(.99, None))

    def test_200_zero_false_isolations_not_enough(self):
        gate = SafetyGate()
        for _ in range(200):
            gate.record_outcome(False)
        gate.update_correlation_estimate('lightgbm', 'tcn', .1)
        self.assertFalse(gate.check(.99, {'lightgbm': .99, 'tcn': .99}))

    def test_unmeasured_dependence_denied_even_with_relaxed_risk(self):
        gate = SafetyGate(epsilon=.05)
        for _ in range(200):
            gate.record_outcome(False)
        self.assertFalse(gate.check(.99, {'lightgbm': .99, 'tcn': .99}))

    def test_measured_evidence_can_pass_with_relaxed_nonproduction_target(self):
        gate = SafetyGate(epsilon=.05)
        for _ in range(200):
            gate.record_outcome(False)
        gate.update_correlation_estimate('lightgbm', 'tcn', .1)
        self.assertTrue(gate.check(.99, {'lightgbm': .99, 'tcn': .99}))


if __name__ == '__main__':
    unittest.main()
