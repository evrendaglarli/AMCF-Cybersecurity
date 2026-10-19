"""Offline controller integration contracts with lightweight in-memory adapters.

The real episodic vector store and SOAR/EDR connectors are intentionally
not loaded: no network, model downloads, or physical containment.
"""
import sys
import types
import unittest
from pathlib import Path
from math import log

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name, relative in [("amcf", "amcf"), ("amcf.core", "amcf/core"),
                       ("amcf.safety", "amcf/safety")]:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / relative)]
        sys.modules[name] = module


class WorkingMemory:
    def __init__(self, **kwargs):
        self.buffer = []

    def append(self, event, posterior):
        self.buffer.append((event, posterior, event.ts))

    def get_recent_events(self, k=50):
        return self.buffer[-k:]


class EpisodicMemory:
    def __init__(self, **kwargs):
        self.outcomes = []

    def retrieve_similar(self, events, top_k=5):
        return []

    def append_outcome(self, *args):
        self.outcomes.append(args)


class ProceduralMemory:
    def __init__(self):
        pass

    def select_playbook(self, *args):
        return "proposal_only"


memory = types.ModuleType("amcf.memory")
memory.WorkingMemory = WorkingMemory
memory.EpisodicMemory = EpisodicMemory
memory.ProceduralMemory = ProceduralMemory
sys.modules["amcf.memory"] = memory

from amcf.safety.safety_gate import SafetyGate
from amcf.safety.enrichment_scheduler import EnrichmentScheduler
sys.modules["amcf.safety"].SafetyGate = SafetyGate
sys.modules["amcf.safety"].EnrichmentScheduler = EnrichmentScheduler

from amcf.core.metacognitive_controller import (
    MetacognitiveController, Event, CostModel
)


class TestControllerConsistency(unittest.TestCase):
    def new_controller(self):
        return MetacognitiveController(2, ["alpha", "beta"], CostModel(), {
            "detector_score_spaces": {"alpha":"probability","beta":"probability"},
            "drift_window": 10,
        })

    def test_calibrator_score_space(self):
        c = self.new_controller()
        self.assertAlmostEqual(c._calibration_logit("alpha", .9), log(9))
        with self.assertRaises(ValueError):
            c._calibration_logit("alpha", 1.1)

    def test_one_stage_event_and_confirmed_feedback_boundary(self):
        c = self.new_controller()
        e = Event(ts=1234, src="host1", dst="host2", kind="edr",
                  attrs={"attack_stage":"execution", "event_id":"evt1"},
                  detector_scores={"alpha":.8, "beta":.7},
                  label=1)
        action, metadata = c.process_event(e)
        self.assertIn(action, ("ignore", "escalate"))
        self.assertAlmostEqual(sum(metadata["hypothesis_beliefs"].values()), 1.0)
        self.assertEqual(metadata["drift"]["status"], "insufficient_evidence")
        self.assertEqual(len(c.episodic_memory.outcomes), 0)  # Unconfirmed inline label
        first = c.hypothesis_tracker.active["host1"].log_evidence
        c.process_event(e)
        self.assertEqual(c.hypothesis_tracker.active["host1"].log_evidence, first)

    def test_gather_proposal_does_not_report_executed(self):
        c = self.new_controller()
        e = Event(ts=1234, src="host1", dst="host2", kind="edr",
                  attrs={}, detector_scores={"alpha":.8, "beta":.7})
        result = c._act(e, "gather", .7, {"ignore":3.0}, {"alpha":.8},
                        {"edr_deep":{"voi":9.0, "cost":5.0},
                         "sandbox":{"voi":8.0, "cost":1.0}})
        self.assertEqual(result["effective_action"], "escalate")
        self.assertEqual(result["enrichments_proposed"], ["edr_deep"])
        self.assertNotIn("enrichments_requested", result)


if __name__ == "__main__":
    unittest.main()
