"""v45 R3 Appendix B/Algorithm 6: causal, immutable, restartable feedback."""
import sys
import types
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name, relative in [
    ("amcf", "amcf"),("amcf.core","amcf/core"),("amcf.safety","amcf/safety")
]:
    if name not in sys.modules:
        module=types.ModuleType(name)
        module.__path__=[str(ROOT/relative)]
        sys.modules[name]=module

from amcf.core.decision_ledger import DecisionLedger, digest
from amcf.core.calibration import fit_confirmed_temperature
from test_r20_controller_contracts import TestControllerConsistency
from amcf.core.metacognitive_controller import Event


class TestLedger(unittest.TestCase):
    def test_snapshot_persists_and_id_is_sha256_canonical(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/"decisions.sqlite3")
            ledger=DecisionLedger(path)
            snapshot={"event_ts_ms":1000,"risks":{"gather":float("inf")},
                      "posterior":0.25,"weights":[.5,.5]}
            d=ledger.record(snapshot)
            self.assertEqual(d,digest(snapshot))
            self.assertEqual(d,ledger.record(snapshot))
            self.assertEqual(ledger.read(d)["risks"]["gather"],"+Infinity")
            self.assertTrue(ledger.verify_chain())
            ledger.close()
            ledger=DecisionLedger(path)
            self.assertEqual(ledger.read(d)["posterior"],.25)
            self.assertTrue(ledger.verify_chain())
            ledger.close()

    def test_numpy_vector_fingerprint_is_canonical(self):
        # A multi-element feature vector must not invoke ndarray.item().
        value={"event_ts_ms":1, "features":np.asarray([0.1,0.2])}
        d=DecisionLedger()
        identifier=d.record(value)
        self.assertEqual(d.read(identifier)["features"],[0.1,0.2])
        self.assertTrue(d.verify_chain())
        d.close()

    def test_feedback_time_duplicate_and_source_guard(self):
        ledger=DecisionLedger()
        d=ledger.record({"event_ts_ms":1000,"posterior":.8})
        state={"weights":[.5,.5]}
        with self.assertRaises(ValueError):
            ledger.write_feedback_and_state(decision_id=d,outcome="TP",
                source="analyst_confirmed",available_at_ms=999,state=state)
        with self.assertRaises(ValueError):
            ledger.write_feedback_and_state(decision_id=d,outcome="TP",
                source="inferred_prediction",available_at_ms=1001,state=state)
        ledger.write_feedback_and_state(decision_id=d,outcome="TP",
            source="analyst_confirmed",available_at_ms=1500,state=state)
        with self.assertRaises(ValueError):
            ledger.write_feedback_and_state(decision_id=d,outcome="FP",
                source="analyst_confirmed",available_at_ms=1600,state=state)
        self.assertTrue(ledger.verify_chain())


class TestControllerCausalFeedback(unittest.TestCase):
    def test_decision_recorded_before_act_label_not_used_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            c=TestControllerConsistency().new_controller()
            # Switch to a file-backed decision ledger for this test.
            c.decision_ledger.close()
            path=str(Path(folder)/"ledger.sqlite3")
            c.decision_ledger=DecisionLedger(path)
            e=Event(ts=1000,src="host1",dst="host2",kind="netflow",
                    attrs={"event_id":"first"},detector_scores={"alpha":.9,"beta":.2},
                    label=1)
            before=c.weight_manager.get_weights()
            action,metadata=c.process_event(e)
            d=metadata["DecisionID"]
            self.assertIn(action,("ignore","escalate"))
            self.assertEqual(c.decision_ledger.read(d)["action"],
                             metadata["outcome"]["action"] if action=="ignore" else
                             c.decision_ledger.read(d)["action"])
            np.testing.assert_allclose(before,c.weight_manager.get_weights())
            self.assertTrue(c.decision_ledger.verify_chain())
            with self.assertRaises(ValueError):
                c.ingest_confirmed_feedback(d,"TP",
                    source="analyst_confirmed",available_at_ms=999)
            feedback=c.ingest_confirmed_feedback(d,"TP",
                source="analyst_confirmed",available_at_ms=1200)
            self.assertEqual(feedback["label"],1)
            with self.assertRaises(ValueError):
                c.ingest_confirmed_feedback(d,"FP",
                    source="analyst_confirmed",available_at_ms=1300)
            confirmed_weights=c.weight_manager.get_weights().copy()
            self.assertEqual(len(c.feedback_window),1)
            self.assertTrue(c.decision_ledger.verify_chain())
            c.decision_ledger.close()
            recovered=TestControllerConsistency().new_controller()
            recovered.decision_ledger.close()
            recovered.decision_ledger=DecisionLedger(path)
            recovered._restore_confirmed_state()
            np.testing.assert_allclose(confirmed_weights,
                                      recovered.weight_manager.get_weights())
            self.assertEqual(len(recovered.feedback_window),1)
            self.assertTrue(recovered.decision_ledger.verify_chain())
            recovered.decision_ledger.close()

    def test_class_enriched_window_and_temperature(self):
        s=np.array([-3.0]*100+[3.0]*100)
        y=np.array([0]*100+[1]*100)
        t=fit_confirmed_temperature(s,y,rho=.99)
        self.assertTrue(np.isfinite(t) and t>0)
        with self.assertRaises(ValueError):
            fit_confirmed_temperature(s,np.zeros(200))

if __name__=="__main__":
    unittest.main()
