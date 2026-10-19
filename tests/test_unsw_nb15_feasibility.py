"""Tests for the UNSW-NB15 feasibility script. Fixtures are NOT real UNSW data."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / 'experiments' / 'unsw_nb15_feasibility.py'
spec = importlib.util.spec_from_file_location('unsw_nb15_feasibility', CODE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestUnswFeasibility(unittest.TestCase):
    def test_missing_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(FileNotFoundError):
                mod.load_release(root / 'missing_train.csv', root / 'missing_test.csv')

    def test_expected_release_shape_protected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            df = pd.DataFrame({'proto': ['tcp','udp'], 'service': ['http','-'],
                               'state': ['FIN','INT'], 'dur': [0.1,0.2],
                               'label': [0,1], 'attack_cat': ['Normal', 'Exploits']})
            a, b = root/'train.csv', root/'test.csv'
            df.to_csv(a, index=False); df.to_csv(b, index=False)
            with self.assertRaises(ValueError):
                mod.load_release(a, b)
            tr, te, feats = mod.load_release(a, b, allow_partial=True)
            self.assertNotIn('label', feats)
            self.assertNotIn('attack_cat', feats)
            self.assertEqual(len(tr), 2)

    def test_threshold_stays_within_validation_budget(self):
        negative_scores = np.array([0.05, 0.08, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                    0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        scores = np.r_[negative_scores, [0.9, 0.8]]
        labels = np.r_[np.zeros(20, dtype=int), [1, 1]]
        threshold = mod.fixed_fpr_threshold(scores, labels, 0.05)
        achieved = np.mean(negative_scores >= threshold)
        self.assertLessEqual(achieved, 0.05 + 1e-12)

    def test_metrics_are_binary_flow_level_only(self):
        y = np.array([0,0,1,1]); p = np.array([0.1,0.7,0.6,0.9])
        result = mod.metrics(y, p, 0.5)
        self.assertEqual((result['tn'],result['fp'],result['fn'],result['tp']), (1,1,0,2))
        self.assertEqual(result['recall'], 1.0)
        self.assertNotIn('ttc', result)
        self.assertNotIn('fp_per_hour', result)


if __name__ == '__main__':
    unittest.main()