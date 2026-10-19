"""Risk-bound arithmetic tests; not a certification of system safety."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'amcf' / 'safety'))
from rare_event_risk import one_sided_binomial_upper

class TestOneSidedBound(unittest.TestCase):
    def test_zero_observations_is_unknown(self):
        self.assertEqual(one_sided_binomial_upper(0,0),1.0)
    def test_200_zero_false_isolations_not_enough_for_1e_minus3(self):
        bound=one_sided_binomial_upper(0,200)
        self.assertGreater(bound,0.001)
        self.assertAlmostEqual(bound, 1 - 0.05**(1/200),places=11)
    def test_3000_zero_trials_enough_only_under_iid_assumption(self):
        self.assertLess(one_sided_binomial_upper(0,3000),0.001)
    def test_more_failures_worse(self):
        self.assertLess(one_sided_binomial_upper(0,200),one_sided_binomial_upper(1,200))
    def test_invalid(self):
        for f,n in [(-1,5),(6,5),(0,-1)]:
            with self.assertRaises(ValueError): one_sided_binomial_upper(f,n)
        with self.assertRaises(ValueError): one_sided_binomial_upper(0,10,confidence=1)
        with self.assertRaises(TypeError): one_sided_binomial_upper(0.0,10)

if __name__=='__main__':unittest.main()