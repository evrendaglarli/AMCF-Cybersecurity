"""Math/protocol-level tests only; do not claim results of real dataset experiment."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
from unsw_nb15_robustness import wilson_ci, paired_delta_bootstrap, fusion
from unsw_nb15_feasibility import fixed_fpr_threshold

class TestRobustness(unittest.TestCase):
    def test_wilson_intervals(self):
        assert wilson_ci(0, 20)[0] == 0
        assert wilson_ci(20, 20)[1] == 1
        lo, hi=wilson_ci(50,100)
        assert lo < 0.5 < hi
        assert abs(0.5-lo-(hi-0.5)) < 1e-12

    def test_paired_delta_identical(self):
        y=np.array([0]*70+[1]*90)
        pred=np.array([0,1]*80)
        ci=paired_delta_bootstrap(y,pred,pred,class_label=1,repetitions=100,seed=42)
        assert ci == (0.,0.)

    def test_paired_difference_positive(self):
        y=np.array([1]*100+[0]*100)
        a=np.array([1]*90+[0]*10+[1]*5+[0]*95)
        b=np.array([1]*50+[0]*50+[1]*5+[0]*95)
        lo,hi=paired_delta_bootstrap(y,a,b,class_label=1,repetitions=1000,seed=1)
        assert 0.3<=lo<hi<=0.5
        assert paired_delta_bootstrap(y,a,b,class_label=0,repetitions=100,seed=1)==(0.,0.)

    def test_fusion_symmetry(self):
        p=np.array([0.1,0.4,0.75])
        assert np.allclose(fusion(p,p),p)
        assert np.allclose(fusion(p,1-p),0.5)

    def test_validation_budget_is_not_test_tuned(self):
        tuning_scores=np.array([0.2,0.4,0.6,0.8,0.9,0.95])
        labels=np.zeros(len(tuning_scores),dtype=int)
        cutoff=fixed_fpr_threshold(tuning_scores,labels,0.1)
        assert ((tuning_scores >= cutoff).mean() <= 0.1)

if __name__=='__main__':unittest.main()