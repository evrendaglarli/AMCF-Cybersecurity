import numpy as np
import pytest
from amcf.utils.statistics import bootstrap_ci, delong_test, holm_bonferroni_correction


def test_bca_interval_for_non_degenerate_recall():
    y = np.r_[np.ones(90, dtype=int), np.zeros(110, dtype=int)]
    preds = np.r_[np.ones(70, dtype=int), np.zeros(20, dtype=int), np.zeros(110, dtype=int)]
    ci = bootstrap_ci(y, preds, n_bootstrap=2000, random_state=11)
    assert 0.65 < ci[0] < 70 / 90 < ci[1] < 0.90
    assert bootstrap_ci(y, preds, n_bootstrap=2000, random_state=11) == ci


def test_bca_rejects_degenerate_input():
    with pytest.raises(ValueError, match='degenerate'):
        bootstrap_ci([1] * 10 + [0] * 10, [1] * 10 + [0] * 10)


def test_paired_delong_is_symmetric_and_identity():
    rng = np.random.default_rng(987)
    y = np.r_[np.ones(100), np.zeros(100)]
    better = y + rng.normal(0, 0.3, len(y))
    worse = rng.normal(0, 1, len(y))
    p = delong_test(y, better, worse)
    assert p < 0.01
    assert p == pytest.approx(delong_test(y, worse, better))
    assert delong_test(y, better, better) == 1.0


def test_paired_delong_rejects_single_class():
    with pytest.raises(ValueError):
        delong_test([1, 1, 1], [0.1, 0.2, 0.3], [0.3, 0.4, 0.5])


def test_holm_adjusted_rejections_follow_monotone_values():
    adj, reject = holm_bonferroni_correction(np.array([0.03, 0.01, 0.04]))
    assert np.allclose(adj, [0.06, 0.03, 0.06])
    assert reject.tolist() == [False, True, False]