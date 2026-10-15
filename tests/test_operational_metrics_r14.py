from types import SimpleNamespace
import pytest
from amcf.utils.metrics import (compute_long_dwell_recall, compute_ttc,
                                compute_escalation_precision, compute_fp_per_endpoint_day)


def test_pre_impact_chain_recall_is_not_event_recall():
    chains = [dict(chain_id=1, impact_ts_ms=1000, first_alert_ts_ms=900),
              dict(chain_id=2, impact_ts_ms=2000, first_alert_ts_ms=2001),
              dict(chain_id=3, impact_ts_ms=4000, first_alert_ts_ms=None)]
    assert compute_long_dwell_recall([], [], chain_records=chains) == pytest.approx(1/3)
    with pytest.raises(ValueError, match='chain'):
        compute_long_dwell_recall([1], [1])


def test_containment_is_actual_action_and_censoring_is_accounted_for():
    cases = [dict(incident_id='a', first_malicious_event_ts_ms=0, containment_ts_ms=600_000),
             dict(incident_id='b', first_malicious_event_ts_ms=0, containment_ts_ms=900_000),
             dict(incident_id='c', first_malicious_event_ts_ms=0, censored_at_ts_ms=1_200_000)]
    assert compute_ttc([], [], [], incident_records=cases) == 15.0
    with pytest.raises(ValueError, match='incident_records'):
        compute_ttc([SimpleNamespace(ts=0)], [1], [1])


def test_escalation_does_not_confuse_auto_positive_with_analyst_handoff():
    rows = [dict(escalation_id='case1', verified_true_positive=True),
            dict(escalation_id='case2', verified_true_positive=False)]
    assert compute_escalation_precision([], [], escalation_records=rows) == 0.5
    with pytest.raises(ValueError, match='records'):
        compute_escalation_precision([1], [1])


def test_fp_rate_requires_logged_active_endpoint_days():
    events = [SimpleNamespace(ts=0, src='host1'), SimpleNamespace(ts=1000, src='host1')]
    with pytest.raises(ValueError, match='active_endpoint_days'):
        compute_fp_per_endpoint_day(events, [1, 1], [0, 1])
    assert compute_fp_per_endpoint_day(events, [1, 1], [0, 1], active_endpoint_days=100) == 0.01