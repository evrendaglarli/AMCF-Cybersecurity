import csv
from audit.enterprise_reconciliation import assess, read_daily_rows


def test_r14_numbers_are_incompatible_if_8400_endpoints_active_15_days():
    check = assess(alerts=338, endpoint_days=8400 * 15, reported_fp_rate=0.061)
    assert not check['feasible_given_fp_not_exceeding_alerts']
    assert check['implied_fp_center_unrounded'] == 7686


def test_small_realistic_aggregate_can_be_consistent():
    check = assess(alerts=100, endpoint_days=1000, reported_fp_rate=0.061,
                   actual_false_positives=61)
    assert check['feasible_given_fp_not_exceeding_alerts']
    assert check['observed_counts_consistent']
    assert check['reported_rate_consistent_with_actual_count']


def test_daily_csv_checks_nonfabricated_numerator_and_denominator(tmp_path):
    path = tmp_path / 'daily.csv'
    with path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'active_endpoints', 'false_positives', 'alerts'])
        writer.writerow(['2026-01-01', '500', '10', '15'])
        writer.writerow(['2026-01-02', '800', '12', '18'])
    assert read_daily_rows(str(path)) == (1300, 33, 22, 2)