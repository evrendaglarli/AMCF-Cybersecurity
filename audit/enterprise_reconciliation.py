"""Check AMCF enterprise alarm counts against endpoint-normalized FP rates.

Requires actual daily active-endpoint counts for a definitive comparison.
Never repairs or synthesizes a study result.  A flat endpoint count is an
explicit *scenario assumption*, not a replacement for operations logs.

Example (hypothetical 8,400 active endpoints every day):
  python audit/enterprise_reconciliation.py --days 15 --active-endpoints 8400 \
      --alerts 338 --reported-fp-rate 0.061

Actual-data check: CSV columns date, active_endpoints, false_positives, alerts
  python audit/enterprise_reconciliation.py --daily-csv verified_daily.csv \
      --reported-fp-rate 0.061 --alerts 338
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def assess(alerts: int, endpoint_days: int, reported_fp_rate: float,
           actual_false_positives: int | None = None,
           tolerance: float = 0.0005) -> dict:
    """Return explicit feasibility checks; do not infer actual FP from rounded rate."""
    if not (isinstance(alerts, int) and alerts >= 0 and
            isinstance(endpoint_days, int) and endpoint_days > 0 and
            0 <= reported_fp_rate and tolerance >= 0):
        raise ValueError('Nonnegative alarms/rate and positive integer endpoint-days required.')
    implied = reported_fp_rate * endpoint_days
    lower = max(0.0, reported_fp_rate - tolerance) * endpoint_days
    upper = (reported_fp_rate + tolerance) * endpoint_days
    feasible = lower <= alerts  # FP <= total alerts, for the same events and time interval
    result = {
        'endpoint_days': endpoint_days,
        'total_alerts': alerts,
        'reported_fp_per_endpoint_day': reported_fp_rate,
        'implied_fp_center_unrounded': implied,
        'implied_fp_interval_due_to_rounding': [lower, upper],
        'feasible_given_fp_not_exceeding_alerts': feasible,
        'maximum_feasible_endpoint_days_at_reported_rate': (
            alerts / max(reported_fp_rate - tolerance, 1e-30)
            if reported_fp_rate > tolerance else None),
    }
    if actual_false_positives is not None:
        if not isinstance(actual_false_positives, int) or actual_false_positives < 0:
            raise ValueError('actual_false_positives must be a nonnegative integer.')
        result['actual_false_positives'] = actual_false_positives
        result['actual_fp_per_endpoint_day'] = actual_false_positives / endpoint_days
        result['observed_counts_consistent'] = (actual_false_positives <= alerts)
        result['reported_rate_consistent_with_actual_count'] = (
            abs(actual_false_positives / endpoint_days - reported_fp_rate) <= tolerance)
    return result


def read_daily_rows(path: str):
    endpoint_days = alerts = false_positives = 0
    seen = set()
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        expected = {'date', 'active_endpoints', 'false_positives', 'alerts'}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError('CSV requires: date, active_endpoints, false_positives, alerts')
        for row in reader:
            day = row['date'].strip()
            if not day or day in seen:
                raise ValueError('Each calendar date must be unique and nonempty.')
            seen.add(day)
            n = int(row['active_endpoints'])
            fp = int(row['false_positives'])
            al = int(row['alerts'])
            if n < 1 or fp < 0 or al < 0 or fp > al:
                raise ValueError(f'Invalid counts on {day}; false positives cannot exceed alerts.')
            endpoint_days += n
            false_positives += fp
            alerts += al
    if not seen:
        raise ValueError('No daily records.')
    return endpoint_days, alerts, false_positives, len(seen)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--daily-csv', type=Path, help='Actual verified per-day counts')
    src.add_argument('--active-endpoints', type=int, help='ASSUMED active endpoints every day')
    parser.add_argument('--days', type=int, help='Required for flat endpoint assumption')
    parser.add_argument('--alerts', type=int, help='Reported total alerts over same window')
    parser.add_argument('--reported-fp-rate', type=float, required=True)
    parser.add_argument('--tolerance', type=float, default=0.0005,
                        help='Rounding half-width for a rate printed to 3 decimals')
    args = parser.parse_args(argv)
    if args.daily_csv:
        ep_days, al, fp, days = read_daily_rows(str(args.daily_csv))
        if args.alerts is not None and args.alerts != al:
            parser.error(f'Reported {args.alerts} alerts but CSV totals {al}.')
        result = assess(al, ep_days, args.reported_fp_rate, fp, args.tolerance)
        result['basis'] = f'VERIFIED_DAILY_CSV: {args.daily_csv}'
        result['days'] = days
    else:
        if not args.days or args.days <= 0 or args.alerts is None:
            parser.error('--days and --alerts are required with --active-endpoints')
        result = assess(args.alerts, args.active_endpoints * args.days,
                        args.reported_fp_rate, tolerance=args.tolerance)
        result['basis'] = 'SCENARIO_ASSUMPTION_NOT_OBSERVED_DATA'
        result['days'] = args.days
        result['assumed_daily_active_endpoints'] = args.active_endpoints
    print(json.dumps(result, indent=2))
    good = result['feasible_given_fp_not_exceeding_alerts']
    good = good and result.get('observed_counts_consistent', True)
    good = good and result.get('reported_rate_consistent_with_actual_count', True)
    if not good:
        print('AUDIT DISCREPANCY: do not alter the paper before validating inputs.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())