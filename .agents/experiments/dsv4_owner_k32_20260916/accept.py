"""Fail closed on K32's numerical, dispatch, lifecycle and ABBA evidence.

The spread check is a conservative screening rule, not a confidence interval.
Acceptance is restricted to the measured C16x32K profile, not a global default.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics


def validate_summary(summary):
    assert summary['status'] == 'complete'
    assert summary['kv_tokens'] == 1048576 and summary['original_weights'] is True
    assert summary['live_comparisons'] == 2688
    assert set(summary['quality']) == {'A1', 'B', 'A2'}
    assert all(q['repeat_exact_out_of16'] == [16] * 3
               for q in summary['quality'].values())
    outputs = summary['cross_arm_continuations']
    assert len(outputs) == 16 and {q['request'] for q in outputs} == set(range(16))
    assert all(q['common_prefix'] == 128 and q['distinct_outputs'] == 1 for q in outputs)
    teachers = summary['teacher_forced']
    assert len(teachers) == 3
    assert {(q['lhs'], q['rhs']) for q in teachers} == {
        ('A1', 'A2'), ('A1', 'B'), ('prior_accepted', 'B')}
    assert all(q['positions'] == 1008 and q['max_abs_logprob'] == 0
               and q['mean_abs_logprob'] == 0 and q['excluded_leading_nulls'] == 16
               and q['top1_same'] == q['top5_records_exact'] == 1008 for q in teachers)
    assert set(summary['timing_paths']) == {'A1', 'B', 'A2'}
    ranks = list(map(str, range(8)))
    for arm, paths in summary['timing_paths'].items():
        assert paths['legacy_splitk'] == []
        assert paths['premix_owner_ranks'] == ranks
        assert paths['k32_ranks'] == (ranks if arm == 'B' else [])
    legs = summary['legs']
    assert set(legs) == {'A1', 'B1', 'B2', 'A2'}
    spreads = {}
    for key, leg in legs.items():
        rates = leg['rates']
        assert len(rates) == 3 and all(math.isfinite(x) and x > 0 for x in rates)
        assert leg['median'] == statistics.median(rates)
        spreads[key] = 100 * (max(rates) - min(rates)) / leg['median']
    control = statistics.mean(legs[k]['median'] for k in ('A1', 'A2'))
    candidate = statistics.mean(legs[k]['median'] for k in ('B1', 'B2'))
    gain = 100 * (candidate / control - 1)
    drift = 100 * (legs['A2']['median'] / legs['A1']['median'] - 1)
    assert math.isclose(control, summary['control_input_tok_s'])
    assert math.isclose(candidate, summary['candidate_input_tok_s'])
    assert math.isclose(gain, summary['gain_pct'], abs_tol=1e-10)
    assert math.isclose(drift, summary['control_drift_pct'], abs_tol=1e-10)
    assert min(legs[k]['median'] for k in ('B1', 'B2')) > max(
        legs[k]['median'] for k in ('A1', 'A2'))
    assert gain > max(abs(drift), *spreads.values())
    return dict(gain_pct=gain, control_drift_pct=drift, within_leg_range_pct=spreads)


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[2]
    target = root / 'acceptance.json'
    assert not target.exists()
    summary = json.loads((root / 'summary.json').read_text())
    screen = validate_summary(summary)
    for arm in ('check-v2', 'A1', 'B', 'A2'):
        directory = root / arm
        plan = json.loads((directory / 'plan.json').read_text())
        assert all(hashlib.sha256((repo / p).read_bytes()).hexdigest() == h
                   for p, h in plan['sources'].items())
        assert not json.loads((directory / f'P32-owner-k32-{arm}.stop.json').read_text())['remaining']
    record = dict(status='accepted_explicit_32k_profile', numerical_exact_on_tested_inputs=True,
                  scope=summary['scope'], default_promoted=False,
                  service_8k_16k_regression_validated=False,
                  universal_batch_invariance_claimed=False, screening=screen,
                  summary_sha256=hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest())
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
