"""Synthetic acceptance-guard tests, NOT GPU or service performance evidence."""
import copy
import unittest

from accept import validate_summary


def fixture():
    ranks = list(map(str, range(8)))
    return dict(status='complete', kv_tokens=1048576, original_weights=True,
                live_comparisons=2688, quality={a: dict(repeat_exact_out_of16=[16]*3)
                                              for a in ('A1', 'B', 'A2')},
                cross_arm_continuations=[dict(request=i, common_prefix=128, distinct_outputs=1)
                                         for i in range(16)],
                teacher_forced=[dict(lhs=a, rhs=b, positions=1008, max_abs_logprob=0,
                                     mean_abs_logprob=0, excluded_leading_nulls=16,
                                     top1_same=1008, top5_records_exact=1008)
                                for a, b in [('A1','A2'), ('A1','B'), ('prior_accepted','B')]],
                timing_paths={a: dict(legacy_splitk=[], premix_owner_ranks=ranks,
                                     k32_ranks=ranks if a=='B' else [])
                              for a in ('A1','B','A2')},
                legs={a: dict(rates=[v-.1, v, v+.1], median=v)
                      for a, v in [('A1',100),('A2',100),('B1',102),('B2',102)]},
                control_input_tok_s=100, candidate_input_tok_s=102,
                gain_pct=2, control_drift_pct=0)


class Acceptance(unittest.TestCase):
    def test_valid(self):
        self.assertAlmostEqual(validate_summary(fixture())['gain_pct'], 2)

    def test_empty_evidence_rejected(self):
        for field in ('quality', 'cross_arm_continuations', 'teacher_forced', 'timing_paths', 'legs'):
            data=fixture(); data[field]=type(data[field])()
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_summary(data)

    def test_drift_and_fallback_rejected(self):
        mutations=[lambda d: d['teacher_forced'][0].update(max_abs_logprob=1e-7),
                   lambda d: d['teacher_forced'][2].update(top5_records_exact=1007),
                   lambda d: d['quality']['B'].update(repeat_exact_out_of16=[16,16,15]),
                   lambda d: d['cross_arm_continuations'][0].update(common_prefix=127),
                   lambda d: d['timing_paths']['B'].update(k32_ranks=[]),
                   lambda d: d.update(kv_tokens=131072),
                   lambda d: d.update(live_comparisons=0)]
        for i, mutate in enumerate(mutations):
            data=copy.deepcopy(fixture()); mutate(data)
            with self.subTest(case=i), self.assertRaises(AssertionError):
                validate_summary(data)

    def test_noisy_speed_rejected(self):
        data=fixture(); data['legs']['B1']['rates']=[90,102,110]
        with self.assertRaises(AssertionError):
            validate_summary(data)


if __name__ == '__main__':
    unittest.main()
