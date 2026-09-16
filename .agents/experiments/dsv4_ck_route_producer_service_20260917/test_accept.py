"""Synthetic negative controls for acceptance; never performance evidence."""
import copy
import unittest

from accept import validate_summary


def fixture():
    ranks = list(map(str, range(8)))
    return dict(status='complete', kv_tokens=1048576, original_weights=True,
        live_comparisons=1376,
        quality={a:dict(repeat_exact_out_of16=[16]*3) for a in ('A1','B','A2')},
        cross_arm_continuations=[dict(request=i, common_prefix=128, distinct_outputs=1)
                                 for i in range(16)],
        teacher_forced=[dict(lhs=a, rhs=b, positions=1008, max_abs_logprob=0,
            mean_abs_logprob=0, excluded_leading_nulls=16, top1_same=1008,
            top5_records_exact=1008)
            for a,b in (('A1','A2'),('A1','B'),('prior_accepted','B'))],
        timing_paths={a:dict(legacy_splitk=[], premix_owner_ranks=ranks,
            k32_ranks=ranks, route_ranks=ranks if a=='B' else []) for a in ('A1','B','A2')},
        legs={a:dict(rates=[v-.01,v,v+.01], median=v)
              for a,v in (('A1',100.),('B1',102.),('B2',102.),('A2',100.))},
        control_input_tok_s=100., candidate_input_tok_s=102., gain_pct=2., control_drift_pct=0.)


class Acceptance(unittest.TestCase):
    def test_valid_synthetic_fixture(self):
        self.assertAlmostEqual(validate_summary(fixture())['gain_pct'], 2.)

    def test_reject_missing_or_changed_evidence(self):
        changes = [
            lambda s:s.update(live_comparisons=0),
            lambda s:s.update(cross_arm_continuations=[]),
            lambda s:s.update(teacher_forced=[]),
            lambda s:s['quality']['B'].update(repeat_exact_out_of16=[16,15,16]),
            lambda s:s['cross_arm_continuations'][0].update(common_prefix=127),
            lambda s:s['teacher_forced'][0].update(max_abs_logprob=1e-7),
            lambda s:s['teacher_forced'][0].update(max_abs_logprob=float('nan')),
            lambda s:s['timing_paths']['B'].update(route_ranks=[]),
            lambda s:s['timing_paths']['A1'].update(route_ranks=['0']),
            lambda s:s['timing_paths']['B'].update(k32_ranks=[]),
            lambda s:s['legs']['B1'].update(rates=[0.,102.,104.]),
            lambda s:s['legs']['B1'].update(rates=[99.,102.,105.]),
            lambda s:s.update(candidate_input_tok_s=103.),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                record=copy.deepcopy(fixture());change(record)
                with self.assertRaises(AssertionError):validate_summary(record)


if __name__=='__main__':unittest.main()
