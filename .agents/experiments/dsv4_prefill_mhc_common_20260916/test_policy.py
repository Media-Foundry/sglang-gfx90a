"""Prototype policy only: does not certify GPU outputs or service speed."""
import unittest
from policy import resolve_batch_hint


class PolicyTest(unittest.TestCase):
    def test_off_and_unscoped_preserve_actual_hint(self):
        for hint in (None, 1, 2, 16, 32, 64):
            for enabled, active in ((False, False), (False, True), (True, False)):
                self.assertEqual(resolve_batch_hint(hint,32768,enabled=enabled,
                    active_scope=active,bf16_dot=True),hint)

    def test_only_admitted_large_prefill(self):
        for rows in (0, 1, 128, 8191, 65537):
            self.assertEqual(resolve_batch_hint(1,rows,enabled=True,active_scope=True),1)
        for rows in (8192, 16384, 32767, 32768, 65536):
            for hint in (None, 1, 2, 16, 32):
                self.assertIsNone(resolve_batch_hint(hint,rows,enabled=True,active_scope=True))

    def test_reject_conflicting_numeric_overrides(self):
        for flag in ('bf16_dot', 'mfma', 'config_iters', 'comb_refine'):
            with self.assertRaises(ValueError):
                resolve_batch_hint(1,32768,enabled=True,active_scope=True,**{flag:True})

    def test_does_not_mutate_scheduler_metadata(self):
        metadata={'batch_size':1, 'tokens':32768}
        self.assertIsNone(resolve_batch_hint(metadata['batch_size'],metadata['tokens'],
            enabled=True,active_scope=True))
        self.assertEqual(metadata,{'batch_size':1,'tokens':32768})


if __name__=='__main__':unittest.main()
