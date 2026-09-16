import os
import unittest
from unittest.mock import patch

from sglang.kernels.ops.debug.dsv4_premix_owner_audit import owned_rows, selected_calls, audit


class TestPremixOwnerAudit(unittest.TestCase):
    def test_partition(self):
        for m in (1, 7, 8, 17, 65, 8192, 32767, 32768, 65536):
            bounds = [owned_rows(m, rank) for rank in range(8)]
            self.assertEqual(bounds[0][0], 0)
            self.assertEqual(bounds[-1][1], m)
            self.assertEqual(sum(b-a for a,b in bounds), m)
            for a,b in zip(bounds, bounds[1:]):
                self.assertEqual(a[1], b[0])
            for start,end in bounds:
                self.assertTrue(start % 8 == 0 or start == m)

    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=True):
            audit(None, None, None, None, None)

    def test_calls(self):
        with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_CALLS': '0,40,84'}):
            self.assertEqual(selected_calls(), frozenset((0,40,84)))
        with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_CALLS': '-1'}):
            with self.assertRaises(ValueError):
                selected_calls()


if __name__ == '__main__':
    unittest.main()
