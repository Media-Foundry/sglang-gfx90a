"""No GPU: native-prefill policy must not change AR/draft or default dispatch."""
import os
import unittest
from unittest.mock import patch

from sglang.kernels.ops.layernorm import gfx90a_mhc_prefill_policy as policy


class UnfusedContractTest(unittest.TestCase):
    def call(self,rows,*,large=False,small=False,enabled=True,common=True):
        flags={policy.ENV:str(int(common)),policy.UNFUSED_ENV:str(int(enabled)),
               'SGLANG_DSV4_GFX90A_BF16_MHC_DOT':'0',
               'SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA':'0',
               'SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS':'0',
               'SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20':'0'}
        with patch.dict(os.environ,flags), \
             patch.object(policy,'mix_pair_active',return_value=large), \
             patch.object(policy,'small_prefill_active',return_value=small):
            return policy.common_unfused_fp32(rows)

    def test_default_and_unscoped_unchanged(self):
        for rows in (1,64,8192,32768,65536):
            self.assertFalse(self.call(rows,large=True,small=True,enabled=False))
            self.assertFalse(self.call(rows,large=True,small=True,common=False))
            self.assertFalse(self.call(rows))

    def test_scopes_do_not_leak_across_bounds(self):
        for rows in (1,64,8191):
            self.assertTrue(self.call(rows,small=True))
            self.assertFalse(self.call(rows,large=True))
        for rows in (8192,32576,65536):
            self.assertTrue(self.call(rows,large=True))
            self.assertFalse(self.call(rows,small=True))
        for rows in (0,65537):
            self.assertFalse(self.call(rows,large=True,small=True))


if __name__=='__main__':unittest.main()
