import os
import unittest
from unittest.mock import patch
import numpy as np
from sglang.kernels.ops.debug import dsv4_first_divergence as audit


class FingerprintTest(unittest.TestCase):
    def test_explicit_diagnostic_coverage(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(audit.selected_layers(), frozenset(range(20,25)))
            self.assertIsNone(audit.selected_stages())
        with patch.dict(os.environ, {
            'SGLANG_DSV4_DEBUG_FIRST_DIV_LAYERS': '0, 1,42',
            'SGLANG_DSV4_DEBUG_FIRST_DIV_STAGES': 'attn_residual,ffn_out',
        }):
            self.assertEqual(audit.selected_layers(), {0,1,42})
            self.assertEqual(audit.selected_stages(), {'attn_residual','ffn_out'})
        for invalid in ('', '-1', '43', '0,,1'):
            with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_FIRST_DIV_LAYERS': invalid}):
                with self.assertRaises(ValueError): audit.selected_layers()
        for invalid in ('', 'ffn_out,', '../x'):
            with patch.dict(os.environ, {'SGLANG_DSV4_DEBUG_FIRST_DIV_STAGES': invalid}):
                with self.assertRaises(ValueError): audit.selected_stages()

    def test_unsampled_row_and_byte_order(self):
        a=np.arange(64*128,dtype=np.uint16).reshape(64,128)
        digest,rows=audit.fingerprint(a,64)
        b=a.copy();b[37,71]^=1
        other,changed=audit.fingerprint(b,64)
        self.assertNotEqual(digest,other)
        self.assertEqual(np.flatnonzero((rows!=changed).any(axis=1)).tolist(),[37])
        c=a.copy();c[0]=c[0,::-1]
        self.assertNotEqual(audit.fingerprint(c,64)[0],digest)
        self.assertEqual(audit.fingerprint(a.copy(),64)[0],digest)

    def test_none_scope_does_not_touch_gpu(self):
        with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_FIRST_DIV_DIR':''}), \
             patch.object(audit.torch.cuda,'is_current_stream_capturing') as gpu:
            self.assertIsNone(audit.get_probe(20,None,None));gpu.assert_not_called()
        with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_FIRST_DIV_DIR':'unused'}), \
             patch('sglang.srt.layers.dsv4_prefill_experiments.mix_pair_active',return_value=False), \
             patch.object(audit.torch.cuda,'is_current_stream_capturing') as gpu:
            self.assertIsNone(audit.get_probe(20,None,None));gpu.assert_not_called()


if __name__=='__main__':unittest.main()
