"""CPU dispatch contract; GPU arithmetic is covered by a separate oracle."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from sglang.kernels.ops.layernorm import gfx90a_mhc_prefill_policy as policy


class CommonPrefillTest(unittest.TestCase):
    def call(self,hint,rows=32768,active=True,flags=None):
        env={policy.ENV:'1','SGLANG_DSV4_GFX90A_BF16_MHC_DOT':'0',
             'SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA':'0',
             'SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS':'0',
             'SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20':'0',**(flags or {})}
        with patch.dict(os.environ,env),patch.object(policy,'mix_pair_active',return_value=active):
            return policy.common_prefill_batch_hint(hint,rows)

    def test_default_off_unscoped_and_small_preserve_hint(self):
        for hint in (None,1,2,16,32,64):
            self.assertEqual(self.call(hint,flags={policy.ENV:'0'}),hint)
            self.assertEqual(self.call(hint,active=False),hint)
            for rows in (0,1,128,8191,65537):self.assertEqual(self.call(hint,rows),hint)

    def test_common_chain_for_all_admitted_batch_hints(self):
        for rows in (8192,32767,32768,65536):
            for hint in (None,1,2,16,32,64):self.assertIsNone(self.call(hint,rows))

    def test_conflicting_numerical_overrides_fail_closed(self):
        for key in ('SGLANG_DSV4_GFX90A_BF16_MHC_DOT','SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA',
                    'SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS','SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20'):
            with self.assertRaises(ValueError):self.call(1,flags={key:'1'})
            self.assertEqual(self.call(1,active=False,flags={key:'1'}),1)

    def test_hint_applied_before_all_gfx90a_shortcuts(self):
        root=Path(__file__).resolve().parents[4]
        tree=ast.parse((root/'python/sglang/kernels/ops/layernorm/mhc.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mhc_fused_post_pre')
        branch=next(n for n in fn.body if isinstance(n,ast.If) and 'sinkhorn_repeat == 20' in ast.unparse(n.test))
        self.assertIn("'gfx90a'",ast.unparse(branch.test))
        first=branch.body[0]
        self.assertEqual(ast.unparse(first.test),'_prefill_common_batch_hint is not None')
        assignment=first.body[0]
        self.assertEqual(ast.unparse(assignment),'global_batch_size = _prefill_common_batch_hint(global_batch_size, num_tokens)')
        self.assertFalse(any(isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)
                             and n.attr=='batch_size' for n in ast.walk(fn)))


if __name__=='__main__':unittest.main()
