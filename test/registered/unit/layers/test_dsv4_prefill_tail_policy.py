"""CPU scope isolation for small-prefill MHC; GPU arithmetic tested separately."""
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from sglang.srt.layers import dsv4_prefill_tail_policy as tail
from sglang.srt.layers import dsv4_prefill_experiments as large
from sglang.kernels.ops.layernorm import gfx90a_mhc_prefill_policy as policy


def inputs(m=256):
    runner=NS(model_config=NS(hf_text_config=NS(model_type='deepseek_v4',num_hidden_layers=43,hidden_size=4096)),
        ps=NS(tp_size=8,attn_tp_size=8,moe_ep_size=1,attn_cp_size=1,attn_dcp_size=1,pp_size=1),is_draft_worker=False)
    batch=NS(input_ids=NS(shape=(m,),device=NS(type='cuda')),spec_algorithm=None,
        forward_mode=NS(name='EXTEND'),tbo_parent_token_range=None,_original_forward_mode=None)
    return runner,batch


class TailPolicy(unittest.TestCase):
    def test_bounds_and_large_scope_stays_off(self):
        for m in (1,64,256,2048,2560,8191):
            runner,batch=inputs(m)
            self.assertTrue(tail.eligible(runner,batch))
            # The legacy large predicate may ask this method before its M guard.
            batch.forward_mode.is_extend_without_speculative=lambda:True
            self.assertFalse(large.post_reuse_eligible(runner,batch))
        for m in (0,8192,32768,65536,65537):self.assertFalse(tail.eligible(*inputs(m)))

    def test_excluded_modes_and_models(self):
        for mode in ('DECODE','TARGET_VERIFY','DRAFT_EXTEND','MIXED','IDLE'):
            runner,batch=inputs();batch.forward_mode.name=mode
            self.assertFalse(tail.eligible(runner,batch))
        for field,value in (('model_type','deepseek_v41'),('num_hidden_layers',40),('hidden_size',2048)):
            runner,batch=inputs();setattr(runner.model_config.hf_text_config,field,value)
            self.assertFalse(tail.eligible(runner,batch))
        for field,value in (('tp_size',4),('attn_tp_size',4),('moe_ep_size',2),('attn_cp_size',2),('attn_dcp_size',2),('pp_size',2)):
            runner,batch=inputs();setattr(runner.ps,field,value)
            self.assertFalse(tail.eligible(runner,batch))
        runner,batch=inputs();runner.is_draft_worker=True;self.assertFalse(tail.eligible(runner,batch))
        runner,batch=inputs();batch.spec_algorithm=NS(is_none=lambda:False);self.assertFalse(tail.eligible(runner,batch))
        for field in ('tbo_parent_token_range','_original_forward_mode'):
            runner,batch=inputs();setattr(batch,field,object());self.assertFalse(tail.eligible(runner,batch))

    def test_disabled_identity_and_required_common_policy(self):
        def fn(*args):return None
        with patch.dict(os.environ,{tail.ENV:'0'}):self.assertIs(tail.instrument(fn),fn)
        with patch.dict(os.environ,{tail.ENV:'1',tail.COMMON_ENV:'0'}):
            with self.assertRaises(ValueError):tail.instrument(fn)

    def test_scope_reset_and_no_large_flags(self):
        runner,batch=inputs();owner=NS(model_runner=runner)
        def fn(self,forward_batch,fail=False):
            self.assertions=(tail.active(),large.post_reuse_active(),large.mix_reuse_active(),large.mix_pair_active())
            if fail:raise RuntimeError('expected')
            return self.assertions
        with patch.dict(os.environ,{tail.ENV:'1',tail.COMMON_ENV:'1'}), \
             patch.object(tail.torch.version,'hip','test'), \
             patch.object(tail.torch.cuda,'get_device_properties',return_value=NS(gcnArchName='gfx90a')), \
             patch.object(tail.torch.cuda,'is_current_stream_capturing',return_value=False):
            wrapped=tail.instrument(fn)
            self.assertEqual(wrapped(owner,forward_batch=batch),(True,False,False,False))
            self.assertFalse(tail.active())
            with self.assertRaises(RuntimeError):wrapped(owner,batch,fail=True)
            self.assertFalse(tail.active())
            with patch.object(tail.torch.cuda,'is_current_stream_capturing',return_value=True):
                self.assertEqual(wrapped(owner,batch),(False,False,False,False))

    def test_only_hint_changes_in_tail(self):
        env={policy.ENV:'1','SGLANG_DSV4_GFX90A_BF16_MHC_DOT':'0',
             'SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA':'0','SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS':'0',
             'SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20':'0'}
        with patch.dict(os.environ,env),patch.object(policy,'mix_pair_active',return_value=False):
            token=tail._active.set(True)
            try:
                for m in (1,64,256,8191):self.assertIsNone(policy.common_prefill_batch_hint(1,m))
                for m in (0,8192,65536):self.assertEqual(policy.common_prefill_batch_hint(1,m),1)
            finally:tail._active.reset(token)
            self.assertEqual(policy.common_prefill_batch_hint(1,256),1)


if __name__=='__main__':unittest.main()
