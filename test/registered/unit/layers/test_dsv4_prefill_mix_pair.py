"""CPU-only scope and source-contract checks for default-off pre-mix pairing."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('pair_scope',ROOT/'python/sglang/srt/layers/dsv4_prefill_experiments.py')
scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)

def fixture(m=32768,mode='EXTEND'):
    runner=NS(model_config=NS(hf_text_config=NS(model_type='deepseek_v4',num_hidden_layers=43,hidden_size=4096)),
        ps=NS(tp_size=8,attn_tp_size=8,moe_ep_size=1,attn_cp_size=1,pp_size=1,attn_dcp_size=1),is_draft_worker=False)
    batch=NS(input_ids=NS(shape=(m,),device=NS(type='cuda')),spec_algorithm=None,
        forward_mode=NS(name=mode,is_extend_without_speculative=lambda:mode in ('EXTEND','MIXED','DLLM_EXTEND')),
        tbo_parent_token_range=None,_original_forward_mode=None)
    return runner,batch

class PairScope(unittest.TestCase):
    def test_bounds_and_modes(self):
        for m in (8192,32767,32768,65536):self.assertTrue(scope.mix_pair_eligible(*fixture(m)))
        for m in (0,1,128,8191,65537):self.assertFalse(scope.mix_pair_eligible(*fixture(m)))
        for mode in ('DECODE','MIXED','DLLM_EXTEND','TARGET_VERIFY','DRAFT_EXTEND'):
            self.assertFalse(scope.mix_pair_eligible(*fixture(mode=mode)))

    def test_parallel_and_model_exclusions(self):
        for name in ('tp_size','attn_tp_size','moe_ep_size','attn_cp_size','pp_size','attn_dcp_size'):
            r,b=fixture();setattr(r.ps,name,4 if name.endswith('tp_size') else 2)
            self.assertFalse(scope.mix_pair_eligible(r,b),name)
        for name,value in [('model_type','deepseek_v41'),('num_hidden_layers',40),('hidden_size',2048)]:
            r,b=fixture();setattr(r.model_config.hf_text_config,name,value)
            self.assertFalse(scope.mix_pair_eligible(r,b))
        r,b=fixture();r.is_draft_worker=True;self.assertFalse(scope.mix_pair_eligible(r,b))
        r,b=fixture();b.spec_algorithm=NS(is_none=lambda:False);self.assertFalse(scope.mix_pair_eligible(r,b))
        for name in ('tbo_parent_token_range','_original_forward_mode'):
            r,b=fixture();setattr(b,name,1);self.assertFalse(scope.mix_pair_eligible(r,b))

    def call(self,flags,fn=None,capture=False,arch='gfx90a'):
        r,b=fixture()
        fn=fn or (lambda self,batch:(scope.mix_reuse_active(),scope.mix_pair_active()))
        with patch.dict(os.environ,flags),patch.object(scope.torch.version,'hip','test'),\
             patch.object(scope.torch.cuda,'is_current_stream_capturing',return_value=capture),\
             patch.object(scope.torch.cuda,'get_device_properties',return_value=NS(gcnArchName=arch)):
            return scope.instrument_prefill_mix_reuse(fn)(NS(model_runner=r),b)

    def test_default_off_and_parent_flag(self):
        self.assertEqual(self.call({scope.MIX_REUSE_ENV:'1',scope.MIX_PAIR_ENV:'0'}),(True,False))
        self.assertEqual(self.call({scope.MIX_REUSE_ENV:'0',scope.MIX_PAIR_ENV:'1'}),(False,False))
        self.assertEqual(self.call({scope.MIX_REUSE_ENV:'1',scope.MIX_PAIR_ENV:'1'}),(True,True))
        self.assertFalse(scope.mix_pair_active());self.assertFalse(scope.mix_reuse_active())

    def test_capture_arch_exception_and_nested_reset(self):
        flags={scope.MIX_REUSE_ENV:'1',scope.MIX_PAIR_ENV:'1'}
        self.assertEqual(self.call(flags,capture=True),(False,False))
        self.assertEqual(self.call(flags,arch='gfx942'),(False,False))
        def fail(self,batch):
            assert scope.mix_pair_active()
            raise ValueError('fixture')
        with self.assertRaises(ValueError):self.call(flags,fail)
        self.assertFalse(scope.mix_pair_active());self.assertFalse(scope.mix_reuse_active())
        token=scope._mix_pair.set(True)
        try:
            self.assertEqual(self.call({scope.MIX_REUSE_ENV:'1',scope.MIX_PAIR_ENV:'0'}),(True,False))
            self.assertTrue(scope.mix_pair_active())
        finally:scope._mix_pair.reset(token)

    def test_integrated_kernel_matches_oracle(self):
        def kernel(path):
            tree=ast.parse((ROOT/path).read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='premix8_pair'))
        self.assertEqual(kernel('python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_pair.py'),
                         kernel('.agents/experiments/dsv4_c16_premix_pair_20260915/candidate.py'))

    def test_caller_and_wrapper_guards(self):
        tree=ast.parse((ROOT/'python/sglang/kernels/ops/layernorm/mhc.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gfx90a_mhc_pre_mix_from_partials_triton')
        branch=next(n for n in fn.body if isinstance(n,ast.If) and '_prefill_mix_reuse_active()' in ast.unparse(n.test))
        self.assertIn('block_k == 1024',ast.unparse(branch.test))
        assignment=next(n for n in branch.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pair_columns' for t in n.targets))
        self.assertIn('group_size == 8',ast.unparse(assignment.value))
        self.assertIn('_prefill_mix_pair_active()',ast.unparse(assignment.value))
        call=next(n for n in ast.walk(branch) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='premix_reuse4')
        self.assertEqual(next(ast.unparse(k.value) for k in call.keywords if k.arg=='pair_columns'),'pair_columns')
        wrapper=ast.parse((ROOT/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_reuse.py').read_text())
        fn=next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='premix_reuse4')
        branch=next(n for n in fn.body if isinstance(n,ast.If) and 'group_size == 8' in ast.unparse(n.test))
        self.assertIn('8192 <= m <= 65536',ast.unparse(branch.test))
        self.assertIsInstance(branch.body[0],ast.If)
        self.assertEqual(ast.unparse(branch.body[0].test),'pair_columns')

if __name__=='__main__':unittest.main()
