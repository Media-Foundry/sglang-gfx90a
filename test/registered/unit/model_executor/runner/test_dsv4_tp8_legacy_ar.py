"""CPU-only scope and forwarding contracts for the default-off AR candidate."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class TestLegacyAR(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[5] / 'python/sglang/srt/distributed/device_communicators/dsv4_ar_experiment.py'
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom)
                     and n.module == 'sglang.srt.environ')]
        self.enabled = True
        self.gate_enabled = False
        self.ns = dict(envs=NS(SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR=NS(get=lambda:self.enabled),
                              SGLANG_DSV4_GFX90A_TP8_M32_GATE_PREFETCH=NS(get=lambda:self.gate_enabled)))
        exec(compile(tree,str(path),'exec'),self.ns)

    def test_predicate(self):
        for enabled,hip,arch,decode,bs,native in itertools.product(
                (False,True),(False,True),('gfx90a','gfx942'),(False,True),
                (1,16,32,64),(False,True)):
            self.assertEqual(self.ns['eligible'](enabled=enabled,hip=hip,arch=arch,
                             decode=decode,batch_size=bs,native=native),
                             enabled and hip and arch=='gfx90a' and decode and bs==32 and native)

    def test_adapter(self):
        import torch
        class Base:
            def all_reduce(self,inp,**kwargs):return kwargs
        self.enabled=False
        self.assertIs(self.ns['adapt_dsv4_ar'](Base),Base)
        self.enabled=True
        cls=self.ns['adapt_dsv4_ar'](Base)
        for active,world,m,dtype,quant,contig in itertools.product(
                (False,True),(4,8),(1,32,64),(torch.bfloat16,torch.float32),
                (False,True),(False,True)):
            obj=cls();obj.world_size=world
            inp=NS(shape=(m,4096),dtype=dtype,is_contiguous=lambda:contig)
            t=self.ns['_active'].set(active)
            try:r=obj.all_reduce(inp,out='sentinel',open_fp8_quant=quant,registered=True)
            finally:self.ns['_active'].reset(t)
            hit=active and world==8 and m==32 and dtype==torch.bfloat16 and not quant and contig
            self.assertEqual(r['use_new'],not hit)
            self.assertEqual(r['out'],'sentinel')
            self.assertTrue(r['registered'])

    def test_scope_restoration_and_spec_exclusion(self):
        import torch
        from unittest.mock import patch
        with patch.object(torch.version,'hip','test'), patch.object(
                torch.cuda,'get_device_properties',return_value=NS(gcnArchName='gfx90a')):
            for native,decode in itertools.product((False,True),(False,True)):
                batch=NS(spec_algorithm=NS(is_none=lambda:native),batch_size=32,
                         forward_mode=NS(is_decode=lambda:decode))
                with self.assertRaisesRegex(RuntimeError,'probe'):
                    with self.ns['dsv4_ar_scope'](batch,'cuda'):
                        self.assertEqual(self.ns['_active'].get(),native and decode)
                        raise RuntimeError('probe')
                self.assertFalse(self.ns['_active'].get())
                self.assertFalse(self.ns['native_m32_active']())

    def test_gate_only_does_not_enable_legacy_ar(self):
        import torch
        from unittest.mock import patch
        self.enabled=False;self.gate_enabled=True
        batch=NS(spec_algorithm=None,batch_size=32,forward_mode=NS(is_decode=lambda:True))
        with patch.object(torch.version,'hip','test'), patch.object(
                torch.cuda,'get_device_properties',return_value=NS(gcnArchName='gfx90a')):
            with self.ns['dsv4_ar_scope'](batch,'cuda'):
                self.assertTrue(self.ns['native_m32_active']())
                self.assertFalse(self.ns['_active'].get())
        self.assertFalse(self.ns['native_m32_active']())

    def test_gate_selector(self):
        path=Path(__file__).resolve().parents[5]/'python/sglang/srt/layers/moe/moe_runner/aiter.py'
        tree=ast.parse(path.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='use_tp8_gate_prefetch' for t in n.targets))
        code=compile(ast.Expression(node.value),str(path),'eval')
        for active,tp,gfx,m,i,assignments,rows,blocks,lds,mfma in itertools.product(
                (False,True),(4,8),(False,True),(1,32),(256,512),(2,4),(1,2),(832,2080),(False,True),(False,True)):
            ns=dict(native_m32_active=lambda:active,get_tensor_model_parallel_world_size=lambda:tp,
                    _is_runtime_gfx90a=lambda:gfx,runner_input=NS(hidden_states=NS(shape=(m,4096)),topk_ids=NS(shape=(m,6))),
                    quant_info=NS(w13_weight=NS(shape=(256,2*i,2048)),w2_weight=NS(shape=(256,4096,i//2))),
                    grouped_assignments=assignments,grouped_gate_rows=rows,gate_blocks=blocks,
                    use_lds_unpack=lds,use_mfma32_prefill=mfma)
            expected=active and tp==8 and gfx and m==32 and i==256 and assignments==4 and rows==2 and blocks==832 and lds and not mfma
            self.assertEqual(eval(code,ns),expected)


if __name__=='__main__':unittest.main()
