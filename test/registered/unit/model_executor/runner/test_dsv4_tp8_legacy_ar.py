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
        self.ns = dict(envs=NS(SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR=NS(get=lambda:self.enabled)))
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


if __name__=='__main__':unittest.main()
