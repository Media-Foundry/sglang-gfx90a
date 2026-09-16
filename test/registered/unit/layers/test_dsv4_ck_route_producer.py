"""CPU dispatch/source contracts; GPU correctness is covered by separate oracles."""
import ast
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[4]
HELPER=ROOT/'python/sglang/kernels/ops/moe/gfx90a_ck_route_producer.py'
CALLER=ROOT/'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py'
TREE=ast.parse(HELPER.read_text())
NODES=[n for n in TREE.body if (isinstance(n,ast.FunctionDef) and n.name=='eligible')
       or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STAGE1' for t in n.targets))]
SCOPE={}
exec(compile(ast.Module(body=NODES,type_ignores=[]),str(HELPER),'exec'),SCOPE)


class RouteProducer(unittest.TestCase):
    def eligible(self,m=32768,i=256,**changes):
        options=dict(native_scope=True,reference=False,shuffle=True,keep=False,raw=False,
            stage2_fp32=True,block_m=64,stage1_kernel=SCOPE['STAGE1'],stage2_kernel='',
            dsv4_activation=True,fixed_slot=True,unique_manifest='unique.json',
            route_manifest='route.json',capturing=False,has_probe=False)
        return SCOPE['eligible'](m,i,**{**options,**changes})

    def test_shape_range(self):
        for m in (16384,32765,32767,32768,36864):self.assertTrue(self.eligible(m))
        for m in (1,32,128,8192,16383,36865,65536):self.assertFalse(self.eligible(m))
        self.assertFalse(self.eligible(i=512))

    def test_contract_fallback(self):
        for key,value in dict(native_scope=False,reference=True,shuffle=False,keep=True,raw=True,
            stage2_fp32=False,block_m=32,stage1_kernel='',stage2_kernel='override',
            dsv4_activation=False,fixed_slot=False,unique_manifest=None,route_manifest=None,
            capturing=True,has_probe=True).items():
            with self.subTest(key=key):self.assertFalse(self.eligible(**{key:value}))

    def test_disabled_path_ast_unchanged(self):
        old=subprocess.check_output(['git','show',
            '9213e5fc77:python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py'],cwd=ROOT,text=True)
        current=ast.parse(CALLER.read_text())
        fn=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name=='gfx90a_bf16_ck_moe')
        added=[n for n in fn.body if isinstance(n,ast.If)
               and 'SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER' in ast.unparse(n.test)]
        self.assertEqual(len(added),1)
        self.assertIn("'0'",ast.unparse(added[0].test))
        fn.body.remove(added[0])
        self.assertEqual(ast.dump(current,include_attributes=False),ast.dump(ast.parse(old),include_attributes=False))

    def test_reference_does_not_inverse_scales_twice(self):
        tree=ast.parse(CALLER.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='gfx90a_bf16_ck_moe'
               and any(k.arg=='scales_shuffled' for k in n.keywords)]
        self.assertEqual(len(calls),1)
        value=next(k.value for k in calls[0].keywords if k.arg=='scales_shuffled')
        self.assertIs(ast.literal_eval(value),False)
        self.assertIn('with route.reference_scope():',CALLER.read_text())

    def test_tested_hip_bodies_preserved(self):
        prior=(ROOT/'.agents/experiments/dsv4_ck_route_major_20260916/route_major.cuh').read_text()
        metadata=(ROOT/'.agents/experiments/dsv4_ck_route_producer_20260916/metadata.cuh').read_text()
        expected=prior.rstrip()+'\n\n'+metadata.replace('#pragma once\n','').replace(
            '#include "../dsv4_ck_route_major_20260916/route_major.cuh"\n','')
        expected=expected.replace('#include "deepseek_v4/gfx90a_ck_fixed_slot.cuh"',
                                  '#include "gfx90a_ck_fixed_slot.cuh"')
        actual=(ROOT/'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_ck_route_producer.cuh').read_text()
        self.assertEqual(actual.rstrip(),expected.rstrip())


if __name__=='__main__':unittest.main()
