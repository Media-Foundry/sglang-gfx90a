"""CPU-only K32 admission and exact tested-kernel provenance."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner_k32.py'
TREE=ast.parse(PATH.read_text())
ELIGIBLE=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='eligible')
scope={}
exec(compile(ast.Module(body=[ELIGIBLE],type_ignores=[]),str(PATH),'exec'),scope)

class OwnerK32(unittest.TestCase):
    def admitted(self,local,width,global_m,**overrides):
        return scope['eligible'](local,width,global_m,**dict(preshuffle_tile=0,dot_fp16=False,fp8_fnuz=False,**overrides))

    def test_supported_geometry(self):
        for local,width,m in [(3072,2048,32768),(3584,4096,32768),(3840,8192,32767),(4096,8191,32768),(1024,2048,8192)]:
            self.assertTrue(self.admitted(local,width,m))

    def test_fallback(self):
        for local,width,m in [(0,2048,32768),(16,513,8192),(768,2048,8192),(1023,2048,32768),
                              (1025,2048,32768),(4112,8192,32768),(3072,2047,32768),
                              (3072,8193,32768),(3072,2048,8191),(3072,2048,65537),(3072,2048,None)]:
            self.assertFalse(self.admitted(local,width,m))
        for shuffle,fp16,fnuz in [(8,False,False),(16,False,False),(0,True,False),(0,False,True)]:
            self.assertFalse(scope['eligible'](3072,2048,32768,preshuffle_tile=shuffle,dot_fp16=fp16,fp8_fnuz=fnuz))

    def test_tested_kernel_preserved(self):
        previous=ast.parse((ROOT/'.agents/experiments/dsv4_indexer_grid_order_20260916/kernel.py').read_text())
        def kernel(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reuse_grouped_grid')
        self.assertEqual(ast.dump(kernel(previous)),ast.dump(kernel(TREE)))
        self.assertIn("do_not_specialize=['M']",ast.unparse(kernel(TREE).decorator_list[0]))

    def test_fixed_launch_contract(self):
        fn=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='prefill_owner_k32')
        launch=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript)
                    and isinstance(n.func.value,ast.Name) and n.func.value.id=='reuse_grouped_grid')
        self.assertEqual([ast.unparse(x) for x in launch.args[-6:]],['16','32','0','tl.bfloat16','tl.float8e4nv','1'])
        self.assertEqual({k.arg:ast.literal_eval(k.value) for k in launch.keywords},{'num_warps':4,'matrix_instr_nonkdim':16})

if __name__=='__main__':unittest.main()
