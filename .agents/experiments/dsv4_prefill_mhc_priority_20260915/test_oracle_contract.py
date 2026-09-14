"""CPU-only checks; these do not establish numerical or performance results."""
import ast
from pathlib import Path
import unittest


class TestPreparedOracle(unittest.TestCase):
    def test_replay_check_does_not_compare_workspace_aliases(self):
        tree=ast.parse(Path(__file__).with_name('oracle.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay_stability')
        class Tensor:
            def __init__(self,value):self.value=value
            def clone(self):return Tensor(self.value)
        def errors(a,b):
            return [dict(bits_exact=x.value==y.value,finite=True)
                    for x,y in zip(a,b,strict=True)]
        ns=dict(errors=errors)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'replay','exec'),ns)
        shared=Tensor(0)
        def drifting_call(arm):
            shared.value+=1
            return [shared]
        result=ns['replay_stability'](drifting_call,'A',3)
        self.assertEqual(result['exact_replays'],0)
        stable=ns['replay_stability'](lambda arm:[shared],'B',3)
        self.assertEqual(stable['exact_replays'],3)

    def test_full_boundary_argument_and_no_persistent_patch(self):
        tree=ast.parse(Path(__file__).with_name('oracle.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=='mhc_fused_post_pre')
        self.assertEqual(ast.literal_eval(call.args[11]),20)
        self.assertEqual(next(k.value.id for k in call.keywords if k.arg=='fn_fp16'),'fn16')
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='call')
        body_try=next(n for n in function.body if isinstance(n,ast.Try))
        self.assertIn('mhc._mhc_fusion_admitted = original_admitted',
                      [ast.unparse(n) for n in body_try.finalbody])

    def test_small_m_and_out_of_scope_keep_legacy(self):
        tree=ast.parse(Path(__file__).with_name('oracle.py').read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='admitted')
        for arm,scope,active,bs,expected in (
            ('A',True,True,1,True),('B',True,True,1,False),
            ('B',False,True,1,True),('B',True,False,1,True),
            ('B',False,False,32,False)):
            ns=dict(arm=arm,scope=scope,original_admitted=lambda bs:bs==1,
                    _mix_reuse=type('Scope',(),{'get':lambda self:active})())
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'admitted','exec'),ns)
            self.assertEqual(ns['admitted'](bs),expected)


if __name__=='__main__':unittest.main()
