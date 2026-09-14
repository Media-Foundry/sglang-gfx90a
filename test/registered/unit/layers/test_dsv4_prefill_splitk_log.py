"""CPU checks for observation-only large-prefill split-K path reporting."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class TestPrefillSplitKLog(unittest.TestCase):
    def test_scope_and_once_per_path(self):
        root=Path(__file__).resolve().parents[4]
        tree=ast.parse((root/'python/sglang/kernels/ops/layernorm/mhc.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_log_prefill_splitk_dispatch')
        messages=[]
        ns=dict(_prefill_splitk_paths_logged=set(),_prefill_mix_reuse_active=None,
                logger=SimpleNamespace(info=lambda *args:messages.append(args)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'log','exec'),ns)
        log=ns[fn.name]
        log('fused_tail',32768,1,'float16')
        self.assertEqual(messages,[])
        ns['_prefill_mix_reuse_active']=lambda:False
        log('fused_tail',32768,1,'float16')
        self.assertEqual(messages,[])
        ns['_prefill_mix_reuse_active']=lambda:True
        log('fused_tail',1,1,'float16')
        self.assertEqual(messages,[])
        log('fused_tail',32767,1,'float16')
        log('fused_tail',32768,1,'float16')
        self.assertEqual(len(messages),1)
        self.assertEqual(messages[0][1:],('fused_tail',32767,1,'float16'))
        log('premix',32768,1,'float32')
        self.assertEqual(len(messages),2)


if __name__=='__main__':unittest.main()
