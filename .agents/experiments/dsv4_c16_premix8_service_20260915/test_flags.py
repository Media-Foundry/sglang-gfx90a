import ast
from pathlib import Path
import subprocess
import unittest

class Flags(unittest.TestCase):
    def test_only_mix_group_changes(self):
        tree=ast.parse((Path(__file__).parent/'run.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='flags' for t in n.targets))
        observed=[]
        for candidate in (False,True):
            ns={'candidate':candidate}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<flags>','exec'),ns)
            r=subprocess.run(['bash','--noprofile','--norc','-c',ns['flags']+
                '\nprintf "%s %s %s %s" "$SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE" '
                '"$SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE" "$SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M" '
                '"$SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES"'],
                check=True,text=True,capture_output=True,env={})
            self.assertEqual(r.stdout,('8' if candidate else '4')+' 16 1 0')
            observed.append(ns['flags'])
        self.assertEqual(observed[0].replace('MIX_GROUP_SIZE=4','MIX_GROUP_SIZE=8'),observed[1])

if __name__=='__main__':unittest.main()
