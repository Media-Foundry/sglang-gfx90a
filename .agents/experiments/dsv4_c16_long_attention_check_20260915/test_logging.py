"""Observation-only change: stripping its debug print restores prior AST."""
import ast
from pathlib import Path
import subprocess


def test_only_new_debug_log():
    repo=Path(__file__).resolve().parents[3]
    path='python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py'
    old=ast.parse(subprocess.check_output(['git','show','3a32a0e237:'+path],cwd=repo).decode())
    new=ast.parse((repo/path).read_text());removed=[]
    class Strip(ast.NodeTransformer):
        def visit_Expr(self,node):
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                and node.value.func.id=='print' and 'prefill attention-stage1 checked:' in ast.unparse(node)):
                removed.append(node);return None
            return self.generic_visit(node)
    new=Strip().visit(new)
    assert len(removed)==1
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
