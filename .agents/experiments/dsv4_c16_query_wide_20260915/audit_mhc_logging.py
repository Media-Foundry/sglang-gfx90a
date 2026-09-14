"""Compare the observation-only MHC change to its frozen pre-change source."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
repo=root.parents[2]
name='python/sglang/kernels/ops/layernorm/mhc.py'
baseline='39bc0e1383'
before=subprocess.check_output(['git','show',f'{baseline}:{name}'],cwd=repo).decode()
after=(repo/name).read_text()
tree=ast.parse(after)
calls=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_log_prefill_splitk_dispatch' for n in ast.walk(tree))
assert calls==2

class Strip(ast.NodeTransformer):
    def visit_FunctionDef(self,n):
        return None if n.name=='_log_prefill_splitk_dispatch' else self.generic_visit(n)
    def visit_Assign(self,n):
        return None if any(isinstance(t,ast.Name) and t.id=='_prefill_splitk_paths_logged' for t in n.targets) else self.generic_visit(n)
    def visit_Expr(self,n):
        return None if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_log_prefill_splitk_dispatch' else self.generic_visit(n)

assert ast.dump(ast.parse(before))==ast.dump(Strip().visit(tree))
result=dict(baseline=baseline,source=name,logging_calls=calls,non_logging_ast_exact=True,
            before_sha256=hashlib.sha256(before.encode()).hexdigest(),
            after_sha256=hashlib.sha256(after.encode()).hexdigest(),
            scope='No changes to MHC arithmetic, tensor operations, kernel launch parameters, return values or dispatch choices. Added one-time active-prefill logs only; not a GPU numerical experiment.')
(root/'mhc-logging-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
