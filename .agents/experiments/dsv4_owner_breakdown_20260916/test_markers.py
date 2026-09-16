"""CPU-only contract: stripping diagnostic marks leaves all owner math unchanged."""
import ast
import difflib
from pathlib import Path
import subprocess
import unittest

repo=Path(__file__).resolve().parents[3]
path='python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py'
old=ast.parse(subprocess.check_output(['git','show','ef3d40aa32:'+path],cwd=repo,text=True))
new=ast.parse((repo/path).read_text())

class StripMarks(ast.NodeTransformer):
    def visit_If(self,node):
        if ast.unparse(node.test)=='_detail_mark is not None': return None
        return self.generic_visit(node)

class Contract(unittest.TestCase):
    def test_math_identical(self):
        reference={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        current={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(set(reference),set(current))
        for name in reference:
            self.assertEqual(ast.dump(reference[name]),ast.dump(StripMarks().visit(current[name])),name)

    def test_only_reserved_slots_and_no_sync(self):
        source=ast.parse((repo/path).read_text())
        marks=[n for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_detail_mark']
        self.assertEqual([n.args[0].value for n in marks],list(range(55,62)))
        self.assertTrue(all(len(n.args)==2 and n.keywords[0].arg=='absolute' and n.keywords[0].value.value is True for n in marks))
        before=subprocess.check_output(['git','show','ef3d40aa32:'+path],cwd=repo,text=True)
        added='\n'.join(difflib.unified_diff(before.splitlines(),(repo/path).read_text().splitlines()))
        lines='\n'.join(l for l in added.splitlines() if l.startswith('+') and not l.startswith('+++'))
        self.assertNotIn('.synchronize(',lines)
        self.assertNotIn('.cpu(',lines)

if __name__=='__main__': unittest.main()
