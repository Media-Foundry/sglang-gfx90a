"""CPU-only checks of the regression's single-variable design and entry gate."""
import ast
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('service.py').read_text()
TREE=ast.parse(SOURCE)


class Setup(unittest.TestCase):
    def flags(self, candidate):
        node=next(n for n in TREE.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='flags' for t in n.targets))
        value=eval(compile(ast.Expression(node.value),'<flags>','eval'),
                   {'candidate':candidate,'checking':False,'int':int})
        return dict(line.removeprefix('export ').split('=',1)
                    for line in value.splitlines() if line.startswith('export '))

    def test_one_variable(self):
        a,b=self.flags(False),self.flags(True)
        self.assertEqual(set(a),set(b))
        self.assertEqual([k for k in a if a[k]!=b[k]],['SGLANG_DSV4_C4_PREFILL_OWNER_K32'])
        self.assertEqual(a['SGLANG_DSV4_PREFILL_MHC_COMMON_FP32'],'1')
        self.assertEqual(b['SGLANG_DSV4_PREFILL_MHC_COMMON_FP32'],'1')
        self.assertEqual(a['SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK'],'0')
        self.assertEqual(a['SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA'],'0')

    def test_entry_gate_before_service(self):
        gate=next(n.lineno for n in TREE.body if isinstance(n,ast.Assert)
                  and "accepted['numerical_exact_on_tested_inputs']" in ast.unparse(n))
        start=next(n.lineno for n in ast.walk(TREE) if isinstance(n,ast.Call)
                   and ast.unparse(n.func)=='life.start')
        self.assertLess(gate,start)
        self.assertIn('gfx90a_indexer_owner_k32.py',SOURCE)
        self.assertIn('max_total_tokens\']==1048576',SOURCE)

    def test_lengths_and_real_output_checks_preserved(self):
        self.assertIn("131069 if args.length == '8k' else 262141",SOURCE)
        self.assertIn('len(ids)==128',SOURCE)
        self.assertIn("response['prompt_token_ids']==request['input_ids']",SOURCE)
        self.assertIn("assert k32_ranks==(list(map(str,range(8))) if candidate else [])",SOURCE)


if __name__=='__main__':
    unittest.main()
