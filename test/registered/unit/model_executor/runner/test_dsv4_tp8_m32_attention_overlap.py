"""CPU-only evaluation of the actual experimental attention selector."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class TestTP8M32AttentionOverlap(unittest.TestCase):
    def test_narrow_selector(self):
        path = Path(__file__).resolve().parents[5] / 'python/sglang/srt/models/deepseek_v4.py'
        tree = ast.parse(path.read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'enable_tp8_m32_hip_streams'
                                  for t in n.targets))
        predicate = compile(ast.Expression(assignment.value), str(path), 'eval')
        for hip, flag, tp, m, bs, ratio, mode, unified in itertools.product(
            (False, True), (False, True), (4,8), (1,4,16,32,64,128),
            (1,32), (4,128), ('decode','extend','verify','draft','idle'), (False, True)
        ):
            env = NS(SGLANG_DSV4_GFX90A_TP8_M32_ATTN_MULTISTREAM=NS(get=lambda: flag))
            batch = NS(batch_size=bs, forward_mode=NS(is_decode=lambda: mode == 'decode'))
            actual = eval(predicate, dict(_is_hip=hip, envs=env,
                          self=NS(attn_tp_size=tp, compress_ratio=ratio),
                          x=NS(shape=(m,4096)),forward_batch=batch,unified_kv=unified))
            expected = hip and flag and tp==8 and m==32 and bs==32 and ratio==4 and mode=='decode' and unified
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
