"""CPU-only guards: shared runtime key and unchanged fixed-shape default."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.moe import gfx90a_fp4_expert_gemv as mod


class TestRuntimeM(unittest.TestCase):
    def test_gate_module_key(self):
        for m in (129, 369, 511, 1023):
            for dynamic in (False, True):
                with patch.object(mod, "_jit_gate_up_grouped") as jit:
                    mod.gfx90a_fp4_expert_gate_up_grouped(
                        torch.zeros(m, 32, dtype=torch.int8),
                        torch.ones(m, 1),
                        torch.zeros(2, 32, 16, dtype=torch.uint8),
                        torch.zeros(2, 32, 1, dtype=torch.uint8),
                        torch.zeros(4, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.zeros(2, dtype=torch.int32), 6, 10., runtime_m=dynamic,
                    )
                    self.assertEqual(jit.call_args.args[1], 0 if dynamic else m)

    def test_down_module_key(self):
        for m in (129, 369):
            for dynamic in (False, True):
                with patch.object(mod, "_jit_down_grouped") as jit:
                    mod.gfx90a_fp4_expert_down_grouped(
                        torch.zeros(m, 6, 32, dtype=torch.int8),
                        torch.ones(m, 6, 1),
                        torch.zeros(2, 16, 16, dtype=torch.uint8),
                        torch.zeros(2, 16, 1, dtype=torch.uint8),
                        torch.zeros(4, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.zeros(2, dtype=torch.int32), torch.ones(m, 6),
                        runtime_m=dynamic,
                    )
                    self.assertEqual(jit.call_args.args[1], 0 if dynamic else m)

    def test_runner_guard_excludes_decode_tiers_and_tp4(self):
        path = Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/moe/moe_runner/aiter.py"
        tree = ast.parse(path.read_text())
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and "SGLANG_DSV4_GFX90A_FP4_RUNTIME_M" in ast.unparse(n.test))
        expression = compile(ast.Expression(guard.test), str(path), "eval")
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 129, 369, 1023, 1024, 8192):
            for enabled in (False, True):
                for tp8 in (False, True):
                    actual = eval(expression, {
                        "envs": SimpleNamespace(SGLANG_DSV4_GFX90A_FP4_RUNTIME_M=SimpleNamespace(get=lambda: enabled)),
                        "num_prefill_tokens": m, "use_mfma32_prefill": False,
                        "quant_info": SimpleNamespace(
                            w13_weight=SimpleNamespace(shape=(256, 512 if tp8 else 1024, 2048)),
                            w2_weight=SimpleNamespace(shape=(256, 4096, 128 if tp8 else 256)),
                        ),
                    })
                    self.assertEqual(actual, enabled and tp8 and 128 < m < 1024)
        self.assertIn("speculative_algorithm is None", ast.unparse(guard))


if __name__ == "__main__":
    unittest.main()
