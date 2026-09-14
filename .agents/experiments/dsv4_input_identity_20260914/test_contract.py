"""CPU tests for the opt-in scope and sampled debug input contract."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from sglang.kernels.ops.debug.dsv4_prefill_attention_ar import eligible, enabled_for
from sglang.kernels.ops.debug.dsv4_sampled_stage_dump import (
    sampled_stage_value, should_dump_stage, stage_dump_scope,
)


class Contract(unittest.TestCase):
    def test_disabled_does_not_touch_gpu_or_batch(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32": "0"}):
            self.assertFalse(enabled_for(None, None, None, None))
        with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32": "1",
                                    "SGLANG_DSV4_DEBUG_PREFILL_WOA_STABLE": "0"}):
            self.assertFalse(enabled_for(None, None, None, None,
                                        flag="SGLANG_DSV4_DEBUG_PREFILL_WOA_STABLE"))

    def test_callback_ownership(self):
        module=SimpleNamespace()
        with stage_dump_scope(module, "outer"):
            self.assertEqual(module._dsv4_stage_dump, "outer")
            with self.assertRaises(ValueError):
                with stage_dump_scope(module, "inner"):
                    raise ValueError("fixture")
            self.assertEqual(module._dsv4_stage_dump, "outer")
        self.assertFalse(hasattr(module, "_dsv4_stage_dump"))

    def test_scope(self):
        base = dict(enabled=True, native=True, extend=True, hip=True,
                    arch="gfx90a", tp=8, attn_tp=8, ep=1, rows=32768)
        self.assertTrue(eligible(**base))
        for key, value in [("enabled", False), ("native", False), ("extend", False),
                           ("hip", False), ("arch", "gfx942"), ("tp", 4),
                           ("attn_tp", 4), ("ep", 2), ("rows", 8191),
                           ("rows", 36865), ("rows", 32), ("rows", 128)]:
            with self.subTest(key=key, value=value):
                self.assertFalse(eligible(**{**base, key: value}))
        self.assertTrue(eligible(**{**base, "rows": 8192}))
        self.assertTrue(eligible(**{**base, "rows": 36864}))

    def test_samples_keep_full_inputs(self):
        positions = torch.tensor([0, 1, 2, 0, 1, 2])
        x = torch.arange(24).view(6, 4)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS": "0,2"}):
                actual = sampled_stage_value("x", x, positions, directory, "p")
                self.assertTrue(torch.equal(actual, x[[0, 2, 3, 5]]))
                self.assertTrue(torch.equal(torch.load(Path(directory) / "p_sample_rows.pt",
                                                      weights_only=True), torch.tensor([0, 2, 3, 5])))
                for name in ("input_ids", "positions"):
                    self.assertTrue(torch.equal(sampled_stage_value(
                        name, positions, positions, directory, "p"), positions))
                # Parameters must not be row-sampled merely because N == M.
                for name in ("projection_wqkv_a", "hc_attn_fn", "router_weight"):
                    self.assertTrue(torch.equal(sampled_stage_value(
                        name, x, positions, directory, "p"), x))
                    with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_STAGE_SKIP_WEIGHTS": "1"}):
                        self.assertFalse(should_dump_stage(name))
                        self.assertTrue(should_dump_stage("ffn_topk_weights"))
            with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS": ""}):
                self.assertTrue(torch.equal(sampled_stage_value("x", x, positions, directory, "p"), x))


if __name__ == "__main__":
    unittest.main()
