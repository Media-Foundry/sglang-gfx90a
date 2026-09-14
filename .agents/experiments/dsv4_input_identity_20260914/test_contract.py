"""CPU tests for the opt-in scope and sampled debug input contract."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.debug.dsv4_prefill_attention_ar import eligible, enabled_for
from sglang.kernels.ops.debug.dsv4_sampled_stage_dump import sampled_stage_value


class Contract(unittest.TestCase):
    def test_disabled_does_not_touch_gpu_or_batch(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32": "0"}):
            self.assertFalse(enabled_for(None, None, None, None))

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
            with patch.dict(os.environ, {"SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS": ""}):
                self.assertTrue(torch.equal(sampled_stage_value("x", x, positions, directory, "p"), x))


if __name__ == "__main__":
    unittest.main()
