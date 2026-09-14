"""CPU-only guard checks: raising the TP8 trial ceiling must not raise TP4."""
import os
import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import bf16_ck_prefill_max_rows


class Capacity(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_TP8_CK_64K':'0'}):
            self.assertEqual(bf16_ck_prefill_max_rows(256),36864)
            self.assertEqual(bf16_ck_prefill_max_rows(512),36864)

    def test_tp8_only(self):
        with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_TP8_CK_64K':'1'}):
            self.assertEqual(bf16_ck_prefill_max_rows(256),65536)
            self.assertEqual(bf16_ck_prefill_max_rows(512),36864)
            self.assertEqual(bf16_ck_prefill_max_rows(128),36864)

    def test_no_gpu_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
