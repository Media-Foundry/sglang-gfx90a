import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "GPU is required")
class TestGfx90aGdnPackedDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        if not arch.startswith("gfx90a"):
            raise unittest.SkipTest("gfx90a-only HIP kernel")

    def test_rows_and_graph(self):
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
        from sglang.kernels.ops.attention.gfx90a_gdn_packed_decode import (
            gfx90a_gdn_packed_decode,
        )

        generator = torch.Generator(device="cuda").manual_seed(20260828)
        mixed = torch.randn(
            1, 2560, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        a = torch.randn(1, 12, generator=generator, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1, 12, generator=generator, device="cuda", dtype=torch.bfloat16)
        A_log = (
            torch.randn(12, generator=generator, device="cuda") * 0.3
        ).to(torch.bfloat16)
        dt_bias = torch.randn(12, generator=generator, device="cuda") * 0.1
        indices = torch.zeros(1, device="cuda", dtype=torch.int32)
        initial = torch.randn(
            2, 12, 128, 128, generator=generator, device="cuda", dtype=torch.float32
        )

        reference_state = initial.clone()
        reference_out = torch.empty(
            1, 1, 12, 128, device="cuda", dtype=torch.bfloat16
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            128**-0.5,
            reference_state,
            reference_out,
            indices,
            True,
        )

        for rows in (4, 8, 16, 32):
            state = initial.clone()
            out = gfx90a_gdn_packed_decode(
                mixed, a, b, A_log, dt_bias, state, indices, rows=rows
            )
            torch.testing.assert_close(out, reference_out, atol=6.2e-5, rtol=0)
            torch.testing.assert_close(state, reference_state, atol=8e-3, rtol=0)

        state = initial.clone()
        out = gfx90a_gdn_packed_decode(
            mixed, a, b, A_log, dt_bias, state, indices, rows=16, waves=2
        )
        torch.testing.assert_close(out, reference_out, atol=6.2e-5, rtol=0)
        torch.testing.assert_close(state, reference_state, atol=8e-3, rtol=0)

        graph_state = initial.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = gfx90a_gdn_packed_decode(
                mixed, a, b, A_log, dt_bias, graph_state, indices, rows=16
            )
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(graph_out).all().item())
        self.assertTrue(torch.isfinite(graph_state).all().item())


if __name__ == "__main__":
    unittest.main()
