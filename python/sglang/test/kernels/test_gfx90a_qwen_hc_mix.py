import pytest
import torch

from sglang.kernels.ops.elementwise.hc_combine import (
    _jit_hc_combine_module,
)
from sglang.kernels.ops.hyperconnection.gfx90a_hc_mix import (
    _module,
    gfx90a_qwen_hc_mix,
)
from sglang.srt.layers.hc_mix_triton import fused_hc_mix
from sglang.srt.utils import is_hip


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen_hc_mix_matches_persistent(monkeypatch):
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(41)
    x = torch.randn(1, 10240, device="cuda", dtype=torch.bfloat16)
    w_down = (torch.randn(320, 10240, device="cuda") * 0.01).bfloat16()
    w_up = (torch.randn(10240, 320, device="cuda") * 0.01).bfloat16()
    monkeypatch.setenv("SGLANG_QWEN4_GFX90A_HC_MIX_HIP", "0")
    expected = fused_hc_mix(x, w_down, w_up, 4, 2560)
    actual = gfx90a_qwen_hc_mix(x, w_down, w_up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen_hc_gate_fusion_is_bitwise_exact():
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(42)
    dtype = torch.bfloat16
    x = torch.randn(1, 10240, device="cuda", dtype=dtype)
    w_down = torch.randn(320, 10240, device="cuda", dtype=dtype)
    w_up = torch.randn(10240, 320, device="cuda", dtype=dtype)
    inject = torch.randn(4, 10240, device="cuda", dtype=dtype)
    block = torch.randn(1, 2560, device="cuda", dtype=dtype)
    residual = torch.randn(1, 10240, device="cuda", dtype=dtype)
    workspace = torch.empty(1, 320, device="cuda", dtype=torch.float32)
    mixed_ref = torch.empty(1, 2560, device="cuda", dtype=dtype)
    mixed_fused = torch.empty_like(mixed_ref)
    partials_ref = torch.empty(1, 8, 4, device="cuda", dtype=torch.float32)
    partials_fused = torch.empty_like(partials_ref)
    out_ref = torch.empty_like(residual)
    out_fused = torch.empty_like(residual)
    mix = _module(1, 1)
    combine = _jit_hc_combine_module(4, 2560, dtype)
    mix.run(x, w_down, w_up, workspace, mixed_ref)
    combine.hc_combine_split(
        block, residual, x, inject, out_ref, partials_ref
    )
    mix.run_with_gate(
        x, w_down, w_up, inject, workspace, partials_fused, mixed_fused
    )
    combine.hc_combine_apply_precomputed(
        block, residual, out_fused, partials_fused
    )
    assert torch.equal(mixed_ref, mixed_fused)
    assert torch.equal(partials_ref, partials_fused)
    assert torch.equal(out_ref, out_fused)
