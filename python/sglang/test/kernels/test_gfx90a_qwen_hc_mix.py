import pytest
import torch

from sglang.kernels.ops.hyperconnection.gfx90a_hc_mix import gfx90a_qwen_hc_mix
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
