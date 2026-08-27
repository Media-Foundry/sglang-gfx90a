import pytest
import torch
from types import SimpleNamespace

from sglang.kernels.ops.moe.gfx90a_mq4g128_moe import (
    mq4g128_grouped,
    mq4g128_indexed,
)
from sglang.srt.layers.quantization.mq4g128 import (
    _requantize_checkpoint_fp8_mq4g128,
    dequantize_mq4g128,
    fwht128,
    quantize_mq4g128,
)
from sglang.srt.utils import is_hip


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_mq4g128_indexed_and_grouped_match_oracle():
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(7)
    e, m, topk, n, k = 3, 5, 2, 64, 256
    weight = torch.randn(e, n, k, device="cuda", dtype=torch.float32) * 0.1
    packed = quantize_mq4g128(weight)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    x_rot = fwht128(x).contiguous()
    expert_ids = torch.tensor(
        [[0, 1], [0, 2], [1, 0], [2, 1], [0, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    dequant = dequantize_mq4g128(packed).reshape(e, n, k)
    expected = torch.empty(m, topk, n, dtype=torch.float32, device="cuda")
    for token in range(m):
        for slot in range(topk):
            expected[token, slot] = dequant[expert_ids[token, slot].item()] @ x_rot[token]
    indexed = mq4g128_indexed(x_rot, packed, expert_ids)
    grouped = mq4g128_grouped(x_rot, packed, expert_ids)
    torch.testing.assert_close(indexed, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(grouped, expected, rtol=2e-5, atol=2e-5)
    # A4 interleaves four accumulators and may schedule FMAs differently from
    # the single-assignment kernel. Both independently match the dequantized
    # oracle; keep the cross-path bound explicit instead of claiming bitwise
    # identity.
    torch.testing.assert_close(grouped, indexed, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not is_hip(), reason="HIP-only FP8 conversion")
def test_streamed_fp8_requant_matches_whole_tensor():
    torch.manual_seed(9)
    weight = (torch.randn(5, 32, 256, device="cuda") * 0.1).to(torch.float8_e4m3fnuz)
    scale = torch.rand(5, 1, 2, device="cuda") * 0.2 + 0.01
    reference = quantize_mq4g128(
        weight.float() * scale.repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :32]
    )
    streamed = _requantize_checkpoint_fp8_mq4g128(weight, scale, expert_chunk=2)
    torch.testing.assert_close(streamed, reference, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen4_routed_method_matches_dequantized_oracle(monkeypatch):
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mq4g128 import Mq4g128RoutedMoEMethod

    monkeypatch.setenv("SGLANG_QWEN4_GFX90A_MQ4G128_GROUPED_OCCUPANCY", "2")
    torch.manual_seed(11)
    e, m, topk, h, i = 2, 1, 10, 2560, 640
    w13 = torch.randn(e, 2 * i, h, device="cuda") * 0.03
    w2 = torch.randn(e, h, i, device="cuda") * 0.03
    p13, p2 = quantize_mq4g128(w13), quantize_mq4g128(w2)
    x = torch.randn(m, h, dtype=torch.bfloat16, device="cuda")
    ids = (torch.arange(topk, device="cuda", dtype=torch.int32) % e).reshape(m, topk)
    weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(weights, ids, torch.empty(0, device="cuda")),
    )
    method = object.__new__(Mq4g128RoutedMoEMethod)
    layer = SimpleNamespace(w13_weight=p13, w2_weight=p2)
    got = method.apply(layer, dispatch).hidden_states.float()

    d13 = dequantize_mq4g128(p13).reshape(e, 2 * i, h)
    d2 = dequantize_mq4g128(p2).reshape(e, h, i)
    xr = fwht128(x.float())
    expected = torch.zeros(m, h, device="cuda")
    for slot in range(topk):
        expert = int(ids[0, slot])
        gu = d13[expert] @ xr[0]
        act = torch.nn.functional.silu(gu[:i]) * gu[i:]
        act_rot = fwht128(act)
        expected[0] += weights[0, slot] * (d2[expert] @ act_rot)
    torch.testing.assert_close(got, expected, rtol=3e-3, atol=3e-2)
