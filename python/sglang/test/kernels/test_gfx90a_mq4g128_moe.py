import pytest
import torch
from types import SimpleNamespace

from sglang.kernels.ops.moe.gfx90a_mq4g128_moe import (
    mq4g128_grouped,
    mq4g128_indexed,
    mq4g128_weighted_reduce,
)
from sglang.kernels.ops.moe.gfx90a_qwen_topk import gfx90a_qwen_topk
from sglang.kernels.ops.moe import topk_softmax
from sglang.srt.layers.quantization.mq4g128 import (
    _requantize_checkpoint_fp8_mq4g128,
    dequantize_mq4g128,
    fwht128,
    quantize_mq4g128,
    swiglu_fwht128,
)
from sglang.srt.utils import is_hip


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_mq4g128_indexed_and_grouped_match_oracle():
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(7)
    # The production Qwen4 shard has 128 local experts.  Keep the unit shape
    # power-of-two as required by the sorter scan while exercising every ID.
    e, m, topk, n, k = 4, 8, 10, 64, 256
    weight = torch.randn(e, n, k, device="cuda", dtype=torch.float32) * 0.1
    packed = quantize_mq4g128(weight)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    x_rot = fwht128(x).contiguous()
    expert_ids = torch.randint(
        -1, e + 1, (m, topk), dtype=torch.int32, device="cuda"
    )
    dequant = dequantize_mq4g128(packed).reshape(e, n, k)
    expected = torch.empty(m, topk, n, dtype=torch.float32, device="cuda")
    for token in range(m):
        for slot in range(topk):
            expert = expert_ids[token, slot].item()
            if 0 <= expert < e:
                expected[token, slot] = dequant[expert] @ x_rot[token]
            else:
                expected[token, slot].zero_()
    indexed = mq4g128_indexed(x_rot, packed, expert_ids)
    grouped = mq4g128_grouped(x_rot, packed, expert_ids)
    torch.testing.assert_close(indexed, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(grouped, expected, rtol=2e-5, atol=2e-5)
    # A4 interleaves four accumulators and may schedule FMAs differently from
    # the single-assignment kernel. Both independently match the dequantized
    # oracle; keep the cross-path bound explicit instead of claiming bitwise
    # identity.
    torch.testing.assert_close(grouped, indexed, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen_weighted_reduce_matches_aten_bitwise():
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    for seed in range(32):
        torch.manual_seed(seed)
        partials = torch.randn(1, 10, 2560, device="cuda")
        weights = torch.softmax(torch.randn(1, 10, device="cuda"), dim=-1)
        expected = (partials * weights.unsqueeze(-1)).sum(1).to(torch.bfloat16)
        actual = mq4g128_weighted_reduce(partials, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen_router_topk_matches_aiter_bitwise():
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    for seed in range(32):
        torch.manual_seed(seed)
        logits = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
        expected_weights = torch.empty(1, 10, device="cuda")
        expected_ids = torch.empty(1, 10, device="cuda", dtype=torch.int32)
        actual_weights = torch.empty_like(expected_weights)
        actual_ids = torch.empty_like(expected_ids)
        topk_softmax(expected_weights, expected_ids, logits, True)
        gfx90a_qwen_topk(logits, actual_weights, actual_ids)
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


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
@pytest.mark.parametrize("shape", [(1, 10, 1280), (4, 10, 1280), (16, 10, 1280)])
def test_swiglu_fwht128_matches_unfused(shape):
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(10)
    gate_up = torch.randn(shape, dtype=torch.float32, device="cuda") * 0.7
    expected = fwht128(
        torch.nn.functional.silu(gate_up[..., :640]) * gate_up[..., 640:]
    )
    actual = swiglu_fwht128(gate_up)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen4_routed_method_matches_dequantized_oracle(monkeypatch):
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mq4g128 import Mq4g128RoutedMoEMethod

    monkeypatch.setenv("SGLANG_QWEN4_GFX90A_MQ4G128_GROUPED_MIN_TOKENS", "1")
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
