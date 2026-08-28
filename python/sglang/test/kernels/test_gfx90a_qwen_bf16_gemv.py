import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import (
    gfx90a_bf16_gate_up_swiglu_subgroup,
    gfx90a_wave64_bf16_gemv,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.utils import is_hip


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
@pytest.mark.parametrize(
    "n,k",
    [
        (4096, 2560),
        (512, 2560),
        (2560, 1536),
        (3584, 2560),
        (640, 2560),
        (320, 2560),
        (24, 2560),
        (2560, 160),
    ],
)
def test_qwen_bf16_gemv_matches_torch(n, k):
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(23 + n + k)
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    expected = F.linear(x, weight)
    actual = gfx90a_wave64_bf16_gemv(x, weight)
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qwen_bf16_gemv_graph_replay():
    if not is_hip() or "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(29)
    x = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16)
    gfx90a_wave64_bf16_gemv(x, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gfx90a_wave64_bf16_gemv(x, weight)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen_shared_gate_up_swiglu_subgroup_matches_old_chain():
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    activation = SiluAndMul()
    for seed in (0, 1, 17, 20260828):
        torch.manual_seed(seed)
        x = (torch.randn(1, 2560, device="cuda") * 0.1).to(torch.bfloat16)
        weight = (torch.randn(320, 2560, device="cuda") * 0.02).to(
            torch.bfloat16
        )
        gate_up = gfx90a_wave64_bf16_gemv(x, weight)
        assert gate_up is not None
        expected = activation(gate_up)
        actual = gfx90a_bf16_gate_up_swiglu_subgroup(x, weight)
        assert actual is not None
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-9)


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
def test_qwen_shared_gate_up_swiglu_subgroup_graph_replay_stable():
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(314159)
    x = (torch.randn(1, 2560, device="cuda") * 0.1).to(torch.bfloat16)
    weight = (torch.randn(320, 2560, device="cuda") * 0.02).to(torch.bfloat16)
    gfx90a_bf16_gate_up_swiglu_subgroup(x, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gfx90a_bf16_gate_up_swiglu_subgroup(x, weight)
    assert actual is not None
    graph.replay()
    torch.cuda.synchronize()
    reference = actual.clone()
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
