import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import (
    gfx90a_wave64_bf16_gemv,
)
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
