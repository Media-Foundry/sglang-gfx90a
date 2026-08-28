from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.gfx90a_qsa_packed_decode import (
    gfx90a_qsa_packed_decode,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    _packed_single_attention_torch,
)
from sglang.srt.utils import is_hip


def test_qwen_qsa_mtp_index_share_override_precedence():
    from sglang.srt.speculative.eagle_worker_v2 import _qsa_index_share_requested

    nested_on = SimpleNamespace(index_share_for_mtp_iteration=True)
    assert _qsa_index_share_requested(SimpleNamespace(text_config=nested_on))
    assert not _qsa_index_share_requested(
        SimpleNamespace(
            text_config=nested_on,
            index_share_for_mtp_iteration=False,
        )
    )


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
@pytest.mark.parametrize("valid", [1, 64, 193, 512, 1024, 2048])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("heads", [6, 12])
def test_gfx90a_qsa_packed_decode_matches_torch(valid, batch, heads):
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(13)
    topk = 2048
    q = torch.randn(batch, heads, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * topk, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    counts = [max(1, valid - row * 7) for row in range(batch)]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    scale = 256**-0.5
    expected = torch.cat(
        [
            _packed_single_attention_torch(
                q[row : row + 1],
                k[offsets[row] :],
                v[offsets[row] :],
                torch.tensor([0, counts[row]], device="cuda", dtype=torch.int32),
                topk,
                scale,
            )
            for row in range(batch)
        ]
    )
    actual = gfx90a_qsa_packed_decode(q, k, v, cu, scale)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_gfx90a_qsa_packed_decode_graph_replay():
    if not is_hip() or "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(17)
    batch, topk = 4, 2048
    q = torch.randn(batch, 6, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * topk, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = torch.tensor([0, 97, 290, 801, 1450], device="cuda", dtype=torch.int32)
    scale = 256**-0.5
    # Warm up JIT and allocations before capture; replay must consume refreshed
    # device-side offsets without any D2H synchronization.
    gfx90a_qsa_packed_decode(q, k, v, cu, scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gfx90a_qsa_packed_decode(q, k, v, cu, scale)
    graph.replay()
    torch.cuda.synchronize()

    offsets = cu.cpu().tolist()
    expected = torch.cat(
        [
            _packed_single_attention_torch(
                q[row : row + 1],
                k[offsets[row] :],
                v[offsets[row] :],
                torch.tensor(
                    [0, offsets[row + 1] - offsets[row]],
                    device="cuda",
                    dtype=torch.int32,
                ),
                topk,
                scale,
            )
            for row in range(batch)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
