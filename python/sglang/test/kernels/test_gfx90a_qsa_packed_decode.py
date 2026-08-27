import pytest
import torch

from sglang.kernels.ops.attention.gfx90a_qsa_packed_decode import (
    gfx90a_qsa_packed_decode,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    _packed_single_attention_torch,
)
from sglang.srt.utils import is_hip


@pytest.mark.skipif(not is_hip(), reason="gfx90a HIP-only kernel")
@pytest.mark.parametrize("valid", [1, 64, 193, 512, 1024, 2048])
def test_gfx90a_qsa_packed_decode_matches_torch(valid):
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx90a")
    torch.manual_seed(13)
    topk = 2048
    q = torch.randn(1, 6, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(topk, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = torch.tensor([0, valid], device="cuda", dtype=torch.int32)
    scale = 256**-0.5
    expected = _packed_single_attention_torch(q, k, v, cu, topk, scale)
    actual = gfx90a_qsa_packed_decode(q, k, v, cu, scale)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
