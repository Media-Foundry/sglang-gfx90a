"""CPU-only dispatch tests. These mocks do not validate GPU arithmetic."""

import types
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.nsa_triton_decode import (
    triton_mla_kernels_decode_optimized as dispatch,
)


def scope(capacity):
    cache = torch.empty(1, 128, 1, 584, dtype=torch.uint8)
    return types.SimpleNamespace(
        blocked_k_quantized=cache, blocked_k=cache,
        indices_in_kvcache=torch.zeros(1, 1, capacity, dtype=torch.int32),
        topk_length=torch.ones(1, dtype=torch.int32),
    )


def result(*args, **kwargs):
    return torch.empty(1, 64, 512), torch.empty(1, 64)


def test_default_retains_legacy_dual_dispatch():
    with patch.object(dispatch, "fused_gather_attn_decode_dsv4_dual_scope_low_overhead", side_effect=result) as old, patch.object(dispatch, "fused_gather_attn_decode_dsv4_dual_scope") as fixed:
        dispatch.triton_sparse_attn_decode(torch.empty(1, 1, 64, 512), scope(128), scope(512), 0.1)
        old.assert_called_once()
        fixed.assert_not_called()


def test_invariant_dual_disables_splitk():
    with patch.object(dispatch, "fused_gather_attn_decode_dsv4_dual_scope_low_overhead") as old, patch.object(dispatch, "fused_gather_attn_decode_dsv4_dual_scope", side_effect=result) as fixed:
        dispatch.triton_sparse_attn_decode(torch.empty(1, 1, 64, 512), scope(128), scope(512), 0.1, invariant_reduction=True)
        old.assert_not_called()
        assert fixed.call_args.kwargs["invariant_reduction"] is True
        assert fixed.call_args.kwargs["force_no_splitk"] is True


def test_invariant_single_propagates():
    with patch.object(dispatch, "fused_gather_attn_decode_dsv4", side_effect=result) as fixed:
        dispatch.triton_sparse_attn_decode(torch.empty(1, 1, 64, 512), scope(128), None, 0.1, invariant_reduction=True)
        assert fixed.call_args.kwargs["invariant_reduction"] is True
