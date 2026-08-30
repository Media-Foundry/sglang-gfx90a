import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.tuned_gemm import tgemm

from sglang.srt.environ import envs

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Use aiter tuned GEMM dispatcher (tgemm.mm) to automatically select the GEMM kernel."""
    if (
        envs.SGLANG_DSV4_GFX90A_M64_ROUTER_HIPBLASLT.get()
        and torch.version.hip
        and hidden_states.shape == (64, 4096)
        and weight.shape == (256, 4096)
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
    ):
        from aiter.tuned_gemm import hipb_gemm

        return hipb_gemm(hidden_states, weight.detach(), 4358)
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
