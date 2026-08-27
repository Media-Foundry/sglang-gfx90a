from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(heads: int, topk: int) -> Module:
    args = make_cpp_args(heads, topk)
    return load_jit(
        "gfx90a_qsa_packed_decode",
        *args,
        cuda_files=["attention/gfx90a_qsa_packed_decode.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aQsaPackedDecode<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_qsa_packed_decode(
    q: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, heads, dim = q.shape
    topk, kv_heads, kv_dim = packed_k.shape
    if (
        batch != 1
        or dim != 256
        or kv_heads != 1
        or kv_dim != 256
        or packed_v.shape != packed_k.shape
        or q.dtype != torch.bfloat16
        or packed_k.dtype != torch.bfloat16
        or packed_v.dtype != torch.bfloat16
        or cu_seqlens_k.shape != (2,)
        or cu_seqlens_k.dtype != torch.int32
    ):
        raise ValueError(
            "gfx90a QSA packed decode requires q=[1,H,256], "
            "K/V=[TOPK,1,256] BF16 and cu_seqlens_k=[2] int32"
        )
    if not all(
        t.is_cuda and t.is_contiguous()
        for t in (q, packed_k, packed_v, cu_seqlens_k)
    ):
        raise ValueError(
            "gfx90a QSA packed decode inputs must be contiguous CUDA tensors"
        )
    out = torch.empty_like(q)
    partial_out = torch.empty(
        (heads, 8, 256), dtype=torch.float32, device=q.device
    )
    partial_max = torch.empty((heads, 8), dtype=torch.float32, device=q.device)
    partial_sum = torch.empty((heads, 8), dtype=torch.float32, device=q.device)
    _module(heads, topk).run(
        q,
        packed_k,
        packed_v,
        cu_seqlens_k,
        partial_out,
        partial_max,
        partial_sum,
        out,
        scale,
    )
    return out
