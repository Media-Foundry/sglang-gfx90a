from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(rows: int) -> Module:
    args = make_cpp_args(rows)
    return load_jit(
        "gfx90a_gdn_packed_decode",
        *args,
        cuda_files=["attention/gfx90a_gdn_packed_decode.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aGdnPackedDecode<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_gdn_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    rows: int = 32,
) -> torch.Tensor:
    """BS1 Qwen3.8Next GDN recurrence specialized for gfx90a."""
    expected = {
        "mixed_qkv": ((1, 7680), torch.bfloat16),
        "a": ((1, 48), torch.bfloat16),
        "b": ((1, 48), torch.bfloat16),
        "A_log": ((48,), torch.float32),
        "dt_bias": ((48,), torch.float32),
    }
    for name, tensor in (
        ("mixed_qkv", mixed_qkv),
        ("a", a),
        ("b", b),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
    ):
        shape, dtype = expected[name]
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must be {shape} {dtype}, got {tensor.shape} {tensor.dtype}")
    if state.ndim != 4 or state.shape[1:] != (48, 128, 128) or state.dtype != torch.bfloat16:
        raise ValueError("state must be [slots,48,128,128] BF16")
    if state_indices.shape != (1,) or state_indices.dtype != torch.int32:
        raise ValueError("state_indices must be [1] int32")
    if not all(t.is_cuda and t.is_contiguous() for t in (mixed_qkv, a, b, A_log, dt_bias, state, state_indices)):
        raise ValueError("all gfx90a GDN inputs must be contiguous CUDA tensors")
    if rows not in (4, 8, 16, 32):
        raise ValueError("rows must be one of 4, 8, 16, 32")
    out = torch.empty((1, 1, 48, 128), dtype=torch.bfloat16, device=mixed_qkv.device)
    _module(rows).run(mixed_qkv, a, b, A_log, dt_bias, state, state_indices, out)
    return out
