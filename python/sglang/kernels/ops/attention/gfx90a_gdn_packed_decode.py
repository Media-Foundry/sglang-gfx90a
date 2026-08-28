from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(rows: int, waves: int, a_bf16: bool, dt_bf16: bool) -> Module:
    args = make_cpp_args(rows, waves, int(a_bf16), int(dt_bf16))
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
    waves: int = 1,
) -> torch.Tensor:
    """BS1 Qwen3.8Next GDN recurrence specialized for gfx90a."""
    expected = {
        "mixed_qkv": ((1, 2560), torch.bfloat16),
        "a": ((1, 12), torch.bfloat16),
        "b": ((1, 12), torch.bfloat16),
    }
    for name, tensor in (
        ("mixed_qkv", mixed_qkv),
        ("a", a),
        ("b", b),
    ):
        shape, dtype = expected[name]
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must be {shape} {dtype}, got {tensor.shape} {tensor.dtype}")
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        if tensor.shape != (12,) or tensor.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name} must be [12] BF16/FP32, got {tensor.shape} {tensor.dtype}")
    if state.ndim != 4 or state.shape[1:] != (12, 128, 128) or state.dtype != torch.float32:
        raise ValueError("state must be [slots,12,128,128] FP32")
    if state_indices.shape != (1,) or state_indices.dtype != torch.int32:
        raise ValueError("state_indices must be [1] int32")
    if not all(t.is_cuda and t.is_contiguous() for t in (mixed_qkv, a, b, A_log, dt_bias, state, state_indices)):
        raise ValueError("all gfx90a GDN inputs must be contiguous CUDA tensors")
    if rows not in (4, 8, 16, 32):
        raise ValueError("rows must be one of 4, 8, 16, 32")
    if waves not in (1, 2) or (128 % (rows * waves)) != 0:
        raise ValueError("waves must be 1/2 and rows*waves must divide 128")
    out = torch.empty((1, 1, 12, 128), dtype=torch.bfloat16, device=mixed_qkv.device)
    _module(
        rows,
        waves,
        A_log.dtype == torch.bfloat16,
        dt_bias.dtype == torch.bfloat16,
    ).run(mixed_qkv, a, b, A_log, dt_bias, state, state_indices, out)
    return out
