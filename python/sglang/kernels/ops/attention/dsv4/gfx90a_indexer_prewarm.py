"""Compile-only priming of explicit query16 runtime-M indexer signatures.

Does not execute queries, allocate dummy KV, or alter dispatch defaults.
The caller must use the same cache, dtype, layout, page strides and compiler
environment as the serving workers. Unsupported/unlisted signatures can still
compile on first use. This is not full-model startup prewarming.
"""


def validate_signature(width, page_columns, page_stride, preshuffle_tile):
    if not all(type(v) is int for v in (width, page_columns, page_stride, preshuffle_tile)):
        raise ValueError("Indexer prewarm dimensions must be integers")
    if not (512 <= width <= 8192 and width % 64 == 0):
        raise ValueError("Prewarm covers page-aligned C4 widths 512..8192")
    if page_columns < width // 64 or page_stride < page_columns:
        raise ValueError("Page table does not cover width or has overlapping rows")
    if preshuffle_tile not in (0, 8, 16):
        raise ValueError("Unsupported indexer cache layout")


def prewarm_query_reuse4(*, width, page_columns, page_stride,
                         preshuffle_tile=16, fp8_fnuz=False, dot_fp16=False):
    """Return an unlaunched CompiledKernel; no kernel execution or KV updates."""
    validate_signature(width, page_columns, page_stride, preshuffle_tile)
    import torch
    import triton.language as tl
    from triton.runtime.jit import MockTensor

    from .gfx90a_indexer_runtime_m import reuse_runtime_m

    if not torch.version.hip or "gfx90a" not in torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName:
        raise RuntimeError("This prewarm helper is scoped to gfx90a")
    pointers = [MockTensor(dtype) for dtype in (
        torch.uint8, torch.uint8, torch.float32,
        torch.int32, torch.int32, torch.float32,
    )]
    # M is do_not_specialize: this primes odd and smaller admitted M as well.
    # MockTensor assumes 16-byte pointer alignment; actual views must match.
    return reuse_runtime_m.warmup(
        *pointers, 32768, width, page_columns, page_stride, 16, 16,
        preshuffle_tile, tl.float16 if dot_fp16 else tl.bfloat16,
        tl.float8e4b8 if fp8_fnuz else tl.float8e4nv,
        grid=(2048, width // 16), num_warps=4,
    )
