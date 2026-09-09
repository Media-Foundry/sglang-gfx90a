"""Opt-in target-only H8 sparse attention; callers own forward-mode guards."""
import torch
from torch.utils.cpp_extension import include_paths
from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        'gfx90a_dsv4_sparse_h8_oracle_v1',
        cuda_files=['deepseek_v4/gfx90a_dsv4_sparse_h8_oracle.cuh'],
        cuda_wrappers=[('run', 'sglang::Gfx90aDsv4SparseH8Oracle::run'),
                       ('run_h16', 'sglang::Gfx90aDsv4UnifiedSparseDecode::run')],
        extra_cuda_cflags=['-O3', '-std=c++20', '-DCK_ENABLE_BF16', '-DCK_USE_XDL'],
        extra_include_paths=[*include_paths(),
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include',
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/library/include'],
    )


def run_if_supported(q, kv, indices, indptr, sink, scale):
    # MQALayer retains a legacy padded 64-head local sink. The first eight
    # entries are this TP8 rank's real heads; slicing is allocation-free.
    if sink.ndim == 1 and sink.shape[0] >= 8:
        sink = sink[:8]
    tokens = q.shape[0]
    if not (tokens in (128, 192) and q.shape == (tokens, 8, 512) and q.dtype == torch.bfloat16
            and kv.ndim == 2 and kv.shape[1] == 512 and kv.dtype == q.dtype
            and indices.dtype == torch.int32 and indptr.dtype == torch.int32
            and indptr.shape == (tokens + 1,) and sink.shape == (8,)
            and sink.dtype == torch.float32
            and all(t.is_cuda and t.is_contiguous() and t.device == q.device
                    for t in (q, kv, indices, indptr, sink))):
        return None
    output = torch.empty_like(q)
    workspace = torch.empty(tokens*2*8*514*4, device=q.device, dtype=torch.uint8)
    _module().run(q, kv, indices, indptr, sink, output, workspace, scale)
    return output
