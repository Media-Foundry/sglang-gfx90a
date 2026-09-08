"""Geometry oracle, also reused by the explicit default-off native M32 trial."""
from torch.utils.cpp_extension import include_paths
from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def module():
    return load_jit(
        'gfx90a_tp8_ar_geometry_oracle',
        cuda_files=['deepseek_v4/gfx90a_tp8_ar_geometry_oracle.cuh'],
        cuda_wrappers=[('run', 'sglang::Gfx90aTp8ArGeometryOracle::run'),
                       ('signal_bytes', 'sglang::Gfx90aTp8ArGeometryOracle::signal_bytes')],
        extra_cuda_cflags=['-O3', '-DUSE_ROCM'],
        extra_include_paths=[*include_paths(),
            '/home/pc/pytorch/third_party/aiter/csrc/include',
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include'],
    )


def run(comm, x, out, blocks):
    assert x.is_contiguous() and out.is_contiguous()
    module().run(comm._ptr, x, out, blocks)
