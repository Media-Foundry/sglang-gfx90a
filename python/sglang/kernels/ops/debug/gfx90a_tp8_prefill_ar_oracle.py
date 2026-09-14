"""Large-prefill legacy AIter grid oracle; caller owns registration and staging."""
from torch.utils.cpp_extension import include_paths
from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def module():
    return load_jit(
        'gfx90a_tp8_prefill_ar_oracle',
        cuda_files=['deepseek_v4/gfx90a_tp8_prefill_ar_oracle.cuh'],
        cuda_wrappers=[('run','sglang::Gfx90aTp8PrefillArOracle::run'),
                       ('signal_bytes','sglang::Gfx90aTp8PrefillArOracle::signal_bytes')],
        extra_cuda_cflags=['-O3','-DUSE_ROCM'],
        extra_include_paths=[*include_paths(),
            '/home/pc/pytorch/third_party/aiter/csrc/include',
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include'])
