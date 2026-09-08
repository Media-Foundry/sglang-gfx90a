"""Call-owned, local-only deferred finalize for the TP8 M32 experiment.

No server selector enables this yet. The caller must join the shared producer
stream before finalize; this object does not introduce or reorder stream waits.
"""
from dataclasses import dataclass

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _jit_finalize_shared():
    args = make_cpp_args(32)
    return load_jit(
        'gfx90a_finalize_shared_local_oracle', args,
        cuda_files=['deepseek_v4/gfx90a_finalize_shared_oracle.cuh'],
        cuda_wrappers=[('run', f'sglang::Gfx90aFinalizeSharedOracle<{args}>::run')],
        extra_cuda_cflags=['-O3'],
    )


@dataclass(frozen=True)
class Gfx90aDeferredFinalize:
    partial: torch.Tensor
    out: torch.Tensor

    def finalize(self, shared: torch.Tensor) -> torch.Tensor:
        _jit_finalize_shared().run(self.partial, shared, self.out)
        return self.out
