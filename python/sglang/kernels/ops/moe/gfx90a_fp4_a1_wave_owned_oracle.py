from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_a1_wave_owned_down(blocks: int) -> Module:
    args = make_cpp_args(blocks)
    return load_jit(
        "gfx90a_fp4_a1_wave_owned_down_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_a1_wave_owned_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4A1WaveOwnedDownOracle<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )
