import torch

from sglang.kernels.jit.utils import cache_once, load_jit

from .utils import make_name


@cache_once
def _module():
    return load_jit(
        make_name("c4_dual_decode"),
        cuda_files=["deepseek_v4/c4_dual_decode.cuh"],
        cuda_wrappers=[("run", "C4DualDecodeKernel::run")],
    )


def c4_dual_decode(
    core_state: torch.Tensor,
    core_input: torch.Tensor,
    core_ape: torch.Tensor,
    index_state: torch.Tensor,
    index_input: torch.Tensor,
    index_ape: torch.Tensor,
    plan: torch.Tensor,
    core_norm: torch.Tensor,
    index_norm: torch.Tensor,
    core_freqs: torch.Tensor,
    index_freqs: torch.Tensor,
    core_out_loc: torch.Tensor,
    index_out_loc: torch.Tensor,
    core_cache: torch.Tensor,
    index_cache: torch.Tensor,
    core_tmp: torch.Tensor,
    index_tmp: torch.Tensor,
    core_eps: float,
    index_eps: float,
) -> None:
    _module().run(
        core_state,
        core_input,
        core_ape,
        index_state,
        index_input,
        index_ape,
        plan,
        core_norm,
        index_norm,
        core_freqs,
        index_freqs,
        core_out_loc,
        index_out_loc,
        core_cache,
        index_cache,
        core_tmp,
        index_tmp,
        core_eps,
        index_eps,
    )
