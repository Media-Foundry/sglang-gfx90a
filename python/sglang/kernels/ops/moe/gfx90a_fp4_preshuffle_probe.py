import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _module(experts: int, rows: int, hidden: int, gate_up: bool):
    args = make_cpp_args(experts, rows, hidden, gate_up)
    return load_jit(
        "gfx90a_fp4_preshuffle_probe",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gemv.cuh",
            "deepseek_v4/gfx90a_fp4_preshuffle_probe.cuh",
        ],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4PreshuffleProbeKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_fp4_preshuffle_probe(
    weight: torch.Tensor,
    scale: torch.Tensor,
    query: torch.Tensor,
    *,
    gate_up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, packed_hidden = weight.shape
    hidden = packed_hidden * 2
    weight_out = torch.empty(query.shape[0], dtype=torch.uint8, device=weight.device)
    scale_out = torch.empty_like(weight_out)
    _module(experts, rows, hidden, gate_up).run(
        weight_out,
        scale_out,
        weight.view(torch.uint8),
        scale.view(torch.uint8).reshape(experts * rows, hidden // 32),
        query,
    )
    return weight_out, scale_out
