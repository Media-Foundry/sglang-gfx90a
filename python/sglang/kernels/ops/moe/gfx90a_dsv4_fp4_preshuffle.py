import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "gfx90a_dsv4_fp4_preshuffle_v3",
        cuda_files=["deepseek_v4/gfx90a_dsv4_fp4_preshuffle.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aDsv4Fp4Preshuffle::run")],
        extra_cuda_cflags=["-O3"],
    )


def preshuffle_into(
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    blocks: int = 832,
) -> None:
    experts, w13_rows, _ = w13.shape
    _, w2_rows, _ = w2.shape
    out_w13, out_s13, out_w2, out_s2 = outputs
    _module().run(
        w13.view(torch.uint8),
        s13.view(torch.uint8).reshape(experts, w13_rows, -1),
        w2.view(torch.uint8),
        s2.view(torch.uint8).reshape(experts, w2_rows, -1),
        out_w13.view(torch.uint8),
        out_s13.view(torch.uint8).reshape(experts, w13_rows, -1),
        out_w2.view(torch.uint8),
        out_s2.view(torch.uint8).reshape(experts, w2_rows, -1),
        blocks,
    )
