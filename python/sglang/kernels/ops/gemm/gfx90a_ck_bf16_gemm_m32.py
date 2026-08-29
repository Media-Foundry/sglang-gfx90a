import os
from pathlib import Path

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module(n: int, k: int):
    args = f"{n}, {k}"
    configured = os.environ.get("SGLANG_CK_ROOT")
    if configured:
        ck = Path(configured)
    else:
        import aiter

        ck = (
            Path(aiter.__file__).resolve().parent.parent
            / "3rdparty"
            / "composable_kernel"
        )
    return load_jit(
        f"gfx90a_ck_bf16_gemm_m32_{n}_{k}",
        cuda_files=["gemm/gfx90a_ck_bf16_gemm_m32.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aCkBf16GemmM32<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[str(ck / "include"), str(ck / "library" / "include")],
    )


def gfx90a_ck_bf16_gemm_m32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] == 32 and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    n, k = weight.shape
    assert (n, k) == (2560, 640), "only the validated Qwen shared-down shape is supported"
    assert x.shape[1] == k
    out = torch.empty((32, n), dtype=torch.bfloat16, device=x.device)
    _module(n, k).run(x, weight, out)
    return out
