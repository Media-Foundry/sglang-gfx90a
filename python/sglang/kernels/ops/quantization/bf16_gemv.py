import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_gemv_kernel(
    x,
    weight,
    out,
    n: tl.constexpr,
    k: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    offs_n = tl.program_id(0) * block_n + tl.arange(0, block_n)
    acc = tl.zeros((block_n,), tl.float32)
    for k_start in tl.static_range(0, k, block_k):
        offs_k = k_start + tl.arange(0, block_k)
        values = tl.load(
            weight + offs_n[:, None] * k + offs_k[None, :],
            mask=(offs_n[:, None] < n) & (offs_k[None, :] < k),
            other=0.0,
        ).to(tl.float32)
        inputs = tl.load(x + offs_k, mask=offs_k < k, other=0.0).to(tl.float32)
        acc += tl.sum(values * inputs[None, :], axis=1)
    tl.store(out + offs_n, acc, mask=offs_n < n)


def gfx90a_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.ndim != 2
        or x.shape[0] != 1
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape[0] < 256
        or weight.shape[0] % 8 != 0
        or weight.shape[1] % 512 != 0
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None

    n, k = weight.shape
    out = torch.empty((1, n), dtype=x.dtype, device=x.device)
    block_n = 8
    block_k = 512
    _bf16_gemv_kernel[(triton.cdiv(n, block_n),)](
        x,
        weight,
        out,
        n=n,
        k=k,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
    )
    return out


@triton.jit
def _bf16_fp32_gemv_kernel(
    x,
    weight,
    out,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k_start in tl.static_range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        values = tl.load(
            weight + offs_n[:, None] * k + offs_k[None, :],
            mask=offs_n[:, None] < n,
            other=0.0,
        ).to(tl.float32)
        inputs = tl.load(x + offs_k).to(tl.float32)
        acc += tl.sum(values * inputs[None, :], axis=1)
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(out + offs_n, rounded, mask=offs_n < n)


def gfx90a_bf16_fp32_gemv(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.shape != (1, 4096)
        or weight.ndim != 2
        or weight.shape[0] not in (512, 1024, 2048)
        or weight.shape[1] != 4096
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None

    n = weight.shape[0]
    out = torch.empty((1, n), dtype=torch.float32, device=x.device)
    block_n = 4
    block_k = 512
    _bf16_fp32_gemv_kernel[(triton.cdiv(n, block_n),)](
        x,
        weight,
        out,
        n=n,
        k=4096,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


@triton.jit
def _bf16_grouped_gemv_kernel(
    x,
    weight,
    out,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    group = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k_start in tl.static_range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        values = tl.load(
            weight
            + group * n * k
            + offs_n[:, None] * k
            + offs_k[None, :],
            mask=offs_n[:, None] < n,
            other=0.0,
        ).to(tl.float32)
        inputs = tl.load(x + group * k + offs_k).to(tl.float32)
        acc += tl.sum(values * inputs[None, :], axis=1)
    tl.store(out + group * n + offs_n, acc, mask=offs_n < n)


def gfx90a_bf16_grouped_gemv(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.shape != (1, 2, 4096)
        or weight.shape != (2, 1024, 4096)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None

    out = torch.empty((1, 2, 1024), dtype=x.dtype, device=x.device)
    block_n = 4
    block_k = 512
    _bf16_grouped_gemv_kernel[(triton.cdiv(1024, block_n), 2)](
        x,
        weight,
        out,
        n=1024,
        k=4096,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out
