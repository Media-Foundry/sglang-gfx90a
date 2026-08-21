import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.bf16_gemv import gfx90a_bf16_gemv


@triton.jit
def _mfma_gemv(x, w, out, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, 16)
    acc = tl.zeros((16, BN), tl.float32)
    for k_start in tl.range(0, K, BK, num_stages=2):
        offs_k = k_start + tl.arange(0, BK)
        a = tl.load(x + offs_m[:, None] * 0 + offs_k[None, :])
        b = tl.load(
            w + offs_k[:, None] + offs_n[None, :] * K,
            mask=offs_n[None, :] < N,
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] == 0) & (offs_n[None, :] < N),
    )


def run_mfma(x, w, block_n, block_k, num_warps):
    out = torch.empty((1, w.shape[0]), device=x.device, dtype=x.dtype)
    _mfma_gemv[(triton.cdiv(w.shape[0], block_n),)](
        x,
        w,
        out,
        N=w.shape[0],
        K=w.shape[1],
        BN=block_n,
        BK=block_k,
        num_warps=num_warps,
    )
    return out


def bench(fn, iterations=200):
    for _ in range(10):
        out = fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        out = fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations, out


def main():
    torch.manual_seed(1)
    for n, k in ((1536, 4096), (8192, 1024), (4096, 2048), (1024, 4096)):
        x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        base_us, base = bench(lambda: gfx90a_bf16_gemv(x, w))
        print(f"shape={n}x{k} scalar_us={base_us:.2f}", flush=True)
        for block_n in (16, 32):
            for block_k in (16, 32):
                for num_warps in (1, 2, 4):
                    try:
                        us, out = bench(
                            lambda bn=block_n, bk=block_k, nw=num_warps: run_mfma(
                                x, w, bn, bk, nw
                            ),
                            iterations=100,
                        )
                        error = (out.float() - base.float()).abs().max().item()
                        print(
                            f"  bn={block_n} bk={block_k} nw={num_warps} "
                            f"us={us:.2f} max_abs={error:.6g}",
                            flush=True,
                        )
                    except Exception as error:
                        print(
                            f"  bn={block_n} bk={block_k} nw={num_warps} "
                            f"ERROR={str(error).splitlines()[0]}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
