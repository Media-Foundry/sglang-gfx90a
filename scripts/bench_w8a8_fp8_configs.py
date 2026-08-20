#!/usr/bin/env python3
import argparse
import itertools
import os
import sys
import time

import torch
import triton


ROOT = os.environ.get("SGLANG_DIR", "/home/pc/Code/sglang")
PYTHON_DIR = os.path.join(ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from sglang.kernels.ops.quantization.fp8_kernel import _w8a8_block_fp8_matmul


DEFAULT_SHAPES = [
    (1536, 4096),
    (4096, 4096),
    (8192, 1024),
    (4096, 2048),
]


def make_tensors(m, n, k, block_n, block_k):
    # Random FP8 payloads are enough here: this script compares kernel time, not accuracy.
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    a_s = torch.rand((m, triton.cdiv(k, block_k)), device="cuda", dtype=torch.float32)
    b_s = torch.rand(
        (triton.cdiv(n, block_n), triton.cdiv(k, block_k)),
        device="cuda",
        dtype=torch.float32,
    )
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    return a, b.contiguous(), c, a_s.contiguous(), b_s.contiguous()


def launch(a, b, c, a_s, b_s, block_n, block_k, config):
    m, k = a.shape
    n = b.shape[0]
    needs_masking = bool(k % config["BLOCK_SIZE_K"] != 0)

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"])
            * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_matmul[grid](
        a,
        b,
        c,
        a_s,
        b_s,
        m,
        n,
        k,
        block_n,
        block_k,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        a_s.stride(0),
        a_s.stride(1),
        b_s.stride(1),
        b_s.stride(0),
        **config,
        needs_masking=needs_masking,
    )
    return c


def bench_one(shape, m, config, warmup, iters):
    n, k = shape
    block_n = 128
    block_k = 128
    a, b, c, a_s, b_s = make_tensors(m, n, k, block_n, block_k)
    for _ in range(warmup):
        launch(a, b, c, a_s, b_s, block_n, block_k, config)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch(a, b, c, a_s, b_s, block_n, block_k, config)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def candidate_configs(mode):
    if mode == "wide":
        choices = ((8, 16, 32), (16, 32, 64, 128), (4, 8), (2, 3), (0, 1, 2))
    else:
        choices = ((16,), (16, 32, 64, 128), (4,), (2,), (0, 1, 2, 4))
    for bm, bn, nw, ns, waves in itertools.product(*choices):
        yield {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": nw,
            "num_stages": ns,
            "waves_per_eu": waves,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Restrict to one N,K shape; may be repeated, e.g. --shape 4096,4096.",
    )
    parser.add_argument("--mode", choices=("narrow", "wide"), default="narrow")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    print(
        "device",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).gcnArchName,
        flush=True,
    )
    configs = list(candidate_configs(args.mode))
    shapes = [
        tuple(int(x) for x in item.split(",", 1))
        for item in args.shape
    ] or DEFAULT_SHAPES
    for shape in shapes:
        for m in args.m:
            rows = []
            for cfg in configs:
                try:
                    ms = bench_one(shape, m, cfg, args.warmup, args.iters)
                except Exception as exc:
                    print(f"FAIL shape={shape} M={m} cfg={cfg} err={exc}", flush=True)
                    continue
                rows.append((ms, cfg))
            rows.sort(key=lambda x: x[0])
            print(f"\nshape N={shape[0]} K={shape[1]} M={m}", flush=True)
            for ms, cfg in rows[: args.topk]:
                print(f"{ms:.4f} ms {cfg}", flush=True)


if __name__ == "__main__":
    main()
