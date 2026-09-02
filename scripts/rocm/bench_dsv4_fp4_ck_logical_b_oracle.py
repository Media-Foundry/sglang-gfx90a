#!/usr/bin/env python3
"""Validate raw DSV4 FP4 reads against CK's logical BF16 B layout."""

import argparse
import hashlib

import torch

from sglang.kernels.jit.utils import load_jit, make_cpp_args
from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import _jit_dequant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--vectors", type=int, default=4096)
    parser.add_argument("--mutations", type=int, default=100)
    args = parser.parse_args()

    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx90a":
        raise RuntimeError("this oracle requires a physical gfx90a GCD")
    if args.n % 16 or args.k % 32:
        raise ValueError("N must be divisible by 16 and K by 32")

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(20260902)
    weight = torch.empty(
        (args.experts, args.n, args.k // 2), dtype=torch.uint8, device=dev
    )
    scale = torch.empty(
        (args.experts, args.n, args.k // 32), dtype=torch.uint8, device=dev
    )
    reference = torch.empty(
        (args.experts, args.n, args.k), dtype=torch.bfloat16, device=dev
    )
    output = torch.empty((args.vectors, 8), dtype=torch.bfloat16, device=dev)
    cpp_args = make_cpp_args(args.experts, args.n, args.k)
    module = load_jit(
        "gfx90a_fp4_ck_logical_b_oracle",
        *cpp_args,
        cuda_files=["deepseek_v4/gfx90a_fp4_ck_logical_b_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4CkLogicalBOracle<{cpp_args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )
    dequant = _jit_dequant(args.experts, args.n, args.k, 256)
    logical_values = args.experts * args.n * args.k

    for mutation in range(args.mutations):
        weight.random_(0, 256, generator=gen)
        # Keep exponents in the finite model-relevant range while explicitly
        # exercising exp=0 on every mutation.
        scale.random_(112, 143, generator=gen)
        scale.view(-1)[mutation % scale.numel()] = 0
        offsets = torch.randint(
            0,
            logical_values // 8,
            (args.vectors,),
            dtype=torch.int64,
            device=dev,
            generator=gen,
        ) * 8
        dequant.run_shuffled(weight, scale, reference)
        module.run(weight, scale, offsets, output)
        expected = reference.view(-1)[
            offsets[:, None] + torch.arange(8, device=dev, dtype=torch.int64)
        ]
        if not torch.equal(output, expected):
            diff = (output.float() - expected.float()).abs()
            raise AssertionError(
                f"mutation {mutation}: max_abs={diff.max().item()} "
                f"mismatches={(output != expected).sum().item()}"
            )

    digest = hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    print(
        f"PASS mutations={args.mutations} vectors={args.vectors} "
        f"shape=E{args.experts}N{args.n}K{args.k} sha256={digest}"
    )


if __name__ == "__main__":
    main()
