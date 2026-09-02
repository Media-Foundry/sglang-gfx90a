#!/usr/bin/env python3
"""Validate raw DSV4 FP4 reads against CK's logical BF16 B layout."""

import argparse
import hashlib
from pathlib import Path

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
    parser.add_argument(
        "--runtime-aiter-layout",
        action="store_true",
        help="exercise the CK DynamicBuffer against AIter-shuffled FP4/scales",
    )
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
    runtime_layout = args.runtime_aiter_layout
    import aiter

    ck_include = (
        Path(aiter.__file__).resolve().parent.parent
        / "3rdparty/composable_kernel/include"
    )
    source = (
        "deepseek_v4/gfx90a_ck_logical_b_runtime_oracle.cuh"
        if runtime_layout
        else "deepseek_v4/gfx90a_fp4_ck_logical_b_oracle.cuh"
    )
    wrapper = (
        "Gfx90aCkLogicalBRuntimeOracle"
        if runtime_layout
        else "Gfx90aFp4CkLogicalBOracle"
    )
    module = load_jit(
        "gfx90a_ck_logical_b_runtime_oracle"
        if runtime_layout
        else "gfx90a_fp4_ck_logical_b_oracle",
        *cpp_args,
        cuda_files=[source],
        cuda_wrappers=[("run", f"sglang::{wrapper}<{cpp_args}>::run")],
        extra_cuda_cflags=[
            "-O3",
            f"-I{ck_include}",
        ],
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
        if runtime_layout:
            from aiter.ops.shuffle import (
                shuffle_scale_a16w4,
                shuffle_weight_a16w4,
            )

            input_weight = shuffle_weight_a16w4(weight, NLane=16, gate_up=True)
            input_scale = shuffle_scale_a16w4(
                scale.reshape(-1, args.k // 32), args.experts, gate_up=True
            )
        else:
            input_weight = weight
            input_scale = scale
        module.run(input_weight, input_scale, offsets, output)
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
        f"shape=E{args.experts}N{args.n}K{args.k} "
        f"runtime_aiter_layout={runtime_layout} sha256={digest}"
    )


if __name__ == "__main__":
    main()
