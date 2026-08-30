#!/usr/bin/env python3
"""Oracle-only TP4/M64 signed-int5 repack versus packed-FP4 LDS decode."""

from __future__ import annotations

import argparse
import statistics

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import _jit_down_grouped
from sglang.kernels.ops.moe.gfx90a_fp4_int5_repack_oracle import (
    repack_fp4_to_signed_int5,
    run_int5_down_partial,
)


E, M, T, N, K = 256, 64, 6, 4096, 512


def time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iterations):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=20)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--recorded-world-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.recorded_world_size) != 0):
        raise RuntimeError("recorded counts are not divisible by world size")
    topk_ids = reconstruct_topk_from_counts(
        raw // args.recorded_world_size, M, T
    ).cuda()
    metadata = make_metadata(topk_ids, assignments=4)

    torch.manual_seed(20260830)
    xq = torch.randint(-127, 128, (M, T, K), dtype=torch.int8, device="cuda")
    xs = torch.rand((M, T, K // 32), dtype=torch.float32, device="cuda")
    topk_weights = torch.rand((M, T), dtype=torch.float32, device="cuda")
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
    scales = torch.full((E, N, K // 32), 127, dtype=torch.uint8, device="cuda")
    dense5 = repack_fp4_to_signed_int5(packed, bitplane=False)
    plane5 = repack_fp4_to_signed_int5(packed, bitplane=True)
    if dense5.nbytes != packed.nbytes * 5 // 4 or plane5.nbytes != dense5.nbytes:
        raise RuntimeError("int5 repack must be exactly 1.25x packed FP4")

    baseline = _jit_down_grouped(E, M, T, N, K, 4, 2, 8, 832, 2)
    partial = {
        name: torch.empty((M, T, N), dtype=torch.float32, device="cuda")
        for name in ("A", "D", "P")
    }

    def run_a() -> None:
        baseline.run_partial(
            xq, xs, packed, scales, metadata.sorted_ids,
            metadata.sorted_experts, metadata.valid, topk_weights, partial["A"],
        )

    def run_d() -> None:
        run_int5_down_partial(
            xq, xs, dense5, scales, metadata.sorted_ids,
            metadata.sorted_experts, metadata.valid, topk_weights, partial["D"],
            bitplane=False,
        )

    def run_p() -> None:
        run_int5_down_partial(
            xq, xs, plane5, scales, metadata.sorted_ids,
            metadata.sorted_experts, metadata.valid, topk_weights, partial["P"],
            bitplane=True,
        )

    runners = {"A": run_a, "D": run_d, "P": run_p}

    def assert_exact(name: str, where: str) -> None:
        if not torch.equal(partial[name], partial["A"]):
            diff = (partial[name] - partial["A"]).abs()
            raise RuntimeError(
                f"{where}: {name} mismatch max_abs={float(diff.max())}"
            )

    for fn in runners.values():
        fn()
    torch.cuda.synchronize()
    assert_exact("D", "initial")
    assert_exact("P", "initial")
    for mutation in range(100):
        xq.random_(-127, 128)
        xs.uniform_(1.0e-5, 0.05)
        topk_weights.uniform_()
        for fn in runners.values():
            fn()
        torch.cuda.synchronize()
        assert_exact("D", f"mutation={mutation}")
        assert_exact("P", f"mutation={mutation}")
    print("CORRECTNESS eager_mutations=100 dense_exact=True bitplane_exact=True")

    # Freeze one reference, then verify graph replay does not stale any input.
    run_a()
    torch.cuda.synchronize()
    for name in ("D", "P"):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                runners[name]()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runners[name]()
        for replay in range(1000):
            graph.replay()
            if replay in (0, 1, 9, 99, 999):
                torch.cuda.synchronize()
                assert_exact(name, f"graph={replay}")
        torch.cuda.synchronize()
        assert_exact(name, "graph=final")
        print(f"CORRECTNESS profile={name} graph_replays=1000 exact=True")

    timings = {name: [] for name in runners}
    for _ in range(args.rounds):
        for name in ("A", "D", "P", "P", "D", "A"):
            timings[name].append(time_us(runners[name], args.warmup, args.iterations))
    for name in ("A", "D", "P"):
        values = timings[name]
        trimmed = sorted(values)[1:-1]
        print(
            f"RESULT profile={name} samples_us="
            + ",".join(f"{v:.3f}" for v in values)
            + f" median_us={statistics.median(values):.3f}"
            + f" trimmed_mean_us={statistics.mean(trimmed):.3f}"
        )


if __name__ == "__main__":
    main()
