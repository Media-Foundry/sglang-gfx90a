#!/usr/bin/env python3
"""Standalone TP4/M32 token-row-owned DSV4 MHC boundary oracle.

This does not alter the model path.  It compares two graph-captured boundaries:

* A: every TP rank evaluates all 32 token rows with the existing gfx90a MHC
  post/pre backend;
* B: rank ``r`` evaluates rows ``[8*r:8*(r+1)]`` and AIter custom all-gather
  publishes the 32 normalized rows again.

The two-boundary graph deliberately includes one publication per boundary.  It
therefore gives a useful upper-bound decision for token-row ownership before a
production reduce-scatter/peer-read implementation is attempted.

Run on four otherwise idle GCDs::

  HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    scripts/rocm/bench_dsv4_tp4_token_row_mhc_oracle.py
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_pre import (
    gfx90a_mhc_post_pre,
    preload_gfx90a_mhc_post_pre,
)


@dataclass
class BoundaryInputs:
    x: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    fn: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    norm: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument(
        "--include-reduce-scatter",
        action="store_true",
        help=(
            "compare stock all-reduce + full-row MHC against reduce-scatter + "
            "owner-local MHC + all-gather"
        ),
    )
    return parser.parse_args()


def randn(shape, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device="cuda", dtype=dtype)


def make_inputs(rows: int, seed_offset: int) -> BoundaryInputs:
    # The magnitudes resemble live residual/MHC metadata while avoiding sigmoid
    # saturation in this component oracle.  Every rank constructs identical
    # tensors, so row ownership is the only semantic difference.
    return BoundaryInputs(
        x=randn((rows, 4096), 100 + seed_offset, torch.bfloat16).mul_(0.125),
        residual=randn(
            (rows, 4, 4096), 200 + seed_offset, torch.bfloat16
        ).mul_(0.125),
        post=randn((rows, 4), 300 + seed_offset, torch.float32).mul_(0.1),
        comb=randn((rows, 4, 4), 400 + seed_offset, torch.float32).mul_(0.1),
        fn=randn((24, 16384), 500 + seed_offset, torch.float16).mul_(0.01),
        scale=torch.tensor([0.25, 0.25, 0.25], device="cuda"),
        base=randn((24,), 600 + seed_offset, torch.float32).mul_(0.1),
        norm=randn((4096,), 700 + seed_offset, torch.bfloat16).mul_(0.1).add_(1),
    )


def run_boundary(inp: BoundaryInputs):
    out = gfx90a_mhc_post_pre(
        inp.x,
        inp.residual,
        inp.post,
        inp.comb,
        inp.fn,
        inp.scale,
        inp.base,
        inp.norm,
        1e-6,
        1e-6,
        2.0,
        1e-6,
    )
    if out is None:
        raise RuntimeError("gfx90a MHC backend rejected the oracle shape")
    return out


def slice_inputs(inp: BoundaryInputs, lo: int, hi: int) -> BoundaryInputs:
    return BoundaryInputs(
        inp.x[lo:hi].contiguous(),
        inp.residual[lo:hi].contiguous(),
        inp.post[lo:hi].contiguous(),
        inp.comb[lo:hi].contiguous(),
        inp.fn,
        inp.scale,
        inp.base,
        inp.norm,
    )


def replace_x(inp: BoundaryInputs, x: torch.Tensor) -> BoundaryInputs:
    return BoundaryInputs(
        x,
        inp.residual,
        inp.post,
        inp.comb,
        inp.fn,
        inp.scale,
        inp.base,
        inp.norm,
    )


def reduced_boundary(inp: BoundaryInputs, comm: CustomAllreduce):
    reduced = comm.custom_all_reduce(inp.x)
    if reduced is None:
        raise RuntimeError("AIter custom all-reduce is unavailable")
    return run_boundary(replace_x(inp, reduced))


def gathered_boundary(
    inp: BoundaryInputs,
    comm: CustomAllreduce,
    rank: int,
    world: int,
    *,
    include_reduce_scatter: bool = False,
):
    rows_per_rank = inp.x.shape[0] // world
    lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
    local_inp = slice_inputs(inp, lo, hi)
    if include_reduce_scatter:
        local_x = torch.empty_like(local_inp.x)
        # AIter's reduce_scatter writes the explicit output buffer but its
        # Python method intentionally has no return value.  The communicator
        # availability was checked once in main(), so consume local_x directly.
        comm.custom_reduce_scatter(inp.x, local_x)
        local_inp = replace_x(local_inp, local_x)
    local = run_boundary(local_inp)
    gathered = comm.custom_all_gather(local[3].contiguous())
    if gathered is None:
        raise RuntimeError("AIter custom all-gather is unavailable")
    full_layer_input = gathered.reshape(inp.x.shape[0], 4096)
    return (*local[:3], local[3], full_layer_input)


def capture(comm: CustomAllreduce, fn):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture():
        with torch.cuda.graph(graph):
            outputs = fn()
    dist.barrier()
    return graph, outputs


def rankmax_reps(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iters: int,
    reps: int,
    world: int,
) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    local: list[float] = []
    for _ in range(reps):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        local.append(start.elapsed_time(end) * 1000.0 / iters)
    all_reps: list[list[float] | None] = [None] * world
    dist.all_gather_object(all_reps, local)
    return [max(all_reps[r][i] for r in range(world)) for i in range(reps)]


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4 or args.rows % world:
        raise ValueError("this oracle requires TP4 and rows divisible by four")

    preload_gfx90a_mhc_post_pre()
    # Rank-distinct X is required when the oracle includes the incoming TP
    # reduction.  The remaining MHC state is intentionally identical because
    # it is replicated model/request metadata in the production boundary.
    inp0 = make_inputs(args.rows, 0)
    inp1 = make_inputs(args.rows, 1000)
    if args.include_reduce_scatter:
        inp0 = replace_x(
            inp0,
            randn((args.rows, 4096), 100 + rank * 10000, torch.bfloat16).mul_(
                0.125
            ),
        )
        inp1 = replace_x(
            inp1,
            randn(
                (args.rows, 4096), 1100 + rank * 10000, torch.bfloat16
            ).mul_(0.125),
        )
    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter custom collectives did not initialize")

    # Eager correctness: local state must equal the corresponding full-row
    # state bit for bit, not merely within a floating-point tolerance.
    reference_fn = reduced_boundary if args.include_reduce_scatter else run_boundary
    ref0 = reference_fn(inp0, comm) if args.include_reduce_scatter else reference_fn(inp0)
    ref1 = reference_fn(inp1, comm) if args.include_reduce_scatter else reference_fn(inp1)
    cand0 = gathered_boundary(
        inp0,
        comm,
        rank,
        world,
        include_reduce_scatter=args.include_reduce_scatter,
    )
    cand1 = gathered_boundary(
        inp1,
        comm,
        rank,
        world,
        include_reduce_scatter=args.include_reduce_scatter,
    )
    rows_per_rank = args.rows // world
    lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
    for reference, candidate in ((ref0, cand0), (ref1, cand1)):
        for ref_tensor, local_tensor in zip(reference[:3], candidate[:3]):
            torch.testing.assert_close(
                local_tensor, ref_tensor[lo:hi], rtol=0, atol=0
            )
        torch.testing.assert_close(candidate[3], reference[3][lo:hi], rtol=0, atol=0)
        torch.testing.assert_close(candidate[4], reference[3], rtol=0, atol=0)
    dist.barrier()
    # Keep graph validation independent of later capture-pool reuse.  AIter's
    # capture context may recycle ordinary graph allocations once a second
    # graph registers its collective buffers.
    frozen_reference = (ref0[3].clone(), ref1[3].clone())

    # Two boundaries, each including publication for B.  This is intentionally
    # stricter than timing one local MHC and extrapolating its compute saving.
    ref_graph, ref_outputs = capture(
        comm,
        lambda: (
            reference_fn(inp0, comm)
            if args.include_reduce_scatter
            else reference_fn(inp0),
            reference_fn(inp1, comm)
            if args.include_reduce_scatter
            else reference_fn(inp1),
        ),
    )
    cand_graph, cand_outputs = capture(
        comm,
        lambda: (
            gathered_boundary(
                inp0,
                comm,
                rank,
                world,
                include_reduce_scatter=args.include_reduce_scatter,
            ),
            gathered_boundary(
                inp1,
                comm,
                rank,
                world,
                include_reduce_scatter=args.include_reduce_scatter,
            ),
        ),
    )
    rows_per_rank = args.rows // world
    lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
    local0 = slice_inputs(inp0, lo, hi)
    local1 = slice_inputs(inp1, lo, hi)
    local_graph, _ = capture(
        comm, lambda: (run_boundary(local0), run_boundary(local1))
    )

    for _ in range(args.graph_replays):
        cand_graph.replay()
    torch.cuda.synchronize()
    for reference, candidate in zip(frozen_reference, cand_outputs):
        torch.testing.assert_close(candidate[4], reference, rtol=0, atol=0)

    # ABBA using slowest-rank time for every repetition.
    a1 = rankmax_reps(
        ref_graph,
        warmup=args.warmup,
        iters=args.iters,
        reps=args.reps,
        world=world,
    )
    b1 = rankmax_reps(
        cand_graph,
        warmup=args.warmup,
        iters=args.iters,
        reps=args.reps,
        world=world,
    )
    b2 = rankmax_reps(
        cand_graph,
        warmup=args.warmup,
        iters=args.iters,
        reps=args.reps,
        world=world,
    )
    a2 = rankmax_reps(
        ref_graph,
        warmup=args.warmup,
        iters=args.iters,
        reps=args.reps,
        world=world,
    )
    a = statistics.median(a1 + a2)
    b = statistics.median(b1 + b2)
    local_only = rankmax_reps(
        local_graph,
        warmup=args.warmup,
        iters=args.iters,
        reps=args.reps,
        world=world,
    )
    local_median = statistics.median(local_only)
    if rank == 0:
        print("correctness=eager_bitwise_exact graph_1000_bitwise_exact", flush=True)
        print(f"A1_rankmax_us={[round(v, 3) for v in a1]}", flush=True)
        print(f"B1_rankmax_us={[round(v, 3) for v in b1]}", flush=True)
        print(f"B2_rankmax_us={[round(v, 3) for v in b2]}", flush=True)
        print(f"A2_rankmax_us={[round(v, 3) for v in a2]}", flush=True)
        print(
            f"A_two_boundaries_us={a:.3f} B_two_boundaries_us={b:.3f} "
            f"saving_us={a - b:.3f} speedup={a / b:.4f} "
            f"gate_55us={'pass' if a - b >= 55.0 else 'fail'}",
            flush=True,
        )
        print(
            f"local_compute_only_two_boundaries_us={local_median:.3f} "
            f"reps={[round(v, 3) for v in local_only]} "
            f"compute_only_saving_us={a - local_median:.3f} "
            f"publication_increment_us={b - local_median:.3f}",
            flush=True,
        )
        print(
            "scope=component_oracle existing_native_mhc_plus_real_aiter_collectives "
            f"incoming_reduce_scatter={'included' if args.include_reduce_scatter else 'not_included'} "
            "production_not_modified",
            flush=True,
        )

    # This pinned AIter build has no exported dispose op; process teardown owns
    # the communicator, matching the existing TP8 standalone oracle.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
