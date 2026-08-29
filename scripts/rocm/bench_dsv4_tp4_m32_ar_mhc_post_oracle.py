#!/usr/bin/env python3
"""Standalone TP4/M32 all-reduce + MHC-post fusion oracle.

This experiment does not alter the production forward path.  It compares:

  A: AIter's real registered BF16 all-reduce, followed by the production
     gfx90a Triton MHC post-combine + RMS-partial kernel;
  B: one HIP kernel which peer-reads the four rank inputs in fixed rank order,
     rounds the reduction to BF16, evaluates MHC post, emits the same 64 RMS
     partials per row, and performs a mandatory system-scope exit handshake.

Run only on four otherwise idle GCDs::

  HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    scripts/rocm/bench_dsv4_tp4_m32_ar_mhc_post_oracle.py
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import aiter as aiter_ops

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from sglang.kernels.ops.layernorm.gfx90a_tp4_m32_ar_mhc_post_oracle import (
    _jit_module,
    run as fused_candidate,
)
from sglang.kernels.ops.layernorm.mhc import mhc_post_combine_rms_triton


ROWS = 32
HIDDEN = 4096
CHANNELS = 4
BLOCKS = 64
WORLD = 4
WORKSPACE_U32 = BLOCKS + 2 * BLOCKS * WORLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--require-gain-us", type=float, default=20.0)
    return parser.parse_args()


def randn(shape, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device="cuda", dtype=dtype)


def capture(comm: CustomAllreduce, fn):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture():
        with torch.cuda.graph(graph):
            outputs = fn()
    dist.barrier()
    return graph, outputs


def rankmax_once(graph: torch.cuda.CUDAGraph, iters: int, world: int) -> float:
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) * 1000.0 / iters
    gathered: list[float | None] = [None] * world
    dist.all_gather_object(gathered, local_us)
    return max(float(value) for value in gathered)


def exact_report(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[bool, float]:
    return torch.equal(lhs, rhs), float(
        (lhs.float() - rhs.float()).abs().max().item()
    )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != WORLD:
        raise RuntimeError(f"requires TP4, got world={world}")
    arch = torch.cuda.get_device_properties(local_rank).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"requires gfx90a, got {arch}")

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter CustomAllreduce did not initialize")

    # Direct HIP allocations have zero IPC offset and avoid the caching
    # allocator suballocation ambiguity in this pinned AIter build.
    input_bytes = ROWS * HIDDEN * torch.bfloat16.itemsize
    input_storage = aiter_ops.allocate_meta_buffer(input_bytes)
    partial = input_storage.view(torch.bfloat16).view(ROWS, HIDDEN)
    sync_workspace = aiter_ops.allocate_meta_buffer(WORKSPACE_U32 * 4)
    sync_workspace.zero_()
    comm.register_buffer(input_storage)
    comm.register_buffer(sync_workspace)

    base = randn((ROWS, HIDDEN), 1000 + rank, torch.bfloat16).mul_(0.0625)
    partial.copy_(base)
    mutation = torch.linspace(
        -1.0, 1.0, partial.numel(), device="cuda", dtype=torch.float32
    ).view_as(partial).to(torch.bfloat16)
    residual = randn((ROWS, CHANNELS, HIDDEN), 2000, torch.bfloat16).mul_(0.125)
    post = randn((ROWS, CHANNELS), 3000, torch.float32).mul_(0.1)
    comb = randn((ROWS, CHANNELS, CHANNELS), 4000, torch.float32).mul_(0.1)

    reduced_a = torch.empty_like(partial)
    out_b = torch.empty_like(residual)
    rms_b = torch.empty((ROWS, 64), dtype=torch.float32, device="cuda")
    reduced_debug = torch.empty_like(partial)
    out_debug = torch.empty_like(residual)
    rms_debug = torch.empty_like(rms_b)

    # Compile and initialize both backends before graph capture.
    _jit_module()
    comm.all_reduce(partial, out=reduced_a, registered=True)
    warm = mhc_post_combine_rms_triton(reduced_a, residual, post, comb)
    if warm is None:
        raise RuntimeError("production MHC post backend rejected M32")
    fused_candidate(
        comm._ptr,
        partial,
        sync_workspace,
        residual,
        post,
        comb,
        out_debug,
        rms_debug,
        reduced_debug,
        rank,
        write_reduced=True,
    )
    torch.cuda.synchronize()
    dist.barrier()

    def reference():
        comm.all_reduce(partial, out=reduced_a, registered=True)
        result = mhc_post_combine_rms_triton(reduced_a, residual, post, comb)
        assert result is not None
        return reduced_a, result[0], result[1]

    def candidate():
        fused_candidate(
            comm._ptr,
            partial,
            sync_workspace,
            residual,
            post,
            comb,
            out_b,
            rms_b,
            reduced_debug,
            rank,
        )
        return out_b, rms_b

    def candidate_debug():
        fused_candidate(
            comm._ptr,
            partial,
            sync_workspace,
            residual,
            post,
            comb,
            out_debug,
            rms_debug,
            reduced_debug,
            rank,
            write_reduced=True,
        )
        return reduced_debug, out_debug, rms_debug

    graph_a, outputs_a = capture(comm, reference)
    graph_b, outputs_b = capture(comm, candidate)
    graph_d, outputs_d = capture(comm, candidate_debug)
    graph_ar, _ = capture(
        comm, lambda: comm.all_reduce(partial, out=reduced_a, registered=True)
    )
    graph_mhc, _ = capture(
        comm,
        lambda: mhc_post_combine_rms_triton(
            reduced_a, residual, post, comb
        ),
    )

    graph_a.replay()
    graph_b.replay()
    graph_d.replay()
    torch.cuda.synchronize()
    initial = [
        exact_report(outputs_a[0], outputs_d[0]),
        exact_report(outputs_a[1], outputs_b[0]),
        exact_report(outputs_a[2], outputs_b[1]),
        exact_report(outputs_b[0], outputs_d[1]),
        exact_report(outputs_b[1], outputs_d[2]),
    ]

    mutation_mismatches = 0
    mutation_max = [0.0, 0.0, 0.0]
    first_mutation = -1
    for iteration in range(args.mutations):
        state = (iteration * 1543 + 17) % 2047
        alpha = (state - 1023) / 32768.0
        partial.copy_(base)
        partial.add_(mutation, alpha=alpha)
        graph_a.replay()
        graph_b.replay()
        graph_d.replay()
        torch.cuda.synchronize()
        checks = (
            exact_report(outputs_a[0], outputs_d[0]),
            exact_report(outputs_a[1], outputs_b[0]),
            exact_report(outputs_a[2], outputs_b[1]),
        )
        if not all(check[0] for check in checks):
            if first_mutation < 0:
                first_mutation = iteration
            mutation_mismatches += 1
        for index, check in enumerate(checks):
            mutation_max[index] = max(mutation_max[index], check[1])

    partial.copy_(base)
    graph_a.replay()
    torch.cuda.synchronize()
    replay_mismatches = 0
    replay_first = -1
    replay_max = [0.0, 0.0]
    for iteration in range(args.replays):
        graph_b.replay()
        torch.cuda.synchronize()
        checks = (
            exact_report(outputs_a[1], outputs_b[0]),
            exact_report(outputs_a[2], outputs_b[1]),
        )
        if not all(check[0] for check in checks):
            if replay_first < 0:
                replay_first = iteration
            replay_mismatches += 1
        for index, check in enumerate(checks):
            replay_max[index] = max(replay_max[index], check[1])

    for _ in range(args.warmup):
        graph_a.replay()
        graph_b.replay()
    torch.cuda.synchronize()
    a1: list[float] = []
    b1: list[float] = []
    b2: list[float] = []
    a2: list[float] = []
    for _ in range(args.reps):
        a1.append(rankmax_once(graph_a, args.iters, world))
        b1.append(rankmax_once(graph_b, args.iters, world))
        b2.append(rankmax_once(graph_b, args.iters, world))
        a2.append(rankmax_once(graph_a, args.iters, world))
    ar_only = [rankmax_once(graph_ar, args.iters, world) for _ in range(args.reps)]
    mhc_only = [rankmax_once(graph_mhc, args.iters, world) for _ in range(args.reps)]

    all_initial: list[list[tuple[bool, float]] | None] = [None] * world
    all_mutations: list[tuple[int, int, list[float]] | None] = [None] * world
    all_replays: list[tuple[int, int, list[float]] | None] = [None] * world
    dist.all_gather_object(all_initial, initial)
    dist.all_gather_object(
        all_mutations, (mutation_mismatches, first_mutation, mutation_max)
    )
    dist.all_gather_object(
        all_replays, (replay_mismatches, replay_first, replay_max)
    )

    if rank == 0:
        a_samples = a1 + a2
        b_samples = b1 + b2
        a_median = statistics.median(a_samples)
        b_median = statistics.median(b_samples)
        saving = a_median - b_median
        exact_initial = all(
            check[0] for rank_checks in all_initial for check in rank_checks
        )
        total_mutation_mismatches = sum(item[0] for item in all_mutations)
        total_replay_mismatches = sum(item[0] for item in all_replays)
        print(f"initial_exact={exact_initial} per_rank={all_initial}", flush=True)
        print(
            f"mutations={args.mutations} rank_mismatches={total_mutation_mismatches} "
            f"details={all_mutations}",
            flush=True,
        )
        print(
            f"graph_replays={args.replays} rank_mismatches={total_replay_mismatches} "
            f"details={all_replays}",
            flush=True,
        )
        print(f"A1_rankmax_us={[round(v, 3) for v in a1]}", flush=True)
        print(f"B1_rankmax_us={[round(v, 3) for v in b1]}", flush=True)
        print(f"B2_rankmax_us={[round(v, 3) for v in b2]}", flush=True)
        print(f"A2_rankmax_us={[round(v, 3) for v in a2]}", flush=True)
        print(
            f"AR_only_rankmax_us={[round(v, 3) for v in ar_only]} "
            f"MHC_only_rankmax_us={[round(v, 3) for v in mhc_only]}",
            flush=True,
        )
        print(
            f"A_ar_plus_production_mhc_us={a_median:.3f} "
            f"B_fused_peer_ar_mhc_us={b_median:.3f} saving_us={saving:.3f} "
            f"speedup={a_median / b_median:.4f} "
            f"gate_{args.require_gain_us:g}us="
            f"{'pass' if saving >= args.require_gain_us else 'fail'}",
            flush=True,
        )
        print(
            "scope=standalone_component_oracle production_not_modified "
            "candidate_has_system_scope_entry_and_exit_epoch_barriers",
            flush=True,
        )

    # Explicit disposal can race process teardown in some pinned builds; keep
    # the lifetime identical to the repository's existing standalone oracles.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
