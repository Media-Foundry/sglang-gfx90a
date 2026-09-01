#!/usr/bin/env python3
"""Exact TP4 M128 draft-early/anchor-late all-reduce oracle."""

from __future__ import annotations

import argparse
import os
import statistics

import aiter as aiter_ops
import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

from sglang.kernels.ops.communication.gfx90a_tp4_m128_progressive_ar_oracle import (
    _jit_module,
    arm,
    progressive,
    wait_draft,
)


ROWS, HIDDEN, WORLD = 128, 4096, 4
BLOCKS = 12
WORKSPACE_U32 = BLOCKS + 2 * BLOCKS * WORLD + WORLD + 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--replays", type=int, default=1000)
    return parser.parse_args()


def capture(comm, fn):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture():
        with torch.cuda.graph(graph):
            outputs = fn()
    dist.barrier()
    return graph, outputs


def rankmax_once(graph, iters, world):
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    local = start.elapsed_time(end) * 1000.0 / iters
    values = [None] * world
    dist.all_gather_object(values, local)
    return max(float(v) for v in values)


def progressive_pair_once(comm, inp, sync, candidate, rank, side):
    main = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    draft_ready = torch.cuda.Event(enable_timing=True)
    full_done = torch.cuda.Event(enable_timing=True)
    start.record(main)
    side.wait_event(start)
    with torch.cuda.stream(side):
        wait_draft(sync)
        draft_ready.record(side)
    progressive(comm._ptr, inp, sync, candidate, rank)
    full_done.record(main)
    full_done.synchronize()
    draft_ready.synchronize()
    return (
        start.elapsed_time(draft_ready) * 1000.0,
        start.elapsed_time(full_done) * 1000.0,
    )


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != WORLD:
        raise RuntimeError(f"requires TP4, got {world}")
    arch = torch.cuda.get_device_properties(local_rank).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"requires gfx90a, got {arch}")

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter CustomAllreduce did not initialize")

    input_bytes = ROWS * HIDDEN * torch.bfloat16.itemsize
    input_storage = aiter_ops.allocate_meta_buffer(input_bytes)
    inp = input_storage.view(torch.bfloat16).view(ROWS, HIDDEN)
    sync = aiter_ops.allocate_meta_buffer(WORKSPACE_U32 * 4)
    sync.zero_()
    comm.register_buffer(input_storage)
    comm.register_buffer(sync)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(9000 + rank)
    base = torch.randn(
        (ROWS, HIDDEN), generator=generator, device="cuda", dtype=torch.bfloat16
    ).mul_(0.0625)
    mutation = torch.linspace(
        -1.0, 1.0, inp.numel(), device="cuda", dtype=torch.float32
    ).view_as(inp).to(torch.bfloat16)
    inp.copy_(base)
    reference = torch.empty_like(inp)
    candidate = torch.empty_like(inp)
    draft_snapshot = torch.empty((32, 3, HIDDEN), dtype=inp.dtype, device="cuda")
    side = torch.cuda.Stream()

    _jit_module()
    comm.all_reduce(inp, out=reference, registered=True)
    with torch.cuda.stream(side):
        wait_draft(sync)
        draft_snapshot.copy_(candidate.view(32, 4, HIDDEN)[:, 1:])
    progressive(comm._ptr, inp, sync, candidate, rank)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    sync.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    def reference_fn():
        comm.all_reduce(inp, out=reference, registered=True)
        return reference

    def candidate_fn():
        main = torch.cuda.current_stream()
        fork = torch.cuda.Event()
        done = torch.cuda.Event()
        fork.record(main)
        side.wait_event(fork)
        with torch.cuda.stream(side):
            wait_draft(sync)
            draft_snapshot.copy_(candidate.view(32, 4, HIDDEN)[:, 1:])
            done.record(side)
        progressive(comm._ptr, inp, sync, candidate, rank)
        main.wait_event(done)
        return candidate, draft_snapshot

    def progressive_only_fn():
        arm(sync)
        progressive(comm._ptr, inp, sync, candidate, rank)
        return candidate

    graph_a, output_a = capture(comm, reference_fn)
    graph_b, outputs_b = capture(comm, candidate_fn)
    graph_p, output_p = capture(comm, progressive_only_fn)
    sync.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    mismatch_mutations = 0
    max_abs = 0.0
    for iteration in range(args.mutations):
        inp.copy_(base)
        inp.add_(mutation, alpha=((iteration * 1543 + 17) % 2047 - 1023) / 32768.0)
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        expected_draft = output_a.view(32, 4, HIDDEN)[:, 1:]
        exact = torch.equal(output_a, outputs_b[0]) and torch.equal(
            expected_draft, outputs_b[1]
        )
        if not exact:
            mismatch_mutations += 1
        max_abs = max(
            max_abs,
            float((output_a.float() - outputs_b[0].float()).abs().max().item()),
            float((expected_draft.float() - outputs_b[1].float()).abs().max().item()),
        )

    inp.copy_(base)
    graph_a.replay()
    torch.cuda.synchronize()
    mismatch_replays = 0
    for _ in range(args.replays):
        graph_b.replay()
        torch.cuda.synchronize()
        if not torch.equal(output_a, outputs_b[0]):
            mismatch_replays += 1

    for _ in range(args.warmup):
        graph_a.replay()
        graph_b.replay()
        graph_p.replay()
    torch.cuda.synchronize()
    a1, b1, b2, a2 = [], [], [], []
    for _ in range(args.rounds):
        a1.append(rankmax_once(graph_a, args.iters, world))
        b1.append(rankmax_once(graph_b, args.iters, world))
        b2.append(rankmax_once(graph_b, args.iters, world))
        a2.append(rankmax_once(graph_a, args.iters, world))

    p_only = [rankmax_once(graph_p, args.iters, world) for _ in range(args.rounds)]
    pair_local = [
        progressive_pair_once(comm, inp, sync, candidate, rank, side)
        for _ in range(max(20, args.iters))
    ]
    pair_reports = [None] * world
    dist.all_gather_object(pair_reports, pair_local)

    reports = [None] * world
    dist.all_gather_object(
        reports, (mismatch_mutations, mismatch_replays, max_abs)
    )
    if rank == 0:
        a = a1 + a2
        b = b1 + b2
        print(f"mutation_reports={reports}", flush=True)
        print(f"A1_rankmax_us={[round(v, 3) for v in a1]}", flush=True)
        print(f"B1_rankmax_us={[round(v, 3) for v in b1]}", flush=True)
        print(f"B2_rankmax_us={[round(v, 3) for v in b2]}", flush=True)
        print(f"A2_rankmax_us={[round(v, 3) for v in a2]}", flush=True)
        print(
            f"P_progressive_only_rankmax_us={[round(v, 3) for v in p_only]}",
            flush=True,
        )
        draft_rankmax = [
            max(float(pair_reports[r][i][0]) for r in range(world))
            for i in range(len(pair_local))
        ]
        full_rankmax = [
            max(float(pair_reports[r][i][1]) for r in range(world))
            for i in range(len(pair_local))
        ]
        print(
            f"eager_draft_ready_us_median={statistics.median(draft_rankmax):.3f} "
            f"eager_full_progressive_us_median={statistics.median(full_rankmax):.3f} "
            f"samples={len(draft_rankmax)}",
            flush=True,
        )
        print(
            f"A_production_ar_us={statistics.median(a):.3f} "
            f"B_progressive_plus_draft_copy_us={statistics.median(b):.3f} "
            f"delta_us={statistics.median(a)-statistics.median(b):.3f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
