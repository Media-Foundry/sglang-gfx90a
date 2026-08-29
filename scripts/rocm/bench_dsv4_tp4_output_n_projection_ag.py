#!/usr/bin/env python3
"""Standalone TP4 C4 output-N projection + registered-AG oracle."""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


PROJECTIONS = (
    ("wqkv_a", 1536),
    ("core_compressor", 2048),
    ("index_compressor", 512),
    ("index_weights", 64),
)
HIDDEN = 4096
TOTAL_N = 4160
HYBRID_N = 1536 + 2048


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=Path("/tmp/dsv4-layer20-m32"))
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--mutations", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--mode", choices=("full", "hybrid"), default="full")
    return parser.parse_args()


def load(case):
    prefix = f"layer_{case.layer}"
    x = torch.load(case.dump_dir / f"{prefix}_attn_norm.pt", weights_only=True)
    weights = [
        torch.load(
            case.dump_dir / f"{prefix}_projection_{name}.pt", weights_only=True
        )
        for name, _ in PROJECTIONS
    ]
    assert x.shape == (32, HIDDEN) and x.dtype == torch.bfloat16
    for weight, (_, width) in zip(weights, PROJECTIONS, strict=True):
        assert weight.shape == (width, HIDDEN) and weight.dtype == torch.bfloat16
    return x.contiguous(), [weight.contiguous() for weight in weights]


def reference(x, weights):
    return torch.cat([F.linear(x, weight) for weight in weights], dim=1).contiguous()


def candidate(x, local_weight, comm, raw_gather, gathered_n, tail_weights):
    local = F.linear(x, local_weight)
    comm.all_gather_reg(local, out=raw_gather)
    gathered = (
        raw_gather.view(comm.world_size, x.shape[0], local.shape[1])
        .movedim(0, 1)
        .contiguous()
        .view(x.shape[0], gathered_n)
    )
    if tail_weights:
        return torch.cat(
            [gathered] + [F.linear(x, weight) for weight in tail_weights], dim=1
        ).contiguous()
    return gathered


def capture(fn, comm=None):
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    if comm is None:
        with torch.cuda.graph(graph):
            output = fn()
    else:
        with comm.capture():
            with torch.cuda.graph(graph):
                output = fn()
    dist.barrier()
    return graph, output


def rankmax(graph, warmup, iters, reps, world):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    local = []
    for _ in range(reps):
        dist.barrier()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        local.append(begin.elapsed_time(end) * 1000.0 / iters)
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    values = [max(row[i] for row in gathered) for i in range(reps)]
    return statistics.median(values), values


def main():
    case = args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"requires TP4, got world={world}")

    x_cpu, weights_cpu = load(case)
    x = x_cpu.cuda()
    base = x.clone()
    mutation = torch.linspace(-1, 1, x.numel(), device="cuda").reshape_as(x).to(
        torch.bfloat16
    )
    weights = [value.cuda() for value in weights_cpu]
    gathered_n = HYBRID_N if case.mode == "hybrid" else TOTAL_N
    gathered_weights = weights[:2] if case.mode == "hybrid" else weights
    tail_weights = weights[2:] if case.mode == "hybrid" else []
    combined = torch.cat(gathered_weights, dim=0).contiguous()
    shard_n = gathered_n // world
    lo, hi = rank * shard_n, (rank + 1) * shard_n
    local_weight = combined[lo:hi].contiguous()

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", rank))
    if comm.disabled:
        raise RuntimeError("AIter registered AG unavailable")
    raw_gather = torch.empty(world * x.shape[0] * shard_n, dtype=x.dtype, device="cuda")
    _ = reference(x, weights)
    local_warmup = F.linear(x, local_weight)
    comm.all_gather_unreg(local_warmup, out=raw_gather)
    torch.cuda.synchronize()
    graph_a, out_a = capture(lambda: reference(x, weights))
    graph_b, out_b = capture(
        lambda: candidate(
            x, local_weight, comm, raw_gather, gathered_n, tail_weights
        ),
        comm,
    )

    graph_a.replay()
    graph_b.replay()
    torch.cuda.synchronize()
    local_reference = F.linear(x, local_weight)
    torch.cuda.synchronize()

    # Consumer contracts are the actual four production boundaries. This also
    # catches a rank-major reorder that is globally shaped but segment-wrong.
    widths = [width for _, width in PROJECTIONS]
    initial_segment_exact = [
        torch.equal(a, b)
        for a, b in zip(torch.split(out_a, widths, 1), torch.split(out_b, widths, 1), strict=True)
    ]
    local_ag_slice_exact = torch.equal(local_reference, out_b[:, :gathered_n][:, lo:hi])
    mismatches = 0
    max_abs = 0.0
    max_rel_l2 = 0.0
    segment_mismatches = [0] * len(PROJECTIONS)
    for iteration in range(case.mutations):
        state = (iteration * 1543 + 17) % 2047
        alpha = (state - 1023) / 32768.0
        x.copy_(base).add_(mutation, alpha=alpha)
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        if not torch.equal(out_a, out_b):
            mismatches += 1
            delta = out_a.float() - out_b.float()
            max_abs = max(max_abs, float(delta.abs().max()))
            max_rel_l2 = max(
                max_rel_l2,
                float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(out_a.float())),
            )
        for index, (a, b) in enumerate(
            zip(torch.split(out_a, widths, 1), torch.split(out_b, widths, 1), strict=True)
        ):
            segment_mismatches[index] += int(not torch.equal(a, b))

    correctness = {
        "rank": rank,
        "initial_segment_exact": initial_segment_exact,
        "local_ag_slice_exact": local_ag_slice_exact,
        "mismatches": mismatches,
        "segment_mismatches": segment_mismatches,
        "max_abs": max_abs,
        "max_rel_l2": max_rel_l2,
    }
    all_correctness = [None] * world
    dist.all_gather_object(all_correctness, correctness)
    if rank == 0:
        print(f"correctness={all_correctness}", flush=True)

    x.copy_(base)
    torch.cuda.synchronize()
    timings = []
    for slot, (name, graph) in enumerate(
        (("A", graph_a), ("B", graph_b), ("B", graph_b), ("A", graph_a)), 1
    ):
        median, values = rankmax(graph, case.warmup, case.iters, case.reps, world)
        timings.append((name, median))
        if rank == 0:
            print(f"slot={slot} {name} median_us={median:.3f} samples={values}", flush=True)
    a_us = statistics.mean(value for name, value in timings if name == "A")
    b_us = statistics.mean(value for name, value in timings if name == "B")
    exact = all(item["mismatches"] == 0 for item in all_correctness)
    if rank == 0:
        print(
            f"mode={case.mode} ABBA A_us={a_us:.3f} B_us={b_us:.3f} saved_us={a_us-b_us:.3f} "
            f"exact={exact} continue_gate={(a_us-b_us)>=30.0 and exact}",
            flush=True,
        )
    comm.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
