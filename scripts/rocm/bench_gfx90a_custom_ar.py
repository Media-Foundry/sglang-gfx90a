#!/usr/bin/env python3
"""Graph-replay microbenchmark for AIter's intra-node custom all-reduce.

Run with:
  torchrun --standalone --nproc-per-node=8 scripts/rocm/bench_gfx90a_custom_ar.py

The default shapes match DSV4 decode residuals at multi-request tiers.  Rank 0
reports the slowest-rank median, which is the relevant graph critical path.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--legacy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    communicator = CustomAllreduce(dist.group.WORLD, local_rank)
    if communicator.disabled:
        raise RuntimeError("AIter custom all-reduce did not initialize")

    for rows in args.rows:
        inp = torch.full(
            (rows, args.hidden),
            rank + 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        out = torch.empty_like(inp)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            communicator.all_reduce(
                inp, out=out, use_new=not args.legacy, registered=True
            )
        communicator.register_graph_buffers()
        dist.barrier()

        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        expected = world * (world + 1) // 2
        if not torch.equal(out, torch.full_like(out, expected)):
            raise AssertionError(
                f"rank={rank} rows={rows}: incorrect all-reduce output"
            )

        local_reps: list[float] = []
        for _ in range(args.reps):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                graph.replay()
            end.record()
            end.synchronize()
            local_reps.append(start.elapsed_time(end) * 1000.0 / args.iters)

        gathered: list[list[float] | None] = [None] * world
        dist.all_gather_object(gathered, local_reps)
        if rank == 0:
            per_rep_critical = [
                max(gathered[r][rep] for r in range(world))
                for rep in range(args.reps)
            ]
            print(
                f"rows={rows} bytes={inp.numel() * inp.element_size()} "
                f"use_new={not args.legacy} "
                f"critical_us={statistics.median(per_rep_critical):.3f} "
                f"reps={[round(x, 3) for x in per_rep_critical]}",
                flush=True,
            )
        del graph, inp, out
        torch.cuda.empty_cache()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
