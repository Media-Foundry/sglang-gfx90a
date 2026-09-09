#!/usr/bin/env python3
"""Graph-replay oracle for TP8 C4-indexer logical-ID exchange.

The row-partition proposal gives each rank 16 of an M128 batch's Top-512
logical IDs.  This benchmark measures the exact int32 collective that would
be required to reconstruct all 128 rows on every rank (256 KiB output).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=9)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"requires TP8, got {world}")

    rccl = dist.new_group(backend="nccl")
    custom = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    local = torch.arange(16 * 512, dtype=torch.int32, device="cuda")
    local.add_(rank * local.numel())
    gathered = torch.empty(world * local.numel(), dtype=torch.int32, device="cuda")

    expected = torch.arange(world * local.numel(), dtype=torch.int32, device="cuda")
    backends = [("rccl", lambda: dist.all_gather_into_tensor(gathered, local, group=rccl))]
    if not custom.disabled:
        # The collective is byte-preserving; reinterpret int32 IDs as float32
        # because AIter's public dtype guard excludes integer payloads.
        backends.append(
            ("aiter-f32-bitcast", lambda: custom.all_gather_unreg(
                local.view(torch.float32), out=gathered.view(torch.float32)))
        )
    for backend, op in backends:
        for _ in range(5):
            op()
        torch.cuda.synchronize()
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op()
        if backend.startswith("aiter"):
            custom.register_graph_buffers()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        local_samples = []
        for _ in range(args.reps):
            dist.barrier()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.iters):
                graph.replay()
            end.record()
            end.synchronize()
            local_samples.append(begin.elapsed_time(end) * 1000.0 / args.iters)
        per_rank: list[list[float] | None] = [None] * world
        dist.all_gather_object(per_rank, local_samples)
        if rank == 0:
            rank_max = [max(samples[i] for samples in per_rank if samples is not None)
                        for i in range(args.reps)]
            trimmed = statistics.mean(sorted(rank_max)[1:-1])
            print(json.dumps({
                "backend": backend,
                "world": world,
                "dtype": "int32",
                "local_shape": [16, 512],
                "output_kib": gathered.numel() * gathered.element_size() / 1024,
                "rank_max_median_us": statistics.median(rank_max),
                "rank_max_trimmed_us": trimmed,
                "rank_max_samples_us": rank_max,
            }), flush=True)
        graph.reset()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
