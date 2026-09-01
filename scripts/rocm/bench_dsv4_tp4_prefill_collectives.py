#!/usr/bin/env python3
"""Graph-replay sweep for TP4 DSV4 prefill row collectives.

Run on an idle four-GCD group with the DS environment::

  HIP_VISIBLE_DEVICES=4,5,6,7 \
    /home/pc/anaconda3/envs/DS/bin/torchrun --standalone --nproc-per-node=4 \
    scripts/rocm/bench_dsv4_tp4_prefill_collectives.py
"""

from __future__ import annotations

import argparse
import os
import statistics
from typing import Callable

import torch
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


H = 4096
WORLD = 4
ROWS = (1024, 1536, 2048, 2304, 2560, 3072, 4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=ROWS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--reps", type=int, default=7)
    return parser.parse_args()


def capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        fn()
    dist.barrier()
    return graph


def rank_max_us(graph: torch.cuda.CUDAGraph, args, world: int) -> tuple[float, list[float]]:
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    local = []
    for _ in range(args.reps):
        dist.barrier()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.iters):
            graph.replay()
        end.record()
        end.synchronize()
        local.append(begin.elapsed_time(end) * 1000.0 / args.iters)
    gathered: list[list[float] | None] = [None] * world
    dist.all_gather_object(gathered, local)
    critical = [max(row[i] for row in gathered if row is not None) for i in range(args.reps)]
    return statistics.median(critical), critical


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != WORLD:
        raise RuntimeError(f"requires TP4, got {world}")
    rccl = dist.new_group(backend="nccl")
    custom = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if custom.disabled:
        raise RuntimeError("AIter custom collectives did not initialize")
    warm = torch.ones(16, dtype=torch.float32, device="cuda")
    dist.all_reduce(warm, group=rccl)
    torch.cuda.synchronize()

    for rows in args.rows:
        if rows % world:
            raise ValueError(f"rows must divide TP4, got {rows}")
        for kind in ("rs", "ag"):
            local_rows = rows // world
            for backend in ("aiter", "rccl"):
                value = float(rank + 1)
                if kind == "rs":
                    inp = torch.full((rows, H), value, dtype=torch.bfloat16, device="cuda")
                    out = torch.empty((local_rows, H), dtype=torch.bfloat16, device="cuda")
                    expected = torch.full_like(out, world * (world + 1) / 2)
                    op = (
                        (lambda: custom.reduce_scatter(inp, out, registered=False))
                        if backend == "aiter"
                        else (lambda: dist.reduce_scatter_tensor(out, inp, group=rccl))
                    )
                else:
                    inp = torch.full((local_rows, H), value, dtype=torch.bfloat16, device="cuda")
                    out = torch.empty((rows, H), dtype=torch.bfloat16, device="cuda")
                    expected = torch.cat(
                        [torch.full_like(inp, float(r + 1)) for r in range(world)]
                    )
                    op = (
                        (lambda: custom.all_gather_unreg(inp, out=out))
                        if backend == "aiter"
                        else (lambda: dist.all_gather_into_tensor(out, inp, group=rccl))
                    )
                graph = capture(op)
                if backend == "aiter":
                    custom.register_graph_buffers()
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                median, reps = rank_max_us(graph, args, world)
                if rank == 0:
                    print(
                        f"rows={rows} kind={kind} backend={backend} "
                        f"payload_mib={rows * H * 2 / 2**20:.1f} "
                        f"rank_max_median_us={median:.3f} "
                        f"reps={[round(x, 3) for x in reps]}",
                        flush=True,
                    )
                del graph, inp, out, expected
                torch.cuda.empty_cache()
                dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
