#!/usr/bin/env python3
"""TP8 graph-replay budget for the proposed hidden-sharded DSV4 path.

This is a standalone diagnostic and is not imported by the serving path.  It
compares AIter peer-read collectives with RCCL for the exact M=32 tensor sizes
needed by the reduce-scatter/sharded-MHC proposal.

Run on an otherwise idle eight-GCD host with the DS environment::

  /home/pc/anaconda3/envs/DS/bin/torchrun --standalone --nproc-per-node=8 \
    scripts/rocm/bench_dsv4_tp8_collective_budget.py
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: torch.dtype


CASES = (
    Case("rs_256k_bf16", "rs", (32, 4096), torch.bfloat16),
    Case("ar_1_fp32", "ar", (32, 1), torch.float32),
    Case("ar_25_fp32", "ar", (32, 25), torch.float32),
    Case("ar_1536_bf16", "ar", (32, 1536), torch.bfloat16),
    Case("ar_1537_bf16", "ar", (32, 1537), torch.bfloat16),
    Case("ar_1537_fp32", "ar", (32, 1537), torch.float32),
    Case("ar_2560_bf16", "ar", (32, 2560), torch.bfloat16),
    Case("ar_4160_bf16", "ar", (32, 4160), torch.bfloat16),
    Case("ag_32k_bf16", "ag", (32, 512), torch.bfloat16),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--cases", nargs="+", choices=[case.name for case in CASES])
    p.add_argument("--backends", nargs="+", choices=("aiter", "rccl"))
    return p.parse_args()


def make_io(case: Case, rank: int, world: int):
    value = float(rank + 1)
    inp = torch.full(case.shape, value, dtype=case.dtype, device="cuda")
    if case.kind == "rs":
        assert inp.numel() % world == 0
        out = torch.empty(inp.numel() // world, dtype=case.dtype, device="cuda")
        expected = torch.full_like(out, world * (world + 1) / 2)
    elif case.kind == "ar":
        out = torch.empty_like(inp)
        expected = torch.full_like(out, world * (world + 1) / 2)
    else:
        out = torch.empty(world * inp.numel(), dtype=case.dtype, device="cuda")
        expected = torch.cat(
            [torch.full_like(inp.flatten(), float(r + 1)) for r in range(world)]
        )
    return inp, out, expected


def capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        fn()
    dist.barrier()
    return graph


def critical_us(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iters: int,
    reps: int,
    world: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    local = []
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
    gathered: list[list[float] | None] = [None] * world
    dist.all_gather_object(gathered, local)
    rank_max = [max(x[i] for x in gathered if x is not None) for i in range(reps)]
    return statistics.median(rank_max), rank_max


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"TP8 benchmark requires world=8, got {world}")
    rccl = dist.new_group(backend="nccl")
    custom = CustomAllreduce(dist.group.WORLD, local_rank)
    if custom.disabled:
        raise RuntimeError("AIter custom collectives did not initialize")

    # Initialize RCCL before graph capture.
    warm = torch.ones(16, dtype=torch.float32, device="cuda")
    dist.all_reduce(warm, group=rccl)
    torch.cuda.synchronize()

    selected_cases = [case for case in CASES if not args.cases or case.name in args.cases]
    selected_backends = args.backends or ("aiter", "rccl")
    for case in selected_cases:
        for backend in selected_backends:
            inp, out, expected = make_io(case, rank, world)
            if backend == "aiter":
                if case.kind == "rs":
                    op = lambda: custom.reduce_scatter(inp, out, registered=True)
                elif case.kind == "ar":
                    op = lambda: custom.all_reduce(inp, out=out, registered=True)
                else:
                    op = lambda: custom.all_gather_reg(inp, out=out)
            else:
                if case.kind == "rs":
                    op = lambda: dist.reduce_scatter_tensor(out, inp, group=rccl)
                elif case.kind == "ar":
                    op = lambda: dist.all_reduce(inp, group=rccl)
                else:
                    op = lambda: dist.all_gather_into_tensor(out, inp, group=rccl)

            graph = capture(op)
            if backend == "aiter":
                custom.register_graph_buffers()
            if backend == "rccl" and case.kind == "ar":
                # RCCL's public all-reduce is in-place.  Reset once for the
                # correctness replay; repeated timing replays may accumulate,
                # but execute the identical communication graph.
                inp.fill_(float(rank + 1))
                graph.replay()
            else:
                for _ in range(5):
                    graph.replay()
            torch.cuda.synchronize()
            actual = inp if backend == "rccl" and case.kind == "ar" else out
            torch.testing.assert_close(actual.flatten(), expected.flatten(), rtol=0, atol=0)
            median, reps = critical_us(
                graph,
                warmup=args.warmup,
                iters=args.iters,
                reps=args.reps,
                world=world,
            )
            if rank == 0:
                in_bytes = inp.numel() * inp.element_size()
                out_bytes = out.numel() * out.element_size()
                print(
                    f"case={case.name} kind={case.kind} backend={backend} "
                    f"input_bytes={in_bytes} output_bytes={out_bytes} "
                    f"rank_max_median_us={median:.3f} "
                    f"rank_max_reps={[round(x, 3) for x in reps]}",
                    flush=True,
                )
            del graph, inp, out, expected
            torch.cuda.empty_cache()
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
