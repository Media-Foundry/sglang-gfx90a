#!/usr/bin/env python3
"""TP4 oracle for DSpark target-verify local top-1 reduction.

Run with::

    HIP_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc-per-node=4 \
      scripts/rocm/bench_dsv4_dspark_target_tp_local_top1.py

The baseline gathers the four vocabulary shards and performs the same global
argmax that greedy target verification consumes.  The candidate first finds a
local maximum, reduces only the M scores, then reduces the winning global token
IDs.  The second reduction uses float32 IDs: the DSV4 vocabulary is below
2**24, so every ID and the sentinel are represented exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class Arm:
    name: str
    fn: callable


def _event_time_us(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iterations)


def _trimmed(values: list[float]) -> float:
    ordered = sorted(values)
    core = ordered[1:-1] if len(ordered) >= 5 else ordered
    return statistics.fmean(core)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mutations", type=int, default=100)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"This DSV4 TP oracle requires four ranks, got {world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if args.vocab_size % world:
        raise ValueError("vocab size must divide TP size for this oracle")
    local_vocab = args.vocab_size // world

    gen = torch.Generator(device=device)
    gen.manual_seed(20260901 + rank)
    local = torch.randn(
        (args.rows, local_vocab), dtype=torch.bfloat16, device=device, generator=gen
    )
    gathered = torch.empty(
        (world * args.rows, local_vocab), dtype=local.dtype, device=device
    )
    local_score = torch.empty((args.rows,), dtype=local.dtype, device=device)
    local_index = torch.empty((args.rows,), dtype=torch.int64, device=device)
    global_score = torch.empty_like(local_score)
    global_id_f32 = torch.empty((args.rows,), dtype=torch.float32, device=device)

    def baseline() -> torch.Tensor:
        dist.all_gather_into_tensor(gathered, local)
        full = (
            gathered.view(world, args.rows, local_vocab)
            .permute(1, 0, 2)
            .reshape(args.rows, args.vocab_size)
        )
        return torch.argmax(full, dim=-1)

    def candidate() -> torch.Tensor:
        score, index = torch.max(local, dim=-1)
        local_score.copy_(score)
        local_index.copy_(index)
        global_score.copy_(local_score)
        dist.all_reduce(global_score, op=dist.ReduceOp.MAX)
        winner = local_score == global_score
        global_id_f32.copy_(
            torch.where(
                winner,
                (local_index + rank * local_vocab).to(torch.float32),
                torch.full_like(global_id_f32, float(1 << 24)),
            )
        )
        dist.all_reduce(global_id_f32, op=dist.ReduceOp.MIN)
        return global_id_f32.to(torch.int64)

    mismatches = 0
    for mutation in range(args.mutations):
        local.normal_(generator=gen)
        # Exercise deterministic global tie-breaking every tenth mutation.
        if mutation % 10 == 0:
            local[mutation % args.rows, 7] = torch.tensor(
                20.0, dtype=local.dtype, device=device
            )
        ref = baseline()
        got = candidate()
        mismatches += int(torch.count_nonzero(ref != got).item())
    mismatch_tensor = torch.tensor([mismatches], dtype=torch.int64, device=device)
    dist.all_reduce(mismatch_tensor, op=dist.ReduceOp.SUM)

    samples: dict[str, list[float]] = {"gather_argmax": [], "local_top1": []}
    arms = [Arm("gather_argmax", baseline), Arm("local_top1", candidate)]
    for round_idx in range(args.rounds):
        order = arms if round_idx % 2 == 0 else list(reversed(arms))
        for arm in order:
            local.normal_(generator=gen)
            elapsed = _event_time_us(
                arm.fn, warmup=args.warmup, iterations=args.iterations
            )
            per_rank = [None for _ in range(world)] if rank == 0 else None
            dist.gather_object(elapsed, per_rank, dst=0)
            if rank == 0:
                samples[arm.name].append(max(float(v) for v in per_rank))

    if rank == 0:
        report = {
            "rows": args.rows,
            "vocab_size": args.vocab_size,
            "world_size": world,
            "mutations": args.mutations,
            "mismatches_across_all_ranks": int(mismatch_tensor.item()),
            "samples_us_rank_max": samples,
            "median_us": {k: statistics.median(v) for k, v in samples.items()},
            "trimmed_mean_us": {k: _trimmed(v) for k, v in samples.items()},
        }
        report["trimmed_gain_pct"] = 100.0 * (
            report["trimmed_mean_us"]["gather_argmax"]
            - report["trimmed_mean_us"]["local_top1"]
        ) / report["trimmed_mean_us"]["gather_argmax"]
        print(json.dumps(report, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
