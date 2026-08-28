#!/usr/bin/env python3
"""Build a capacity-preserving Qwen4-Exp EP4 placement for BS32 decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def owners_from_physical(physical: np.ndarray, ep_size: int) -> np.ndarray:
    layers, experts = physical.shape
    per_rank = experts // ep_size
    owners = np.empty((layers, experts), dtype=np.int16)
    for layer in range(layers):
        for position, expert in enumerate(physical[layer]):
            owners[layer, expert] = position // per_rank
    return owners


def mean_max_load(ids: np.ndarray, owners: np.ndarray, ep_size: int) -> float:
    values = []
    for layer in range(ids.shape[1]):
        routed_ranks = owners[layer][ids[:, layer]]
        loads = np.stack(
            [(routed_ranks == rank).sum(axis=(1, 2)) for rank in range(ep_size)],
            axis=1,
        )
        values.append(loads.max(axis=1))
    return float(np.stack(values, axis=1).mean())


def optimize_layer(
    ids: np.ndarray,
    experts: int,
    ep_size: int,
    iterations: int,
    rng: np.random.Generator,
) -> list[int]:
    passes, _, _ = ids.shape
    per_rank = experts // ep_size
    counts = np.zeros((passes, experts), dtype=np.int16)
    for index in range(passes):
        counts[index] = np.bincount(ids[index].ravel(), minlength=experts)
    total = counts.sum(axis=0)

    # Capacity-constrained LPT is a strong deterministic starting point.
    bins: list[list[int]] = [[] for _ in range(ep_size)]
    rank_totals = [0] * ep_size
    for expert in np.argsort(-total, kind="stable"):
        rank = min(
            (r for r in range(ep_size) if len(bins[r]) < per_rank),
            key=lambda r: (rank_totals[r], r),
        )
        bins[rank].append(int(expert))
        rank_totals[rank] += int(total[expert])

    owner = np.empty(experts, dtype=np.int16)
    for rank, bucket in enumerate(bins):
        owner[bucket] = rank
    loads = np.stack(
        [counts[:, owner == rank].sum(axis=1) for rank in range(ep_size)], axis=1
    )
    score = float(loads.max(axis=1).mean())

    # Hot experts dominate stragglers. Swap them with any expert on a peer
    # rank, preserving exactly `experts / ep_size` residents per rank.
    hot = np.argsort(-total)[: experts // 2]
    for _ in range(iterations):
        expert_a = int(hot[rng.integers(len(hot))])
        rank_a = int(owner[expert_a])
        rank_b = int(rng.integers(ep_size))
        if rank_a == rank_b:
            continue
        index_b = int(rng.integers(per_rank))
        expert_b = bins[rank_b][index_b]
        trial = loads.copy()
        trial[:, rank_a] += counts[:, expert_b] - counts[:, expert_a]
        trial[:, rank_b] += counts[:, expert_a] - counts[:, expert_b]
        trial_score = float(trial.max(axis=1).mean())
        if trial_score + 1.0e-9 >= score:
            continue
        index_a = bins[rank_a].index(expert_a)
        bins[rank_a][index_a] = expert_b
        bins[rank_b][index_b] = expert_a
        owner[expert_a] = rank_b
        owner[expert_b] = rank_a
        loads = trial
        score = trial_score

    return [expert for bucket in bins for expert in bucket]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    records = [record for record in payload["records"] if record["forward_mode"] == 2]
    ids = torch.stack([record["topk_ids_of_layer"] for record in records]).numpy()
    old_physical = np.asarray(payload["last_physical_to_logical_map"])
    if ids.ndim != 4 or ids.shape[2:] != (32, 10):
        raise RuntimeError(f"expected [passes,layers,32,10], got {ids.shape}")
    if old_physical.shape[0] != ids.shape[1]:
        raise RuntimeError("recorder layer count does not match placement")
    experts = old_physical.shape[1]
    if experts % args.ep_size:
        raise RuntimeError("expert count must be divisible by EP size")
    if ids.min() < 0 or ids.max() >= experts:
        raise RuntimeError("recorder contains invalid logical expert IDs")

    rng = np.random.default_rng(args.seed)
    optimized = np.asarray(
        [
            optimize_layer(
                ids[:, layer], experts, args.ep_size, args.iterations, rng
            )
            for layer in range(ids.shape[1])
        ]
    )
    for layer in optimized:
        if sorted(layer.tolist()) != list(range(experts)):
            raise RuntimeError("optimized placement is not a permutation")

    old_score = mean_max_load(
        ids, owners_from_physical(old_physical, args.ep_size), args.ep_size
    )
    new_score = mean_max_load(
        ids, owners_from_physical(optimized, args.ep_size), args.ep_size
    )
    output = Path(args.output)
    output.write_text(
        json.dumps({"physical_to_logical_map": optimized.tolist()}, separators=(",", ":"))
        + "\n"
    )
    print(
        json.dumps(
            {
                "passes": ids.shape[0],
                "layers": ids.shape[1],
                "old_mean_max_rank_assignments": old_score,
                "new_mean_max_rank_assignments": new_score,
                "relative_reduction": 1.0 - new_score / old_score,
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
