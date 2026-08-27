#!/usr/bin/env python3
"""TP8 output-N-sharded C4 projection + AIter all-gather oracle.

This diagnostic is intentionally independent of the serving selector.  It uses
the real layer-20 M32 dump and compares:

* A: the four production-shaped BF16 ``F.linear`` calls, concatenated as
  ``[M, 4160]``;
* B: one BF16 ``F.linear`` per rank with a contiguous 520-row output-weight
  shard, followed by AIter registered all-gather and a rank-major-to-output-N
  reorder.

The installed AIter version exposes the original rank-major all-gather API, so
the reorder is part of B's captured graph and timing.  No production module or
AIter source is modified by this script.

Example::

  /home/pc/anaconda3/envs/DS/bin/torchrun --standalone --nproc-per-node=8 \
    scripts/rocm/bench_dsv4_tp8_output_n_projection_ag.py \
    --dump-dir /tmp/dsv4-layer20-m32
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
from pathlib import Path
from typing import Callable

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
TOTAL_OUTPUT = sum(width for _, width in PROJECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-dir", type=Path, default=Path("/tmp/dsv4-layer20-m32")
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--mutations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument(
        "--allow-nonexact",
        action="store_true",
        help="Report strict mismatches without returning a failing exit status.",
    )
    return parser.parse_args()


def load_dump(args: argparse.Namespace) -> tuple[torch.Tensor, list[torch.Tensor]]:
    prefix = f"layer_{args.layer}"
    x = torch.load(
        args.dump_dir / f"{prefix}_attn_norm.pt",
        map_location="cpu",
        weights_only=True,
    )
    weights = [
        torch.load(
            args.dump_dir / f"{prefix}_projection_{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for name, _ in PROJECTIONS
    ]
    if x.ndim != 2 or x.shape[1] != HIDDEN or x.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 [M,{HIDDEN}] activation, got {x.shape} {x.dtype}")
    for (name, width), weight in zip(PROJECTIONS, weights, strict=True):
        if weight.shape != (width, HIDDEN) or weight.dtype != torch.bfloat16:
            raise ValueError(
                f"{name}: expected BF16 [{width},{HIDDEN}], "
                f"got {weight.shape} {weight.dtype}"
            )
    return x.contiguous(), [weight.contiguous() for weight in weights]


def reference_projection(
    x: torch.Tensor, weights: list[torch.Tensor]
) -> torch.Tensor:
    # Keep the four GEMMs separate so A retains production shape selection and
    # BF16 output rounding.  Concatenation materializes the common [M,4160]
    # contract used by the strict oracle.
    return torch.cat([F.linear(x, weight) for weight in weights], dim=1).contiguous()


def candidate_projection(
    x: torch.Tensor,
    weight_shard: torch.Tensor,
    comm: CustomAllreduce,
    raw_gather: torch.Tensor,
) -> torch.Tensor:
    local = F.linear(x, weight_shard)
    comm.all_gather_reg(local, out=raw_gather)
    # Old AIter AG concatenates flattened rank buffers.  Convert
    # [rank, token, output_shard] to [token, rank * output_shard].  contiguous()
    # is intentionally captured and timed: consumers require physical N4160.
    return (
        raw_gather.view(comm.world_size, x.shape[0], weight_shard.shape[0])
        .movedim(0, 1)
        .contiguous()
        .view(x.shape[0], TOTAL_OUTPUT)
    )


def capture(fn: Callable[[], torch.Tensor], comm: CustomAllreduce | None = None):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    if comm is None:
        with torch.cuda.graph(graph):
            output = fn()
    else:
        with comm.capture():
            with torch.cuda.graph(graph):
                output = fn()
    dist.barrier()
    return graph, output


def rank_max_samples(
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
    gathered: list[list[float] | None] = [None] * world
    dist.all_gather_object(gathered, local)
    rank_max = [
        max(samples[index] for samples in gathered if samples is not None)
        for index in range(reps)
    ]
    return statistics.median(rank_max), rank_max


def digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def mutation_sweep(
    x: torch.Tensor,
    base_x: torch.Tensor,
    mutation: torch.Tensor,
    graph_a: torch.cuda.CUDAGraph,
    output_a: torch.Tensor,
    graph_b: torch.cuda.CUDAGraph,
    output_b: torch.Tensor,
    count: int,
) -> tuple[int, float, float, int]:
    mismatches = 0
    max_abs = 0.0
    max_rel_l2 = 0.0
    first_mismatch = -1
    for iteration in range(count):
        # A deterministic 2047-state sequence gives 1000 distinct perturbation
        # coefficients.  The amplitude crosses BF16 ulps without changing the
        # real activation's overall distribution.
        state = (iteration * 1543 + 17) % 2047
        alpha = (state - 1023) / 32768.0
        x.copy_(base_x)
        x.add_(mutation, alpha=alpha)
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        if not torch.equal(output_a, output_b):
            if first_mismatch < 0:
                first_mismatch = iteration
            mismatches += 1
            delta = output_a.float() - output_b.float()
            max_abs = max(max_abs, delta.abs().max().item())
            rel_l2 = (
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(output_a.float()).clamp_min(1e-12)
            ).item()
            max_rel_l2 = max(max_rel_l2, rel_l2)
    return mismatches, max_abs, max_rel_l2, first_mismatch


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"this oracle requires exactly 8 ranks, got {world}")
    if TOTAL_OUTPUT % world != 0:
        raise RuntimeError(f"N={TOTAL_OUTPUT} is not divisible by world={world}")

    x_cpu, weights_cpu = load_dump(args)
    x = x_cpu.cuda(non_blocking=False)
    base_x = x.clone()
    # Bounded, non-symmetric values make stale/rank-order errors visible while
    # retaining the scale of the captured activation.
    mutation = torch.linspace(
        -1.0, 1.0, x.numel(), dtype=torch.float32, device="cuda"
    ).view_as(x).to(torch.bfloat16)
    weights = [weight.cuda(non_blocking=False) for weight in weights_cpu]
    combined_weight = torch.cat(weights, dim=0).contiguous()
    shard_width = TOTAL_OUTPUT // world
    lo, hi = rank * shard_width, (rank + 1) * shard_width
    weight_shard = combined_weight[lo:hi].contiguous()

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter CustomAllreduce did not initialize")
    input_bytes = x.shape[0] * shard_width * x.element_size()
    if input_bytes % 16 != 0 or input_bytes > comm.max_size // (world * 2):
        raise RuntimeError(
            f"AG input violates alignment/pool gate: bytes={input_bytes} "
            f"limit={comm.max_size // (world * 2)}"
        )

    raw_gather = torch.empty(
        world * x.shape[0] * shard_width,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Initialize hipBLAS and the unregistered AIter path before graph capture;
    # neither library is allowed to perform first-use allocation in capture.
    _ = reference_projection(x, weights)
    local_warmup = F.linear(x, weight_shard)
    comm.all_gather_unreg(local_warmup, out=raw_gather)
    torch.cuda.synchronize()
    graph_a, output_a = capture(lambda: reference_projection(x, weights))
    graph_b, output_b = capture(
        lambda: candidate_projection(x, weight_shard, comm, raw_gather), comm
    )

    # Initial exactness and rank-order check before the longer mutation sweep.
    graph_a.replay()
    graph_b.replay()
    torch.cuda.synchronize()
    initial_exact = torch.equal(output_a, output_b)
    initial_max_abs = (output_a.float() - output_b.float()).abs().max().item()
    local_slice = output_b[:, lo:hi]
    local_exact = torch.equal(local_warmup, local_slice)
    local_max_abs = (local_warmup.float() - local_slice.float()).abs().max().item()
    segment_max_abs = [
        (part_a.float() - part_b.float()).abs().max().item()
        for part_a, part_b in zip(
            torch.split(output_a, [width for _, width in PROJECTIONS], dim=1),
            torch.split(output_b, [width for _, width in PROJECTIONS], dim=1),
            strict=True,
        )
    ]

    mismatch_count, max_abs, max_rel_l2, first_mismatch = mutation_sweep(
        x,
        base_x,
        mutation,
        graph_a,
        output_a,
        graph_b,
        output_b,
        args.mutations,
    )
    local_correctness = {
        "rank": rank,
        "initial_exact": initial_exact,
        "initial_max_abs": initial_max_abs,
        "local_ag_slice_exact": local_exact,
        "local_ag_slice_max_abs": local_max_abs,
        "segment_max_abs": segment_max_abs,
        "mismatches": mismatch_count,
        "max_abs": max_abs,
        "max_rel_l2": max_rel_l2,
        "first_mismatch": first_mismatch,
        "a_hash": digest(output_a),
        "b_hash": digest(output_b),
    }
    correctness: list[dict | None] = [None] * world
    dist.all_gather_object(correctness, local_correctness)
    if rank == 0:
        print(
            f"correctness rows={x.shape[0]} K={HIDDEN} N={TOTAL_OUTPUT} "
            f"shard_N={shard_width} mutations={args.mutations} "
            f"per_rank={correctness}",
            flush=True,
        )

    # Restore the real tensor before timing.  Each letter contains seven
    # rank-max samples by default; A/B/B/A controls thermal/order drift.
    x.copy_(base_x)
    torch.cuda.synchronize()
    schedule = (("A", graph_a), ("B", graph_b), ("B", graph_b), ("A", graph_a))
    timings: list[tuple[str, float, list[float]]] = []
    for slot, (name, graph) in enumerate(schedule, start=1):
        median, samples = rank_max_samples(
            graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        timings.append((name, median, samples))
        if rank == 0:
            print(
                f"slot={slot} mode={name} rank_max_median_us={median:.3f} "
                f"samples={[round(value, 3) for value in samples]}",
                flush=True,
            )

    a_medians = [median for name, median, _ in timings if name == "A"]
    b_medians = [median for name, median, _ in timings if name == "B"]
    a_paired = statistics.mean(a_medians)
    b_paired = statistics.mean(b_medians)
    if rank == 0:
        print(
            f"ABBA A_mean_us={a_paired:.3f} B_mean_us={b_paired:.3f} "
            f"delta_pct={(b_paired / a_paired - 1.0) * 100.0:+.2f}",
            flush=True,
        )

    any_mismatch = any(
        item is None
        or not item["initial_exact"]
        or item["mismatches"] != 0
        for item in correctness
    )
    comm.close()
    dist.barrier()
    dist.destroy_process_group()
    if any_mismatch and not args.allow_nonexact:
        raise AssertionError(
            "output-N projection+AG was not BF16 bitwise exact; "
            "see per-rank correctness report"
        )


if __name__ == "__main__":
    main()
