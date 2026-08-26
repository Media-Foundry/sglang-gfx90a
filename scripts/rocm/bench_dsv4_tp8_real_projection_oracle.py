#!/usr/bin/env python3
"""Real-tensor TP8 row-sharded projection oracle for DeepSeek-V4 C4 layers.

The dump is produced by the opt-in debug hook in ``deepseek_v4.py``.  This
script stays outside the model selector: it compares the four production
full-K BF16 projections with a single K-sharded projection and AIter all-reduce.

Example:

  torchrun --standalone --nproc-per-node=8 \
    scripts/rocm/bench_dsv4_tp8_real_projection_oracle.py \
    --dump-dir /tmp/dsv4-layer20-m32
"""

from __future__ import annotations

import argparse
import hashlib
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
TUNED_FP16_SOLUTIONS = {1536: 12183, 2560: 12008, 4160: 11948}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--partial-dtypes",
        nargs="+",
        choices=("bf16", "fp16", "fp32", "fp32_cached_x"),
        default=("bf16", "fp16", "fp32", "fp32_cached_x"),
    )
    parser.add_argument("--untuned-fp16", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument(
        "--pair-abba",
        action="store_true",
        help="Time exactly two candidate modes in A/B/B/A order.",
    )
    return parser.parse_args()


def load_dump(args: argparse.Namespace) -> tuple[torch.Tensor, list[torch.Tensor]]:
    prefix = f"layer_{args.layer}"
    activation = torch.load(
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
    if activation.ndim != 2 or activation.shape[1] != 4096:
        raise ValueError(f"expected activation [M,4096], got {activation.shape}")
    if activation.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 activation, got {activation.dtype}")
    for (name, expected_n), weight in zip(PROJECTIONS, weights, strict=True):
        if weight.shape != (expected_n, 4096):
            raise ValueError(
                f"{name}: expected [{expected_n},4096], got {weight.shape}"
            )
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"{name}: expected logical BF16 weight, got {weight.dtype}")
    return activation.contiguous(), [weight.contiguous() for weight in weights]


def mixed_outputs(packed: torch.Tensor) -> tuple[torch.Tensor, ...]:
    qkv, core, index_core, index_weight = torch.split(
        packed, tuple(width for _, width in PROJECTIONS), dim=1
    )
    outputs = (
        qkv.to(torch.bfloat16).contiguous(),
        core.float().contiguous(),
        index_core.float().contiguous(),
        index_weight.to(torch.bfloat16).contiguous(),
    )
    assert all(output.is_contiguous() for output in outputs)
    return outputs


def full_reference(
    x: torch.Tensor, weights: list[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    # Keep the four calls separate to match production rounding and dtype:
    # wqkv/index weights are BF16; both compressor score tensors are FP32.
    raw = [F.linear(x, weight) for weight in weights]
    outputs = (
        raw[0].contiguous(),
        raw[1].float().contiguous(),
        raw[2].float().contiguous(),
        raw[3].contiguous(),
    )
    assert all(output.is_contiguous() for output in outputs)
    return outputs


def sharded_projection(
    mode: str,
    x_shard: torch.Tensor,
    weight_shard_bf16: torch.Tensor,
    weight_shard_fp16: torch.Tensor,
    weight_shard_fp32: torch.Tensor,
    x_shard_fp32: torch.Tensor,
    comm: CustomAllreduce,
    *,
    registered: bool,
    tuned_fp16: bool,
) -> tuple[torch.Tensor, ...]:
    if mode == "bf16":
        local = F.linear(x_shard, weight_shard_bf16)
        return mixed_outputs(comm.all_reduce(local, registered=registered))
    if mode == "fp16":
        x_fp16 = x_shard.to(torch.float16)
        if tuned_fp16:
            from aiter.tuned_gemm import hipb_gemm

            solution = TUNED_FP16_SOLUTIONS[weight_shard_fp16.shape[0]]
            local = hipb_gemm(
                x_fp16,
                weight_shard_fp16,
                solution,
                otype=torch.float16,
            )
        else:
            local = F.linear(x_fp16, weight_shard_fp16)
        return mixed_outputs(comm.all_reduce(local, registered=registered))
    if mode in ("fp32", "fp32_cached_x"):
        # Logical weights are static and would be cached by any production
        # prototype.  Only the small per-step activation cast belongs in the
        # fp32 graph; fp32_cached_x is the ideal lower bound that removes it.
        x_fp32 = x_shard.float() if mode == "fp32" else x_shard_fp32
        local = F.linear(x_fp32, weight_shard_fp32)
        return mixed_outputs(comm.all_reduce(local, registered=registered))
    raise ValueError(mode)


def error_triplet(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_f = candidate.float().flatten()
    reference_f = reference.float().flatten()
    delta = candidate_f - reference_f
    return (
        delta.abs().max(),
        torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_f),
        F.cosine_similarity(
            candidate_f.unsqueeze(0), reference_f.unsqueeze(0)
        ).squeeze(0),
    )


def gather_metrics(
    candidate: torch.Tensor, reference: torch.Tensor, world: int
) -> tuple[list[float], int]:
    metrics = torch.stack(error_triplet(candidate, reference)).float()
    max_rel = metrics[:2].clone()
    cosine = metrics[2:].clone()
    dist.all_reduce(max_rel, op=dist.ReduceOp.MAX)
    dist.all_reduce(cosine, op=dist.ReduceOp.MIN)
    metrics[:2] = max_rel
    metrics[2:] = cosine
    digest = hashlib.sha256(
        candidate.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest)
    return [value.item() for value in metrics], len(set(digests))


def flatten_mixed(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([output.float() for output in outputs], dim=1)


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
    rank_max = [max(gathered[r][i] for r in range(world)) for i in range(reps)]
    return statistics.median(rank_max), rank_max


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError(f"this oracle requires TP8, got world={world}")

    x_cpu, weights_cpu = load_dump(args)
    x = x_cpu.cuda(non_blocking=False)
    weights = [weight.cuda(non_blocking=False) for weight in weights_cpu]
    combined_weight = torch.cat(weights, dim=0).contiguous()
    hidden_shard = x.shape[1] // world
    lo, hi = rank * hidden_shard, (rank + 1) * hidden_shard
    x_shard = x[:, lo:hi].contiguous()
    weight_shard_bf16 = combined_weight[:, lo:hi].contiguous()
    weight_shard_fp16 = weight_shard_bf16.to(torch.float16)
    weight_shard_fp32 = weight_shard_bf16.float()
    x_shard_fp32 = x_shard.float()

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter custom all-reduce did not initialize")

    reference = full_reference(x, weights)
    candidates: dict[str, tuple[torch.Tensor, ...]] = {}
    for mode in args.partial_dtypes:
        candidate = sharded_projection(
            mode,
            x_shard,
            weight_shard_bf16,
            weight_shard_fp16,
            weight_shard_fp32,
            x_shard_fp32,
            comm,
            registered=False,
            tuned_fp16=not args.untuned_fp16,
        )
        candidates[mode] = candidate
        overall, unique_hashes = gather_metrics(
            flatten_mixed(candidate), flatten_mixed(reference), world
        )
        segment_metrics = []
        for (name, _), candidate_part, reference_part in zip(
            PROJECTIONS, candidate, reference, strict=True
        ):
            segment_metrics.append(
                (
                    name,
                    str(candidate_part.dtype),
                    *gather_metrics(candidate_part, reference_part, world)[0],
                )
            )
        if rank == 0:
            print(
                f"mode={mode} rows={x.shape[0]} overall_max_abs={overall[0]:.8g} "
                f"overall_rel_l2={overall[1]:.8g} cosine={overall[2]:.9f} "
                f"rank_hashes_unique={unique_hashes} segments={segment_metrics}",
                flush=True,
            )

    if not args.skip_graphs:
        reference_graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(reference_graph):
            reference_graph_output = full_reference(x, weights)
        dist.barrier()
        reference_us, reference_reps = critical_us(
            reference_graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        if rank == 0:
            print(
                f"reference_critical_us={reference_us:.3f} "
                f"reps={[round(v, 3) for v in reference_reps]}",
                flush=True,
            )

        mode_graphs: dict[str, torch.cuda.CUDAGraph] = {}
        graph_outputs: dict[str, tuple[torch.Tensor, ...]] = {}
        for mode in args.partial_dtypes:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with comm.capture():
                with torch.cuda.graph(graph):
                    graph_output = sharded_projection(
                        mode,
                        x_shard,
                        weight_shard_bf16,
                        weight_shard_fp16,
                        weight_shard_fp32,
                        x_shard_fp32,
                        comm,
                        registered=True,
                        tuned_fp16=not args.untuned_fp16,
                    )
            dist.barrier()
            mode_graphs[mode] = graph
            graph_outputs[mode] = graph_output

        if args.pair_abba:
            if len(args.partial_dtypes) != 2:
                raise ValueError("--pair-abba requires exactly two candidate modes")
            first, second = args.partial_dtypes
            schedule = (first, second, second, first)
        else:
            schedule = tuple(args.partial_dtypes)

        for slot, mode in enumerate(schedule, start=1):
            graph = mode_graphs[mode]
            mode_us, mode_reps = critical_us(
                graph,
                warmup=args.warmup,
                iters=args.iters,
                reps=args.reps,
                world=world,
            )
            for graph_part, eager_part in zip(
                graph_outputs[mode], candidates[mode], strict=True
            ):
                torch.testing.assert_close(graph_part, eager_part, rtol=0, atol=0)
            if rank == 0:
                print(
                    f"slot={slot} mode={mode} candidate_critical_us={mode_us:.3f} "
                    f"delta_pct={(mode_us / reference_us - 1) * 100:+.2f} "
                    f"reps={[round(v, 3) for v in mode_reps]}",
                    flush=True,
                )
        _ = reference_graph_output

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
