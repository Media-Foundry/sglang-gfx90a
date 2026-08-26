#!/usr/bin/env python3
"""TP8 hidden-sharded DSV4 MHC structural oracle and graph budget.

This is deliberately a standalone experiment.  It does not select a model
kernel or modify the production forward path.  The candidate decomposition is

  row-parallel partial [M,H]
    -> reduce-scatter hidden [M,H/P]
    -> sharded mHC post
    -> AR([M, 24 dot partials + 1 residual sumsq])
    -> sharded mHC pre weighted sum
    -> sharded weighted sum (rounded to BF16)
    -> AR([M,1] FP32 sumsq)
    -> sharded RMSNorm with its production BF16 output rounding
    -> row-sharded replicated projections
    -> one BF16 AR([M,N]).

The script first compares that decomposition with a full-H reference, then
captures the candidate and reference sequences in HIP graphs and reports the
slowest-rank median.  The pointwise PyTorch implementation is an oracle, not a
proposed production kernel; only the collective shapes and dependency chain
are representative until a fused sharded-MHC kernel exists.

Run (only on an otherwise idle eight-GCD node):

  SGLANG_DP_USE_REDUCE_SCATTER=1 torchrun --standalone --nproc-per-node=8 \
    scripts/rocm/bench_dsv4_tp8_rs_mhc_oracle.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


@dataclass
class Inputs:
    partial: torch.Tensor
    partial_rank_major: torch.Tensor
    residual_full: torch.Tensor
    residual_shard: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    fn_full: torch.Tensor
    fn_shard: torch.Tensor
    pre_scale: torch.Tensor
    pre_base: torch.Tensor
    norm_full: torch.Tensor
    norm_shard: torch.Tensor
    wqkv_full: torch.Tensor
    wqkv_shard: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--hc", type=int, default=4)
    p.add_argument("--mixes", type=int, default=24)
    p.add_argument("--qkv", type=int, default=1536)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--skip-graphs", action="store_true")
    p.add_argument("--skip-breakdown", action="store_true")
    return p.parse_args()


def _randn(shape, *, seed: int, dtype: torch.dtype) -> torch.Tensor:
    # Generate identical model/state tensors on every rank without a host
    # broadcast, which also keeps graph setup outside the measured region.
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, device="cuda", dtype=dtype)


def make_inputs(args: argparse.Namespace, rank: int, world: int) -> Inputs:
    assert world == 8, "this oracle is specifically for the TP8 design"
    assert args.hidden % world == 0
    hs = args.hidden // world
    lo, hi = rank * hs, (rank + 1) * hs

    # Each rank contributes a different row-parallel output partial.
    partial = _randn(
        (args.rows, args.hidden), seed=1000 + rank, dtype=torch.bfloat16
    ).mul_(0.0625)
    partial_rank_major = (
        partial.view(args.rows, world, hs)
        .movedim(1, 0)
        .reshape(world * args.rows, hs)
        .contiguous()
    )
    residual_full = _randn(
        (args.rows, args.hc, args.hidden), seed=2000, dtype=torch.bfloat16
    ).mul_(0.125)
    residual_shard = residual_full[:, :, lo:hi].contiguous()
    post = _randn((args.rows, args.hc), seed=3000, dtype=torch.float32).mul_(0.1)
    comb = _randn(
        (args.rows, args.hc, args.hc), seed=4000, dtype=torch.float32
    ).mul_(0.1)

    fn_full = _randn(
        (args.mixes, args.hc, args.hidden), seed=5000, dtype=torch.float32
    ).mul_(0.01)
    fn_shard = fn_full[:, :, lo:hi].contiguous()
    pre_scale = torch.tensor(0.25, device="cuda", dtype=torch.float32)
    pre_base = _randn((args.hc,), seed=6000, dtype=torch.float32).mul_(0.1)
    norm_full = _randn((args.hidden,), seed=7000, dtype=torch.bfloat16).mul_(0.1).add_(1)
    norm_shard = norm_full[lo:hi].contiguous()

    # The real fused wqkv_a is [1536,4096].  K-sharding is contiguous.
    wqkv_full = _randn(
        (args.qkv, args.hidden), seed=8000, dtype=torch.bfloat16
    ).mul_(0.015625)
    wqkv_shard = wqkv_full[:, lo:hi].contiguous()
    return Inputs(
        partial,
        partial_rank_major,
        residual_full,
        residual_shard,
        post,
        comb,
        fn_full,
        fn_shard,
        pre_scale,
        pre_base,
        norm_full,
        norm_shard,
        wqkv_full,
        wqkv_shard,
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    # comb is [old_channel,new_channel], matching DeepSeek-V4/SGLang.
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + torch.einsum("moi,moh->mih", comb, residual.float())
    ).to(torch.bfloat16)


def pre_weights_from_stats(
    stats: torch.Tensor,
    *,
    hc: int,
    hidden: int,
    scale: torch.Tensor,
    base: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dots, residual_sumsq = stats[:, :-1], stats[:, -1:]
    inv_rms = torch.rsqrt(residual_sumsq / (hc * hidden) + eps)
    mixes = dots * inv_rms
    # Only the first hc MHC outputs produce pre weights.  post/comb Sinkhorn
    # outputs do not affect the layer-input and wqkv_a budget being tested.
    return torch.sigmoid(mixes[:, :hc] * scale + base) + eps


def full_reference(
    args: argparse.Namespace,
    inp: Inputs,
    comm: CustomAllreduce,
    *,
    registered: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reduced = comm.all_reduce(inp.partial, registered=registered)
    residual = mhc_post(reduced, inp.residual_full, inp.post, inp.comb)
    residual_f = residual.float()
    dots = torch.einsum("mch,kch->mk", residual_f, inp.fn_full)
    sumsq = residual_f.square().sum((1, 2), keepdim=False).unsqueeze(1)
    stats = torch.cat((dots, sumsq), dim=1)
    pre = pre_weights_from_stats(
        stats,
        hc=args.hc,
        hidden=args.hidden,
        scale=inp.pre_scale,
        base=inp.pre_base,
        eps=args.eps,
    )
    y = torch.einsum("mc,mch->mh", pre, residual_f).to(torch.bfloat16)
    y_f = y.float()
    y_sumsq = y_f.square().sum(1, keepdim=True)
    layer_input = (
        y_f
        * torch.rsqrt(y_sumsq / args.hidden + args.eps)
        * inp.norm_full.float()
    ).to(torch.bfloat16)
    qkv = torch.matmul(layer_input, inp.wqkv_full.T)
    return residual, layer_input, qkv


def sharded_candidate(
    args: argparse.Namespace,
    inp: Inputs,
    comm: CustomAllreduce,
    rank: int,
    *,
    registered: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hs = args.hidden // dist.get_world_size()
    reduced_shard = torch.empty(
        (args.rows, hs), dtype=torch.bfloat16, device="cuda"
    )
    comm.reduce_scatter(
        inp.partial_rank_major, reduced_shard, registered=registered
    )
    residual = mhc_post(reduced_shard, inp.residual_shard, inp.post, inp.comb)
    residual_f = residual.float()
    dots_local = torch.einsum("mch,kch->mk", residual_f, inp.fn_shard)
    sumsq_local = residual_f.square().sum((1, 2), keepdim=False).unsqueeze(1)
    stats_local = torch.cat((dots_local, sumsq_local), dim=1)
    stats = comm.all_reduce(stats_local, registered=registered)
    pre = pre_weights_from_stats(
        stats,
        hc=args.hc,
        hidden=args.hidden,
        scale=inp.pre_scale,
        base=inp.pre_base,
        eps=args.eps,
    )
    y = torch.einsum("mc,mch->mh", pre, residual_f).to(torch.bfloat16)
    y_f = y.float()
    y_sumsq_local = y_f.square().sum(1, keepdim=True)

    y_sumsq = comm.all_reduce(y_sumsq_local, registered=registered)
    inv_rms = torch.rsqrt(y_sumsq / args.hidden + args.eps)
    layer_input = (
        y_f * inv_rms * inp.norm_shard.float()
    ).to(torch.bfloat16)
    qkv_local = torch.matmul(layer_input, inp.wqkv_shard.T)
    qkv = comm.all_reduce(qkv_local, registered=registered)
    return residual, layer_input, qkv


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
    per_rep = [max(gathered[r][i] for r in range(world)) for i in range(reps)]
    return statistics.median(per_rep), per_rep


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    inp = make_inputs(args, rank, world)
    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter custom collectives did not initialize")

    dist.barrier()
    ref_residual, ref_layer_input, ref_qkv = full_reference(
        args, inp, comm, registered=False
    )
    cand_residual, cand_layer_input, cand_qkv = sharded_candidate(
        args, inp, comm, rank, registered=False
    )
    hs = args.hidden // world
    lo, hi = rank * hs, (rank + 1) * hs
    torch.testing.assert_close(
        cand_residual, ref_residual[:, :, lo:hi], rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        cand_layer_input, ref_layer_input[:, lo:hi], rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(cand_qkv, ref_qkv, rtol=3e-2, atol=5e-2)
    def error_triplet(candidate: torch.Tensor, reference: torch.Tensor):
        candidate_f = candidate.float().flatten()
        reference_f = reference.float().flatten()
        delta = candidate_f - reference_f
        return (
            delta.abs().max(),
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_f),
            torch.nn.functional.cosine_similarity(
                candidate_f.unsqueeze(0), reference_f.unsqueeze(0)
            ).squeeze(0),
        )

    local_metrics = torch.stack(
        [
            *error_triplet(cand_residual, ref_residual[:, :, lo:hi]),
            *error_triplet(cand_layer_input, ref_layer_input[:, lo:hi]),
            *error_triplet(cand_qkv, ref_qkv),
        ]
    ).to(dtype=torch.float32)
    # max_abs/relL2 use MAX. Cosine uses MIN to report the worst rank.
    max_rel = local_metrics[[0, 1, 3, 4, 6, 7]].clone()
    cosines = local_metrics[[2, 5, 8]].clone()
    dist.all_reduce(max_rel, op=dist.ReduceOp.MAX)
    dist.all_reduce(cosines, op=dist.ReduceOp.MIN)
    local_metrics[[0, 1, 3, 4, 6, 7]] = max_rel
    local_metrics[[2, 5, 8]] = cosines
    qkv_hash = hashlib.sha256(
        cand_qkv.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    qkv_hashes: list[str | None] = [None] * world
    dist.all_gather_object(qkv_hashes, qkv_hash)
    dist.barrier()
    if rank == 0:
        print(
            "structure_tolerance=pass "
            f"RS={args.rows}x{args.hidden}bf16->{args.rows}x{hs}bf16 "
            f"stats_AR={args.rows}x{args.mixes + 1}fp32 "
            f"norm_AR={args.rows}x1fp32 "
            f"projection_AR={args.rows}x{args.qkv}bf16 "
            f"residual_max_abs={local_metrics[0].item():.6g} "
            f"residual_rel_l2={local_metrics[1].item():.6g} "
            f"residual_cosine={local_metrics[2].item():.9f} "
            f"layer_input_max_abs={local_metrics[3].item():.6g} "
            f"layer_input_rel_l2={local_metrics[4].item():.6g} "
            f"layer_input_cosine={local_metrics[5].item():.9f} "
            f"qkv_max_abs={local_metrics[6].item():.6g} "
            f"qkv_rel_l2={local_metrics[7].item():.6g} "
            f"qkv_cosine={local_metrics[8].item():.9f} "
            f"rank_hashes_unique={len(set(qkv_hashes))}",
            flush=True,
        )

    if not args.skip_graphs:
        # Capture separately: registering all graph input pointers is part of
        # AIter's normal graph lifecycle, while graph replay is what matters to
        # the model critical path.
        ref_graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with comm.capture():
            with torch.cuda.graph(ref_graph):
                ref_graph_outputs = full_reference(
                    args, inp, comm, registered=True
                )
        dist.barrier()

        cand_graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with comm.capture():
            with torch.cuda.graph(cand_graph):
                cand_graph_outputs = sharded_candidate(
                    args, inp, comm, rank, registered=True
                )
        dist.barrier()

        ref_median, ref_reps = critical_us(
            ref_graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        cand_median, cand_reps = critical_us(
            cand_graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        # Keep outputs live through timing and validate graph replay too.
        torch.testing.assert_close(
            cand_graph_outputs[2], ref_graph_outputs[2], rtol=3e-2, atol=5e-2
        )
        if rank == 0:
            print(
                f"reference_critical_us={ref_median:.3f} "
                f"reps={[round(x, 3) for x in ref_reps]}",
                flush=True,
            )
            print(
                f"candidate_critical_us={cand_median:.3f} "
                f"reps={[round(x, 3) for x in cand_reps]} "
                f"delta_pct={(cand_median / ref_median - 1.0) * 100.0:+.2f}",
                flush=True,
            )
            print(
                "note=PyTorch pointwise/GEMM math is an oracle; do not treat "
                "this delta as a production-kernel result",
                flush=True,
            )

        if not args.skip_breakdown:
            # Isolate the strict-semantics communication floor.
            stats_seed = _randn(
                (args.rows, args.mixes + 1), seed=9000 + rank, dtype=torch.float32
            )
            norm_seed = _randn(
                (args.rows, 1), seed=10000 + rank, dtype=torch.float32
            )
            projection_bf16 = _randn(
                (args.rows, args.qkv), seed=11000 + rank, dtype=torch.bfloat16
            )

            def capture_collective(fn):
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with comm.capture():
                    with torch.cuda.graph(graph):
                        output = fn()
                dist.barrier()
                return graph, output

            rs_out = torch.empty(
                (args.rows, args.hidden // world),
                dtype=torch.bfloat16,
                device="cuda",
            )
            rs_graph, rs_keepalive = capture_collective(
                lambda: (
                    comm.reduce_scatter(
                        inp.partial_rank_major, rs_out, registered=True
                    ),
                    rs_out,
                )[1]
            )
            stats_graph, stats_keepalive = capture_collective(
                lambda: comm.all_reduce(stats_seed, registered=True)
            )
            norm_graph, norm_keepalive = capture_collective(
                lambda: comm.all_reduce(norm_seed, registered=True)
            )
            proj16_graph, proj16_keepalive = capture_collective(
                lambda: comm.all_reduce(projection_bf16, registered=True)
            )
            breakdown = []
            for name, graph in (
                ("RS_32x4096_bf16", rs_graph),
                ("AR_32x25_fp32", stats_graph),
                ("AR_32x1_fp32", norm_graph),
                (f"AR_32x{args.qkv}_bf16", proj16_graph),
            ):
                median, reps = critical_us(
                    graph,
                    warmup=args.warmup,
                    iters=args.iters,
                    reps=args.reps,
                    world=world,
                )
                breakdown.append((name, median, reps))
            # Explicitly retain captured outputs until every replay completes.
            _ = (rs_keepalive, stats_keepalive, norm_keepalive, proj16_keepalive)
            if rank == 0:
                for name, median, reps in breakdown:
                    print(
                        f"primitive={name} critical_us={median:.3f} "
                        f"reps={[round(x, 3) for x in reps]}",
                        flush=True,
                    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
