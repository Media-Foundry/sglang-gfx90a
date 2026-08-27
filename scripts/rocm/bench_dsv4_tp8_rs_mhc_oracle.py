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
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

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
    fn_shard_fp16: torch.Tensor
    pre_scale: torch.Tensor
    pre_base: torch.Tensor
    norm_full: torch.Tensor
    norm_shard: torch.Tensor
    wqkv_full: torch.Tensor
    wqkv_shard: torch.Tensor
    wqkv_shard_fp16: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--hc", type=int, default=4)
    p.add_argument("--mixes", type=int, default=24)
    p.add_argument("--qkv", type=int, default=1536)
    p.add_argument(
        "--dump-dir",
        help="Load real layer-20 tensors produced by the DSV4 debug dump.",
    )
    p.add_argument("--model-dir", default="/home/pc/models/modelscope")
    p.add_argument("--layer-id", type=int, default=20)
    p.add_argument(
        "--projection-partial-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    p.add_argument("--isolate-reference-layer-input", action="store_true")
    p.add_argument(
        "--include-expert-all-gather",
        action="store_true",
        help="Model the FFN boundary by restoring full-H expert input.",
    )
    p.add_argument(
        "--tuned-fp16",
        action="store_true",
        help="Use the isolated hipBLASLt solution tuned for the selected N.",
    )
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--skip-graphs", action="store_true")
    p.add_argument("--skip-breakdown", action="store_true")
    p.add_argument(
        "--native-sharded-mhc",
        action="store_true",
        help="Use the standalone gfx90a H512 stage1/2/3 HIP kernels.",
    )
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

    if args.dump_dir:
        dump_dir = Path(args.dump_dir)

        def load(name: str) -> torch.Tensor:
            return torch.load(
                dump_dir / f"layer_20_rank_{rank}_{name}.pt",
                map_location="cpu",
                weights_only=True,
            ).cuda()

        partial = load("wo_b_partial").contiguous()
        partial_rank_major = (
            partial.view(args.rows, world, hs)
            .movedim(1, 0)
            .reshape(world * args.rows, hs)
            .contiguous()
        )
        residual_full = load("ffn_mhc_residual").contiguous()
        residual_shard = residual_full[:, :, lo:hi].contiguous()
        post = load("ffn_mhc_post").contiguous()
        comb = load("ffn_mhc_comb").contiguous()
        fn_full = load("hc_ffn_fn").reshape(args.mixes, args.hc, args.hidden)
        fn_shard = fn_full[:, :, lo:hi].contiguous()
        fn_shard_fp16 = (
            fn_shard.to(torch.float16).reshape(args.mixes, -1).contiguous()
        )
        pre_scale = load("hc_ffn_scale").contiguous()
        pre_base = load("hc_ffn_base").contiguous()
        norm_full = load("ffn_norm_weight").contiguous()
        norm_shard = norm_full[lo:hi].contiguous()
        wqkv_full = load("router_weight").contiguous()
        if wqkv_full.shape != (args.qkv, args.hidden):
            raise ValueError(
                f"dumped projection shape {tuple(wqkv_full.shape)} does not "
                f"match --qkv={args.qkv}"
            )
        wqkv_shard = wqkv_full[:, lo:hi].contiguous()
        wqkv_shard_fp16 = wqkv_shard.to(torch.float16)
        return Inputs(
            partial,
            partial_rank_major,
            residual_full,
            residual_shard,
            post,
            comb,
            fn_full,
            fn_shard,
            fn_shard_fp16,
            pre_scale,
            pre_base,
            norm_full,
            norm_shard,
            wqkv_full,
            wqkv_shard,
            wqkv_shard_fp16,
        )

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
    fn_shard_fp16 = (
        fn_shard.to(torch.float16).reshape(args.mixes, -1).contiguous()
    )
    pre_scale = torch.tensor(0.25, device="cuda", dtype=torch.float32)
    pre_base = _randn((args.hc,), seed=6000, dtype=torch.float32).mul_(0.1)
    norm_full = (
        _randn((args.hidden,), seed=7000, dtype=torch.bfloat16).mul_(0.1).add_(1)
    )
    norm_shard = norm_full[lo:hi].contiguous()

    # The real fused wqkv_a is [1536,4096].  K-sharding is contiguous.
    wqkv_full = _randn(
        (args.qkv, args.hidden), seed=8000, dtype=torch.bfloat16
    ).mul_(0.015625)
    wqkv_shard = wqkv_full[:, lo:hi].contiguous()
    wqkv_shard_fp16 = wqkv_shard.to(torch.float16)
    return Inputs(
        partial,
        partial_rank_major,
        residual_full,
        residual_shard,
        post,
        comb,
        fn_full,
        fn_shard,
        fn_shard_fp16,
        pre_scale,
        pre_base,
        norm_full,
        norm_shard,
        wqkv_full,
        wqkv_shard,
        wqkv_shard_fp16,
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
    return torch.sigmoid(mixes[:, :hc] * scale.flatten()[0] + base[:hc]) + eps


def full_reference(
    args: argparse.Namespace,
    inp: Inputs,
    comm: CustomAllreduce,
    *,
    registered: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return residual, layer_input, qkv, layer_input


def sharded_candidate(
    args: argparse.Namespace,
    inp: Inputs,
    comm: CustomAllreduce,
    rank: int,
    *,
    registered: bool,
    layer_input_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    hs = args.hidden // dist.get_world_size()
    reduced_shard = torch.empty(
        (args.rows, hs), dtype=torch.bfloat16, device="cuda"
    )
    comm.reduce_scatter(
        inp.partial_rank_major, reduced_shard, registered=registered
    )
    if args.native_sharded_mhc:
        from sglang.kernels.ops.layernorm.gfx90a_sharded_mhc import (
            gfx90a_sharded_mhc_stage1,
            gfx90a_sharded_mhc_stage2,
            gfx90a_sharded_mhc_stage3,
        )

        residual, stats_local = gfx90a_sharded_mhc_stage1(
            reduced_shard,
            inp.residual_shard,
            inp.post,
            inp.comb,
            inp.fn_shard_fp16,
        )
        stats = comm.all_reduce(stats_local, registered=registered)
        post, comb, y, y_sumsq_local = gfx90a_sharded_mhc_stage2(
            residual,
            stats,
            inp.pre_scale,
            inp.pre_base,
            args.eps,
            args.eps,
            2.0,
        )
        y_sumsq = comm.all_reduce(y_sumsq_local, registered=registered)
        layer_input = gfx90a_sharded_mhc_stage3(
            y,
            y_sumsq,
            inp.norm_shard,
            args.eps,
        )
    else:
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
    projection_input = (
        layer_input if layer_input_override is None else layer_input_override
    )
    if args.projection_partial_dtype == "fp32":
        qkv_local = torch.matmul(
            projection_input.float(), inp.wqkv_shard.float().T
        )
        qkv = comm.all_reduce(qkv_local, registered=registered).to(torch.bfloat16)
    elif args.projection_partial_dtype == "fp16":
        projection_input_fp16 = projection_input.to(torch.float16)
        if args.tuned_fp16:
            solutions = {1536: 12183, 2560: 12008, 4160: 11948}
            if args.qkv not in solutions:
                raise ValueError(f"no tuned FP16 solution for N={args.qkv}")
            from aiter.tuned_gemm import hipb_gemm

            # hipb_gemm owns the per-process hipBLASLt extension lifecycle;
            # calling raw hipb_mm before hipb_create_extension segfaults.
            qkv_local = hipb_gemm(
                projection_input_fp16,
                inp.wqkv_shard_fp16,
                solutions[args.qkv],
                otype=torch.float16,
            )
        else:
            qkv_local = torch.matmul(
                projection_input_fp16, inp.wqkv_shard_fp16.T
            )
        qkv = comm.all_reduce(qkv_local, registered=registered).to(torch.bfloat16)
    else:
        qkv_local = torch.matmul(projection_input, inp.wqkv_shard.T)
        qkv = comm.all_reduce(qkv_local, registered=registered)
    expert_input = None
    if args.include_expert_all_gather:
        gathered = comm.custom_all_gather(layer_input.contiguous())
        if gathered is None:
            raise RuntimeError("AIter custom all-gather is unavailable")
        expert_input = (
            gathered.view(dist.get_world_size(), args.rows, hs)
            .movedim(0, 1)
            .reshape(args.rows, args.hidden)
            .contiguous()
        )
    return residual, layer_input, qkv, expert_input


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

    if args.native_sharded_mhc:
        from sglang.kernels.ops.layernorm.gfx90a_sharded_mhc import (
            gfx90a_sharded_mhc_pre_stats,
        )

        pre_local = gfx90a_sharded_mhc_pre_stats(
            inp.residual_shard, inp.fn_shard_fp16
        )
        residual_local_f = inp.residual_shard.float()
        pre_local_ref = torch.cat(
            (
                torch.einsum(
                    "mch,kch->mk",
                    residual_local_f,
                    inp.fn_shard_fp16.view(args.mixes, args.hc, -1).float(),
                ),
                residual_local_f.square().sum((1, 2)).unsqueeze(1),
            ),
            dim=1,
        )
        torch.testing.assert_close(pre_local, pre_local_ref, rtol=3e-3, atol=5e-2)
        pre_global = comm.all_reduce(pre_local, registered=False)
        pre_global_ref = torch.cat(
            (
                torch.einsum(
                    "mch,kch->mk", inp.residual_full.float(), inp.fn_full.half().float()
                ),
                inp.residual_full.float().square().sum((1, 2)).unsqueeze(1),
            ),
            dim=1,
        )
        pre_delta = (pre_global - pre_global_ref).abs()
        pre_rel_l2 = torch.linalg.vector_norm(pre_global - pre_global_ref) / (
            torch.linalg.vector_norm(pre_global_ref).clamp_min(1e-12)
        )
        if rank == 0:
            print(
                f"native_pre_stats_max_abs={pre_delta.max().item():.8g} "
                f"native_pre_stats_rel_l2={pre_rel_l2.item():.8g}",
                flush=True,
            )

    dist.barrier()
    ref_residual, ref_layer_input, ref_qkv, ref_expert_input = full_reference(
        args, inp, comm, registered=False
    )
    hs = args.hidden // world
    lo, hi = rank * hs, (rank + 1) * hs
    projection_input_override = (
        ref_layer_input[:, lo:hi].contiguous()
        if args.isolate_reference_layer_input
        else None
    )
    cand_residual, cand_layer_input, cand_qkv, cand_expert_input = sharded_candidate(
        args,
        inp,
        comm,
        rank,
        registered=False,
        layer_input_override=projection_input_override,
    )
    torch.testing.assert_close(
        cand_residual, ref_residual[:, :, lo:hi], rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        cand_layer_input, ref_layer_input[:, lo:hi], rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(cand_qkv, ref_qkv, rtol=3e-2, atol=5e-2)
    if args.include_expert_all_gather:
        assert cand_expert_input is not None
        torch.testing.assert_close(
            cand_expert_input, ref_expert_input, rtol=2e-2, atol=2e-2
        )
    topk_summary = None
    if args.dump_dir and args.qkv == 256:
        from safetensors import safe_open

        key = f"layers.{args.layer_id}.ffn.gate.bias"
        index_path = Path(args.model_dir) / "model.safetensors.index.json"
        shard_name = json.loads(index_path.read_text())["weight_map"][key]
        with safe_open(
            Path(args.model_dir) / shard_name, framework="pt", device="cpu"
        ) as handle:
            router_bias = handle.get_tensor(key).cuda().float()
        ref_choice = torch.nn.functional.softplus(ref_qkv.float()).sqrt()
        ref_choice = ref_choice + router_bias
        cand_choice = torch.nn.functional.softplus(cand_qkv.float()).sqrt()
        cand_choice = cand_choice + router_bias
        ref_top7 = torch.topk(ref_choice, 7, dim=-1, sorted=True)
        cand_top6 = torch.topk(cand_choice, 6, dim=-1, sorted=False).indices
        ref_top6 = ref_top7.indices[:, :6]
        ref_sets = torch.sort(ref_top6, dim=-1).values
        cand_sets = torch.sort(cand_top6, dim=-1).values
        topk_summary = (
            int(torch.equal(ref_sets, cand_sets)),
            int((ref_sets != cand_sets).any(dim=-1).sum()),
            float((ref_top7.values[:, 5] - ref_top7.values[:, 6]).min()),
            float((cand_choice - ref_choice).abs().max()),
        )

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
            f"projection_AR={args.rows}x{args.qkv}{args.projection_partial_dtype} "
            f"projection_partial_dtype={args.projection_partial_dtype} "
            f"isolate_reference_layer_input={args.isolate_reference_layer_input} "
            f"tuned_fp16={args.tuned_fp16} "
            f"include_expert_all_gather={args.include_expert_all_gather} "
            f"native_sharded_mhc={args.native_sharded_mhc} "
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
        if topk_summary is not None:
            print(
                "router_topk_set_exact="
                f"{bool(topk_summary[0])} mismatched_rows={topk_summary[1]} "
                f"min_6v7_margin={topk_summary[2]:.8g} "
                f"choice_max_abs={topk_summary[3]:.8g}",
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
                    args,
                    inp,
                    comm,
                    rank,
                    registered=True,
                    layer_input_override=projection_input_override,
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
        # A/B/B/A guards against one-time clock and cache state.  Report both
        # individual legs and the combined medians; every leg still uses the
        # slowest rank for each repetition.
        cand_median_2, cand_reps_2 = critical_us(
            cand_graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        ref_median_2, ref_reps_2 = critical_us(
            ref_graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        ref_abba = statistics.median(ref_reps + ref_reps_2)
        cand_abba = statistics.median(cand_reps + cand_reps_2)
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
                f"abba_A2_reference_us={ref_median_2:.3f} "
                f"reps={[round(x, 3) for x in ref_reps_2]}",
                flush=True,
            )
            print(
                f"abba_B2_candidate_us={cand_median_2:.3f} "
                f"reps={[round(x, 3) for x in cand_reps_2]}",
                flush=True,
            )
            print(
                f"abba_reference_median_us={ref_abba:.3f} "
                f"abba_candidate_median_us={cand_abba:.3f} "
                f"abba_delta_pct={(cand_abba / ref_abba - 1.0) * 100.0:+.2f}",
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
            projection_dtype = (
                torch.float32
                if args.projection_partial_dtype == "fp32"
                else (
                    torch.float16
                    if args.projection_partial_dtype == "fp16"
                    else torch.bfloat16
                )
            )
            projection = _randn(
                (args.rows, args.qkv),
                seed=11000 + rank,
                dtype=projection_dtype,
            )
            expert_shard = _randn(
                (args.rows, args.hidden // world),
                seed=12000 + rank,
                dtype=torch.bfloat16,
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
                lambda: comm.all_reduce(projection, registered=True)
            )
            expert_ag_graph, expert_ag_keepalive = capture_collective(
                lambda: comm.custom_all_gather(expert_shard)
            )
            breakdown = []
            for name, graph in (
                ("RS_32x4096_bf16", rs_graph),
                ("AR_32x25_fp32", stats_graph),
                ("AR_32x1_fp32", norm_graph),
                (
                    f"AR_32x{args.qkv}_{args.projection_partial_dtype}",
                    proj16_graph,
                ),
                ("AG_32x512_bf16", expert_ag_graph),
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
            _ = (
                rs_keepalive,
                stats_keepalive,
                norm_keepalive,
                proj16_keepalive,
                expert_ag_keepalive,
            )
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
