#!/usr/bin/env python3
"""Read-only TP4/M32 FFN-boundary rank-max diagnostic.

The service-marker half parses a trace produced by the current TP4 BS32
profile with real diverse requests.  The component half replays the exact
gfx90a MHC decomposition on the real layer-20 M32 dump.  No model selector or
production source is changed.

Run on four idle GCDs::

  HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
    scripts/rocm/bench_dsv4_tp4_ffn_boundary_rankmax_v2.py
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from sglang.kernels.ops.layernorm.mhc import (
    _gfx90a_mhc_rmsnorm_kernel,
    gfx90a_mhc_pre_mix_from_partials_triton,
    hc_split_sinkhorn,
    mhc_post_combine_rms_triton,
    mhc_weighted_sum_triton,
    mhc_fused_post_pre,
)
import sglang.kernels.ops.layernorm.mhc as mhc_module


TRACE_RE = re.compile(
    r"rank=(?P<rank>\d+).*?deltas_us=(?P<coarse>\[[^\]]+\])"
    r".*?moe_us=(?P<moe>\[[^\]]+\])"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("/tmp/dsv4_tp4_marker.log"))
    parser.add_argument(
        "--dump-dir", type=Path, default=Path("/tmp/dsv4_ffn_dump.f3ZQ89")
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=7)
    return parser.parse_args()


def parse_service_groups(path: Path) -> list[dict[str, float]]:
    groups: list[dict[int, tuple[list[float], list[float]]]] = []
    current: dict[int, tuple[list[float], list[float]]] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = TRACE_RE.search(line)
        if match is None:
            continue
        rank = int(match.group("rank"))
        coarse = [float(x) for x in ast.literal_eval(match.group("coarse"))]
        moe = [float(x) for x in ast.literal_eval(match.group("moe"))]
        if len(coarse) < 7 or len(moe) < 10:
            continue
        if rank in current:
            current = {}
        current[rank] = (coarse, moe)
        if len(current) == 4:
            groups.append(current)
            current = {}

    reports: list[dict[str, float]] = []
    for group in groups:
        # Exclude capture/JIT outliers; the accepted hot layer is about 1.2 ms.
        if max(values[0][2] for values in group.values()) > 1000.0:
            continue
        ffn_mhc = [values[0][5] for values in group.values()]
        router = [values[1][1] for values in group.values()]
        topk = [values[1][2] for values in group.values()]
        routed = [values[1][3] for values in group.values()]
        join = [values[1][4] for values in group.values()]
        add = [values[1][5] for values in group.values()]
        ar = [values[1][8] for values in group.values()]
        reports.append(
            {
                "ffn_mhc_rankmax": max(ffn_mhc),
                "ffn_mhc_spread": max(ffn_mhc) - min(ffn_mhc),
                "router_rankmax": max(router),
                "router_spread": max(router) - min(router),
                "topk_rankmax": max(topk),
                "routed_rankmax": max(routed),
                "join_rankmax": max(join),
                "add_rankmax": max(add),
                "ar_rankmax": max(ar),
                # Since all ranks execute the same AR shape, duration spread is
                # a conservative upper bound on arrival wait at its rendezvous.
                "ar_arrival_upper": max(ar) - min(ar),
            }
        )
    return reports


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    return graph, outputs


def capture_comm(comm: CustomAllreduce, fn):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture():
        with torch.cuda.graph(graph):
            outputs = fn()
    dist.barrier()
    return graph, outputs


def rankmax(graph: torch.cuda.CUDAGraph, args: argparse.Namespace, world: int):
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.reps):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            graph.replay()
        end.record()
        end.synchronize()
        local = start.elapsed_time(end) * 1000.0 / args.iters
        gathered: list[float | None] = [None] * world
        dist.all_gather_object(gathered, local)
        samples.append(max(float(value) for value in gathered))
    return samples


def rankmax_once(graph: torch.cuda.CUDAGraph, iters: int, world: int) -> float:
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    local = start.elapsed_time(end) * 1000.0 / iters
    gathered: list[float | None] = [None] * world
    dist.all_gather_object(gathered, local)
    return max(float(value) for value in gathered)


def load_dump(args: argparse.Namespace, rank: int, name: str) -> torch.Tensor:
    path = args.dump_dir / f"layer_{args.layer}_rank_{rank}_{name}.pt"
    return torch.load(path, map_location="cpu", weights_only=True).cuda().contiguous()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"requires TP4, got {world}")
    # mhc_fused_post_pre only uses the TP group to request symmetric output
    # allocation.  The standalone oracle has no SGLang parallel-state object;
    # allocation symmetry is disabled here, so a dummy group faithfully selects
    # the production kernel sequence without introducing a collective.
    mhc_module.get_tp_group = lambda: None

    x = load_dump(args, rank, "attn_out")
    residual = load_dump(args, rank, "ffn_mhc_residual")
    post = load_dump(args, rank, "ffn_mhc_post")
    comb = load_dump(args, rank, "ffn_mhc_comb")
    fn = load_dump(args, rank, "hc_ffn_fn")
    scale = load_dump(args, rank, "hc_ffn_scale")
    base = load_dump(args, rank, "hc_ffn_base")
    norm = load_dump(args, rank, "ffn_norm_weight")
    router_weight = load_dump(args, rank, "router_weight")

    post_result = mhc_post_combine_rms_triton(x, residual, post, comb)
    if post_result is None:
        raise RuntimeError("MHC post rejected real M32 dump")
    residual_cur, rms_partials = post_result
    mixes = gfx90a_mhc_pre_mix_from_partials_triton(
        residual_cur, fn, rms_partials, 1e-6
    )
    if mixes is None:
        raise RuntimeError("MHC pre-mix rejected real M32 dump")
    pre, next_post, next_comb = hc_split_sinkhorn(
        mixes, scale, base, 4, 20, 1e-6, None
    )
    weighted = mhc_weighted_sum_triton(residual_cur, pre.squeeze(1))
    if weighted is None:
        raise RuntimeError("MHC weighted sum rejected real M32 dump")
    normalized = torch.empty_like(weighted)
    _gfx90a_mhc_rmsnorm_kernel[(32,)](
        weighted,
        norm,
        normalized,
        hidden_size=4096,
        eps=1e-6,
        BLOCK_H=4096,
        num_warps=8,
    )
    _ = F.linear(normalized, router_weight)
    torch.cuda.synchronize()

    # Each graph exposes one precise producer/consumer segment.  The full graph
    # is built from exactly the same functions and therefore also serves as the
    # component-boundary correctness oracle.
    graph_post, out_post = capture(
        lambda: mhc_post_combine_rms_triton(x, residual, post, comb)
    )
    graph_mix, out_mix = capture(
        lambda: gfx90a_mhc_pre_mix_from_partials_triton(
            residual_cur, fn, rms_partials, 1e-6
        )
    )
    graph_sinkhorn, out_sinkhorn = capture(
        lambda: hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6, None)
    )
    graph_weighted, out_weighted = capture(
        lambda: mhc_weighted_sum_triton(residual_cur, pre.squeeze(1))
    )

    def rms_only():
        out = torch.empty_like(weighted)
        _gfx90a_mhc_rmsnorm_kernel[(32,)](
            weighted,
            norm,
            out,
            hidden_size=4096,
            eps=1e-6,
            BLOCK_H=4096,
            num_warps=8,
        )
        return out

    graph_rms, out_rms = capture(rms_only)
    graph_router, out_router = capture(lambda: F.linear(normalized, router_weight))

    def full_boundary():
        result = mhc_post_combine_rms_triton(x, residual, post, comb)
        assert result is not None
        r, partials = result
        m = gfx90a_mhc_pre_mix_from_partials_triton(r, fn, partials, 1e-6)
        assert m is not None
        p, po, co = hc_split_sinkhorn(m, scale, base, 4, 20, 1e-6, None)
        y = mhc_weighted_sum_triton(r, p.squeeze(1))
        assert y is not None
        out = torch.empty_like(y)
        _gfx90a_mhc_rmsnorm_kernel[(32,)](
            y,
            norm,
            out,
            hidden_size=4096,
            eps=1e-6,
            BLOCK_H=4096,
            num_warps=8,
        )
        return r, partials, m, p, po, co, y, out

    graph_full, out_full = capture(full_boundary)

    def production_boundary():
        return mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            norm_weight=norm,
            norm_eps=1e-6,
        )

    # This is the exact backend reached by apply_mhc_post_pre_boundary on
    # gfx90a.  Capture it beside the explicit staged graph so allocator and
    # Python dispatch are absent from both replay measurements.
    production_eager = production_boundary()
    torch.cuda.synchronize()
    graph_production, out_production = capture(production_boundary)

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter CustomAllreduce unavailable")
    ar_input = x.clone()
    ar_output = torch.empty_like(ar_input)
    comm.register_buffer(ar_input)
    graph_ar, _ = capture_comm(
        comm, lambda: comm.all_reduce(ar_input, out=ar_output, registered=True)
    )

    graphs = {
        "post_plus_rms_partials": graph_post,
        "pre_mix": graph_mix,
        "sinkhorn": graph_sinkhorn,
        "weighted_sum": graph_weighted,
        "rmsnorm": graph_rms,
        "router_bf16_structural": graph_router,
        "full_mhc_boundary": graph_full,
        "production_mhc_boundary": graph_production,
        "tp4_ar": graph_ar,
    }
    timings = {name: rankmax(graph, args, world) for name, graph in graphs.items()}

    graph_post.replay(); graph_mix.replay(); graph_sinkhorn.replay()
    graph_weighted.replay(); graph_rms.replay(); graph_full.replay()
    graph_production.replay()
    torch.cuda.synchronize()
    exact = (
        torch.equal(out_post[0], out_full[0])
        and torch.equal(out_post[1], out_full[1])
        and torch.equal(out_mix, out_full[2])
        and all(torch.equal(a, b) for a, b in zip(out_sinkhorn, out_full[3:6]))
        and torch.equal(out_weighted, out_full[6])
        and torch.equal(out_rms, out_full[7])
    )
    # production returns residual, post, comb, layer_input.  The explicit
    # staged graph additionally exposes its internal partials/mixes.
    production_pairs = (
        (out_production[0], out_full[0]),
        (out_production[1], out_full[4]),
        (out_production[2], out_full[5]),
        (out_production[3], out_full[7]),
    )
    production_exact = [torch.equal(a, b) for a, b in production_pairs]
    production_max_abs = [
        float((a.float() - b.float()).abs().max().item())
        for a, b in production_pairs
    ]
    eager_capture_exact = [
        torch.equal(a, b) for a, b in zip(production_eager, out_production)
    ]

    # Fixed teacher-forced state plus 100 bounded hidden mutations catches
    # accidental equivalence at only one activation without changing weights,
    # residual state, or routing metadata.
    original_x = x.clone()
    mutation = torch.linspace(-1.0, 1.0, x.numel(), device=x.device).view_as(x)
    mutation = mutation.to(torch.bfloat16)
    mutation_mismatches = 0
    mutation_max_abs = [0.0] * 4
    for iteration in range(100):
        state = (iteration * 1543 + 17) % 2047
        alpha = (state - 1023) / 32768.0
        x.copy_(original_x)
        x.add_(mutation, alpha=alpha)
        graph_production.replay()
        graph_full.replay()
        torch.cuda.synchronize()
        pairs = (
            (out_production[0], out_full[0]),
            (out_production[1], out_full[4]),
            (out_production[2], out_full[5]),
            (out_production[3], out_full[7]),
        )
        if not all(torch.equal(a, b) for a, b in pairs):
            mutation_mismatches += 1
        for index, (a, b) in enumerate(pairs):
            mutation_max_abs[index] = max(
                mutation_max_abs[index],
                float((a.float() - b.float()).abs().max().item()),
            )
    x.copy_(original_x)

    # Paired ABBA avoids interpreting slow service/process drift as a backend
    # gain.  Both graphs have already received identical warmups above.
    prod_a1, staged_b1, staged_b2, prod_a2 = [], [], [], []
    for _ in range(args.reps):
        prod_a1.append(rankmax_once(graph_production, args.iters, world))
        staged_b1.append(rankmax_once(graph_full, args.iters, world))
        staged_b2.append(rankmax_once(graph_full, args.iters, world))
        prod_a2.append(rankmax_once(graph_production, args.iters, world))
    all_exact: list[bool | None] = [None] * world
    dist.all_gather_object(all_exact, exact)
    correctness_payload = (
        production_exact,
        production_max_abs,
        eager_capture_exact,
        mutation_mismatches,
        mutation_max_abs,
    )
    all_production: list[tuple | None] = [None] * world
    dist.all_gather_object(all_production, correctness_payload)

    if rank == 0:
        service = parse_service_groups(args.trace)
        print(f"component_correctness_all_ranks={all(all_exact)} details={all_exact}")
        print(f"production_vs_staged_correctness={all_production}")
        for name, samples in timings.items():
            print(
                f"component={name} rankmax_us={[round(v, 3) for v in samples]} "
                f"median_us={statistics.median(samples):.3f}"
            )
        print(f"service_trace={args.trace} accepted_hot_groups={len(service)}")
        if service:
            for key in service[0]:
                values = [row[key] for row in service]
                print(
                    f"service={key} values={[round(v, 3) for v in values]} "
                    f"median_us={statistics.median(values):.3f}"
                )
        print(
            "correctness_scope=read_only_real_layer20_tensor_exact_component_chaining; "
            "service_markers_are_current_real-diverse-request_execution and add no math"
        )
        prod_samples = prod_a1 + prod_a2
        staged_samples = staged_b1 + staged_b2
        prod_median = statistics.median(prod_samples)
        staged_median = statistics.median(staged_samples)
        print(f"production_A1_rankmax_us={[round(v, 3) for v in prod_a1]}")
        print(f"staged_B1_rankmax_us={[round(v, 3) for v in staged_b1]}")
        print(f"staged_B2_rankmax_us={[round(v, 3) for v in staged_b2]}")
        print(f"production_A2_rankmax_us={[round(v, 3) for v in prod_a2]}")
        print(
            f"production_median_us={prod_median:.3f} "
            f"staged_median_us={staged_median:.3f} "
            f"saving_us={prod_median - staged_median:.3f} "
            f"gate_20us={'pass' if prod_median - staged_median >= 20.0 else 'fail'}"
        )

    # The pinned Python wrapper may expose no dispose op during interpreter
    # teardown.  The OS owns this short-lived diagnostic communicator.
    comm._ptr = 0
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
