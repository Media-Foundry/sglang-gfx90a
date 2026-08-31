#!/usr/bin/env python3
"""Standalone M128 MHC post->pre two-stage fusion oracle on gfx90a.

The default input repeats the accepted real M32 layer-20/rank-0 dump four
times, preserving real tensor ranges while exercising the physical M128
shape.  No model selector or production wrapper imports this oracle.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.layernorm.gfx90a_m128_mhc_two_stage_oracle import (
    jit_gfx90a_m128_mhc_two_stage_oracle,
)
from sglang.kernels.ops.layernorm.mhc import (
    _gfx90a_mhc_rmsnorm_kernel,
    gfx90a_mhc_pre_mix_from_partials_triton,
    hc_split_sinkhorn,
    mhc_post_combine_rms_triton,
    mhc_weighted_sum_triton,
)


M, HC, H, MIX, SPLITS = 128, 4, 4096, 24, 16
RMS_EPS = 1e-6
SINKHORN_EPS = 1e-6
POST_MULTIPLIER = 2.0
NORM_EPS = 1e-6


def trimmed(values: list[float]) -> float:
    return statistics.fmean(sorted(values)[1:-1])


def metric(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    delta = actual.float() - expected.float()
    denom = torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
    return {
        "exact": torch.equal(expected, actual),
        "max_abs": float(delta.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(delta) / denom),
    }


def elapsed(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def load_m128(dump_dir: Path, layer: int, rank: int, suffix: str) -> torch.Tensor:
    path = dump_dir / f"layer_{layer}_rank_{rank}_{suffix}.pt"
    value = torch.load(path, map_location="cuda", weights_only=True).contiguous()
    if value.shape[0] == 32:
        value = value.repeat((4,) + (1,) * (value.ndim - 1)).contiguous()
    return value


def make_synthetic_inputs() -> tuple[torch.Tensor, ...]:
    """Create bounded, deterministic tensors when the ephemeral dump is gone.

    Generate the large ``fn`` matrix in FP16 first and promote it for the
    baseline so both arms consume exactly the same representable weights.
    """
    torch.manual_seed(20260901)
    device = torch.device("cuda")
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device) * 0.125
    residual = (
        torch.randn((M, HC, H), dtype=torch.bfloat16, device=device) * 0.125
    )
    previous_post = torch.sigmoid(
        torch.randn((M, HC), dtype=torch.float32, device=device)
    )
    previous_comb = torch.softmax(
        torch.randn((M, HC, HC), dtype=torch.float32, device=device), dim=1
    )
    fn_fp16 = (
        torch.randn((MIX, HC * H), dtype=torch.float16, device=device) * 0.0078125
    )
    fn = fn_fp16.float()
    hc_scale = torch.tensor([1.0, 1.0, 1.0], device=device)
    hc_base = torch.zeros((MIX,), dtype=torch.float32, device=device)
    norm_weight = torch.ones((H,), dtype=torch.bfloat16, device=device)
    return (
        x,
        residual,
        previous_post,
        previous_comb,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=Path("/tmp/dsv4_ffn_dump.f3ZQ89"))
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000 or args.rounds != 7:
        raise ValueError("formal oracle requires >=100 mutations, >=1000 replays, 7 rounds")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("this oracle requires gfx90a")

    dump_probe = args.dump_dir / f"layer_{args.layer}_rank_{args.rank}_attn_out.pt"
    if dump_probe.exists():
        x = load_m128(args.dump_dir, args.layer, args.rank, "attn_out")
        residual = load_m128(
            args.dump_dir, args.layer, args.rank, "ffn_mhc_residual"
        )
        previous_post = load_m128(
            args.dump_dir, args.layer, args.rank, "ffn_mhc_post"
        )
        previous_comb = load_m128(
            args.dump_dir, args.layer, args.rank, "ffn_mhc_comb"
        )
        fn = load_m128(args.dump_dir, args.layer, args.rank, "hc_ffn_fn")
        hc_scale = load_m128(args.dump_dir, args.layer, args.rank, "hc_ffn_scale")
        hc_base = load_m128(args.dump_dir, args.layer, args.rank, "hc_ffn_base")
        norm_weight = load_m128(
            args.dump_dir, args.layer, args.rank, "ffn_norm_weight"
        ).bfloat16()
        input_source = "real_dump"
    else:
        (
            x,
            residual,
            previous_post,
            previous_comb,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
        ) = make_synthetic_inputs()
        input_source = "bounded_synthetic"
    print(f"INPUT source={input_source}", flush=True)
    if x.shape != (M, H) or residual.shape != (M, HC, H):
        raise RuntimeError(f"expected M128 inputs, got x={x.shape} residual={residual.shape}")
    if fn.shape != (MIX, HC * H):
        raise RuntimeError(f"expected fn [24,16384], got {fn.shape}")
    fn_fp16 = fn.half().contiguous()

    module = jit_gfx90a_m128_mhc_two_stage_oracle(20)
    candidate = {
        "residual": torch.empty_like(residual),
        "rms_partial": torch.empty((SPLITS, M), dtype=torch.float32, device="cuda"),
        "dot_partial": torch.empty((SPLITS, M, MIX), dtype=torch.float32, device="cuda"),
        "mixes": torch.empty((M, MIX), dtype=torch.float32, device="cuda"),
        "post": torch.empty((M, HC), dtype=torch.float32, device="cuda"),
        "comb": torch.empty((M, HC, HC), dtype=torch.float32, device="cuda"),
        "layer_input": torch.empty((M, H), dtype=torch.bfloat16, device="cuda"),
    }

    def baseline_run():
        post_result = mhc_post_combine_rms_triton(
            x, residual, previous_post, previous_comb
        )
        if post_result is None:
            raise RuntimeError("baseline MHC post rejected M128")
        residual_cur, rms_partial = post_result
        mixes = gfx90a_mhc_pre_mix_from_partials_triton(
            residual_cur, fn, rms_partial, RMS_EPS
        )
        if mixes is None:
            raise RuntimeError("baseline MHC pre-mix rejected M128")
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, HC, 20, SINKHORN_EPS, None
        )
        weighted = mhc_weighted_sum_triton(residual_cur, pre.squeeze(1))
        if weighted is None:
            raise RuntimeError("baseline weighted sum rejected M128")
        layer_input = torch.empty_like(weighted)
        _gfx90a_mhc_rmsnorm_kernel[(M,)](
            weighted,
            norm_weight,
            layer_input,
            hidden_size=H,
            eps=NORM_EPS,
            BLOCK_H=H,
            num_warps=8,
        )
        return residual_cur, mixes.view(M, MIX), post.squeeze(1), comb, layer_input

    def producer_run():
        module.producer(
            x,
            residual,
            previous_post,
            previous_comb,
            fn_fp16,
            candidate["residual"],
            candidate["rms_partial"],
            candidate["dot_partial"],
        )

    def consumer_run():
        module.consumer(
            candidate["residual"],
            candidate["rms_partial"],
            candidate["dot_partial"],
            hc_scale,
            hc_base,
            norm_weight,
            candidate["mixes"],
            candidate["post"],
            candidate["comb"],
            candidate["layer_input"],
            RMS_EPS,
            SINKHORN_EPS,
            POST_MULTIPLIER,
            NORM_EPS,
        )

    def candidate_run():
        producer_run(); consumer_run()
        return (
            candidate["residual"], candidate["mixes"], candidate["post"],
            candidate["comb"], candidate["layer_input"],
        )

    # Compile every backend outside capture.
    baseline_out = baseline_run()
    candidate_out = candidate_run()
    torch.cuda.synchronize()

    names = ("residual", "mixes", "post", "comb", "layer_input")
    initial = {
        name: metric(expected, actual)
        for name, expected, actual in zip(names, baseline_out, candidate_out)
    }
    print(f"INITIAL {json.dumps(initial, sort_keys=True)}", flush=True)

    mutation_metrics = {
        name: {"all_exact": True, "max_abs": 0.0, "max_rel_l2": 0.0}
        for name in names
    }
    torch.manual_seed(20260901)
    for mutation in range(args.mutations):
        # Bounded perturbations keep the real tensor distributions while
        # changing every producer/consumer boundary.
        x.add_(torch.randn_like(x) * 0.00390625)
        residual.add_(torch.randn_like(residual) * 0.00390625)
        previous_post.add_(torch.randn_like(previous_post) * 1e-4)
        previous_comb.add_(torch.randn_like(previous_comb) * 1e-4)
        baseline_out = baseline_run(); candidate_out = candidate_run()
        torch.cuda.synchronize()
        for name, expected, actual in zip(names, baseline_out, candidate_out):
            current = metric(expected, actual)
            record = mutation_metrics[name]
            record["all_exact"] = bool(record["all_exact"] and current["exact"])
            record["max_abs"] = max(float(record["max_abs"]), float(current["max_abs"]))
            record["max_rel_l2"] = max(
                float(record["max_rel_l2"]), float(current["rel_l2"])
            )
    print(f"MUTATIONS n={args.mutations} {json.dumps(mutation_metrics, sort_keys=True)}", flush=True)

    graph_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a):
        graph_baseline_out = baseline_run()
    graph_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_b):
        graph_candidate_out = candidate_run()

    replay_metrics = {
        name: {"all_exact": True, "max_abs": 0.0, "max_rel_l2": 0.0}
        for name in names
    }
    for replay in range(args.graph_replays):
        x.add_(torch.randn_like(x) * 0.0009765625)
        residual.add_(torch.randn_like(residual) * 0.0009765625)
        graph_a.replay(); graph_b.replay()
        torch.cuda.synchronize()
        for name, expected, actual in zip(
            names, graph_baseline_out, graph_candidate_out
        ):
            current = metric(expected, actual)
            record = replay_metrics[name]
            record["all_exact"] = bool(record["all_exact"] and current["exact"])
            record["max_abs"] = max(float(record["max_abs"]), float(current["max_abs"]))
            record["max_rel_l2"] = max(
                float(record["max_rel_l2"]), float(current["rel_l2"])
            )
    print(f"GRAPH n={args.graph_replays} {json.dumps(replay_metrics, sort_keys=True)}", flush=True)

    graph_producer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_producer):
        producer_run()
    graph_consumer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_consumer):
        consumer_run()

    samples = {"baseline": [], "candidate": [], "producer": [], "consumer": []}
    for round_index in range(args.rounds):
        order = ("baseline", "candidate", "candidate", "baseline")
        if round_index % 2:
            order = tuple(reversed(order))
        for arm in order:
            graph = graph_a if arm == "baseline" else graph_b
            samples[arm].append(elapsed(graph, args.warmup, args.iterations))
        samples["producer"].append(
            elapsed(graph_producer, args.warmup, args.iterations)
        )
        samples["consumer"].append(
            elapsed(graph_consumer, args.warmup, args.iterations)
        )

    timings = {
        name: {
            "median_us": statistics.median(values),
            "trimmed_mean_us": trimmed(values),
            "samples_us": values,
        }
        for name, values in samples.items()
    }
    base_us = timings["baseline"]["trimmed_mean_us"]
    candidate_us = timings["candidate"]["trimmed_mean_us"]
    decision = {
        "saved_us": base_us - candidate_us,
        "gain_pct": (base_us / candidate_us - 1.0) * 100.0,
        "passes_36us": base_us - candidate_us >= 36.0,
        "bitwise_all_boundaries": all(
            bool(value["all_exact"]) for value in mutation_metrics.values()
        ),
    }
    report = {
        "shape": {"M": M, "HC": HC, "H": H, "mix": MIX, "splits": SPLITS},
        "initial": initial,
        "mutations": mutation_metrics,
        "graph_replays": replay_metrics,
        "timings": timings,
        "decision": decision,
    }
    print(f"TIMINGS {json.dumps(timings, sort_keys=True)}", flush=True)
    print(f"DECISION {json.dumps(decision, sort_keys=True)}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"REPORT {args.output}", flush=True)


if __name__ == "__main__":
    main()
