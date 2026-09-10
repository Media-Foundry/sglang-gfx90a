#!/usr/bin/env python3
"""M128 MHC weighted-sum -> RMSNorm fusion oracle on gfx90a.

The production dispatcher gates `gfx90a_mhc_weighted_rmsnorm_triton` on
`global_batch_size == 1`, so the C32/M128 target verify executes
`mhc_weighted_sum_triton` (writing a bf16 [M,4096] tensor) followed by a
separate `_gfx90a_mhc_rmsnorm_kernel` that reads it back.  The fused kernel is
shape-generic and keeps the same intermediate bf16 rounding, so this oracle
measures what removing that round trip is worth and whether both arms stay
bitwise identical.  No production selector is changed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.layernorm.mhc import (
    _gfx90a_mhc_rmsnorm_kernel,
    gfx90a_mhc_pre_mix_triton,
    gfx90a_mhc_weighted_rmsnorm_triton,
    hc_split_sinkhorn4_triton,
    mhc_weighted_sum_triton,
)

HC, H, MIX = 4, 4096, 24
RMS_EPS = 1e-6
SINKHORN_EPS = 1e-6
SINKHORN_ITERS = 20
NORM_EPS = 1e-6
DUMP = Path("/tmp/dsv4_tp8_rowstable_router_all_20260908")


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


def load_real_inputs(
    dump_dir: Path, layer: int, rank: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take the leading `rows` real rows of the layer dump, no tiling."""

    def get(suffix: str) -> torch.Tensor:
        path = dump_dir / f"layer_{layer}_rank_{rank}_{suffix}.pt"
        return torch.load(path, map_location="cuda", weights_only=True)

    residual = get("ffn_mhc_residual")
    if residual.shape[0] < rows:
        raise RuntimeError(f"dump has {residual.shape[0]} rows, need {rows}")
    residual = residual[:rows].contiguous()
    return (
        residual,
        get("hc_ffn_fn").contiguous(),
        get("hc_ffn_scale").contiguous(),
        get("hc_ffn_base").contiguous(),
        get("ffn_norm_weight").bfloat16().contiguous(),
    )


def make_synthetic_inputs(rows: int):
    torch.manual_seed(20260910)
    device = torch.device("cuda")
    residual = torch.randn((rows, HC, H), dtype=torch.bfloat16, device=device) * 0.125
    fn = (torch.randn((MIX, HC * H), dtype=torch.float16, device=device) * 0.0078125).float()
    hc_scale = torch.ones((3,), dtype=torch.float32, device=device)
    hc_base = torch.zeros((MIX,), dtype=torch.float32, device=device)
    norm_weight = torch.ones((H,), dtype=torch.bfloat16, device=device)
    return residual, fn, hc_scale, hc_base, norm_weight


def derive_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> torch.Tensor:
    """Run the production pre-mix + sinkhorn so `pre` has real coefficients."""
    mixes = gfx90a_mhc_pre_mix_triton(residual, fn, RMS_EPS)
    if mixes is None:
        raise RuntimeError("gfx90a pre-mix rejected the oracle shape")
    # Call the Triton reference directly: the gfx90a dispatcher imports the
    # model runner, which this standalone oracle does not need.
    split = hc_split_sinkhorn4_triton(
        mixes, hc_scale, hc_base, HC, SINKHORN_ITERS, SINKHORN_EPS
    )
    if split is None:
        raise RuntimeError("hc_split_sinkhorn4_triton rejected the oracle shape")
    pre = split[0]
    return pre.squeeze(1).contiguous() if pre.ndim == 3 else pre.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=DUMP)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rows", type=int, default=128)
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

    probe = args.dump_dir / f"layer_{args.layer}_rank_{args.rank}_ffn_mhc_residual.pt"
    if probe.exists():
        residual, fn, hc_scale, hc_base, norm_weight = load_real_inputs(
            args.dump_dir, args.layer, args.rank, args.rows
        )
        input_source = "real_dump"
    else:
        residual, fn, hc_scale, hc_base, norm_weight = make_synthetic_inputs(args.rows)
        input_source = "bounded_synthetic"
    rows = residual.shape[0]
    print(f"INPUT source={input_source} rows={rows}", flush=True)

    pre = derive_pre(residual, fn, hc_scale, hc_base)
    if pre.shape != (rows, HC):
        raise RuntimeError(f"expected pre [{rows},{HC}], got {tuple(pre.shape)}")

    # Baseline: exactly the production fallback -- weighted sum writes a bf16
    # [M,4096] tensor, then a second kernel reads it back and normalizes.
    normalized = torch.empty((rows, H), dtype=torch.bfloat16, device=residual.device)

    def baseline_run() -> torch.Tensor:
        summed = mhc_weighted_sum_triton(residual, pre)
        if summed is None:
            raise RuntimeError("weighted-sum rejected its shape")
        _gfx90a_mhc_rmsnorm_kernel[(rows,)](
            summed,
            norm_weight,
            normalized,
            hidden_size=H,
            eps=NORM_EPS,
            BLOCK_H=4096,
            num_warps=8,
        )
        return normalized

    def candidate_run() -> torch.Tensor:
        fused = gfx90a_mhc_weighted_rmsnorm_triton(residual, pre, norm_weight, NORM_EPS)
        if fused is None:
            raise RuntimeError("fused weighted-rmsnorm rejected its shape")
        return fused

    initial = metric(baseline_run().clone(), candidate_run())
    torch.cuda.synchronize()
    print(f"INITIAL {json.dumps(initial, sort_keys=True)}", flush=True)

    record = {"all_exact": True, "max_abs": 0.0, "max_rel_l2": 0.0}
    torch.manual_seed(20260910)
    for _ in range(args.mutations):
        # Bounded perturbation keeps the real distribution while moving every
        # weighted-sum input and re-deriving real pre coefficients.
        residual.add_(torch.randn_like(residual) * 0.00390625)
        pre.copy_(derive_pre(residual, fn, hc_scale, hc_base))
        expected = baseline_run().clone()
        actual = candidate_run()
        torch.cuda.synchronize()
        current = metric(expected, actual)
        record["all_exact"] = bool(record["all_exact"] and current["exact"])
        record["max_abs"] = max(record["max_abs"], float(current["max_abs"]))
        record["max_rel_l2"] = max(record["max_rel_l2"], float(current["rel_l2"]))
    print(f"MUTATIONS n={args.mutations} {json.dumps(record, sort_keys=True)}", flush=True)

    graph_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a):
        graph_baseline_out = baseline_run()
    graph_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_b):
        graph_candidate_out = candidate_run()

    replay = {"all_exact": True, "max_abs": 0.0, "max_rel_l2": 0.0}
    for _ in range(args.graph_replays):
        residual.add_(torch.randn_like(residual) * 0.0009765625)
        pre.copy_(derive_pre(residual, fn, hc_scale, hc_base))
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        current = metric(graph_baseline_out, graph_candidate_out)
        replay["all_exact"] = bool(replay["all_exact"] and current["exact"])
        replay["max_abs"] = max(replay["max_abs"], float(current["max_abs"]))
        replay["max_rel_l2"] = max(replay["max_rel_l2"], float(current["rel_l2"]))
    print(f"GRAPH n={args.graph_replays} {json.dumps(replay, sort_keys=True)}", flush=True)

    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for round_index in range(args.rounds):
        order = ("baseline", "candidate", "candidate", "baseline")
        if round_index % 2:
            order = tuple(reversed(order))
        for arm in order:
            graph = graph_a if arm == "baseline" else graph_b
            samples[arm].append(elapsed(graph, args.warmup, args.iterations))

    timings = {
        name: {
            "median": statistics.median(values),
            "trimmed_mean": trimmed(values),
            "min": min(values),
            "max": max(values),
        }
        for name, values in samples.items()
    }
    saved = timings["baseline"]["median"] - timings["candidate"]["median"]
    for name, stats in timings.items():
        print(
            f"{name:10s} median={stats['median']:9.3f} us "
            f"trimmed={stats['trimmed_mean']:9.3f} us",
            flush=True,
        )
    print(
        f"SAVED median={saved:.3f} us/boundary "
        f"({saved / timings['baseline']['median'] * 100.0:.2f}%)",
        flush=True,
    )

    payload = {
        "input_source": input_source,
        "rows": rows,
        "layer": args.layer,
        "rank": args.rank,
        "initial": initial,
        "mutations": {"n": args.mutations, **record},
        "graph_replays": {"n": args.graph_replays, **replay},
        "timings_us": timings,
        "samples_us": samples,
        "saved_us_per_boundary": saved,
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
