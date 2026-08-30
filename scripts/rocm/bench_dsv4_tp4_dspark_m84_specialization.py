#!/usr/bin/env python3
"""ABBA oracle for extending the accepted TP4/M64 routed stack to DSpark tiers.

The route is reconstructed from a target-only ``per_pass`` expert recorder.
All arms keep A4/R2/G2080/D832, the LDS E2M1 lookup, group-32 activation
quantization and fixed-order FP32 partial reduction.  The only changes are:

* A: generic shuffle gate and generic W8 down;
* B: DPP gate and generic W8 down;
* C: row-prefetch DPP gate and logical-scale W4 down.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_down_grouped_row_prefetch_logical_scale,
    _jit_gate_up_grouped,
    _jit_gate_up_grouped_dpp,
    _jit_gate_up_grouped_row_prefetch,
)
from sglang.kernels.ops.quantization.int8_kernel import (
    _per_token_group_quant_int8,
)


E, M, T, H, I, N = 256, 84, 6, 4096, 512, 4096
ASSIGNMENTS, ROWS, GATE_WAVES = 4, 2, 8
GATE_BLOCKS, DOWN_BLOCKS, LDS_LUT = 2080, 832, 2


def quant_into(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> None:
    groups = x.numel() // 32
    _per_token_group_quant_int8[(groups,)](
        x,
        q,
        scale,
        32,
        32,
        1e-10,
        int8_min=-128,
        int8_max=127,
        BLOCK=32,
        num_warps=1,
        num_stages=1,
    )


def time_us(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def trimmed_mean(values: list[float]) -> float:
    ordered = sorted(values)
    return statistics.fmean(ordered[1:-1])


def main() -> None:
    global M
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", type=Path, required=True)
    parser.add_argument("--pass-index", type=int, default=0)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--recorded-world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=M)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--candidate-gate-blocks", type=int, default=GATE_BLOCKS)
    parser.add_argument("--candidate-down-blocks", type=int, default=DOWN_BLOCKS)
    parser.add_argument("--candidate-down-waves", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    M = args.batch_size

    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")
    if args.rounds != 7 or args.mutations < 100 or args.graph_replays < 1000:
        raise ValueError("formal oracle requires 7 rounds, 100 mutations and 1000 replays")

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.recorded_world_size) != 0):
        raise RuntimeError("recorded counts are not divisible by world size")
    counts = raw // args.recorded_world_size
    topk_ids = reconstruct_topk_from_counts(counts, m=M, topk=T).cuda()
    metadata = make_metadata(topk_ids, assignments=ASSIGNMENTS)
    active = int((counts > 0).sum())
    scans = int(metadata.sorted_experts.numel())
    print(
        f"ROUTE active_experts={active} assignments={int(counts.sum())} "
        f"a4_scans={scans} max_occupancy={int(counts.max())}",
        flush=True,
    )

    torch.manual_seed(20260831)
    device = torch.device("cuda")
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq = torch.empty((M, H), dtype=torch.int8, device=device)
    xscale = torch.empty((M, H // 32), dtype=torch.float32, device=device)
    quant_into(x, xq, xscale)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device
    )
    s13 = torch.full(
        (E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device
    )
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)

    gate_modules = {
        "A": _jit_gate_up_grouped(
            E, M, T, I, H, ASSIGNMENTS, ROWS, GATE_WAVES, GATE_BLOCKS, LDS_LUT
        ),
        "B": _jit_gate_up_grouped_dpp(
            E,
            M,
            T,
            I,
            H,
            ASSIGNMENTS,
            ROWS,
            GATE_WAVES,
            args.candidate_gate_blocks,
            LDS_LUT,
        ),
        "C": _jit_gate_up_grouped_row_prefetch(
            E,
            M,
            T,
            I,
            H,
            ASSIGNMENTS,
            ROWS,
            GATE_WAVES,
            args.candidate_gate_blocks,
            LDS_LUT,
        ),
    }
    down_modules = {
        "generic": _jit_down_grouped(
            E, M, T, N, I, ASSIGNMENTS, ROWS, 8, DOWN_BLOCKS, LDS_LUT
        ),
        "logical": _jit_down_grouped_row_prefetch_logical_scale(
            E,
            M,
            T,
            N,
            I,
            ASSIGNMENTS,
            args.candidate_down_waves,
            args.candidate_down_blocks,
            LDS_LUT,
        ),
    }

    states: dict[str, dict[str, torch.Tensor]] = {}
    stages = {}
    for name in ("A", "B", "C"):
        state = {
            "intermediate": torch.empty(
                (M, T, I), dtype=torch.bfloat16, device=device
            ),
            "iq": torch.empty((M, T, I), dtype=torch.int8, device=device),
            "iscale": torch.empty(
                (M, T, I // 32), dtype=torch.float32, device=device
            ),
            "partial": torch.empty((M, T, N), dtype=torch.float32, device=device),
            "output": torch.empty((M, N), dtype=torch.bfloat16, device=device),
        }
        gate = gate_modules[name]
        down = down_modules["logical" if name == "C" else "generic"]

        def gate_stage(gate=gate, state=state) -> None:
            gate.run(
                xq,
                xscale,
                w13,
                s13,
                metadata.sorted_ids,
                metadata.sorted_experts,
                metadata.valid,
                state["intermediate"],
                10.0,
            )

        def quant_stage(state=state) -> None:
            quant_into(state["intermediate"], state["iq"], state["iscale"])

        def down_stage(down=down, state=state) -> None:
            down.run_partial(
                state["iq"],
                state["iscale"],
                w2,
                s2,
                metadata.sorted_ids,
                metadata.sorted_experts,
                metadata.valid,
                topk_weights,
                state["partial"],
            )

        def reduce_stage(down=down, state=state) -> None:
            down.reduce(state["partial"], state["output"])

        def full_stage(
            gate_stage=gate_stage,
            quant_stage=quant_stage,
            down_stage=down_stage,
            reduce_stage=reduce_stage,
        ) -> None:
            gate_stage()
            quant_stage()
            down_stage()
            reduce_stage()

        states[name] = state
        stages[name] = {
            "gate": gate_stage,
            "quant": quant_stage,
            "down": down_stage,
            "reduce": reduce_stage,
            "full": full_stage,
        }

    def assert_exact(candidate: str, label: str) -> None:
        for tensor_name in ("intermediate", "iq", "iscale", "partial", "output"):
            actual = states[candidate][tensor_name]
            expected = states["A"][tensor_name]
            if not torch.equal(actual, expected):
                diff = (actual.float() - expected.float()).abs()
                raise RuntimeError(
                    f"{label}: {candidate}.{tensor_name} mismatch "
                    f"max_abs={float(diff.max())}"
                )

    for mutation in range(args.mutations):
        x.normal_()
        topk_weights.uniform_()
        quant_into(x, xq, xscale)
        for name in ("A", "B", "C"):
            stages[name]["full"]()
        torch.cuda.synchronize()
        assert_exact("B", f"mutation={mutation}")
        assert_exact("C", f"mutation={mutation}")
    print(f"CORRECT mutations={args.mutations} exact=true", flush=True)

    graphs = {}
    for name in ("A", "B", "C"):
        for _ in range(10):
            stages[name]["full"]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stages[name]["full"]()
        graphs[name] = graph
    x.normal_()
    topk_weights.uniform_()
    quant_into(x, xq, xscale)
    for name in ("A", "B", "C"):
        for _ in range(args.graph_replays):
            graphs[name].replay()
    torch.cuda.synchronize()
    assert_exact("B", "graph")
    assert_exact("C", "graph")
    print(f"GRAPH replays={args.graph_replays} exact=true", flush=True)

    timings = {name: [] for name in ("A", "B", "C")}
    for round_idx in range(args.rounds):
        for name in ("A", "B", "C", "C", "B", "A"):
            timings[name].append(
                time_us(
                    stages[name]["full"],
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )
        print(f"ROUND {round_idx + 1}/{args.rounds}", flush=True)

    report = {
        "route": {
            "active_experts": active,
            "assignments": int(counts.sum()),
            "a4_scans": scans,
            "max_occupancy": int(counts.max()),
        },
        "candidate_geometry": {
            "gate_blocks": args.candidate_gate_blocks,
            "down_blocks": args.candidate_down_blocks,
            "down_waves": args.candidate_down_waves,
        },
        "results": {},
    }
    for name in ("A", "B", "C"):
        values = timings[name]
        result = {
            "samples_us": values,
            "median_us": statistics.median(values),
            "trimmed_mean_us": trimmed_mean(values),
        }
        report["results"][name] = result
        print(
            f"RESULT arm={name} median_us={result['median_us']:.3f} "
            f"trimmed_mean_us={result['trimmed_mean_us']:.3f} samples_us="
            + ",".join(f"{value:.3f}" for value in values),
            flush=True,
        )

    for stage_name in ("gate", "quant", "down", "reduce"):
        for name in ("A", "B", "C"):
            values = [
                time_us(
                    stages[name][stage_name],
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                for _ in range(args.rounds)
            ]
            print(
                f"STAGE arm={name} stage={stage_name} "
                f"median_us={statistics.median(values):.3f}",
                flush=True,
            )

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
