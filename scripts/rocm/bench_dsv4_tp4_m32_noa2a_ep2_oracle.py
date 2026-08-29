#!/usr/bin/env python3
"""Component oracle for TP4 M32 no-A2A expert-owner layouts.

This is deliberately not production wiring.  It reconstructs one real,
diverse M32/top-6 route and compares the current per-rank routed stage

  A: E256, expert-TP4 I512, all assignments

with two or four candidate expert-owner groups

  EP2: E128, expert-TP2 I1024, only owned assignments;
  EP4: E64, expert-TP1 I2048, only owned assignments.

Ranks within an owner group have the same kernel shape, so the candidate
compute lower bound is the slowest owner.  It excludes the final TP/global
collective and cannot establish end-to-end numerical equivalence because this
standalone oracle does not load and concatenate all four real TP weight
shards.  Packed FP4 weights, INT8 activation quantization, A4 sorting and the
fixed-slot reduction implementation are otherwise unchanged.  B includes the
partial-buffer clear required by the existing masked grouped-down protocol.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    Metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


M, T, H, N = 32, 6, 4096, 4096
ASSIGNMENTS, ROWS, WAVES, LDS_LUT = 4, 2, 8, 2


@dataclass
class Stage:
    name: str
    e: int
    i: int
    metadata: Metadata
    xq: torch.Tensor
    xs: torch.Tensor
    topk_weights: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    intermediate: torch.Tensor
    owned_intermediate: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    gate_blocks: int
    down_blocks: int
    clear_partial: bool

    def gate(self, blocks: int | None = None) -> None:
        blocks = self.gate_blocks if blocks is None else blocks
        _jit_gate_up_grouped(
            self.e, M, T, self.i, H, ASSIGNMENTS, ROWS, WAVES, blocks, LDS_LUT
        ).run(
            self.xq,
            self.xs,
            self.w13,
            self.s13,
            self.metadata.sorted_ids,
            self.metadata.sorted_experts,
            self.metadata.valid,
            self.intermediate,
            10.0,
        )

    def quant(self) -> None:
        self.iq, self.isc = per_token_group_quant_int8(self.intermediate, 32)

    def owned_quant_ideal(self) -> None:
        """Optimistic cost for quantizing only valid owner assignments."""
        per_token_group_quant_int8(self.owned_intermediate, 32)

    def down(self, blocks: int | None = None) -> None:
        blocks = self.down_blocks if blocks is None else blocks
        if self.clear_partial:
            self.partial.zero_()
        _jit_down_grouped(
            self.e, M, T, N, self.i, ASSIGNMENTS, ROWS, WAVES, blocks, LDS_LUT
        ).run_partial(
            self.iq,
            self.isc,
            self.w2,
            self.s2,
            self.metadata.sorted_ids,
            self.metadata.sorted_experts,
            self.metadata.valid,
            self.topk_weights,
            self.partial,
        )

    def reduce(self) -> None:
        _jit_down_grouped(
            self.e,
            M,
            T,
            N,
            self.i,
            ASSIGNMENTS,
            ROWS,
            WAVES,
            self.down_blocks,
            LDS_LUT,
        ).reduce(self.partial, self.output)

    def full(self) -> None:
        self.gate()
        self.quant()
        self.down()
        self.reduce()


def owner_metadata(topk_ids: torch.Tensor, owned: list[int]) -> Metadata:
    """Make A4 metadata for one expert owner, preserving original slot IDs."""
    buckets: list[list[int]] = [[] for _ in range(len(owned))]
    remap = {expert: local for local, expert in enumerate(owned)}
    for token, experts in enumerate(topk_ids.cpu().tolist()):
        for slot, expert in enumerate(experts):
            if expert in remap:
                buckets[remap[expert]].append((slot << 24) | token)
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), ASSIGNMENTS):
            block = bucket[offset : offset + ASSIGNMENTS]
            ids.extend(block)
            ids.extend([M] * (ASSIGNMENTS - len(block)))
            experts.append(expert)
    device = topk_ids.device
    return Metadata(
        ASSIGNMENTS,
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


def balanced_owners(counts: torch.Tensor, num_groups: int) -> tuple[list[int], ...]:
    """Balance A4 scans and assignments with equal expert counts per group."""
    groups: list[list[int]] = [[] for _ in range(num_groups)]
    scans = [0] * num_groups
    assignments = [0] * num_groups
    experts_per_group = 256 // num_groups
    ordered = sorted(
        range(256),
        key=lambda expert: (
            (int(counts[expert]) + ASSIGNMENTS - 1) // ASSIGNMENTS,
            int(counts[expert]),
            -expert,
        ),
        reverse=True,
    )
    for expert in ordered:
        choices = [
            group for group in range(num_groups)
            if len(groups[group]) < experts_per_group
        ]
        group = min(
            choices,
            key=lambda value: (scans[value], assignments[value], len(groups[value])),
        )
        count = int(counts[expert])
        groups[group].append(expert)
        scans[group] += (count + ASSIGNMENTS - 1) // ASSIGNMENTS
        assignments[group] += count
    return tuple(sorted(group) for group in groups)


def full_metadata(topk_ids: torch.Tensor) -> Metadata:
    buckets: list[list[int]] = [[] for _ in range(256)]
    for token, experts in enumerate(topk_ids.cpu().tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), ASSIGNMENTS):
            block = bucket[offset : offset + ASSIGNMENTS]
            ids.extend(block)
            ids.extend([M] * (ASSIGNMENTS - len(block)))
            experts.append(expert)
    device = topk_ids.device
    return Metadata(
        ASSIGNMENTS,
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


def make_stage(
    name: str,
    e: int,
    i: int,
    metadata: Metadata,
    xq: torch.Tensor,
    xs: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_blocks: int,
    down_blocks: int,
    clear_partial: bool,
) -> Stage:
    valid_assignments = int(
        (metadata.sorted_ids.remainder(1 << 24) < M).sum().item()
    )
    return Stage(
        name=name,
        e=e,
        i=i,
        metadata=metadata,
        xq=xq,
        xs=xs,
        topk_weights=topk_weights,
        w13=torch.randint(
            0, 256, (e, 2 * i, H // 2), dtype=torch.uint8, device="cuda"
        ),
        s13=torch.full(
            (e, 2 * i, H // 32), 127, dtype=torch.uint8, device="cuda"
        ),
        w2=torch.randint(
            0, 256, (e, N, i // 2), dtype=torch.uint8, device="cuda"
        ),
        s2=torch.full(
            (e, N, i // 32), 127, dtype=torch.uint8, device="cuda"
        ),
        intermediate=torch.empty((M, T, i), dtype=torch.bfloat16, device="cuda"),
        owned_intermediate=torch.empty(
            (valid_assignments, i), dtype=torch.bfloat16, device="cuda"
        ),
        partial=torch.empty((M, T, N), dtype=torch.float32, device="cuda"),
        output=torch.empty((M, N), dtype=torch.bfloat16, device="cuda"),
        gate_blocks=gate_blocks,
        down_blocks=down_blocks,
        clear_partial=clear_partial,
    )


def time_us(fn, warmup: int, iterations: int) -> float:
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


def trimmed(values: list[float]) -> float:
    return statistics.mean(sorted(values)[1:-1])


def describe_metadata(name: str, metadata: Metadata) -> None:
    encoded = metadata.sorted_ids.cpu()
    valid = int((encoded.remainder(1 << 24) < M).sum())
    scans = int(metadata.sorted_experts.numel())
    active = int(metadata.sorted_experts.unique().numel())
    print(
        f"ROUTE profile={name} active={active} assignments={valid} "
        f"scans={scans} padded={scans * ASSIGNMENTS - valid}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--recorded-world-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="greedily balance A4 scans instead of contiguous expert IDs",
    )
    parser.add_argument("--owner-groups", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--candidate-blocks",
        type=int,
        nargs="+",
        default=(416, 624, 832, 1040, 1248, 1664, 2080),
        help="gate/down block counts to sweep for owner candidates",
    )
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a required, got {arch}")

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.recorded_world_size) != 0):
        raise RuntimeError("recorded counts are not divisible by world size")
    counts = raw // args.recorded_world_size
    topk_ids = reconstruct_topk_from_counts(counts).cuda()
    owners = (
        balanced_owners(counts, args.owner_groups)
        if args.balanced
        else tuple(
            list(range(group * (256 // args.owner_groups),
                       (group + 1) * (256 // args.owner_groups)))
            for group in range(args.owner_groups)
        )
    )
    print(f"SPLIT mode={'balanced' if args.balanced else 'contiguous'}", flush=True)
    metadata = {"A": full_metadata(topk_ids)}
    metadata.update({
        f"B{group}": owner_metadata(topk_ids, owner)
        for group, owner in enumerate(owners)
    })
    for name, value in metadata.items():
        describe_metadata(name, value)

    torch.manual_seed(20260830)
    x = torch.randn((M, H), dtype=torch.bfloat16, device="cuda")
    xq, xs = per_token_group_quant_int8(x, 32)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device="cuda")
    stages = {
        "A": make_stage(
            "A", 256, 512, metadata["A"], xq, xs, topk_weights,
            2080, 832, False,
        ),
    }
    local_e = 256 // args.owner_groups
    local_i = 512 * args.owner_groups
    for group in range(args.owner_groups):
        name = f"B{group}"
        stages[name] = make_stage(
            name, local_e, local_i, metadata[name], xq, xs, topk_weights,
            2080, 832, True,
        )

    # Compile and populate consumer inputs once before the isolated sweeps.
    for stage in stages.values():
        stage.full()
    torch.cuda.synchronize()

    candidate_blocks = tuple(args.candidate_blocks)
    candidate_names = tuple(f"B{group}" for group in range(args.owner_groups))
    for name in candidate_names:
        stage = stages[name]
        gate_results: dict[int, list[float]] = {}
        down_results: dict[int, list[float]] = {}
        for blocks in candidate_blocks:
            gate_results[blocks] = [
                time_us(lambda b=blocks: stage.gate(b), args.warmup, args.iterations)
                for _ in range(3)
            ]
            stage.gate()
            stage.quant()
            down_results[blocks] = [
                time_us(lambda b=blocks: stage.down(b), args.warmup, args.iterations)
                for _ in range(3)
            ]
        stage.gate_blocks = min(gate_results, key=lambda b: statistics.median(gate_results[b]))
        stage.down_blocks = min(down_results, key=lambda b: statistics.median(down_results[b]))
        for kind, results in (("gate", gate_results), ("down", down_results)):
            for blocks, values in results.items():
                print(
                    f"SWEEP profile={name} stage={kind} blocks={blocks} "
                    f"samples_us=" + ",".join(f"{v:.3f}" for v in values)
                    + f" median_us={statistics.median(values):.3f}",
                    flush=True,
                )
        print(
            f"SELECT profile={name} gate_blocks={stage.gate_blocks} "
            f"down_blocks={stage.down_blocks}",
            flush=True,
        )

    # ABBA-like A/B0/B1/B1/B0/A ordering.  Report rank/group slowest B.
    samples = {name: [] for name in stages}
    component_samples = {
        name: {
            part: []
            for part in ("gate", "quant", "owned_quant_ideal", "down", "reduce")
        }
        for name in stages
    }
    order = ("A",) + candidate_names + tuple(reversed(candidate_names)) + ("A",)
    for _ in range(args.rounds):
        for name in order:
            stage = stages[name]
            samples[name].append(time_us(stage.full, args.warmup, args.iterations))
            stage.gate()
            component_samples[name]["gate"].append(
                time_us(stage.gate, args.warmup, args.iterations)
            )
            component_samples[name]["quant"].append(
                time_us(stage.quant, args.warmup, args.iterations)
            )
            component_samples[name]["owned_quant_ideal"].append(
                time_us(stage.owned_quant_ideal, args.warmup, args.iterations)
            )
            stage.quant()
            component_samples[name]["down"].append(
                time_us(stage.down, args.warmup, args.iterations)
            )
            component_samples[name]["reduce"].append(
                time_us(stage.reduce, args.warmup, args.iterations)
            )

    summary = {}
    optimistic_summary = {}
    for name in stages:
        values = samples[name]
        summary[name] = trimmed(values)
        print(
            f"RESULT profile={name} gate_blocks={stages[name].gate_blocks} "
            f"down_blocks={stages[name].down_blocks} samples_us="
            + ",".join(f"{v:.3f}" for v in values)
            + f" median_us={statistics.median(values):.3f} "
            f"trimmed_mean_us={summary[name]:.3f}",
            flush=True,
        )
        for part, part_values in component_samples[name].items():
            print(
                f"COMPONENT profile={name} stage={part} "
                f"median_us={statistics.median(part_values):.3f} "
                f"trimmed_mean_us={trimmed(part_values):.3f}",
                flush=True,
            )

        dense_quant = trimmed(component_samples[name]["quant"])
        owned_quant = trimmed(component_samples[name]["owned_quant_ideal"])
        optimistic_summary[name] = summary[name] - dense_quant + owned_quant
        print(
            f"OPTIMISTIC profile={name} dense_full_us={summary[name]:.3f} "
            f"owned_quant_full_us={optimistic_summary[name]:.3f} "
            f"quant_replacement_us={owned_quant:.3f}",
            flush=True,
        )

    candidate = max(optimistic_summary[name] for name in candidate_names)
    gain = (summary["A"] / candidate - 1.0) * 100.0
    print(
        f"DECISION baseline_us={summary['A']:.3f} "
        f"candidate_rankmax_us={candidate:.3f} gain_pct={gain:.3f} "
        f"passes_10pct={gain >= 10.0} correctness=performance_lower_bound_only "
        "excluded=collectives,real_weight_shard_concatenation",
        flush=True,
    )


if __name__ == "__main__":
    main()
