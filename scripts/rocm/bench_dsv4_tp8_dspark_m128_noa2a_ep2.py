#!/usr/bin/env python3
"""M128 component oracle for TP8 EP1 versus no-A2A EP2/expert-TP4.

This is a layout cost screen, not production wiring or a cross-layout output
oracle.  Two owner groups are assumed to execute concurrently, so candidate
latency is the slower owner.  Initial hidden quantization, the final TP8
collective, shared expert and sorter cost are excluded.
"""

import argparse
import json
import statistics

import torch

import scripts.rocm.bench_dsv4_tp4_m32_noa2a_ep2_oracle as base
from scripts.rocm.bench_dsv4_tp4_m32_paged_decode_geometry import capture, time_graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distribution", choices=("balanced", "skewed"), required=True)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--mutations", type=int, default=100)
    p.add_argument("--graph-replays", type=int, default=1000)
    args = p.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000:
        raise ValueError("formal screen requires 100 mutations and 1000 graph replays")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("gfx90a required")

    base.M = 128
    base.LDS_LUT = False
    torch.manual_seed(20260910)
    probability = torch.ones(256)
    if args.distribution == "skewed":
        probability[:32] = 16
    ids = torch.stack(
        [torch.multinomial(probability, 6, replacement=False) for _ in range(base.M)]
    ).to(torch.int32).cuda()
    counts = torch.bincount(ids.cpu().long().flatten(), minlength=256)
    owners = base.balanced_owners(counts, 2)
    xq = torch.randint(-127, 128, (base.M, 4096), dtype=torch.int8, device="cuda")
    xs = torch.rand((base.M, 128), device="cuda") * 0.01
    weights = torch.rand((base.M, 6), device="cuda")
    stages = {
        "A": base.make_stage(
            "A", 256, 256, base.full_metadata(ids), xq, xs, weights, 832, 832, False
        )
    }
    for owner, experts in enumerate(owners):
        stages[f"B{owner}"] = base.make_stage(
            f"B{owner}", 128, 512, base.owner_metadata(ids, experts),
            xq, xs, weights, 832, 832, True,
        )

    graphs = {}
    for name, stage in stages.items():
        for mutation in range(args.mutations):
            xq.random_(-127, 128)
            xs.uniform_(0.001, 0.01)
            weights.uniform_()
            stage.full()
        torch.cuda.synchronize()
        def full(stage=stage):
            stage.full()
            return stage.output
        graphs[name], out = capture(full)
        expected = out.clone()
        for _ in range(args.graph_replays):
            graphs[name].replay()
        torch.cuda.synchronize()
        if not torch.equal(expected, out):
            raise AssertionError(f"{name} graph replay drift")

    samples = {name: [] for name in stages}
    for name in ("A", "B0", "B1", "B1", "B0", "A"):
        value = time_graph(graphs[name], 20, args.iterations)
        samples[name].append(value)
        print("ABBA_SAMPLE", name, value, flush=True)
    means = {name: statistics.mean(values) for name, values in samples.items()}
    candidate = max(means["B0"], means["B1"])
    report = {
        "distribution": args.distribution,
        "baseline_us": means["A"],
        "owner_us": [means["B0"], means["B1"]],
        "candidate_rankmax_us": candidate,
        "gain_pct": (means["A"] / candidate - 1.0) * 100.0,
        "active_experts": int((counts > 0).sum()),
        "max_occupancy": int(counts.max()),
        "scans": {name: int(stage.metadata.sorted_experts.numel()) for name, stage in stages.items()},
        "correctness": "100 input mutations executed per shape; 1000 graph replays bitwise stable; no cross-layout equality claim",
    }
    print("RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
