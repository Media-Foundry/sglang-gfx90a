#!/usr/bin/env python3
"""Standalone exact/ABBA oracle for TP4 M32 E256 top6 A4 sorting."""

import argparse
import statistics

import torch
from aiter.fused_moe import moe_sorting

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    reconstruct_topk_from_counts,
)
from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def candidate_module():
    return load_jit(
        "gfx90a_m32_a4_sorter_oracle",
        cuda_files=["deepseek_v4/gfx90a_m32_a4_sorter_oracle.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aM32A4SorterOracle::run")],
        extra_cuda_cflags=["-O3"],
    )


def baseline(topk_ids, topk_weights):
    return moe_sorting(
        topk_ids, topk_weights, 256, 4096, torch.bfloat16, block_size=4
    )[:4]


def candidate(module, topk_ids, topk_weights, outputs=None):
    if outputs is None:
        outputs = (
            torch.empty(1210, dtype=torch.int32, device="cuda"),
            torch.empty(1210, dtype=torch.float32, device="cuda"),
            torch.empty(303, dtype=torch.int32, device="cuda"),
            torch.empty(2, dtype=torch.int32, device="cuda"),
        )
    module.run(topk_ids, topk_weights, *outputs)
    return outputs


def assert_exact(a, b, label):
    if not torch.equal(a[3], b[3]):
        raise RuntimeError(f"{label} num_valid mismatch {a[3].cpu()} {b[3].cpu()}")
    valid = int(a[3][0].item())
    blocks = (valid + 3) // 4
    for name, av, bv in (
        ("sorted_ids", a[0][:valid], b[0][:valid]),
        ("sorted_weights", a[1][:valid], b[1][:valid]),
        ("sorted_experts", a[2][:blocks], b[2][:blocks]),
    ):
        if not torch.equal(av, bv):
            mismatch = int(torch.count_nonzero(av != bv).item())
            raise RuntimeError(f"{label} {name} mismatch count={mismatch}")


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    torch.cuda.synchronize()
    return graph, outputs


def time_graph(graph, warmup, iterations):
    for _ in range(warmup): graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations): graph.replay()
    end.record(); end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def trimmed(values):
    values = sorted(values)
    return statistics.mean(values[1:-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recorder", default="/tmp/expert_distribution_recorder_1787803355.1855972.pt")
    p.add_argument("--mutations", type=int, default=100)
    p.add_argument("--replays", type=int, default=1000)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iterations", type=int, default=500)
    args = p.parse_args()

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    real_ids = reconstruct_topk_from_counts(payload["logical_count"][37, 34] // 8).cuda()
    ids = real_ids.clone()
    weights = torch.rand((32, 6), dtype=torch.float32, device="cuda")
    module = candidate_module()

    torch.manual_seed(20260830)
    for mutation in range(args.mutations):
        if mutation == 0:
            ids.copy_(real_ids)
        else:
            scores = torch.rand((32, 256), device="cuda")
            ids.copy_(torch.topk(scores, 6, dim=1, sorted=False).indices.int())
        weights.uniform_()
        a = baseline(ids, weights); b = candidate(module, ids, weights)
        torch.cuda.synchronize(); assert_exact(a, b, f"mutation={mutation}")
    print(f"CORRECTNESS mutations={args.mutations} prefix_metadata_exact=True")

    ids.copy_(real_ids); weights.uniform_()
    graph_a, out_a = capture(lambda: baseline(ids, weights))
    cand_outputs = candidate(module, ids, weights)
    graph_b, out_b = capture(lambda: candidate(module, ids, weights, cand_outputs))
    for replay in range(args.replays):
        graph_a.replay(); graph_b.replay()
        if replay % 100 == 99:
            torch.cuda.synchronize(); assert_exact(out_a, out_b, f"replay={replay+1}")
    print(f"GRAPH replays={args.replays} exact=True")

    values = {"A": [], "B": []}
    for _ in range(args.rounds):
        for name in ("A", "B", "B", "A"):
            values[name].append(time_graph(graph_a if name == "A" else graph_b, 30, args.iterations))
    av, bv = trimmed(values["A"]), trimmed(values["B"])
    saving = av - bv
    print(f"RESULT production_us={av:.3f} candidate_us={bv:.3f} saving_us={saving:.3f} ratio={bv/av:.4f}")
    print(f"DECISION pass={saving >= 8.0 and bv <= 0.60 * av}")


if __name__ == "__main__": main()
