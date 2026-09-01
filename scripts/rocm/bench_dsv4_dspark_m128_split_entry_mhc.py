#!/usr/bin/env python3
"""DSpark gamma-three M128 -> M96/M32 production entry-MHC oracle."""

from __future__ import annotations

import argparse
import statistics

import torch

import sglang.kernels.ops.layernorm.mhc as mhc_module
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre


M, DRAFT_M, ANCHOR_M, HC, H, MIX = 128, 96, 32, 4, 4096, 24


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def elapsed(graph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--with-producers", action="store_true")
    args = parser.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000 or args.rounds != 7:
        raise ValueError("formal oracle requires 100 mutations, 1000 replays, 7 rounds")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"requires gfx90a, got {arch}")

    # The production op only consults TP state for optional symmetric output
    # allocation. Disable that allocator in this local row-math oracle.
    mhc_module.get_tp_group = lambda: None
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    x = torch.randn((M, H), generator=generator, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(
        (M, HC, H), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    post = torch.sigmoid(
        torch.randn((M, HC), generator=generator, device="cuda")
    )
    comb = torch.softmax(
        torch.randn((M, HC, HC), generator=generator, device="cuda"), dim=1
    )
    fn = (
        torch.randn((MIX, HC * H), generator=generator, device="cuda")
        * 0.0078125
    )
    scale = torch.ones((3,), dtype=torch.float32, device="cuda")
    base = torch.zeros((MIX,), dtype=torch.float32, device="cuda")
    norm = torch.ones((H,), dtype=torch.bfloat16, device="cuda")
    projection_weights = tuple(
        torch.randn(
            (n, H), generator=generator, device="cuda", dtype=torch.bfloat16
        ).mul_(0.015625)
        for n in ((1536, 2048, 512, 64) if args.with_producers else ())
    )
    draft_rows = torch.tensor(
        [row for row in range(M) if row % 4], dtype=torch.int64, device="cuda"
    )
    anchor_rows = torch.arange(0, M, 4, dtype=torch.int64, device="cuda")

    def with_producers(mhc_output):
        layer_input = mhc_output[3]
        projections = tuple(
            torch.mm(layer_input, weight.t()) for weight in projection_weights
        )
        return (*mhc_output, *projections)

    def run_rows(rows):
        return with_producers(mhc_fused_post_pre(
            x.index_select(0, rows),
            residual.index_select(0, rows),
            post.index_select(0, rows),
            comb.index_select(0, rows),
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
        ))

    def run_full():
        return with_producers(mhc_fused_post_pre(
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
        ))

    graph_full, full = capture(run_full)
    graph_draft, draft = capture(lambda: run_rows(draft_rows))
    graph_anchor, anchor = capture(lambda: run_rows(anchor_rows))
    x_delta = torch.sin(
        torch.arange(x.numel(), dtype=torch.float32, device="cuda")
    ).view_as(x).to(torch.bfloat16)
    residual_delta = torch.cos(
        torch.arange(residual.numel(), dtype=torch.float32, device="cuda")
    ).view_as(residual).to(torch.bfloat16)

    def exact_rows() -> bool:
        return all(
            torch.equal(part, whole.index_select(0, draft_rows))
            for whole, part in zip(full, draft)
        ) and all(
            torch.equal(part, whole.index_select(0, anchor_rows))
            for whole, part in zip(full, anchor)
        )

    mutation_failures = 0
    for mutation in range(args.mutations):
        alpha = ((mutation * 1543 + 17) % 2047 - 1023) / 32768.0
        x.add_(x_delta, alpha=alpha)
        residual.add_(residual_delta, alpha=alpha * 0.25)
        graph_full.replay(); graph_draft.replay(); graph_anchor.replay()
        torch.cuda.synchronize()
        mutation_failures += int(not exact_rows())

    graph_full.replay(); graph_draft.replay(); graph_anchor.replay()
    torch.cuda.synchronize()
    stable = [tensor.clone() for tensor in (*full, *draft, *anchor)]
    for _ in range(args.graph_replays):
        graph_full.replay(); graph_draft.replay(); graph_anchor.replay()
    torch.cuda.synchronize()
    replay_stable = all(
        torch.equal(expected, actual)
        for expected, actual in zip(stable, (*full, *draft, *anchor))
    )

    full_a1, draft_b1, anchor_c1, anchor_c2, draft_b2, full_a2 = (
        [] for _ in range(6)
    )
    for _ in range(args.rounds):
        full_a1.append(elapsed(graph_full, args.warmup, args.iterations))
        draft_b1.append(elapsed(graph_draft, args.warmup, args.iterations))
        anchor_c1.append(elapsed(graph_anchor, args.warmup, args.iterations))
        anchor_c2.append(elapsed(graph_anchor, args.warmup, args.iterations))
        draft_b2.append(elapsed(graph_draft, args.warmup, args.iterations))
        full_a2.append(elapsed(graph_full, args.warmup, args.iterations))

    full_us = statistics.median(full_a1 + full_a2)
    draft_us = statistics.median(draft_b1 + draft_b2)
    anchor_us = statistics.median(anchor_c1 + anchor_c2)
    print(
        f"with_producers={args.with_producers} exact={exact_rows()} "
        f"mutation_failures={mutation_failures}/{args.mutations} "
        f"replay_stable={replay_stable} replays={args.graph_replays}",
        flush=True,
    )
    print(f"full_a1={[round(v,3) for v in full_a1]}", flush=True)
    print(f"draft_b1={[round(v,3) for v in draft_b1]}", flush=True)
    print(f"anchor_c1={[round(v,3) for v in anchor_c1]}", flush=True)
    print(f"anchor_c2={[round(v,3) for v in anchor_c2]}", flush=True)
    print(f"draft_b2={[round(v,3) for v in draft_b2]}", flush=True)
    print(f"full_a2={[round(v,3) for v in full_a2]}", flush=True)
    print(
        f"full_us={full_us:.3f} draft_us={draft_us:.3f} "
        f"anchor_us={anchor_us:.3f} serial_split_us={draft_us+anchor_us:.3f} "
        f"hideable_us={full_us-anchor_us:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
