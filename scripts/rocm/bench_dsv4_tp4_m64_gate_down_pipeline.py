#!/usr/bin/env python3
"""Standalone two-stream M64 routed gate/down pipeline oracle.

Split the stable expert-block list into two contiguous chunks.  The main
stream computes gate chunk 0 then gate chunk 1, while an auxiliary stream
consumes chunk 0 with the exact LDS-quant down kernel as soon as it is ready.
Both streams join before the unchanged fixed-slot reduction.  No production
selector is modified.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_down_consumer_quant_oracle import (
    gfx90a_fp4_down_consumer_quant_oracle,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8

E, M, T, H, I, N = 256, 64, 6, 4096, 512, 4096
A, R, W, LDS = 4, 2, 8, 2


def reconstruct_topk(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.int64).cpu()
    assert tuple(counts.shape) == (E,) and counts.sum().item() == M * T
    rows: list[list[int]] = [[] for _ in range(M)]
    for expert in torch.argsort(counts, descending=True).tolist():
        for _ in range(int(counts[expert])):
            choices = [t for t in range(M) if len(rows[t]) < T and expert not in rows[t]]
            token = min(choices, key=lambda t: (len(rows[t]), t))
            rows[token].append(expert)
    return torch.tensor(rows, dtype=torch.int32)


def metadata(topk: torch.Tensor):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(topk.cpu().tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    ids, experts = [], []
    sentinel = (T << 24) | M
    for expert, bucket in enumerate(buckets):
        for off in range(0, len(bucket), A):
            block = bucket[off : off + A]
            ids.extend(block + [sentinel] * (A - len(block)))
            experts.append(expert)
    device = topk.device
    return (
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
    )


def timed(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lo, hi = torch.cuda.Event(True), torch.cuda.Event(True)
    lo.record()
    for _ in range(iterations):
        fn()
    hi.record(); hi.synchronize()
    return lo.elapsed_time(hi) * 1000.0 / iterations


def main() -> None:
    global M
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, choices=(32, 64), default=64)
    ap.add_argument("--recorder", default="/tmp/expert_distribution_recorder_1788072257.651073.pt")
    ap.add_argument("--pass-index", type=int, default=20)
    ap.add_argument("--layer", type=int, default=34)
    ap.add_argument("--gate-blocks", type=int, nargs="+", default=[832, 1040, 2080])
    ap.add_argument("--ctas", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--mutations", type=int, default=100)
    args = ap.parse_args()
    M = args.m

    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("gfx90a ROCm required")
    recorder = Path(args.recorder)
    if recorder.is_file():
        payload = torch.load(recorder, map_location="cpu", weights_only=False)
        raw = payload["logical_count"][args.pass_index, args.layer]
        if torch.any(raw.remainder(4) != 0):
            raise RuntimeError("TP4 recorder counts must be divisible by four")
        counts = raw // 4
    elif M == 32:
        counts = torch.zeros(E, dtype=torch.int64)
        counts[:7] = 5
        counts[7:65] = 2
        counts[65:106] = 1
        print(
            f"recorder {recorder} missing; using synthetic diverse M32 route "
            "(assignments=192 active=106 a4_blocks=113)"
        )
    else:
        raise FileNotFoundError(recorder)
    device = torch.device("cuda")
    topk = reconstruct_topk(counts).to(device)
    ids, experts = metadata(topk)
    blocks = experts.numel()
    split = blocks // 2
    # Keep the split on an expert boundary so no expert's A4 run crosses streams.
    while split < blocks and split > 0 and experts[split] == experts[split - 1]:
        split += 1
    chunks = []
    for lo, hi in ((0, split), (split, blocks)):
        chunks.append((
            ids[lo * A : hi * A], experts[lo:hi],
            torch.tensor([(hi - lo) * A, M], dtype=torch.int32, device=device),
        ))
    print(f"route active={(counts>0).sum().item()} blocks={blocks} split={split}/{blocks-split}")

    torch.manual_seed(47)
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq, xs = per_token_group_quant_int8(x, 32)
    tw = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)
    inter_a = torch.empty((M, T, I), dtype=torch.bfloat16, device=device)
    inter_b = torch.empty_like(inter_a)
    part_a = torch.empty((M, T, N), dtype=torch.float32, device=device)
    part_b = torch.empty_like(part_a)
    out_a = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    out_b = torch.empty_like(out_a)
    full_valid = torch.tensor([ids.numel(), M], dtype=torch.int32, device=device)
    gate_full = _jit_gate_up_grouped(E, M, T, I, H, A, R, W, 2080, LDS)
    down = _jit_down_grouped(E, M, T, N, I, A, R, W, 832, LDS)

    def baseline():
        gate_full.run(xq, xs, w13, s13, ids, experts, full_valid, inter_a, 10.0)
        iq, isc = per_token_group_quant_int8(inter_a, 32)
        down.run_partial(iq, isc, w2, s2, ids, experts, full_valid, tw, part_a)
        down.reduce(part_a, out_a)

    base_stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()

    def make_candidate(gate_blocks: int, ctas: int):
        gate = _jit_gate_up_grouped(E, M, T, I, H, A, R, W, gate_blocks, LDS)
        ready0, ready1 = torch.cuda.Event(), torch.cuda.Event()

        def run():
            i0, e0, v0 = chunks[0]; i1, e1, v1 = chunks[1]
            gate.run(xq, xs, w13, s13, i0, e0, v0, inter_b, 10.0)
            ready0.record(base_stream)
            with torch.cuda.stream(side):
                side.wait_event(ready0)
                gfx90a_fp4_down_consumer_quant_oracle(inter_b, w2, s2, i0, e0, v0, tw, part_b, ctas_per_expert=ctas)
            gate.run(xq, xs, w13, s13, i1, e1, v1, inter_b, 10.0)
            ready1.record(base_stream)
            with torch.cuda.stream(side):
                side.wait_event(ready1)
                gfx90a_fp4_down_consumer_quant_oracle(inter_b, w2, s2, i1, e1, v1, tw, part_b, ctas_per_expert=ctas)
            base_stream.wait_stream(side)
            down.reduce(part_b, out_b)
        return run

    baseline(); torch.cuda.synchronize(); reference = out_a.clone()
    candidates = {(g, c): make_candidate(g, c) for g in args.gate_blocks for c in args.ctas}
    for key, fn in candidates.items():
        fn(); torch.cuda.synchronize()
        if not torch.equal(part_a, part_b) or not torch.equal(reference, out_b):
            raise AssertionError(f"candidate {key} mismatch max={(reference.float()-out_b.float()).abs().max().item()}")
    for mutation in range(args.mutations):
        xq.add_((mutation % 7) + 1)
        baseline(); reference.copy_(out_a)
        for key, fn in candidates.items():
            fn(); torch.cuda.synchronize()
            if not torch.equal(reference, out_b):
                raise AssertionError(f"mutation={mutation} candidate={key} mismatch")
    print(f"correctness mutations={args.mutations} exact=True")

    for key, fn in candidates.items():
        aa, bb = [], []
        for _ in range(args.rounds):
            aa.append(timed(baseline, args.warmup, args.iterations))
            bb.append(timed(fn, args.warmup, args.iterations))
            bb.append(timed(fn, args.warmup, args.iterations))
            aa.append(timed(baseline, args.warmup, args.iterations))
        am, bm = statistics.median(aa), statistics.median(bb)
        print(f"ABBA gate_blocks={key[0]} ctas={key[1]} A_us={am:.3f} B_us={bm:.3f} delta={(bm/am-1)*100:+.2f}%")


if __name__ == "__main__":
    main()
