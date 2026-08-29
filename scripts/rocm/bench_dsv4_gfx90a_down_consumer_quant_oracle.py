#!/usr/bin/env python3
"""Standalone M32 down-consumer group32-quant fusion oracle.

The production A4 gate and fixed-slot FP32 reduction stay unchanged.  The
candidate down CTA loads the gate's BF16 bounded-SwiGLU output, produces the
same group32 INT8 values/scales in LDS, and consumes them immediately.  No
production selector is modified.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_down_consumer_quant_oracle import (
    gfx90a_fp4_down_consumer_quant_oracle,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, M, T, H, N = 256, 32, 6, 4096, 4096
ASSIGNMENTS, ROWS, WAVES, BLOCKS, LDS = 4, 2, 8, 832, 2


def reconstruct_topk(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.int64).cpu()
    if tuple(counts.shape) != (E,) or counts.sum().item() != M * T:
        raise ValueError("invalid M32 recorder counts")
    rows: list[list[int]] = [[] for _ in range(M)]
    for expert in torch.argsort(counts, descending=True).tolist():
        for _ in range(int(counts[expert])):
            choices = [
                token
                for token in range(M)
                if len(rows[token]) < T and expert not in rows[token]
            ]
            if not choices:
                raise RuntimeError("could not reconstruct duplicate-free top-k")
            token = min(choices, key=lambda value: (len(rows[value]), value))
            rows[token].append(expert)
    result = torch.tensor(rows, dtype=torch.int32)
    if any(len(set(row)) != T for row in result.tolist()):
        raise RuntimeError("duplicate expert in reconstructed row")
    return result


def make_metadata(topk_ids: torch.Tensor):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(topk_ids.cpu().tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), ASSIGNMENTS):
            block = bucket[offset : offset + ASSIGNMENTS]
            ids.extend(block + [M] * (ASSIGNMENTS - len(block)))
            experts.append(expert)
    device = topk_ids.device
    return (
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


def timed(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def abba(fn_a, fn_b, warmup: int, iterations: int, rounds: int):
    a: list[float] = []
    b: list[float] = []
    for _ in range(rounds):
        a.append(timed(fn_a, warmup, iterations))
        b.append(timed(fn_b, warmup, iterations))
        b.append(timed(fn_b, warmup, iterations))
        a.append(timed(fn_a, warmup, iterations))
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-replays", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        choices=(256, 512),
        default=256,
        help="expert intermediate shard: TP8=256, TP4=512",
    )
    parser.add_argument(
        "--ctas-per-expert", type=int, nargs="+", default=[4, 6, 7, 8, 9, 10, 12]
    )
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a required, got {arch}")
    torch.manual_seed(7)
    device = torch.device("cuda")
    intermediate_size = args.intermediate_size

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(8) != 0):
        raise RuntimeError("TP8 recorder count is not divisible by eight")
    counts = raw // 8
    topk_ids = reconstruct_topk(counts).to(device)
    sorted_ids, sorted_experts, valid = make_metadata(topk_ids)
    print(
        f"routing pass={args.pass_index} layer={args.layer} "
        f"active={int((counts > 0).sum())} max_occ={int(counts.max())} "
        f"a4_blocks={sorted_experts.numel()}",
        flush=True,
    )

    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq, xs = per_token_group_quant_int8(x, 32)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(
        0, 256, (E, 2 * intermediate_size, H // 2), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0, 256, (E, N, intermediate_size // 2), dtype=torch.uint8, device=device
    )
    s13 = torch.full(
        (E, 2 * intermediate_size, H // 32), 127, dtype=torch.uint8, device=device
    )
    s2 = torch.full(
        (E, N, intermediate_size // 32), 127, dtype=torch.uint8, device=device
    )
    intermediate_a = torch.empty(
        (M, T, intermediate_size), dtype=torch.bfloat16, device=device
    )
    intermediate_b = torch.empty_like(intermediate_a)
    partial_a = torch.empty((M, T, N), dtype=torch.float32, device=device)
    partial_b = torch.empty_like(partial_a)
    out_a = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    out_b = torch.empty_like(out_a)

    gate_module = _jit_gate_up_grouped(
        E, M, T, intermediate_size, H, ASSIGNMENTS, ROWS, WAVES, BLOCKS, LDS
    )
    down_module = _jit_down_grouped(
        E, M, T, N, intermediate_size, ASSIGNMENTS, ROWS, WAVES, BLOCKS, LDS
    )

    def gate(out: torch.Tensor) -> None:
        gate_module.run(
            xq,
            xs,
            w13,
            s13,
            sorted_ids,
            sorted_experts,
            valid,
            out,
            10.0,
        )

    def reduce(partial: torch.Tensor, out: torch.Tensor) -> None:
        down_module.reduce(partial, out)

    def baseline_down(intermediate: torch.Tensor) -> None:
        iq, isc = per_token_group_quant_int8(intermediate, 32)
        down_module.run_partial(
            iq,
            isc,
            w2,
            s2,
            sorted_ids,
            sorted_experts,
            valid,
            topk_weights,
            partial_a,
        )

    def run_a() -> torch.Tensor:
        gate(intermediate_a)
        baseline_down(intermediate_a)
        reduce(partial_a, out_a)
        return out_a

    def make_b(ctas_per_expert: int):
        def candidate_down(intermediate: torch.Tensor) -> None:
            gfx90a_fp4_down_consumer_quant_oracle(
                intermediate,
                w2,
                s2,
                sorted_ids,
                sorted_experts,
                valid,
                topk_weights,
                partial_b,
                ctas_per_expert=ctas_per_expert,
            )

        def run_b() -> torch.Tensor:
            gate(intermediate_b)
            candidate_down(intermediate_b)
            reduce(partial_b, out_b)
            return out_b

        return candidate_down, run_b

    reference = run_a().clone()
    torch.cuda.synchronize()
    for ctas in args.ctas_per_expert:
        _, run_b = make_b(ctas)
        candidate = run_b()
        torch.cuda.synchronize()
        partial_exact = torch.equal(partial_a, partial_b)
        output_exact = torch.equal(reference, candidate)
        max_abs = (reference.float() - candidate.float()).abs().max().item()
        print(
            f"correctness ctas_per_expert={ctas} partial_exact={partial_exact} "
            f"output_exact={output_exact} max_abs={max_abs:.8g}",
            flush=True,
        )
        if not (partial_exact and output_exact):
            raise AssertionError(f"CTA{ctas} candidate is not bitwise exact")

    stress_ctas = 12 if 12 in args.ctas_per_expert else args.ctas_per_expert[0]
    _, stress_b = make_b(stress_ctas)
    for replay in range(args.correctness_replays):
        xq.add_((replay % 7) + 1)
        reference = run_a().clone()
        candidate = stress_b()
        torch.cuda.synchronize()
        if not torch.equal(reference, candidate):
            diff = (reference.float() - candidate.float()).abs().max().item()
            raise AssertionError(f"replay={replay} mismatch max_abs={diff}")
    print(
        f"correctness_mutations={args.correctness_replays} "
        f"ctas_per_expert={stress_ctas} output_exact=True",
        flush=True,
    )

    # Capture only after all modules and allocator paths are warm.  Both graphs
    # share immutable inputs but own their intermediate/partial/output buffers.
    graph_a = torch.cuda.CUDAGraph()
    run_a()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph_a):
        run_a()
    _, graph_run_b = make_b(stress_ctas)
    graph_b = torch.cuda.CUDAGraph()
    graph_run_b()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph_b):
        graph_run_b()
    for replay in range(args.graph_replays):
        xq.add_((replay % 5) + 1)
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        if not torch.equal(partial_a, partial_b) or not torch.equal(out_a, out_b):
            diff = (out_a.float() - out_b.float()).abs().max().item()
            raise AssertionError(f"graph replay={replay} mismatch max_abs={diff}")
    print(
        f"graph_replays={args.graph_replays} ctas_per_expert={stress_ctas} "
        "partial_exact=True output_exact=True",
        flush=True,
    )

    gate(intermediate_a)
    torch.cuda.synchronize()
    for ctas in args.ctas_per_expert:
        candidate_down, run_b = make_b(ctas)
        a_samples, b_samples = abba(
            run_a, run_b, args.warmup, args.iterations, args.rounds
        )
        chain_a, chain_b = abba(
            lambda: baseline_down(intermediate_a),
            lambda: candidate_down(intermediate_a),
            args.warmup,
            args.iterations,
            args.rounds,
        )
        am, bm = statistics.median(a_samples), statistics.median(b_samples)
        cam, cbm = statistics.median(chain_a), statistics.median(chain_b)
        print(
            f"ABBA ctas_per_expert={ctas} A_full_us={am:.3f} B_full_us={bm:.3f} "
            f"full_delta_pct={(bm/am-1)*100:+.2f} "
            f"A_quant_down_us={cam:.3f} B_fused_down_us={cbm:.3f} "
            f"down_delta_pct={(cbm/cam-1)*100:+.2f} "
            f"A_samples={[round(v,3) for v in a_samples]} "
            f"B_samples={[round(v,3) for v in b_samples]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
