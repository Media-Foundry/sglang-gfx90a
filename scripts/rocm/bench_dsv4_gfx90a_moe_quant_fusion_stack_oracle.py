#!/usr/bin/env python3
"""Full M32 MHC/RMS-to-routed-MoE quant-fusion stack oracle."""

from __future__ import annotations

import argparse
import statistics

import torch

from bench_dsv4_gfx90a_down_consumer_quant_oracle import (
    make_metadata,
    reconstruct_topk,
)
from sglang.kernels.ops.layernorm.gfx90a_mhc_rms_quant_oracle import (
    gfx90a_mhc_rms_quant_oracle,
)
from sglang.kernels.ops.layernorm.mhc import (
    _gfx90a_mhc_rmsnorm_kernel,
    mhc_weighted_sum_triton,
)
from sglang.kernels.ops.moe.gfx90a_fp4_down_consumer_quant_oracle import (
    gfx90a_fp4_down_consumer_quant_oracle,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, M, T, H, I, N = 256, 32, 6, 4096, 256, 4096
A, ROWS, WAVES, BLOCKS, LDS = 4, 2, 8, 832, 2


def timed(fn, warmup: int, iterations: int) -> float:
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


def abba(fn_a, fn_b, warmup: int, iterations: int, rounds: int):
    sa: list[float] = []
    sb: list[float] = []
    for _ in range(rounds):
        sa.append(timed(fn_a, warmup, iterations))
        sb.append(timed(fn_b, warmup, iterations))
        sb.append(timed(fn_b, warmup, iterations))
        sa.append(timed(fn_a, warmup, iterations))
    return sa, sb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", default="/tmp/dsv4_ffn_dump.f3ZQ89")
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--dump-layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-replays", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm required")
    device = torch.device("cuda")
    torch.manual_seed(7)

    prefix = f"{args.dump_dir}/layer_{args.dump_layer}_rank_{args.rank}"
    residual = torch.load(
        f"{prefix}_ffn_mhc_residual.pt", map_location=device, weights_only=False
    ).contiguous()
    norm_weight = torch.load(
        f"{prefix}_ffn_norm_weight.pt", map_location=device, weights_only=False
    ).contiguous()
    pre = torch.softmax(torch.randn((M, 4), device=device), dim=-1)
    if residual.shape != (M, 4, H) or norm_weight.shape != (H,):
        raise RuntimeError("unexpected real MHC dump shape")

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw_counts = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw_counts.remainder(8) != 0):
        raise RuntimeError("TP8 recorder count is not divisible by eight")
    counts = raw_counts // 8
    topk_ids = reconstruct_topk(counts).to(device)
    sorted_ids, sorted_experts, valid = make_metadata(topk_ids)
    print(
        f"routing pass={args.pass_index} layer={args.layer} "
        f"active={int((counts > 0).sum())} max_occ={int(counts.max())} "
        f"a4_blocks={sorted_experts.numel()}",
        flush=True,
    )

    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    hidden_a = torch.empty((M, H), dtype=torch.bfloat16, device=device)
    hidden_b = torch.empty_like(hidden_a)
    gate_q_b = torch.empty_like(hidden_b, dtype=torch.int8)
    gate_s_b = torch.empty((M, H // 32), dtype=torch.float32, device=device)
    intermediate_a = torch.empty((M, T, I), dtype=torch.bfloat16, device=device)
    intermediate_b = torch.empty_like(intermediate_a)
    partial_a = torch.empty((M, T, N), dtype=torch.float32, device=device)
    partial_b = torch.empty_like(partial_a)
    out_a = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    out_b = torch.empty_like(out_a)
    gate_module = _jit_gate_up_grouped(E, M, T, I, H, A, ROWS, WAVES, BLOCKS, LDS)
    down_module = _jit_down_grouped(E, M, T, N, I, A, ROWS, WAVES, BLOCKS, LDS)

    def weighted() -> torch.Tensor:
        result = mhc_weighted_sum_triton(residual, pre)
        if result is None:
            raise RuntimeError("MHC weighted sum rejected real shape")
        return result

    def rms_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _gfx90a_mhc_rmsnorm_kernel[(M,)](
            x,
            norm_weight,
            hidden_a,
            hidden_size=H,
            eps=args.eps,
            BLOCK_H=H,
            num_warps=8,
        )
        return per_token_group_quant_int8(hidden_a, 32)

    def rms_candidate(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gfx90a_mhc_rms_quant_oracle(
            x, norm_weight, hidden_b, gate_q_b, gate_s_b, args.eps
        )
        return gate_q_b, gate_s_b

    def gate(q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
        gate_module.run(
            q, scale, w13, s13, sorted_ids, sorted_experts, valid, out, 10.0
        )

    def run_a():
        weighted_a = weighted()
        gate_q_a, gate_s_a = rms_reference(weighted_a)
        gate(gate_q_a, gate_s_a, intermediate_a)
        down_q_a, down_s_a = per_token_group_quant_int8(intermediate_a, 32)
        down_module.run_partial(
            down_q_a,
            down_s_a,
            w2,
            s2,
            sorted_ids,
            sorted_experts,
            valid,
            topk_weights,
            partial_a,
        )
        down_module.reduce(partial_a, out_a)
        return gate_q_a, gate_s_a, down_q_a, down_s_a, out_a

    def run_b():
        weighted_b = weighted()
        gate_q, gate_s = rms_candidate(weighted_b)
        gate(gate_q, gate_s, intermediate_b)
        gfx90a_fp4_down_consumer_quant_oracle(
            intermediate_b,
            w2,
            s2,
            sorted_ids,
            sorted_experts,
            valid,
            topk_weights,
            partial_b,
            ctas_per_expert=16,
        )
        down_module.reduce(partial_b, out_b)
        return gate_q, gate_s, out_b

    gate_q_a, gate_s_a, down_q_a, down_s_a, reference = run_a()
    gate_q_b_out, gate_s_b_out, candidate = run_b()
    torch.cuda.synchronize()
    exact = {
        "hidden": torch.equal(hidden_a, hidden_b),
        "gate_q": torch.equal(gate_q_a, gate_q_b_out),
        "gate_scale": torch.equal(gate_s_a, gate_s_b_out),
        "intermediate": torch.equal(intermediate_a, intermediate_b),
        "partial": torch.equal(partial_a, partial_b),
        "final": torch.equal(reference, candidate),
    }
    print(f"correctness exact={exact}", flush=True)
    if not all(exact.values()):
        raise AssertionError("full fusion stack is not bitwise exact")

    for replay in range(args.correctness_replays):
        residual.add_(
            torch.tensor((replay % 5) - 2, dtype=torch.bfloat16, device=device)
            / 128
        )
        gate_q_a, gate_s_a, _, _, reference = run_a()
        gate_q_b_out, gate_s_b_out, candidate = run_b()
        torch.cuda.synchronize()
        if not (
            torch.equal(hidden_a, hidden_b)
            and torch.equal(gate_q_a, gate_q_b_out)
            and torch.equal(gate_s_a, gate_s_b_out)
            and torch.equal(intermediate_a, intermediate_b)
            and torch.equal(partial_a, partial_b)
            and torch.equal(reference, candidate)
        ):
            raise AssertionError(f"mutated replay {replay} mismatch")
    print(
        f"correctness_mutations={args.correctness_replays} all_exact=True",
        flush=True,
    )

    run_a()
    run_b()
    torch.cuda.synchronize()
    graph_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a):
        graph_a_outputs = run_a()
    graph_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_b):
        graph_b_outputs = run_b()
    for replay in range(args.graph_replays):
        residual.add_(
            torch.tensor((replay % 3) - 1, dtype=torch.bfloat16, device=device)
            / 256
        )
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        if not (
            torch.equal(hidden_a, hidden_b)
            and torch.equal(graph_a_outputs[0], graph_b_outputs[0])
            and torch.equal(graph_a_outputs[1], graph_b_outputs[1])
            and torch.equal(intermediate_a, intermediate_b)
            and torch.equal(partial_a, partial_b)
            and torch.equal(graph_a_outputs[-1], graph_b_outputs[-1])
        ):
            raise AssertionError(f"graph replay {replay} mismatch")
    print(f"graph_replays={args.graph_replays} all_exact=True", flush=True)

    graph_a_samples, graph_b_samples = abba(
        graph_a.replay,
        graph_b.replay,
        args.warmup,
        args.iterations,
        args.rounds,
    )
    graph_am = statistics.median(graph_a_samples)
    graph_bm = statistics.median(graph_b_samples)
    print(
        f"GRAPH_ABBA A_full_us={graph_am:.3f} B_full_us={graph_bm:.3f} "
        f"saved_us={graph_am-graph_bm:.3f} "
        f"delta_pct={(graph_bm/graph_am-1)*100:+.2f} "
        f"A_samples={[round(v,3) for v in graph_a_samples]} "
        f"B_samples={[round(v,3) for v in graph_b_samples]}",
        flush=True,
    )

    a_samples, b_samples = abba(
        run_a, run_b, args.warmup, args.iterations, args.rounds
    )
    am, bm = statistics.median(a_samples), statistics.median(b_samples)
    print(
        f"ABBA A_full_us={am:.3f} B_full_us={bm:.3f} saved_us={am-bm:.3f} "
        f"delta_pct={(bm/am-1)*100:+.2f} "
        f"A_samples={[round(v,3) for v in a_samples]} "
        f"B_samples={[round(v,3) for v in b_samples]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
