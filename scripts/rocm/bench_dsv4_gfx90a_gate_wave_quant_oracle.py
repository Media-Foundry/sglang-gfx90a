#!/usr/bin/env python3
"""Standalone wave-owned A4 gate + bounded-SwiGLU + INT8 quant oracle."""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.moe.gfx90a_fp4_gate_wave_quant_oracle import (
    gfx90a_fp4_gate_wave_quant_oracle,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, M, T, H, I, N = 256, 32, 6, 4096, 256, 4096
A, ROWS, WAVES, BLOCKS, LDS = 4, 2, 8, 832, 2


def reconstruct_topk(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.int64).cpu()
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
    if tuple(result.shape) != (M, T) or any(
        len(set(row)) != T for row in result.tolist()
    ):
        raise RuntimeError("invalid reconstructed top-k")
    return result


def make_metadata(topk_ids: torch.Tensor):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(topk_ids.cpu().tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), A):
            block = bucket[offset : offset + A]
            ids.extend(block + [M] * (A - len(block)))
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
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("NAME", "RECORDER", "PASS", "LAYER"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-replays", type=int, default=100)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a required, got {arch}")
    torch.manual_seed(7)
    device = torch.device("cuda")

    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq, xs = per_token_group_quant_int8(x, 32)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)
    intermediate_a = torch.empty((M, T, I), dtype=torch.bfloat16, device=device)
    intermediate_b = torch.empty_like(intermediate_a)
    iq_b = torch.empty((M, T, I), dtype=torch.int8, device=device)
    is_b = torch.empty((M, T, I // 32), dtype=torch.float32, device=device)
    partial_a = torch.empty((M, T, N), dtype=torch.float32, device=device)
    partial_b = torch.empty_like(partial_a)
    out_a = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    out_b = torch.empty_like(out_a)
    gate_module = _jit_gate_up_grouped(E, M, T, I, H, A, ROWS, WAVES, BLOCKS, LDS)
    down_module = _jit_down_grouped(E, M, T, N, I, A, ROWS, WAVES, BLOCKS, LDS)

    for name, recorder, pass_text, layer_text in args.case:
        pass_index, layer = int(pass_text), int(layer_text)
        payload = torch.load(recorder, map_location="cpu", weights_only=False)
        raw = payload["logical_count"][pass_index, layer]
        if torch.any(raw.remainder(8) != 0):
            raise RuntimeError(f"{name}: counts not divisible by TP8")
        counts = raw // 8
        topk_ids = reconstruct_topk(counts).to(device)
        sorted_ids, sorted_experts, valid = make_metadata(topk_ids)
        print(
            f"case={name} pass={pass_index} layer={layer} "
            f"active={int((counts > 0).sum())} max_occ={int(counts.max())} "
            f"a4_blocks={sorted_experts.numel()}",
            flush=True,
        )

        def gate_a() -> tuple[torch.Tensor, torch.Tensor]:
            gate_module.run(
                xq, xs, w13, s13, sorted_ids, sorted_experts, valid,
                intermediate_a, 10.0,
            )
            return per_token_group_quant_int8(intermediate_a, 32)

        def gate_b() -> tuple[torch.Tensor, torch.Tensor]:
            gfx90a_fp4_gate_wave_quant_oracle(
                xq, xs, w13, s13, sorted_ids, sorted_experts, valid,
                intermediate_b, iq_b, is_b, 10.0,
            )
            return iq_b, is_b

        def run_a() -> torch.Tensor:
            iq, isc = gate_a()
            down_module.run_partial(
                iq, isc, w2, s2, sorted_ids, sorted_experts, valid,
                topk_weights, partial_a,
            )
            down_module.reduce(partial_a, out_a)
            return out_a

        def run_b() -> torch.Tensor:
            iq, isc = gate_b()
            down_module.run_partial(
                iq, isc, w2, s2, sorted_ids, sorted_experts, valid,
                topk_weights, partial_b,
            )
            down_module.reduce(partial_b, out_b)
            return out_b

        reference = run_a().clone()
        iq_a, is_a = gate_a()
        candidate = run_b()
        torch.cuda.synchronize()
        exact = {
            "intermediate": torch.equal(intermediate_a, intermediate_b),
            "q": torch.equal(iq_a, iq_b),
            "scale": torch.equal(is_a, is_b),
            "partial": torch.equal(partial_a, partial_b),
            "output": torch.equal(reference, candidate),
        }
        max_scale = (is_a - is_b).abs().max().item()
        max_output = (reference.float() - candidate.float()).abs().max().item()
        print(
            f"correctness case={name} exact={exact} "
            f"max_scale={max_scale:.8g} max_output={max_output:.8g}",
            flush=True,
        )
        if not all(exact.values()):
            raise AssertionError(f"{name}: wave producer not bitwise exact")

        for replay in range(args.correctness_replays):
            xq.add_((replay % 7) + 1)
            reference = run_a().clone()
            candidate = run_b()
            torch.cuda.synchronize()
            if not torch.equal(reference, candidate):
                diff = (reference.float() - candidate.float()).abs().max().item()
                raise AssertionError(f"{name}: replay={replay} max_abs={diff}")
        print(
            f"correctness_mutations case={name} replays={args.correctness_replays} "
            "output_exact=True",
            flush=True,
        )

        gate_a_samples, gate_b_samples = abba(
            gate_a, gate_b, args.warmup, args.iterations, args.rounds
        )
        a_samples, b_samples = abba(
            run_a, run_b, args.warmup, args.iterations, args.rounds
        )
        gam, gbm = statistics.median(gate_a_samples), statistics.median(gate_b_samples)
        am, bm = statistics.median(a_samples), statistics.median(b_samples)
        print(
            f"ABBA case={name} A_gate_quant_us={gam:.3f} B_wave_gate_quant_us={gbm:.3f} "
            f"gate_delta_pct={(gbm/gam-1)*100:+.2f} "
            f"A_full_us={am:.3f} B_full_us={bm:.3f} "
            f"full_delta_pct={(bm/am-1)*100:+.2f} "
            f"A_samples={[round(v,3) for v in a_samples]} "
            f"B_samples={[round(v,3) for v in b_samples]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
