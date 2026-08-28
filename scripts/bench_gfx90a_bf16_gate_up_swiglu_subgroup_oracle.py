"""Independent ABBA and graph replay audit for the subgroup oracle."""

from __future__ import annotations

import statistics

import torch

from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import (
    _jit_gfx90a_bf16_gemv_module,
    gfx90a_bf16_gate_up_swiglu_subgroup_oracle,
)
from sglang.srt.layers.activation import silu_and_mul


def elapsed_once(fn) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0


def stats(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return statistics.median(samples), statistics.fmean(ordered[2:-2])


def run_group(seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    x = (torch.randn(1, 2560, device="cuda") * 0.1).to(torch.bfloat16)
    weight = (torch.randn(320, 2560, device="cuda") * 0.02).to(torch.bfloat16)
    gate_up = torch.empty(1, 320, dtype=torch.bfloat16, device="cuda")
    old_out = torch.empty(1, 160, dtype=torch.bfloat16, device="cuda")
    fused_out = torch.empty_like(old_out)
    gemv = _jit_gfx90a_bf16_gemv_module(1, 320, 2560)

    def chain():
        gemv.run(x, weight, gate_up)
        silu_and_mul(gate_up, old_out)

    def fused():
        gfx90a_bf16_gate_up_swiglu_subgroup_oracle(x, weight, fused_out)

    for _ in range(100):
        chain()
        fused()
    torch.cuda.synchronize()
    chain()
    fused()
    torch.cuda.synchronize()
    delta = (fused_out.float() - old_out.float()).abs()

    samples = {"chain": [], "fused": []}
    for _ in range(31):
        samples["chain"].append(elapsed_once(chain))
        samples["fused"].append(elapsed_once(fused))
        samples["fused"].append(elapsed_once(fused))
        samples["chain"].append(elapsed_once(chain))
    chain_median, chain_trim = stats(samples["chain"])
    fused_median, fused_trim = stats(samples["fused"])
    result = {
        "chain_median": chain_median,
        "chain_trim": chain_trim,
        "fused_median": fused_median,
        "fused_trim": fused_trim,
        "median_speedup": chain_median / fused_median - 1.0,
        "trim_speedup": chain_trim / fused_trim - 1.0,
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "mismatch": int((fused_out != old_out).sum().item()),
    }
    print(f"seed={seed} {result}")
    return result


def graph_replay_audit() -> None:
    torch.manual_seed(314159)
    x = (torch.randn(1, 2560, device="cuda") * 0.1).to(torch.bfloat16)
    weight = (torch.randn(320, 2560, device="cuda") * 0.02).to(torch.bfloat16)
    out = torch.empty(1, 160, dtype=torch.bfloat16, device="cuda")
    gfx90a_bf16_gate_up_swiglu_subgroup_oracle(x, weight, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gfx90a_bf16_gate_up_swiglu_subgroup_oracle(x, weight, out)
    graph.replay()
    torch.cuda.synchronize()
    reference = out.clone()
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize()
    print(
        "graph_replay=1000 "
        f"finite={bool(torch.isfinite(out).all())} "
        f"stable={bool(torch.equal(out, reference))}"
    )
    assert torch.isfinite(out).all() and torch.equal(out, reference)


def main() -> None:
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("gfx90a required")
    results = [run_group(seed) for seed in (0, 1, 17, 20260828)]
    graph_replay_audit()
    print(
        "all_median_gt_10pct="
        f"{all(item['median_speedup'] > 0.10 for item in results)} "
        "all_trim_gt_10pct="
        f"{all(item['trim_speedup'] > 0.10 for item in results)}"
    )


if __name__ == "__main__":
    main()
