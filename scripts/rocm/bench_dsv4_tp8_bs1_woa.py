#!/usr/bin/env python3
"""TP8's one-group wo_a: existing GEMV implementations vs einsum fallback."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import (
    _jit_gfx90a_bf16_grouped_gemv_module,
    gfx90a_wave64_bf16_gemv,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(20908)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 1, 1024, device="cuda", dtype=torch.bfloat16)
    module = _jit_gfx90a_bf16_grouped_gemv_module(1, 1)

    def grouped():
        module.run(x, w, out)
        return out

    funcs = {
        "einsum": lambda: torch.einsum("tgd,grd->tgr", x, w),
        "linear": lambda: gfx90a_wave64_bf16_gemv(x.view(1,4096), w[0]).view(1,1,1024),
        "grouped_g1": grouped,
    }
    graphs = {}
    for name, fn in funcs.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fn()
        graphs[name] = g, y
    result = {"mutations":[], "timings_us":{k:[] for k in funcs}}
    for i in range(100):
        x.normal_()
        if i % 25 == 0:
            w.normal_()
        for g, _ in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        ref = torch.matmul(x.float(), w[0].float().T)
        base = graphs["einsum"][1]
        row = {"i":i}
        for name in ("linear", "grouped_g1"):
            g,y = graphs[name]
            before = y.clone()
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            row[name] = {
                "replay_exact": torch.equal(before,y),
                "einsum_exact": torch.equal(base,y),
                "max_abs_vs_einsum": float((base.float()-y.float()).abs().max()),
                "relative_l2_vs_fp32": float((y.float()-ref).norm()/ref.norm()),
                "finite": bool(torch.isfinite(y).all()),
            }
        result["mutations"].append(row)
    bursts = {}
    for name,fn in funcs.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(64):
                fn()
        bursts[name] = g
    for name in ["einsum", "linear", "grouped_g1", "grouped_g1", "linear", "einsum"]*3:
        bursts[name].replay()
        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            bursts[name].replay()
        end.record();end.synchronize()
        result["timings_us"][name].append(start.elapsed_time(end)*1000/(64*20))
    result["median_us"]={k:statistics.median(v) for k,v in result["timings_us"].items()}
    result["summary"]={name:{
        "replay_exact":sum(r[name]["replay_exact"] for r in result["mutations"]),
        "einsum_exact":sum(r[name]["einsum_exact"] for r in result["mutations"]),
        "max_abs_vs_einsum":max(r[name]["max_abs_vs_einsum"] for r in result["mutations"]),
        "max_relative_l2":max(r[name]["relative_l2_vs_fp32"] for r in result["mutations"]),
    } for name in ("linear","grouped_g1")}
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ["median_us","summary"]}),flush=True)


if __name__ == "__main__":
    main()
