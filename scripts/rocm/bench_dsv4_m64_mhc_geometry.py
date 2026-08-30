#!/usr/bin/env python3
"""Exact wave64 MHC pre-mix rows-per-CTA oracle for gfx90a M64."""

import statistics
import torch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

ROWS=(1,2,3,4,6,8)

@cache_once
def module(rows:int):
    args=make_cpp_args(rows)
    return load_jit(
        "gfx90a_mhc_pre_mix_geometry_oracle", *args,
        cuda_files=["gemm/gfx90a_mhc_pre_mix_geometry_oracle.cuh"],
        cuda_wrappers=[("run",f"sglang::Gfx90aMhcPreMixGeometryOracle<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )

def time_us(fn,warmup=10,iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    b.record(); b.synchronize()
    return a.elapsed_time(b)*1000/iters

def main():
    if not torch.version.hip: raise RuntimeError("ROCm required")
    torch.manual_seed(20260830)
    x=torch.randn((64,4,4096),device="cuda",dtype=torch.bfloat16)
    fn=torch.randn((24,16384),device="cuda",dtype=torch.float32)*0.01
    out={r:torch.empty((64,1,24),device="cuda",dtype=torch.float32) for r in ROWS}
    for r in ROWS: module(r).run(x,fn,out[r],1e-6)
    torch.cuda.synchronize()
    ref=out[3]
    for r in ROWS:
        d=(out[r]-ref).abs()
        print(f"CORRECT rows={r} exact={torch.equal(out[r],ref)} max_abs={d.max().item():.8g}")
    # Mutate the activation while preserving identical weights and addresses.
    for mutation in range(100):
        x.normal_()
        for r in ROWS: module(r).run(x,fn,out[r],1e-6)
        torch.cuda.synchronize()
        for r in ROWS:
            if not torch.equal(out[r],out[3]):
                raise RuntimeError(f"mutation={mutation} rows={r} mismatch")
    print("CORRECT mutations=100 all_exact=True")
    samples={r:[] for r in ROWS}
    for _ in range(7):
        for r in (*ROWS, *reversed(ROWS)):
            samples[r].append(time_us(lambda r=r:module(r).run(x,fn,out[r],1e-6)))
    for r in ROWS:
        v=samples[r]; trim=statistics.mean(sorted(v)[1:-1])
        print(f"RESULT rows={r} median_us={statistics.median(v):.3f} trimmed_us={trim:.3f} samples="+",".join(f"{x:.3f}" for x in v))

if __name__=="__main__": main()

