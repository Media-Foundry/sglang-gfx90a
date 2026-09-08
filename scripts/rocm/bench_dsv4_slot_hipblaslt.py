#!/usr/bin/env python3
"""Screen supported explicit hipBLASLt solutions for row-position invariance.

No tuned-table writes or runtime dispatch changes. Real layer0 M736 fixture.
Initial timing is a screen, not ABBA or E2E; full checks are only for finalists.
"""
import json
from pathlib import Path
import torch
from aiter import hipb_create_extension, hipb_findallsols
from aiter.tuned_gemm import hipb_gemm


def main():
    p=Path('/tmp/dsv4_tp8_slot_layer0_20260908')
    original=torch.load(p/'layer_0_rank_0_attn_norm.pt',weights_only=True).cuda()
    x=original.clone()
    w=torch.load(p/'layer_0_rank_0_projection_wqkv_a.pt',weights_only=True).cuda()
    ref=(x[:46].double()@w.double().t()).bfloat16()
    hipb_create_extension()
    solutions=hipb_findallsols(x,w.t(),out_dtype=x.dtype)
    print('supported',len(solutions),flush=True)
    viable=[]
    rejected={'nonfinite':0,'row_variant':0,'reference_error':0}
    def invariant(y):
        z=y.reshape(16,46,-1)
        return torch.equal(z,z[:1].expand_as(z))
    for solution in solutions:
        y=hipb_gemm(x,w,solution)
        if not torch.isfinite(y).all():
            rejected['nonfinite']+=1
            continue
        if not invariant(y):
            rejected['row_variant']+=1
            continue
        error=(y[:46].float()-ref.float()).abs().max().item()
        if error>0.015625:
            rejected['reference_error']+=1
            continue
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=hipb_gemm(x,w,solution)
        for _ in range(5):g.replay()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):g.replay()
        end.record();end.synchronize()
        row={'solution':solution,'screen_us':start.elapsed_time(end)*1000/30,'fp64_max_abs':error}
        viable.append(row)
    print('rejected',json.dumps(rejected),flush=True)
    print('stable_candidates',json.dumps(sorted(viable,key=lambda r:r['screen_us'])),flush=True)
    torch.manual_seed(20908)
    for row in sorted(viable,key=lambda r:r['screen_us'])[:3]:
        solution=row['solution'];g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):y=hipb_gemm(x,w,solution)
        exact=0
        for _ in range(100):
            x.copy_(torch.randn_like(x[:46]).repeat(16,1));g.replay()
            exact+=int(invariant(y))
        witness=y.clone()
        for _ in range(1000):g.replay()
        print('finalist',json.dumps({**row,'mutation_exact':exact,'stable_1000':torch.equal(y,witness)}),flush=True)
    x.copy_(original)


if __name__=='__main__':main()
