#!/usr/bin/env python3
"""Standalone TP8 I256 gate screen; no production selector or weight cache."""
import json
import statistics
import argparse

import torch
from bench_dsv4_gfx90a_occupancy_bucket_oracle import make_metadata
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_gate_up_grouped, _jit_gate_up_grouped_row_prefetch, _jit_down_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    full=parser.parse_args().full
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
    torch.manual_seed(20908)
    m,t,e,i,k=32,6,256,256,4096
    x=torch.randint(-127,128,(m,k),dtype=torch.int8,device='cuda')
    scale=torch.full((m,k//32),.01,device='cuda')
    w=torch.randint(0,256,(e,2*i,k//2),dtype=torch.uint8,device='cuda')
    ws=torch.full((e,2*i,k//32),127,dtype=torch.uint8,device='cuda')
    args=(e,m,t,i,k,4,2,8,832,2)
    mods=[_jit_gate_up_grouped(*args),_jit_gate_up_grouped_row_prefetch(*args)]
    if full:
        down=_jit_down_grouped(e,m,t,k,i,4,2,8,832,2)
        w2=torch.randint(0,256,(e,k,i//2),dtype=torch.uint8,device='cuda')
        s2=torch.randint(122,128,(e,k,i//32),dtype=torch.uint8,device='cuda')
        router_weights=torch.rand((m,t),device='cuda')
    for expert_pool in (256,128,32):
        ids=torch.rand((m,expert_pool),device='cuda').topk(t,dim=1).indices.int()
        meta=make_metadata(ids,assignments=4)
        outputs=[torch.empty((m,t,i),dtype=torch.bfloat16,device='cuda') for _ in mods]
        finals=[torch.empty((m,k),dtype=torch.bfloat16,device='cuda') for _ in mods] if full else outputs
        partials=[torch.empty((m,t,k),dtype=torch.float32,device='cuda') for _ in mods] if full else []
        graphs=[]
        for arm,(mod,out) in enumerate(zip(mods,outputs)):
            def run():
                mod.run(x,scale,w,ws,meta.sorted_ids,meta.sorted_experts,meta.valid,out,10.)
                if full:
                    q,qs=per_token_group_quant_int8(out,32)
                    down.run_partial(q,qs,w2,s2,meta.sorted_ids,meta.sorted_experts,
                                     meta.valid,router_weights,partials[arm])
                    down.reduce(partials[arm],finals[arm])
            run();torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):run()
            graphs.append(g)
        exact=0;max_abs=0.;stable=True
        for n in range(100):
            x.random_(-127,128)
            scale.uniform_(.001,.02)
            if n%25==0:ws.random_(122,128)
            for g in graphs:g.replay()
            torch.cuda.synchronize()
            exact+=int(torch.equal(*outputs) and torch.equal(*finals))
            max_abs=max(max_abs,float((finals[0]-finals[1]).abs().max()))
            assert all(torch.isfinite(o).all() for o in finals)
            expected=finals[1].clone()
            for _ in range(10):graphs[1].replay()
            torch.cuda.synchronize()
            stable=stable and torch.equal(expected,finals[1])
        samples=[[],[]]
        for _ in range(5):
            for arm in (0,1,1,0):
                for _ in range(20):graphs[arm].replay()
                torch.cuda.synchronize()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):graphs[arm].replay()
                end.record();end.synchronize()
                samples[arm].append(begin.elapsed_time(end)*10)
        print(json.dumps(dict(full_stage=full,expert_pool=expert_pool,active_experts=ids.unique().numel(),
                              scans=meta.sorted_experts.numel(),exact_mutations=exact,
                              max_abs=max_abs,stable=stable,samples_us=samples,
                              trimmed_us=[statistics.mean(sorted(v)[1:-1]) for v in samples])),flush=True)


if __name__=='__main__':main()
