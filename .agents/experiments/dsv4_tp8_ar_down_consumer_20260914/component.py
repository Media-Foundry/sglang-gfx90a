"""Bounded M32 down-chain check: varied physical scales/routes, current uniform control.

Synthetic inputs, not a captured model-layer oracle. Both paths consume the
same physical scale bytes. No weight loading, requantization, or service edits.
"""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/home/pc/Code/sglang/scripts/rocm')
import torch
from bench_dsv4_gfx90a_down_consumer_quant_oracle import make_metadata
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import _jit_down_grouped_uniform, _jit_down_grouped
from sglang.kernels.ops.moe.gfx90a_fp4_down_consumer_quant_oracle import gfx90a_fp4_down_consumer_quant_oracle
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


def main():
    output = ROOT/'component.json'
    assert not output.exists()
    gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    active={p['process_info']['pid'] for g in gpu for p in g.get('process_list', [])
            if isinstance(p.get('process_info'), dict)}
    # Import-time device detection may already create this process's context.
    assert active <= {os.getpid()}, f'External GPU owners: {active-{os.getpid()}}'
    (ROOT/'component.gpu-before.json').write_text(json.dumps(gpu, indent=2))
    torch.manual_seed(20260914)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx90a'
    m, t, k, n, e = 32, 6, 256, 4096, 256
    x = torch.empty((m,t,k), dtype=torch.bfloat16, device='cuda')
    w = torch.randint(0,256,(e,n,k//2), dtype=torch.uint8, device='cuda')
    scale = torch.randint(119,130,(e,n,k//32),dtype=torch.uint8,device='cuda')
    weights = torch.empty((m,t),dtype=torch.float32,device='cuda')
    pa = torch.empty((m,t,n),dtype=torch.float32,device='cuda')
    pb = torch.empty_like(pa)
    oa = torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
    ob = torch.empty_like(oa)
    control = _jit_down_grouped_uniform(e,m,t,n,k,4,2,8,832,2)
    reducer = _jit_down_grouped(e,m,t,n,k,4,2,8,832,2)
    records=[]
    for distribution in ('diverse', 'skewed'):
        ids = torch.full((2048,),m,dtype=torch.int32,device='cuda')
        experts = torch.full((512,),-1,dtype=torch.int32,device='cuda')
        valid = torch.zeros(2,dtype=torch.int32,device='cuda')
        def mutate(i):
            x.normal_(0,0.5 + (i%4)*0.5)
            if i%7==0:
                x[:,:,::32]=0
            weights.uniform_(0.01,1)
            weights.div_(weights.sum(-1,keepdim=True))
            # Every row has six distinct experts; both routing regimes vary.
            pool = e if distribution == 'diverse' else 24
            topk = torch.stack([torch.randperm(pool,device='cuda')[:t] for _ in range(m)]).int()
            si,se,nv=make_metadata(topk)
            ids.fill_(m); experts.fill_(-1)
            ids[:si.numel()].copy_(si); experts[:se.numel()].copy_(se); valid.copy_(nv)
            # Change scales/weights as well as activation and sorter metadata.
            scale.random_(119,130)
            w[:,i%n,:].random_(0,256)
        def a():
            iq,isc=per_token_group_quant_int8(x,32)
            control.run(iq,isc,w,scale,ids,experts,valid,weights,pa,oa)
        def b():
            gfx90a_fp4_down_consumer_quant_oracle(x,w,scale,ids,experts,valid,weights,pb,ctas_per_expert=16)
            reducer.reduce(pb,ob)
        mutate(0); a(); b(); torch.cuda.synchronize()
        ga,gb=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga): a()
        with torch.cuda.graph(gb): b()
        for i in range(100):
            mutate(i)
            for _ in range(10):
                ga.replay(); gb.replay()
                assert torch.equal(pa,pb), (distribution,i,'FP32 partial mismatch')
                assert torch.equal(oa,ob), (distribution,i,'BF16 mismatch')
                assert torch.isfinite(ob).all()
        def timed(graph):
            for _ in range(10): graph.replay()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100): graph.replay()
            end.record();end.synchronize()
            return begin.elapsed_time(end)*10
        samples={'A':[],'B':[]}
        for _ in range(3):
            for arm in ('A','B','B','A'):
                samples[arm].append(timed(ga if arm=='A' else gb))
        med={arm:statistics.median(v) for arm,v in samples.items()}
        record=dict(distribution=distribution, mutations=100, checked_graph_replays=1000,
                    partial_exact=True, output_exact=True, samples_us=samples,median_us=med,
                    time_saved_pct=100*(1-med['B']/med['A']))
        records.append(record);print(json.dumps(record),flush=True)
    output.write_text(json.dumps(dict(scope='synthetic down+quant+reduction, not full MoE/E2E', records=records),indent=2)+'\n')


if __name__ == '__main__': main()
