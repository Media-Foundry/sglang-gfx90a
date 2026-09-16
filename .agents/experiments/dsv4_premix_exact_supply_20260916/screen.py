"""Exact cooperative supply: current full pre-mix and full-boundary graph ABBA."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import triton
from module import load
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_wave import post_wave_module
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
wide='--wide' in sys.argv
root=Path(__file__).resolve().parent;target=root/('wide-screen.json' if wide else 'screen.json');assert not target.exists()
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{n}.pt' for n in ['attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','hc_ffn_fn']]
seed=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
mod,postmod=load(),post_wave_module();torch.manual_seed(2026091607)
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32);assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',pci=pci.value.decode(),scope='Sampled real layer0 repeated/scaled to occupancy, not live independent full-M or service',
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,Path(__file__),root/'module.py',root/'exact.cuh']},cases=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
for m in [8192,32767,32768,65536]:
    x,res,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed[:4]]
    post,comb,fn=post.view(m,4),comb.view(m,4,4),seed[4].clone().contiguous()
    hidden=torch.empty_like(res);rms=torch.empty((m,64),device='cuda')
    a=torch.empty((m,24),device='cuda');b=torch.empty_like(a)
    def generate():postmod.run(x,res,post,comb,hidden,rms)
    def baseline():return premix8_pair[(12,triton.cdiv(m,8))](hidden,fn,rms,a,m,1.e-6,num_warps=1)
    generate()
    for name in (['s6','s12'] if wide else ['d1','s2','s4']):
        def candidate():getattr(mod,name)(hidden.view(m,16384),fn,rms,b)
        for mutation in range(10):
            if mutation:
                x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
                fn.add_(torch.randn_like(fn)*1.e-6);generate()
            b.fill_(float('nan'));baseline();candidate()
            assert torch.equal(a.view(torch.int32),b.view(torch.int32)),(m,name,mutation,(a-b).abs().max().item())
        perm=torch.randperm(m,device='cuda');inv=torch.argsort(perm);xp=hidden[perm].view(m,16384).contiguous();rp=rms[perm].contiguous()
        getattr(mod,name)(xp,fn,rp,b)
        assert torch.equal(b[inv].view(torch.int32),a.view(torch.int32))
        del xp,rp
        for boundary in [False,True]:
            def call_a():
                if boundary:generate()
                baseline()
            def call_b():
                if boundary:generate()
                candidate()
            graphs={}
            for arm,call in [('A',call_a),('B',call_b)]:
                for _ in range(3):call()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):call()
                graphs[arm]=g
            for _ in range(100):graphs['B'].replay()
            assert torch.equal(a.view(torch.int32),b.view(torch.int32))
            samples=[]
            for cycle in range(3):
                for arm in ['A','B','B','A']:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(5):graphs[arm].replay()
                    end.record();end.synchronize();samples.append(dict(arm=arm,cycle=cycle,ms=begin.elapsed_time(end)/5))
            med={arm:statistics.median(v['ms'] for v in samples if v['arm']==arm) for arm in ['A','B']}
            result=dict(m=m,name=name,includes_post=boundary,median_ms=med,samples=samples,mutation10_exact=True,permutation_exact=True,replay100_exact=True)
            report['cases'].append(result);save();print(m,name,boundary,med,flush=True)
            del g,graphs
    del x,res,post,comb,fn,hidden,rms,a,b
    torch.cuda.empty_cache()
report['status']='complete';save()
