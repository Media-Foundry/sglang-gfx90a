"""Time exact HIP post plus complete MFMA pre-mix against current full boundary."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import triton
import triton.language as tl
from module import load_cooperative
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_wave import post_wave_module
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair

@triton.jit
def finish(parts, rms, out, M, S:tl.constexpr):
    row=tl.program_id(0);c=tl.arange(0,32)
    acc=tl.full((32,),0,tl.float32)
    for s in tl.static_range(S):
        acc+=tl.load(parts+(s*M+row)*24+c,c<24,0)
    p=tl.arange(0,64)
    inv=tl.rsqrt(tl.sum(tl.load(rms+row*64+p),0)/16384+1.e-6)
    tl.store(out+row*24+c,acc*inv,c<24)

root=Path(__file__).resolve().parent;target=root/'boundary.json';assert not target.exists()
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{n}.pt' for n in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','hc_ffn_fn')]
seed=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
mod,postmod=load_cooperative(),post_wave_module()
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',pci_bus=pci.value.decode(),cases=[],
    scope='Exact production HIP post included in both arms; full pre-mix and finish; sampled real layer0 expanded to occupancy, not service',
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,Path(__file__),root/'module.py',root/'cooperative.cuh']})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
for m in (8192,32767,32768):
    x,res,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed[:4]]
    post,comb,fn=post.view(m,4),comb.view(m,4,4),seed[4].contiguous()
    hidden=torch.empty_like(res);rms=torch.empty((m,64),device='cuda')
    a=torch.empty((m,24),device='cuda');b=torch.empty_like(a)
    def baseline():
        postmod.run(x,res,post,comb,hidden,rms)
        premix8_pair[(12,triton.cdiv(m,8))](hidden,fn,rms,a,m,1.e-6,num_warps=1)
    for s in (4,16):
        buf=torch.empty((s,m,24),device='cuda')
        def candidate():
            postmod.run(x,res,post,comb,hidden,rms)
            getattr(mod,f's{s}')(hidden.view(m,16384),fn,buf)
            finish[(m,)](buf,rms,b,m,s,num_warps=1)
        graphs={}
        for name,call in [('A',baseline),('B',candidate)]:
            for _ in range(3):call()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):call()
            graphs[name]=g
        assert torch.isfinite(b).all()
        samples=[]
        for cycle in range(3):
            for arm in ('A','B','B','A'):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(5):graphs[arm].replay()
                end.record();end.synchronize()
                samples.append(dict(cycle=cycle,arm=arm,ms=begin.elapsed_time(end)/5))
        med={arm:statistics.median(v['ms'] for v in samples if v['arm']==arm) for arm in ('A','B')}
        result=dict(m=m,split=s,scratch_bytes=buf.numel()*4,median_ms=med,samples=samples,
            max_abs=(a-b).abs().max().item(),relative_l2=((a-b).double().norm()/a.double().norm()).item())
        report['cases'].append(result);save();print(m,s,med,flush=True)
        del g,graphs,buf
    del x,res,post,comb,hidden,rms,a,b
    torch.cuda.empty_cache()
report['status']='complete';save()
