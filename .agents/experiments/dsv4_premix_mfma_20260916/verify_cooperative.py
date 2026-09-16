"""Component checks only; random activations are not full-model quality evidence."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import triton
import triton.language as tl
from safetensors import safe_open
from module import load_cooperative
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

root=Path(__file__).resolve().parent;target=root/'cooperative-validation.json'
assert not target.exists()
mod=load_cooperative();torch.manual_seed(2026091604)
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',pci_bus=pci.value.decode(),shapes=[],checkpoint_fn=[],
    scope='Component arithmetic/batch stability only; no service/logits acceptance',
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'cooperative.cuh',root/'module.py']})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def stats(out,ref):
    d=out.double()-ref.double()
    return dict(max_abs=d.abs().max().item(),relative_l2=(d.norm()/ref.double().norm()).item(),
                bit_exact=torch.equal(out.view(torch.int32),ref.view(torch.int32)))
def call(x,fn,rms,out,buf,s):
    getattr(mod,f's{s}')(x,fn,buf)
    finish[(x.shape[0],)](buf,rms,out,x.shape[0],s,num_warps=1)

model=Path('/home/pc/models/modelscope')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
names=[n for n in index if n.startswith('layers.') and n.endswith(('.hc_attn_fn','.hc_ffn_fn')) and int(n.split('.')[1])<43]
assert len(names)==86
for number,name in enumerate(names):
    with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:fn=f.get_tensor(name).float().reshape(24,16384).cuda()
    x=torch.randn((65,16384),device='cuda',dtype=torch.bfloat16)
    rms=x.float().view(65,64,256).square().sum(-1).contiguous()
    ref=torch.empty((65,24),device='cuda')
    premix8_pair[(12,9)](x,fn,rms,ref,65,1.e-6,num_warps=1)
    cases=[]
    for s in (4,16):
        buf=torch.full((s,65,24),float('nan'),device='cuda');out=torch.empty_like(ref)
        call(x,fn,rms,out,buf,s)
        v=stats(out,ref)
        assert torch.isfinite(buf).all() and torch.isfinite(out).all() and v['relative_l2']<1.e-5
        cases.append(dict(split=s,**v))
    report['checkpoint_fn'].append(dict(name=name,checks=cases));save()
    if number%20==0:print('checkpoint Fn',number+1,'/',len(names),flush=True)

for m in (1,17,63,64,65,127,129,8191,8192,8193,32767,32768,32769,65536):
    x=torch.randn((m,16384),device='cuda',dtype=torch.bfloat16)
    # Independent temporary random weights; checkpoint files are never changed.
    fn=torch.randn((24,16384),device='cuda')*.02
    rms=x.float().view(m,64,256).square().sum(-1).contiguous()
    ref=torch.empty((m,24),device='cuda');out=torch.empty_like(ref)
    result=dict(m=m,checks=[])
    for s in (4,16):
        buf=torch.empty((s,m,24),device='cuda');worst_abs=worst_rel=0.
        for mutation in range(10):
            if mutation:fn.add_(torch.randn_like(fn)*.0001)
            buf.fill_(float('nan'));out.fill_(float('nan'))
            premix8_pair[(12,triton.cdiv(m,8))](x,fn,rms,ref,m,1.e-6,num_warps=1)
            call(x,fn,rms,out,buf,s)
            v=stats(out,ref);worst_abs=max(worst_abs,v['max_abs']);worst_rel=max(worst_rel,v['relative_l2'])
            assert torch.isfinite(buf).all() and torch.isfinite(out).all() and v['relative_l2']<1.e-5
        original=out.clone()
        perm=torch.randperm(m,device='cuda');inv=torch.argsort(perm)
        xp=x[perm].contiguous();rp=rms[perm].contiguous()
        call(xp,fn,rp,out,buf,s)
        perm_exact=torch.equal(out[inv].view(torch.int32),original.view(torch.int32))
        del xp,rp
        n=min(m,65);small=torch.empty((n,24),device='cuda');sb=torch.empty((s,n,24),device='cuda')
        call(x[:n],fn,rms[:n],small,sb,s)
        prefix_exact=torch.equal(small.view(torch.int32),original[:n].view(torch.int32))
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):call(x,fn,rms,out,buf,s)
        for _ in range(1000 if m==8192 else 100):g.replay()
        replay_exact=torch.equal(out.view(torch.int32),original.view(torch.int32))
        fn.mul_(.97)
        g.replay();changed=out.clone()
        call(x,fn,rms,out,buf,s)
        mutation_exact=torch.equal(out.view(torch.int32),changed.view(torch.int32))
        assert perm_exact and prefix_exact and replay_exact and mutation_exact
        result['checks'].append(dict(split=s,max_abs=worst_abs,relative_l2=worst_rel,
            permutation_exact=perm_exact,prefix_exact=prefix_exact,replay_exact=replay_exact,
            changed_input_exact=mutation_exact,replays=1000 if m==8192 else 100))
        del g,buf,original,small,sb,changed
    report['shapes'].append(result);save();print('shape',m,result['checks'],flush=True)
    del x,fn,rms,ref,out
    torch.cuda.empty_cache()
report['status']='complete';save()
