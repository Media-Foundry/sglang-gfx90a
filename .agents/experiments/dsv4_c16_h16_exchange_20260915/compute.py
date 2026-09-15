"""H8 full-M vs real H16 half-M compute bound. No communication/E2E claim.

Both peers have independent nonzero Q/sinks for exactly the same query IDs.
Candidate joins HEADS, never neighboring query selections. All KV selections,
duplicates, sentinels, softmax grouping and stages1 remain unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'compute.json';assert not target.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in owners for p in g.get('process_list',[]))
import numpy as np
import torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import _sparse_attn_v4_paged_prefill_kernel as kernel

torch.manual_seed(20260915);rng=np.random.default_rng(20260915)
result=dict(scope=__doc__,status='running',real_capture=False,cases=[],sources={
    str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
    (Path(__file__),repo/'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py')})
def save():target.write_text(json.dumps(result,indent=2)+'\n')
def measure(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):fn()
    b.record();b.synchronize();return a.elapsed_time(b)/5

for family,m,req_len in [('c128',8192,4096),('c4',8192,4096),
                          ('c128',32768,8192),('c4',32768,8192)]:
    ratio=128 if family=='c128' else 4
    pi=[];ei=[];pp=[0];ep=[0]
    for row in range(m):
        request,pos=divmod(row,req_len)
        history=(pos+1)//ratio
        selected=np.sort(rng.choice(history,512,replace=False)) if history>512 else np.arange(history)
        pi.extend((selected+request*(req_len//ratio)).tolist());pp.append(len(pi))
        ei.extend(range(row-min(pos,127),row+1));ep.append(len(ei))
    pi,pp,ei,ep=[torch.tensor(x,device='cuda',dtype=torch.int32) for x in (pi,pp,ei,ep)]
    q=torch.randn((m,16,512),device='cuda',dtype=torch.bfloat16).mul_(.3)
    # All16 heads are real, not8 useful heads plus zeros.
    assert torch.count_nonzero(q[:,8:])>0
    pk=torch.randn((m//ratio,512),device='cuda',dtype=torch.bfloat16).mul_(.5)
    ek=torch.randn((m,512),device='cuda',dtype=torch.bfloat16).mul_(.8)
    sink=torch.linspace(-4,4,16,device='cuda')
    qa,qb=q[:,:8],q[:,8:]
    oa,ob=torch.empty_like(qa),torch.empty_like(qb)
    joined=torch.empty_like(q)
    def launch(query,out,sinks,begin=0):
        return kernel[(len(query),1)](query,pk,pi,pp[begin:],ek,ei,ep[begin:],sinks,out,
            *query.stride(),*pk.stride(),*ek.stride(),*out.stride(),query.shape[1],512,
            512**-.5,BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1,num_stages=1)
    def a0():return launch(qa,oa,sink[:8])
    def a1():return launch(qb,ob,sink[8:])
    def b0():return launch(q[:m//2],joined[:m//2],sink)
    def b1():return launch(q[m//2:],joined[m//2:],sink,m//2)
    checks=[]
    for mutation in range(5):
        if mutation:
            q.mul_(1.001);pk.mul_(.999);ek.mul_(1.0001);sink.add_(.001)
        if mutation==2:
            pi[::97]=-1;ei[::113]=-1
        a0();a1();joined.fill_(float('nan'));b0();b1()
        a=torch.cat((oa,ob),dim=1)
        check=dict(mutation=mutation,bits_exact=torch.equal(a.view(torch.uint8),joined.view(torch.uint8)),
            max_abs=float((a.float()-joined.float()).abs().max()),finite=bool(torch.isfinite(joined).all()))
        checks.append(check)
    operations={'a0':a0,'a1':a1,'b0':b0,'b1':b1}
    for fn in operations.values():fn()
    samples={'A':[],'B':[]}
    # Same serial screen GPU, but compare the slowest peer's compute, NOT sum.
    raw=[]
    for _ in range(3):
        for arm in ('A','B','B','A'):
            times=[measure(operations[arm.lower()+str(i)]) for i in range(2)]
            samples[arm].append(max(times));raw.append(dict(arm=arm,peer_ms=times))
    centers={k:statistics.median(v) for k,v in samples.items()}
    resources={}
    for k,fn in operations.items():
        c=fn();resources[k]=dict(registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared,
            artifact=c.hash,hsaco_sha256=hashlib.sha256(c.asm['hsaco']).hexdigest())
    item=dict(family=family,m=m,request_len=req_len,checks=checks,raw_samples=raw,
        rankmax_samples_ms=samples,median_ms=centers,compute_speedup=centers['A']/centers['B'],
        maximum_exchange_pack_budget_ms=centers['A']-centers['B'],resources=resources,
        send_bytes_per_rank_q_and_o=m//2*8*512*2*2)
    result['cases'].append(item);save();print(family,m,centers,checks,flush=True)
result['status']='complete';save()
