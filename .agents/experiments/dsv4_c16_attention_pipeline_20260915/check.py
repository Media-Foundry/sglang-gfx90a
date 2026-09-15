"""Stages1 vs unchanged two-stage attention: ragged bounds and replay oracle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import statistics

root=Path(__file__).resolve().parent;repo=root.parents[2]
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in owners for p in g.get('process_list',[]))
import numpy as np
import torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import _sparse_attn_v4_paged_prefill_kernel as kernel

torch.manual_seed(1829);rng=np.random.default_rng(1829)
target=root/'check-strided.json';assert not target.exists()
results=[]
for m in (1,17,129,8191,8192,32767,32768,65536):
    pl=np.resize(np.array([0,1,15,16,17,127,128,512]),m)
    el=np.resize(np.array([0,0,1,15,16,17,127,128]),m)
    ip=np.r_[0,np.cumsum(pl)].astype(np.int32)
    ie=np.r_[0,np.cumsum(el)].astype(np.int32)
    ids=rng.integers(0,2048,int(ip[-1]),dtype=np.int32)
    eis=rng.integers(0,4096,int(ie[-1]),dtype=np.int32)
    ids[::19]=-1;eis[::23]=-1
    # Repeated entries remain separate softmax occurrences.
    if len(ids)>40:ids[32:40]=3
    pi,pp,ei,ep=[torch.tensor(v,device='cuda') for v in (ids,ip,eis,ie)]
    # Non-contiguous token and KV row strides, as accepted by the public ABI.
    q=torch.randn((m,16,512),device='cuda',dtype=torch.bfloat16).mul_(.3)[:,::2,:]
    pk=torch.randn((2048,1024),device='cuda',dtype=torch.bfloat16).mul_(.5)[:,::2]
    ek=torch.randn((4096,1024),device='cuda',dtype=torch.bfloat16).mul_(.8)[:,::2]
    assert q.stride()==(8192,1024,1) and pk.stride()==ek.stride()==(1024,2)
    sink=torch.linspace(-20,20,8,device='cuda')
    a,b=torch.empty_like(q),torch.empty_like(q)
    def launch(out,stage,query=q,pids=pi,pind=pp,eids=ei,eind=ep):
        return kernel[(len(query),1)](query,pk,pids,pind,ek,eids,eind,sink,out,
            *query.stride(),*pk.stride(),*ek.stride(),*out.stride(),8,512,512**-.5,
            BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1,num_stages=stage)
    launch(a,2);launch(b,1);torch.cuda.synchronize()
    repetitions=100 if m==129 else 3
    for rep in range(repetitions):
        q.mul_(1.001);pk.mul_(.9999);ek.mul_(1.0001);sink.add_(.001)
        if len(pi)>0:pi[0]=rep%2048 if rep%2 else -1
        if len(ei)>0:ei[0]=rep%4096 if rep%2 else -1
        b.fill_(float('nan'));launch(a,2);launch(b,1)
        assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),(m,rep)
        assert torch.isfinite(b).all() and torch.count_nonzero(b[0])==0
    row_perm=False;replays=0
    if m==129:
        order=rng.permutation(m);permutation=torch.tensor(order,device='cuda')
        # Use current mutated indices, preserving duplicates and ordering.
        ids=pi.cpu().numpy();eis=ei.cpu().numpy()
        rp=np.concatenate([ids[ip[i]:ip[i+1]] for i in order])
        re=np.concatenate([eis[ie[i]:ie[i+1]] for i in order])
        rpi,rpp,rei,rep=[torch.tensor(v,device='cuda',dtype=torch.int32) for v in
            (rp,np.r_[0,np.cumsum(pl[order])],re,np.r_[0,np.cumsum(el[order])])]
        pq=q.index_select(0,permutation);po=torch.empty_like(pq)
        launch(po,1,pq,rpi,rpp,rei,rep)
        assert torch.equal(po.view(torch.uint8),a.index_select(0,permutation).view(torch.uint8))
        row_perm=True
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch(b,1)
        for _ in range(1000):
            b.fill_(float('nan'));graph.replay()
            assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
        replays=1000
    def timing(out,st):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):launch(out,st)
        end.record();end.synchronize();return start.elapsed_time(end)/5
    samples={'A':[],'B':[]}
    for cycle in range(3):
        for label,out,st in [('A',a,2),('B',b,1),('B',b,1),('A',a,2)]:
            samples[label].append(timing(out,st))
    row=dict(m=m,exact_checks=repetitions,empty_row_exact=True,row_permutation_exact=row_perm,
             graph_replays_exact=replays,samples_ms=samples,
             median_ms={k:statistics.median(v) for k,v in samples.items()},q_stride=q.stride(),kv_stride=pk.stride())
    results.append(row);print(row,flush=True)
target.write_text(json.dumps(dict(status='complete',results=results,scope=__doc__,
    source_sha256=hashlib.sha256((repo/'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py').read_bytes()).hexdigest()),indent=2)+'\n')
