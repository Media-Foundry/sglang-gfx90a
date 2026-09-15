"""Synthetic H8 two-bank prefill consumer screen, including all new GPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--rows',type=int,default=8192,choices=(8192,32768))
p.add_argument('--strided-q',action='store_true')
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(r.get('process_info'),dict) for g in owners for r in g.get('process_list',[])),owners

import numpy as np
import torch
from oracle import Runner, reference, ROOT, REPO
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import _sparse_attn_v4_paged_prefill_kernel as base
gate=json.loads((ROOT/'smoke-refined.json').read_text())
assert gate['status']=='complete' and gate['graph_mutation_exact']
source_paths=[Path(__file__), ROOT/'oracle.py',ROOT/'dsv4_prefill_two_source_core.cuh',
    REPO/'python/sglang/kernels/jit/csrc/debug/gfx90a_prefill_two_source_ck_oracle.cuh',
    REPO/'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py']
result=dict(status='running',scope=__doc__,rows=args.rows,strided_q=args.strided_q,
    metric='GPU consumer with pre-existing ragged metadata: candidate encoder+core+reducer; allocations and common metadata builder excluded. Optional Q copy is timed.',
    real_service_capture=False,results=[],sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
def measure(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):fn()
    b.record();b.synchronize();return a.elapsed_time(b)/5
def ints(x):return torch.as_tensor(x,dtype=torch.int32,device='cuda')
torch.manual_seed(20260915);rng=np.random.default_rng(20260915)
save()
for family in ('c128','c4_reused','c4_varied'):
    ratio=128 if family=='c128' else 4
    requests=args.rows//8192;slots_per_request=8192//ratio
    pids=[];eids=[];pp=[0];ep=[0]
    for row in range(args.rows):
        request,pos=divmod(row,8192);history=(pos+1)//ratio
        if family=='c4_varied' and history>512:
            selected=np.sort(rng.choice(history,512,replace=False))
        else:
            selected=np.arange(max(0,history-512),history)
        pids.extend((selected+request*slots_per_request).tolist());pp.append(len(pids))
        eids.extend(range(request*8192+max(0,pos-127),row+1));ep.append(len(eids))
    pi,pp,ei,ep=map(ints,(pids,pp,eids,ep))
    storage=torch.randn((args.rows,16 if args.strided_q else 8,512),device='cuda',dtype=torch.bfloat16)*.3
    q=storage[:,:8,:]
    q_for_ck=q.contiguous()
    pkv=torch.randn((requests*slots_per_request,512),device='cuda',dtype=torch.bfloat16)*.5
    ekv=torch.randn((args.rows,512),device='cuda',dtype=torch.bfloat16)*.8
    sink=torch.linspace(-2,2,8,device='cuda')
    baseline=torch.empty(q.shape,device='cuda',dtype=torch.bfloat16)
    runner=Runner(q_for_ck,pkv,pi,pp,ekv,ei,ep,sink,refined=True)
    def control():
        return base[(args.rows,1)](q,pkv,pi,pp,ekv,ei,ep,sink,baseline,
            *q.stride(),*pkv.stride(),*ekv.stride(),*baseline.stride(),8,512,512**-.5,
            BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1)
    def candidate(splits):
        if args.strided_q:q_for_ck.copy_(q)
        return runner(splits)
    control();candidate(1);candidate(2);torch.cuda.synchronize()
    sample_rows=sorted(set((0,1,15,16,127,128,2047,2048,4095,4096,8191,args.rows-1)))
    expected=[]
    for row in sample_rows:
        ps=int(pp[row]);pe=int(pp[row+1]);es=int(ep[row]);ee=int(ep[row+1])
        expected.append(reference(q[row:row+1],pkv,pi[ps:pe],ints([0,pe-ps]),
            ekv,ei[es:ee],ints([0,ee-es]),sink)[0])
    expected=torch.stack(expected)
    checks={}
    for name,fn in [('triton',control),('ck1',lambda:candidate(1)),('ck2',lambda:candidate(2))]:
        fn();actual=baseline if name=='triton' else runner.out
        assert bool(torch.isfinite(actual).all())
        sampled=actual[sample_rows].float()
        checks[name]=dict(sampled_max_abs=float((sampled-expected).abs().max()))
        torch.testing.assert_close(sampled,expected,atol=.003,rtol=.02)
    samples={'triton':[],'ck1':[],'ck2':[]}
    for splits in (1,2):
        for cycle in range(3):
            for name,fn in [('triton',control),(f'ck{splits}',lambda:candidate(splits)),
                            (f'ck{splits}',lambda:candidate(splits)),('triton',control)]:
                samples[name].append(measure(fn))
    kernel=control()
    item=dict(family=family,samples_ms=samples,median_ms={n:median(v) for n,v in samples.items()},
        checks=checks,prefix_indices=pi.numel(),extend_indices=ei.numel(),
        combined_index_bytes=runner.combined.numel()*4,workspace_bytes=runner.scratch.numel(),
        q_copy_bytes=q.numel()*q.element_size() if args.strided_q else 0,
        baseline_code=dict(registers=kernel.n_regs,spills=kernel.n_spills,lds=kernel.metadata.shared,
                           hsaco_sha256=hashlib.sha256(kernel.asm['hsaco']).hexdigest()))
    result['results'].append(item);save();print({k:v for k,v in item.items() if k not in ('samples_ms','baseline_code')},flush=True)
    del runner,q,storage,q_for_ck,pkv,ekv,pi,pp,ei,ep,baseline,actual,sampled,expected
    torch.cuda.empty_cache()
result['status']='complete';save()
