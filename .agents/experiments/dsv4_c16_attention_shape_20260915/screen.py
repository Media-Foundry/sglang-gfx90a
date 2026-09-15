"""H8 native two-source prefill geometry screen; no production selector changes."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

root=Path(__file__).resolve().parent;repo=root.parents[2]
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in owners for p in g.get('process_list',[]))
import numpy as np
import torch
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import _sparse_attn_v4_paged_prefill_kernel as kernel

torch.manual_seed(20260915);rng=np.random.default_rng(20260915)
target=root/'screen.json';assert not target.exists()
result=dict(scope=__doc__,real_service_capture=False,rows=8192,results=[],compile_errors={},status='running',
    source_sha256=hashlib.sha256((repo/'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py').read_bytes()).hexdigest())
def save():target.write_text(json.dumps(result,indent=2)+'\n')
def ints(x):return torch.as_tensor(x,dtype=torch.int32,device='cuda')
def measure(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):fn()
    b.record();b.synchronize();return a.elapsed_time(b)/5

configs=[('H16W1',16,1),('H8W1',8,1),('H16W2',16,2)]
for family in ('c128','c4_varied'):
    m=8192;ratio=128 if family=='c128' else 4
    ids=[];eid=[];ptr=[0];eptr=[0]
    for row in range(m):
        history=(row+1)//ratio
        selected=np.sort(rng.choice(history,512,replace=False)) if history>512 else np.arange(history)
        ids.extend(selected.tolist());ptr.append(len(ids))
        eid.extend(range(max(0,row-127),row+1));eptr.append(len(eid))
    pi,pp,ei,ep=map(ints,(ids,ptr,eid,eptr))
    q=torch.randn((m,8,512),device='cuda',dtype=torch.bfloat16)*.3
    pk=torch.randn((m//ratio,512),device='cuda',dtype=torch.bfloat16)*.5
    ek=torch.randn((m,512),device='cuda',dtype=torch.bfloat16)*.8
    sink=torch.linspace(-2,2,8,device='cuda')
    outs={name:torch.empty_like(q) for name,_,_ in configs}
    def launch(config):
        name,h,w=config;o=outs[name]
        return kernel[(m,1)](q,pk,pi,pp,ek,ei,ep,sink,o,*q.stride(),*pk.stride(),*ek.stride(),
            *o.stride(),8,512,512**-.5,BLOCK_H=h,BLOCK_D=512,BLOCK_K=16,num_warps=w)
    baseline=configs[0];launch(baseline)
    item=dict(family=family,candidates={});result['results'].append(item)
    for config in configs[1:]:
        name=config[0]
        if name in result['compile_errors']:continue
        try:compiled=launch(config)
        except (AssertionError,ValueError,Exception) as error:
            # Compilation errors only are recoverable. A device failure must
            # not be treated as an ordinary unsupported geometry.
            from triton.compiler.errors import CompilationError
            if not isinstance(error,(CompilationError,AssertionError,ValueError)):raise
            result['compile_errors'][name]=repr(error);save();continue
        torch.cuda.synchronize()
        checks=[]
        for mutation in range(5):
            if mutation:
                q.mul_(1.005);pk.mul_(.999);ek.mul_(1.001);sink.add_(.01)
            launch(baseline);launch(config)
            a,b=outs[baseline[0]],outs[name]
            checks.append(dict(bits_exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),
                max_abs=float((a.float()-b.float()).abs().max()),finite=bool(torch.isfinite(b).all())))
        samples={'A':[],'B':[]}
        for cycle in range(3):
            for label,cfg in [('A',baseline),('B',config),('B',config),('A',baseline)]:
                samples[label].append(measure(lambda:launch(cfg)))
        codes={}
        for label,cfg in [('A',baseline),('B',config)]:
            c=launch(cfg)
            codes[label]=dict(registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared,
                hsaco_sha256=hashlib.sha256(c.asm['hsaco']).hexdigest())
        med={key:statistics.median(value) for key,value in samples.items()}
        item['candidates'][name]=dict(checks=checks,samples_ms=samples,median_ms=med,
            rate_gain_pct=100*(med['A']/med['B']-1),resources=codes)
        save();print(family,name,med,codes,flush=True)
result['status']='complete';save()
