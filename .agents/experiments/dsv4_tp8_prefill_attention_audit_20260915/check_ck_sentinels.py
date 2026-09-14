"""Pending GPU contract check for existing H8 CK; no production edits.

Q=0, valid KV rows=1, sink=0 gives output valid_count/(valid_count+1).
Negative slots must not add softmax mass. This is not a speed benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
active={r['process_info']['pid'] for g in owners for r in g.get('process_list',[])
        if isinstance(r.get('process_info'),dict)}
assert active<={os.getpid()},f'GPU occupied: {active}'

import torch
from sglang.kernels.ops.attention.dsv4.gfx90a_sparse_h8 import _module

repo=Path(__file__).resolve().parents[3]
sources=['python/sglang/kernels/ops/attention/dsv4/gfx90a_sparse_h8.py',
    'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_dsv4_sparse_h8_oracle.cuh',
    'python/sglang/kernels/jit/csrc/deepseek_v4/dsv4_unified_sparse_decode_ck.cuh']
result=dict(status='running',scope=__doc__,cases=[],
    source_sha256={s:hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in sources})
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
save()
q=torch.zeros((1,8,512),device='cuda',dtype=torch.bfloat16)
kv=torch.ones((2,512),device='cuda',dtype=torch.bfloat16)
sink=torch.zeros(8,device='cuda',dtype=torch.float32)
workspace=torch.empty(2*8*514*4,device='cuda',dtype=torch.uint8)
for name,slots in [('valid',[0]),('mixed_negative',[0,-1]),
                   ('other_negative',[-2,1]),('duplicate',[0,0]),
                   ('all_negative',[-1,-2]),('empty',[])]:
    indices=torch.tensor(slots or [-1],device='cuda',dtype=torch.int32)
    indptr=torch.tensor([0,len(slots)],device='cuda',dtype=torch.int32)
    output=torch.empty_like(q)
    _module().run(q,kv,indices,indptr,sink,output,workspace,512**-.5)
    count=sum(s>=0 for s in slots)
    reference=torch.full_like(q,count/(count+1))
    delta=float((output.float()-reference.float()).abs().max())
    exact=torch.equal(output.view(torch.uint8),reference.view(torch.uint8))
    result['cases'].append(dict(name=name,slots=slots,valid_occurrences=count,
        expected=float(reference.flatten()[0]),actual=float(output.flatten()[0]),
        all_finite=bool(torch.isfinite(output).all()),max_abs=delta,exact=exact))
    save();print(json.dumps(result['cases'][-1]),flush=True)
result['status']='complete';result['contract_pass']=all(c['exact'] for c in result['cases']);save()
assert result['contract_pass'],'CK sentinel contract mismatch; results retained for diagnosis'
