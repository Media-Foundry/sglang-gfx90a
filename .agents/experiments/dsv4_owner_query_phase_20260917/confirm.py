"""Confirm fixed Lt owner projections against current full-M math, all owners."""
import ast
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5' and os.environ.get('ROCBLAS_USE_HIPBLASLT')=='1'
root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'confirm.json';assert not target.exists()
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.kernels.ops.attention.dsv4.gfx90a_rocblas_linear import Linear
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan

torch.set_num_threads(1);torch.set_grad_enabled(False)
assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>50*1024**3
torch.backends.cuda.preferred_blas_library('cublaslt')
source=root.parent/'dsv4_qkv_fixed_solution_20260917/oracle.py'
old=json.loads(source.with_suffix('.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==old['source_sha256']
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Fixed')
namespace={'Linear':Linear,'torch':torch,'C':C}
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
fixed=namespace['Fixed']()
fixture=root.parent/'dsv4_c16_owner_producer_20260915/capture/oracle'
cap=json.loads((fixture/'report.json').read_text())
assert hashlib.sha256((fixture/'inputs.pt').read_bytes()).hexdigest()==cap['inputs_file_sha256']
data=torch.load(fixture/'inputs.pt',map_location='cpu',weights_only=True)
digest=lambda t:hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
assert digest(data['q_lora'])==cap['q_lora_sha256']
model=Path('/home/pc/models/modelscope');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
key='layers.20.attn.indexer.wq_b'
with safe_open(str(model/index[key+'.weight']),framework='pt',device='cpu') as handle:
    w=(handle.get_tensor(key+'.weight').float().reshape(64,128,8,128)*
       handle.get_tensor(key+'.scale').float()[:,None,:,None]).reshape(8192,1024).bfloat16().cuda()
assert digest(w)==cap['parameters']['wq_b']['weight']['sha256']
seed=data['q_lora'].cuda()
report=dict(status='running',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    raw_only=True,fixture_input_sha256=cap['q_lora_sha256'],weight_sha256=digest(w),cases=[],timings=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def diff(a,b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    bad=a!=b
    return dict(exact=not bool(bad.any()),different=int(bad.sum()),rows=int(bad.any(1).sum()),
                max_abs=float((a.float()-b.float()).abs().max()))
def extensions(m):
    result=[]
    while m:
        n=min(m,8192);result.append(n);m-=n
    return result
def bench(a,b):
    graphs={}
    for name,fn in [('A',a),('B',b)]:
        for _ in range(3):fn()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):fn()
        graphs[name]=graph
    values={'A':[],'B':[]}
    for _ in range(4):
        for name in ('A','B','B','A'):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graphs[name].replay();end.record();end.synchronize()
            values[name].append(start.elapsed_time(end)/3)
    return dict(samples=values,full_ms=statistics.median(values['A']),owner_ms=statistics.median(values['B']))
save()
try:
    for m in (8192,16383,32765,32767,32768):
        ext=extensions(m);pre=[0]*len(ext)
        original=torch.cat((seed,seed[:1]))[:m].contiguous()
        for mutation in range(3):
            x=original if mutation==0 else (original.roll(17*mutation,0).float()*(1-.011*mutation)).bfloat16()
            full=F.linear(x,w)
            assert torch.equal(full,F.linear(x,w)), 'Unstable control'
            for owner in range(8):
                lengths,ids,valid,_=host_plan(ext,pre,owner)
                mask=torch.from_numpy(valid.astype(bool)&(lengths[ids]>512)).cuda()
                ids=torch.from_numpy(ids).cuda();part=x.index_select(0,ids)
                available=fixed.solutions(part,w)
                for solution in (3778,3953):
                    assert solution in available
                    actual=fixed.run(part,w,solution)
                    item=dict(m=m,owner=owner,mutation=mutation,solution=solution,local_m=len(part),
                              active=int(mask.sum()),versus_full=diff(actual[mask],full.index_select(0,ids)[mask]))
                    report['cases'].append(item)
                if mutation==0 and owner==0:
                    whole=torch.empty_like(full);out=torch.empty((len(part),8192),device=x.device,dtype=x.dtype)
                    for solution in (3778,3953):
                        # Includes query-row gather on candidate; full reference output also preallocated.
                        timing=bench(lambda:torch.mm(x,w.t(),out=whole),
                                     lambda:fixed.run(x.index_select(0,ids),w,solution,out))
                        report['timings'].append(dict(m=m,solution=solution,**timing))
            save();print('CASE COMPLETE',m,mutation,flush=True)
    report['exact_cases']=sum(v['versus_full']['exact'] for v in report['cases'])
    report['total_cases']=len(report['cases'])
    report['status']='complete';save()
finally:
    if report['status']!='complete':report['status']='failed';report['error']=traceback.format_exc();save()
    fixed.close()
