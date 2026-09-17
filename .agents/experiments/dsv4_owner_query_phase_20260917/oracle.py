"""Real indexer projection: fixed-Lt algorithm plus row-phase-preserving ownership.

No service edits. Compare current full-M raw BF16 projection against compact
owner rows, and whole global row tiles reassigned to owners without changing
their row-within-tile phase. Only valid active rows count toward correctness.
"""
import ast
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES') == '5'
assert os.environ.get('ROCBLAS_USE_HIPBLASLT') == '1'
root=Path(__file__).resolve().parent
repo=root.parents[2]
target=root/'oracle.json'
assert not target.exists()
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
capture=json.loads((fixture/'report.json').read_text())
assert hashlib.sha256((fixture/'inputs.pt').read_bytes()).hexdigest()==capture['inputs_file_sha256']
data=torch.load(fixture/'inputs.pt',map_location='cpu',weights_only=True)
digest=lambda x:hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
assert digest(data['q_lora'])==capture['q_lora_sha256']
model=Path('/home/pc/models/modelscope');idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
key='layers.20.attn.indexer.wq_b'
with safe_open(str(model/idx[key+'.weight']),framework='pt',device='cpu') as handle:
    weight=(handle.get_tensor(key+'.weight').float().reshape(64,128,8,128)*
            handle.get_tensor(key+'.scale').float()[:,None,:,None]).reshape(8192,1024).bfloat16()
assert digest(weight)==capture['parameters']['wq_b']['weight']['sha256']
x=data['q_lora'].cuda();w=weight.cuda();m=len(x)
ext,pre=capture['extend_lens'],capture['prefix_lens']
plans=[host_plan(ext,pre,r) for r in range(8)]
owned=np.sort(np.concatenate([p[1][p[2].astype(bool)] for p in plans]))
assert len(np.unique(owned))==len(owned)
active=np.flatnonzero(plans[0][0]>512)
assert np.isin(active,owned).all()
active_mask=np.zeros(m,dtype=bool);active_mask[active]=True


def rows_for(owner, phase):
    if phase is None:
        ids=plans[owner][1];valid=plans[owner][2].astype(bool)&active_mask[ids]
    else:
        tiles=np.unique(active//phase)[owner::8]
        ids=(tiles[:,None]*phase+np.arange(phase)[None,:]).reshape(-1)
        valid=(ids<m)&active_mask[np.minimum(ids,m-1)]
        assert np.all(ids%phase==np.arange(len(ids))%phase)
        ids=np.minimum(ids,m-1)
    return torch.from_numpy(ids).cuda(),torch.from_numpy(valid).cuda()


def difference(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    bad=a!=b
    return dict(exact=not bool(bad.any()),different=int(bad.sum()),rows=int(bad.any(1).sum()),
                max_abs=float((a.float()-b.float()).abs().max()))


def timing(call):
    for _ in range(3):call()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):call()
    samples=[]
    for _ in range(4):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize()
        samples.append(a.elapsed_time(b)/3)
    return statistics.median(samples)


report=dict(status='running',scope=__doc__,input_sha256=capture['q_lora_sha256'],
    weight_sha256=digest(w),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    library=str(fixed.path),query_backend_env=os.environ['ROCBLAS_USE_HIPBLASLT'],
    owned_rows=len(owned),active_rows=len(active),
    variants={},screen=[],validation=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    reference=F.linear(x,w)
    assert torch.equal(reference,F.linear(x,w)), 'Full reference itself changed before the screen'
    variants={}
    for name,phase in [('compact16',None),('phase32',32),('phase64',64),('phase128',128),('phase256',256)]:
        ids,valid=rows_for(0,phase);part=x.index_select(0,ids)
        variants[name]=(ids,valid,part,reference.index_select(0,ids),phase)
        available=fixed.solutions(part,w)
        report['variants'][name]=dict(m=len(part),active=int(valid.sum()),solutions=available,
            automatic=difference(F.linear(part,w)[valid],reference.index_select(0,ids)[valid]))
        save()
    available=fixed.solutions(x,w)
    report['full_solutions']=available
    common=sorted(set(available).intersection(*(set(v['solutions']) for v in report['variants'].values())))
    assert common and all(s>0 for s in common), 'Expected explicit Lt indices from the forced backend'
    count=min(48,len(common))
    selected=[common[(i*(len(common)-1))//max(1,count-1)] for i in range(count)]
    report['common_count']=len(common);report['screen_indices']=selected;save()
    for solution in selected:
        item=dict(solution=solution,variants={})
        for name,(ids,valid,part,expected,phase) in variants.items():
            actual=fixed.run(part,w,solution)
            item['variants'][name]=difference(actual[valid],expected[valid])
        report['screen'].append(item);save()
        print('SCREEN',solution,{name:v['different'] for name,v in item['variants'].items()},flush=True)
    finalists=[(item['solution'],name) for item in report['screen']
               for name,v in item['variants'].items() if v['exact']]
    report['exact_finalists']=finalists
    for solution,name in finalists[:6]:
        phase=variants[name][-1]
        item=dict(solution=solution,variant=name,owners=[])
        for owner in range(8):
            ids,valid=rows_for(owner,phase);part=x.index_select(0,ids)
            assert solution in fixed.solutions(part,w)
            actual=fixed.run(part,w,solution)
            item['owners'].append(dict(owner=owner,m=len(part),active=int(valid.sum()),
                versus_full=difference(actual[valid],reference.index_select(0,ids)[valid])))
        ids,valid,part,expected,_=variants[name]
        out=torch.empty((len(part),8192),device=x.device,dtype=x.dtype)
        item['projection_only_ms']=timing(lambda:fixed.run(part,w,solution,out))
        report['validation'].append(item);save()
    report['status']='complete';save()
finally:
    if report['status']!='complete':
        report['status']='failed';report['error']=traceback.format_exc();save()
    fixed.close()
