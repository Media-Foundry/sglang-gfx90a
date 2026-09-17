"""Fresh explicit-Lt process without ROCBLAS_USE_HIPBLASLT or global preference changes."""
import hashlib
import json
import os
from pathlib import Path
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5' and 'ROCBLAS_USE_HIPBLASLT' not in os.environ
root=Path(__file__).resolve().parent
target=root/'abi.json';assert not target.exists()
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from direct import Direct
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan

torch.set_num_threads(1);torch.set_grad_enabled(False)
assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>50*1024**3
preference=str(torch.backends.cuda.preferred_blas_library())
fixture=root.parent/'dsv4_c16_owner_producer_20260915/capture/oracle'
capture=json.loads((fixture/'report.json').read_text())
assert hashlib.sha256((fixture/'inputs.pt').read_bytes()).hexdigest()==capture['inputs_file_sha256']
data=torch.load(fixture/'inputs.pt',map_location='cpu',weights_only=True)
digest=lambda t:hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
assert digest(data['q_lora'])==capture['q_lora_sha256']
model=Path('/home/pc/models/modelscope');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
key='layers.20.attn.indexer.wq_b'
with safe_open(str(model/index[key+'.weight']),framework='pt',device='cpu') as f:
    w=(f.get_tensor(key+'.weight').float().reshape(64,128,8,128)*f.get_tensor(key+'.scale').float()[:,None,:,None]).reshape(8192,1024).bfloat16().cuda()
assert digest(w)==capture['parameters']['wq_b']['weight']['sha256']
seed=data['q_lora'].cuda()
op=Direct(root/'build-v1/manifest.json')
report=dict(status='running',preference=preference,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    direct_sha256=hashlib.sha256((root/'direct.py').read_bytes()).hexdigest(),cases=[],graph_mutations=0)
def save():target.write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    # This creates the Lt plan before any torch GEMM, testing true cold handle setup.
    op.prepare(3072,8192,1024)
    for m in (8192,32767,32768):
        ext=[8192]*(m//8192)+([m%8192] if m%8192 else [])
        original=torch.cat((seed,seed[:1]))[:m].contiguous()
        for mutation in range(3):
            x=(original.roll(mutation*17,0).float()*(1-.011*mutation)).bfloat16()
            full=F.linear(x,w)
            for owner in range(8):
                lengths,ids,valid,_=host_plan(ext,[0]*len(ext),owner)
                mask=torch.from_numpy(valid.astype(bool)&(lengths[ids]>512)).cuda()
                ids=torch.from_numpy(ids).cuda();part=x.index_select(0,ids)
                actual=op(part,w);expected=full.index_select(0,ids)
                assert torch.equal(actual[mask],expected[mask]), (m,mutation,owner)
                report['cases'].append(dict(m=m,mutation=mutation,owner=owner,active=int(mask.sum()),exact=True))
            save()
    # Graph replays change actual inputs while retaining all captured addresses.
    ext=capture['extend_lens'];pre=capture['prefix_lens']
    lengths,ids,valid,_=host_plan(ext,pre,0)
    mask=torch.from_numpy(valid.astype(bool)&(lengths[ids]>512)).cuda();ids=torch.from_numpy(ids).cuda()
    x=seed.clone();part=x.index_select(0,ids)
    out=torch.empty((len(part),8192),device=part.device,dtype=part.dtype)
    graph_stream=torch.cuda.Stream();graph_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph_stream):
        graph_op=Direct(root/'build-v1/manifest.json')
        try:
            graph_op(part,w,out)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=graph_stream):graph_op(part,w,out)
            for iteration in range(100):
                x.copy_((seed.roll(iteration,0).float()*(1-iteration*.0001)).bfloat16())
                part.copy_(x.index_select(0,ids));graph.replay()
                expected=F.linear(x,w).index_select(0,ids)
                assert torch.equal(out[mask],expected[mask]), iteration
                report['graph_mutations']+=1
        finally:graph_op.close()
    torch.cuda.current_stream().wait_stream(graph_stream)
    alternate=torch.cuda.Stream();alternate.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alternate):
        other=Direct(root/'build-v1/manifest.json')
        try:
            alternative=other(part,w)
            alternate.synchronize()
            assert torch.equal(alternative[mask],expected[mask])
            report['alternate_stream_exact']=True
        finally:other.close()
    assert str(torch.backends.cuda.preferred_blas_library())==preference
    assert 'ROCBLAS_USE_HIPBLASLT' not in os.environ
    report['workspace_bytes']=0 if op.workspace is None else op.workspace.numel()
    report['plan_workspaces']={str(k):v[1] for k,v in op.plans.items()}
    report['status']='complete';save()
finally:
    if report['status']!='complete':report['status']='failed';report['error']=traceback.format_exc();save()
    op.close()
