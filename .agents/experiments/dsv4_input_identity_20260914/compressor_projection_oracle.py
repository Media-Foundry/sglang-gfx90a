"""Real layer2 inputs: isolate compressor projection row/shape sensitivity.

This does not claim service kv_score reproduction: those outputs have not yet
been captured. It calls the runtime projection entry with checkpoint weights.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
from safetensors import safe_open
from woa_tiles import project
from sglang.kernels.ops.attention.dsv4.gemm import linear_bf16_fp32

assert os.environ.get('HIP_VISIBLE_DEVICES')=='4' and os.environ.get('SGLANG_USE_AITER')=='1'
torch.set_num_threads(4)
root=Path(__file__).resolve().parent
run=root/'stable-layer2-prepare';out=root/'compressor-projection-oracle.json'
assert (run/'complete.json').exists() and not out.exists()
xs=[];sources={}
for arm,m in [('A1',32768),('B1',32767)]:
    p=run/('trace-'+arm)/'layer_2_rank_0_prepare_full_input.pt'
    x=torch.load(p,weights_only=True);assert x.shape==(m,4096)
    sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    xs.append(x.cuda())
assert torch.equal(xs[0][-8192:],xs[1][-8192:])
def time_ms(fn):
    fn();fn();times=[]
    for _ in range(9):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();times.append(a.elapsed_time(b))
    return statistics.median(times)
records=[]
for kind,prefix in [('core','layers.2.attn.compressor'),('index','layers.2.attn.indexer.compressor')]:
    with safe_open('/home/pc/models/modelscope/model-00004-of-00048.safetensors',framework='pt',device='cpu') as f:
        raw=[f.get_tensor(prefix+'.'+name+'.weight') for name in ('wkv','wgate')]
    w=torch.cat(raw).bfloat16().contiguous().cuda()
    assert w.shape==((2048 if kind=='core' else 512),4096)
    for backend in ('runtime','linear','fixed128'):
        def fn(x):
            if backend=='runtime':return linear_bf16_fp32(x,w)
            if backend=='linear':return F.linear(x,w).float()
            return project(x,w,(128,128,128,8)).float()
        yy=[fn(x) for x in xs]
        assert all(torch.isfinite(y).all() for y in yy)
        if backend=='runtime':runtime=[y[-8192:].clone() for y in yy]
        a,b=[y[-8192:] for y in yy];different=a!=b
        padded=fn(F.pad(xs[1],(0,0,0,1)))[-8193:-1]
        changed_rows=different.any(dim=1).nonzero().flatten().tolist()
        row=dict(kind=kind,backend=backend,shape=list(w.shape),
                 changed_elements=int(different.sum()),changed_rows=changed_rows,
                 max_abs=float((a-b).abs().max()),
                 first8_changes=different[:8].nonzero().tolist(),
                 same_m_shifted=int(torch.count_nonzero(a!=padded)),
                 same_rows_changed_m=int(torch.count_nonzero(b!=padded)),
                 changed_from_runtime=[int(torch.count_nonzero(y!=r)) for y,r in zip((a,b),runtime)],
                 ms=[time_ms(lambda x=x:fn(x)) for x in xs])
        if backend=='fixed128':assert row['changed_elements']==row['same_m_shifted']==row['same_rows_changed_m']==0
        records.append(row)
        print(json.dumps({**row,'changed_rows_count':len(changed_rows),'changed_rows':changed_rows[:20]}),flush=True)
    del w,yy,runtime
out.write_text(json.dumps(dict(records=records,sources=sources,scope=__doc__),indent=2)+'\n')
