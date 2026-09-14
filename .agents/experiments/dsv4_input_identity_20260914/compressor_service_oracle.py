"""Reproduce service projection with actual captured weights and full inputs."""
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from woa_tiles import project
from sglang.kernels.ops.attention.dsv4.gemm import linear_bf16_fp32

assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
assert os.environ.get('SGLANG_USE_AITER') == '1'
torch.set_num_threads(4)
root = Path(__file__).resolve().parent / 'stable-layer2-compressor'
assert (root/'complete.json').exists()
out = root/'service-projection-oracle.json'
assert not out.exists()
sources = {}
def read(arm, kind, name):
    p=root/('trace-'+arm)/f'layer_2_rank_0_prepare_compressor_{kind}_{name}.pt'
    sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return torch.load(p, weights_only=True)
results = []
for kind,prefix in [('core','layers.2.attn.compressor'),('index','layers.2.attn.indexer.compressor')]:
    w = read('A1',kind,'weight')
    with safe_open('/home/pc/models/modelscope/model-00004-of-00048.safetensors',framework='pt',device='cpu') as f:
        raw=torch.cat([f.get_tensor(prefix+'.'+n+'.weight') for n in ('wkv','wgate')]).to(w.dtype)
    assert torch.equal(w,raw)
    w=w.cuda(); xx=[]; yy=[]; fixed=[]
    for arm in ('A1','B1','A2'):
        x=read(arm,kind,'input').cuda(); service=read(arm,kind,'projection').cuda()
        replay=linear_bf16_fp32(x,w)
        exact=torch.equal(replay,service)
        assert exact, (kind,arm,'runtime not reproduced')
        y=project(x,w,(128,128,128,8)).float()
        xx.append(x); yy.append(service); fixed.append(y)
        row=dict(kind=kind,arm=arm,checkpoint_weight_exact=True,
                 full_service_reproduced=exact,rows=x.shape[0],
                 fixed_vs_service_changed=int(torch.count_nonzero(y!=service)),
                 fixed_vs_service_max_abs=float((y-service).abs().max()))
        results.append(row); print(json.dumps(row),flush=True)
    assert torch.equal(xx[0][-8192:],xx[1][-8192:])
    assert torch.equal(fixed[0][-8192:],fixed[1][-8192:])
    assert torch.equal(fixed[0],fixed[2])
    # Exercise real per-row values with ten deterministic changes and both row offsets.
    mutations=[]
    for i in range(10):
        a,b=xx[0].clone(),xx[1].clone()
        delta=(i+1)/128
        for x in (a,b): x[-8192:, i*31] += delta
        fa,fb=[project(x,w,(128,128,128,8)).float()[-8192:] for x in (a,b)]
        assert torch.equal(fa,fb) and torch.isfinite(fa).all()
        mutations.append(dict(index=i,exact=True))
    results.append(dict(kind=kind,changed_rows_exact=True,mutations=mutations))
    del xx,yy,fixed,w,a,b,fa,fb,x,y,service,replay
out.write_text(json.dumps(dict(results=results,sources=sources),indent=2)+'\n')
