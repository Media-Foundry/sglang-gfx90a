"""Frozen wo_b service replay and independent row/shape causality checks.

Only physical GCD4. Sampled rows are restored at their actual service offsets;
other rows are zeros. Reproduction of every retained service value is required
before drawing a causal conclusion. No serving selector is changed.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import statistics

from safetensors import safe_open
import torch
import torch.nn.functional as F
from woa_tiles import project

assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
torch.set_num_threads(4)
root = Path(__file__).resolve().parent
run = root / 'layer0-stable-qkv-wqb'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--service',action='store_true')
args=parser.parse_args()
if args.service:
    from sglang.kernels.ops.debug.dsv4_prefill_wqb import project as service_project
output = root / ('wob-service-oracle.json' if args.service else 'wob-oracle.json')
assert (run / 'complete.json').exists() and not output.exists()
model = Path('/home/pc/models/modelscope')
index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
sources = {}

def load(arm, rank, name):
    path = run / ('trace-'+arm) / f'layer_0_rank_{rank}_{name}.pt'
    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return torch.load(path, weights_only=True)

def weight_tensor(name):
    with safe_open(str(model/index[name]), framework='pt', device='cpu') as f:
        return f.get_tensor(name)

def elapsed(fn):
    fn(); fn(); values=[]
    for _ in range(9):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return statistics.median(values)

raw=weight_tensor('layers.0.attn.wo_b.weight')
scale=weight_tensor('layers.0.attn.wo_b.scale')
assert raw.shape==(4096,8192) and scale.shape==(32,64)
records=[]
for rank in range(8):
    w=(raw[:,rank*1024:(rank+1)*1024].float().reshape(32,128,8,128)
       *scale[:,rank*8:(rank+1)*8].float()[:,None,:,None]).reshape(4096,1024).bfloat16().cuda()
    xs=[];rows=[];refs=[];keys=[]
    for arm,m in [('A1',32768),('B1',32767)]:
        rr=load(arm,rank,'sample_rows')
        pos=load(arm,rank,'positions')
        source=load(arm,rank,'wo_a').flatten(1).cuda()
        assert source.shape==(len(rr),1024)
        x=torch.zeros((m,1024),device='cuda',dtype=torch.bfloat16)
        x.index_copy_(0,rr.cuda(),source)
        xs.append(x);rows.append(rr.cuda());refs.append(load(arm,rank,'wo_b_partial').cuda())
        keys.append({int(pos[r]):i for i,r in enumerate(rr.tolist()) if r>=m-8192})
    common=sorted(set(keys[0])&set(keys[1]))
    assert len(common)==153
    aa=torch.tensor([int(rows[0][keys[0][p]]) for p in common],device='cuda')
    bb=torch.tensor([int(rows[1][keys[1][p]]) for p in common],device='cuda')
    assert torch.equal(xs[0][aa],xs[1][bb]),rank
    for backend in ('linear','fixed128'):
        fn=(lambda x:F.linear(x,w)) if backend=='linear' else (lambda x:project(x,w,(128,128,128,8)))
        if args.service and backend=='fixed128':
            fn=lambda x:service_project(x,w,projection_name='wo_b')
        yy=[fn(x) for x in xs]
        if args.service and backend=='fixed128':
            assert all(torch.equal(y,project(x,w,(128,128,128,8))) for x,y in zip(xs,yy))
        reproduced=[int(torch.count_nonzero(y[r]!=ref)) for y,r,ref in zip(yy,rows,refs)]
        if backend=='linear':assert reproduced==[0,0],(rank,reproduced)
        delta=yy[0][aa]!=yy[1][bb]
        record=dict(rank=rank,backend=backend,service_mismatches=reproduced,
                    shifted_elements=int(delta.sum()),shifted_rows=int(delta.any(dim=1).sum()),
                    max_abs=float((yy[0][aa].float()-yy[1][bb].float()).abs().max()),
                    ms=[elapsed(lambda x=x:fn(x)) for x in xs])
        # Isolate row placement with constant M and shape with constant row.
        z=F.pad(xs[1],(0,0,0,1))
        yz=fn(z)
        record['same_m_adjacent_rows_changed']=int(torch.count_nonzero(yy[0][aa]!=yz[bb]))
        record['same_rows_changed_m_changed']=int(torch.count_nonzero(yy[1][bb]!=yz[bb]))
        if backend=='fixed128':
            assert record['shifted_elements']==record['same_m_adjacent_rows_changed']==record['same_rows_changed_m_changed']==0
            samples=xs[0][aa].clone()
            for trial in range(100):
                ar=25088+128*(trial%4);br=ar-(1,63,64,127,128,129,255,256)[trial%8]
                sample=(samples[trial%len(samples)].float()*(1+(trial%9-4)/64)).bfloat16()
                xs[0][ar]=sample;xs[1][br]=sample
                assert torch.equal(fn(xs[0])[ar],fn(xs[1])[br]),(rank,trial)
            record['row_mutation_passes']=100
        records.append(record);print(json.dumps(record),flush=True)
    del xs,yy,w,z,yz
output.write_text(json.dumps(dict(records=records,sources=sources,
    scope='Layer0 wo_b retained 153 case15 positions per TP shard, physical GCD4 only; not whole-model determinism.'),indent=2)+'\n')
