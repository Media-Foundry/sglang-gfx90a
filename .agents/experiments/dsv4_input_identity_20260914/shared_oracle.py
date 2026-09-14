"""Frozen full shared-expert replay, then independently fix either GEMM."""
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

assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
torch.set_num_threads(4)
root=Path(__file__).resolve().parent
run=root/'layer0-stable-qkv-wqb-wob'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--service',action='store_true')
parser.add_argument('--axes',action='store_true')
args=parser.parse_args()
assert not (args.service and args.axes)
target=root/('shared-service-oracle.json' if args.service else
             'shared-axes-oracle.json' if args.axes else 'shared-oracle.json')
assert (run/'complete.json').exists() and not target.exists()
model=Path('/home/pc/models/modelscope')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
sources={}
def load(arm,rank,name):
    path=run/('trace-'+arm)/f'layer_0_rank_{rank}_{name}.pt'
    sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return torch.load(path,weights_only=True)
def read(key):
    with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)
def dequant(name):
    w=read(f'layers.0.ffn.shared_experts.{name}.weight')
    s=read(f'layers.0.ffn.shared_experts.{name}.scale')
    n,k=w.shape
    return (w.float().view(n//128,128,k//128,128)*s.float()[:,None,:,None]).reshape(n,k).bfloat16()
def time_ms(fn):
    fn();fn();values=[]
    for _ in range(9):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return statistics.median(values)
wg,wu,wd=[dequant(n) for n in ('w1','w3','w2')]
records=[]
for rank in range(8):
    gate=torch.cat([w[rank*256:(rank+1)*256] for w in (wg,wu)]).contiguous().cuda()
    down=wd[:,rank*256:(rank+1)*256].contiguous().cuda()
    xs=[];rows=[];refs=[];keys=[]
    for arm,m in [('A1',32768),('B1',32767)]:
        rr=load(arm,rank,'sample_rows');pos=load(arm,rank,'positions')
        x=torch.zeros((m,4096),device='cuda',dtype=torch.bfloat16)
        x.index_copy_(0,rr.cuda(),load(arm,rank,'ffn_input').cuda())
        xs.append(x);rows.append(rr.cuda());refs.append(load(arm,rank,'ffn_shared').cuda())
        keys.append({int(pos[r]):i for i,r in enumerate(rr.tolist()) if r>=m-8192})
    common=sorted(set(keys[0])&set(keys[1]));assert len(common)==153
    aa=torch.tensor([int(rows[0][keys[0][p]]) for p in common],device='cuda')
    bb=torch.tensor([int(rows[1][keys[1][p]]) for p in common],device='cuda')
    assert torch.equal(xs[0][aa],xs[1][bb])
    for mode in ('linear','fixed-gate','fixed-down','fixed-both'):
        def fn(x):
            gu=project(x,gate,(128,128,128,8)) if mode in ('fixed-gate','fixed-both') else F.linear(x,gate)
            g,u=gu.chunk(2,dim=-1)
            mid=F.silu(g.clamp(max=10.))*u.clamp(min=-10.,max=10.)
            y=project(mid,down,(128,128,128,8)) if mode in ('fixed-down','fixed-both') else F.linear(mid,down)
            return gu,mid,y
        outputs=[fn(x) for x in xs]
        mismatches=[int(torch.count_nonzero(y[2][r]!=ref)) for y,r,ref in zip(outputs,rows,refs)]
        if mode=='linear':assert mismatches==[0,0],(rank,mismatches)
        stages={name:dict(elements=int(torch.count_nonzero(outputs[0][i][aa]!=outputs[1][i][bb])),
                            rows=int(torch.any(outputs[0][i][aa]!=outputs[1][i][bb],dim=1).sum()))
                for i,name in enumerate(('gate_up','bounded_swiglu','down'))}
        if mode=='fixed-both':assert all(v['elements']==0 for v in stages.values())
        row=dict(rank=rank,mode=mode,service_mismatches=mismatches,stages=stages,
                 ms=[time_ms(lambda x=x:fn(x)) for x in xs])
        if args.axes:
            padded=F.pad(xs[1],(0,0,0,1))
            yz=fn(padded)
            row['same_m_shifted']={name:int(torch.count_nonzero(outputs[0][i][aa]!=yz[i][bb]))
                                  for i,name in enumerate(('gate_up','bounded_swiglu','down'))}
            row['same_rows_changed_m']={name:int(torch.count_nonzero(outputs[1][i][bb]!=yz[i][bb]))
                                       for i,name in enumerate(('gate_up','bounded_swiglu','down'))}
            del padded,yz
        records.append(row);print(json.dumps(row),flush=True)
    if args.service:
        from sglang.kernels.ops.debug.dsv4_prefill_shared import shared
        for x,y in zip(xs,outputs):assert torch.equal(shared(x,gate,down,10.),y[2])
        samples=xs[0][aa].clone()
        for trial in range(100):
            ar=25088+128*(trial%4);br=ar-(1,63,64,127,128,129,255,256)[trial%8]
            sample=(samples[trial%len(samples)].float()*(1+(trial%9-4)/64)).bfloat16()
            xs[0][ar]=sample;xs[1][br]=sample
            assert torch.equal(shared(xs[0],gate,down,10.)[ar],shared(xs[1],gate,down,10.)[br]),(rank,trial)
        records.append(dict(rank=rank,mode='service-mutations',passes=100))
        print('service mutations pass',rank,100,flush=True)
    del xs,outputs,gate,down
target.write_text(json.dumps(dict(records=records,sources=sources,
    scope='Layer0 retained case15 rows only. Fixed variants do not claim old-GEMM parity.'),indent=2)+'\n')
