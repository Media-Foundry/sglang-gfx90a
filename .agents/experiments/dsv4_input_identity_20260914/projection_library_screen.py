"""Real-input QKV/wq_b component screen: validate service reproduction first."""
import json
import os
from pathlib import Path
import statistics
import argparse

from safetensors import safe_open
import torch
import torch.nn.functional as F
from woa_tiles import project
from triton.runtime.errors import OutOfResources

root = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--service-kernel',action='store_true')
args=parser.parse_args()
if args.service_kernel:
    from sglang.kernels.ops.debug.dsv4_prefill_wqb import project as service_project
assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
output = root / ('wqb-service-oracle.json' if args.service_kernel else 'projection-library-screen.json')
assert not output.exists()
run = root / 'layer0-stable-qkv'
assert (run / 'complete.json').exists()

def load(arm, rank, name):
    return torch.load(run / ('trace-' + arm) / f'layer_0_rank_{rank}_{name}.pt', weights_only=True)

def time_ms(fn):
    fn(); fn(); values=[]
    for _ in range(9):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();values.append(a.elapsed_time(b))
    return statistics.median(values)

records=[]
with safe_open('/home/pc/models/modelscope/model-00002-of-00048.safetensors',framework='pt',device='cpu') as f:
    raw=f.get_tensor('layers.0.attn.wq_b.weight')
    scales=f.get_tensor('layers.0.attn.wq_b.scale')
assert raw.shape==(32768,1024) and scales.shape==(256,8)
for rank in range(8):
    weight=(raw[rank*4096:(rank+1)*4096].float().view(32,128,8,128)
            *scales[rank*32:(rank+1)*32].float()[:,None,:,None]).reshape(4096,1024).bfloat16().cuda()
    xs=[];refs=[];sample_rows=[];keys=[]
    for arm,m in [('A1',32768),('B1',32767)]:
        positions=load(arm,rank,'positions').tolist()
        rows=load(arm,rank,'sample_rows')
        source=load(arm,rank,'prepare_q_lora_norm').cuda()
        x=torch.zeros((m,1024),device='cuda',dtype=torch.bfloat16)
        x.index_copy_(0,rows.cuda(),source)
        xs.append(x);sample_rows.append(rows.cuda())
        refs.append(load(arm,rank,'prepare_q_before_norm_rope').cuda())
        # Case15 is the final 8192 rows in both verified trace layouts.
        keys.append({positions[r]:i for i,r in enumerate(rows.tolist()) if r>=m-8192})
    common=sorted(set(keys[0])&set(keys[1]))
    for name in (['linear','service'] if args.service_kernel else
                 ['linear','transposed-mm','tile128','tile64','wide128','wide64']):
        def fn(x):
            if name=='linear':return F.linear(x,weight)
            if name=='service':return service_project(x,weight)
            if name=='transposed-mm':return (weight@x.t()).t().contiguous()
            tile={'tile128':(128,128,128,8),'tile64':(64,128,128,4),
                  'wide128':(128,256,64,8),'wide64':(64,256,128,8)}[name]
            return project(x,weight,tile)
        try:
            ys=[fn(x).index_select(0,r) for x,r in zip(xs,sample_rows)]
        except OutOfResources as exc:
            records.append(dict(rank=rank,backend=name,status='out-of-resources',error=str(exc)))
            print(records[-1],flush=True)
            continue
        reproduced=[int(torch.count_nonzero(y!=ref)) for y,ref in zip(ys,refs)]
        if name=='linear': assert reproduced==[0,0],(rank,reproduced)
        if name=='service':
            assert all(torch.equal(y,project(x,weight,(128,128,128,8)).index_select(0,r))
                       for x,r,y in zip(xs,sample_rows,ys)),rank
        a=torch.stack([ys[0][keys[0][p]] for p in common])
        b=torch.stack([ys[1][keys[1][p]] for p in common])
        row=dict(rank=rank,backend=name,changed_vs_service=reproduced,
                 shifted_changed=int(torch.count_nonzero(a!=b)),
                 shifted_rows=int(torch.count_nonzero(torch.any(a!=b,dim=1))),
                 max_shift=float((a.float()-b.float()).abs().max()),
                 ms=[time_ms(lambda x=x:fn(x)) for x in xs])
        records.append(row);print(row,flush=True)
    if args.service_kernel:
        source=load('A1',rank,'prepare_q_lora_norm').cuda()
        mutations=[]
        for trial in range(100):
            a=25088+(trial%4)*128
            b=a-(1,63,64,127,128,129,255,256)[trial%8]
            sample=(source[trial%len(source)].float()*(1+(trial%9-4)/64)).bfloat16()
            xs[0][a].copy_(sample);xs[1][b].copy_(sample)
            ya=service_project(xs[0],weight)[a]
            yb=service_project(xs[1],weight)[b]
            assert torch.equal(ya,yb),(rank,trial)
            mutations.append(dict(trial=trial,row_a=a,row_b=b,exact=True))
        records.append(dict(rank=rank,backend='service-mutations',trials=mutations))
        print('mutation passes',rank,len(mutations),flush=True)
    del xs,weight,ys
output.write_text(json.dumps(records,indent=2)+'\n')
