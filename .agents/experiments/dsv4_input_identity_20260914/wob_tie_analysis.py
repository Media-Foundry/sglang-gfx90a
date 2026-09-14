"""CPU FP64 dot reference for the actual 40 changed wo_b output elements."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

torch.set_num_threads(4)
root=Path(__file__).resolve().parent
run=root/'layer0-stable-qkv-wqb'
target=root/'wob-ties.json'
assert not target.exists()
def load(arm,rank,name):
    return torch.load(run/('trace-'+arm)/f'layer_0_rank_{rank}_{name}.pt',weights_only=True)
with safe_open('/home/pc/models/modelscope/model-00002-of-00048.safetensors',framework='pt',device='cpu') as f:
    raw=f.get_tensor('layers.0.attn.wo_b.weight')
    scale=f.get_tensor('layers.0.attn.wo_b.scale')
records=[]
for rank in range(8):
    w=(raw[:,rank*1024:(rank+1)*1024].float().reshape(32,128,8,128)
       *scale[:,rank*8:(rank+1)*8].float()[:,None,:,None]).reshape(4096,1024).bfloat16()
    keys=[];ys=[];xs=[]
    for arm,m in [('A1',32768),('B1',32767)]:
        rows=load(arm,rank,'sample_rows');pos=load(arm,rank,'positions')
        keys.append({int(pos[r]):i for i,r in enumerate(rows.tolist()) if r>=m-8192})
        ys.append(load(arm,rank,'wo_b_partial'));xs.append(load(arm,rank,'wo_a').flatten(1))
    for pos in sorted(set(keys[0])&set(keys[1])):
        i,j=keys[0][pos],keys[1][pos]
        assert torch.equal(xs[0][i],xs[1][j])
        for col in (ys[0][i]!=ys[1][j]).nonzero().flatten().tolist():
            a=float(ys[0][i,col]);b=float(ys[1][j,col])
            reference=float(torch.dot(xs[0][i].double(),w[col].double()))
            midpoint=(a+b)/2
            spacing=abs(a-b)
            lo=torch.tensor(min(a,b),dtype=torch.bfloat16)
            adjacent=float(torch.nextafter(lo,torch.tensor(float('inf'),dtype=torch.bfloat16)))==max(a,b)
            records.append(dict(rank=rank,position=pos,column=col,a=a,b=b,adjacent=adjacent,
                reference_fp64=reference,midpoint=midpoint,
                midpoint_error=reference-midpoint,error_fraction_of_spacing=abs(reference-midpoint)/spacing,
                bf16_fp64_reference=float(torch.tensor(reference,dtype=torch.float64).bfloat16())))
assert len(records)==40
result=dict(rows=records,all_adjacent=all(r['adjacent'] for r in records),
    max_midpoint_error=max(abs(r['midpoint_error']) for r in records),
    max_fraction_of_spacing=max(r['error_fraction_of_spacing'] for r in records),
    exact_midpoints=sum(r['midpoint_error']==0 for r in records))
target.write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='rows'})
