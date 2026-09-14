"""Check whether retained layer0 router drift touches a selected expert."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
root=Path(__file__).resolve().parent
run=root/'layer0-shared-wired'
target=root/'router-membership-audit.json'
assert (run/'complete.json').exists() and not target.exists()
sources={}
def load(arm,rank,name):
    p=run/('trace-'+arm)/f'layer_0_rank_{rank}_{name}.pt'
    sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return torch.load(p,weights_only=True)
records=[]
for rank in range(8):
    values=[];keys=[];tops=[];weights=[];inputs=[]
    for arm in ('A1','B1'):
        pos=load(arm,rank,'positions');rows=load(arm,rank,'sample_rows')
        keys.append({int(pos[row]):i for i,row in enumerate(rows.tolist()) if row>=len(pos)-8192})
        values.append(load(arm,rank,'ffn_router_logits'))
        tops.append(load(arm,rank,'ffn_topk_ids'))
        weights.append(load(arm,rank,'ffn_topk_weights'))
        inputs.append(load(arm,rank,'ffn_input'))
    common=sorted(set(keys[0])&set(keys[1]));assert len(common)==153
    for pos in common:
        i,j=keys[0][pos],keys[1][pos]
        assert torch.equal(inputs[0][i],inputs[1][j])
        assert torch.equal(tops[0][i],tops[1][j]) and torch.equal(weights[0][i],weights[1][j])
        for expert in (values[0][i]!=values[1][j]).nonzero().flatten().tolist():
            selected=tops[0][i].tolist()
            records.append(dict(rank=rank,case=15,position=pos,expert=expert,
                a=float(values[0][i,expert]),b=float(values[1][j,expert]),
                selected_ids=selected,changed_expert_selected=expert in selected))
assert len(records)==8 and not any(r['changed_expert_selected'] for r in records)
result=dict(records=records,sources=sources,scope='Retained layer0 case15 positions only, hash-router layer; not a learned-router invariance proof.')
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
