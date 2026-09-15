"""Exploratory cross-revision comparison; current A request did NOT complete."""
import hashlib
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
paths=[root.parent/'dsv4_c16_stage_frontier_20260915/A/data',root/'A/data']
stem='layer-22-rank-0'
metas=[torch.load(p/(stem+'-metadata.pt'),weights_only=True,map_location='cpu') for p in paths]
assert metas[0]['metadata']==metas[1]['metadata']
assert all(torch.equal(metas[0][k],metas[1][k]) for k in ('input_ids','positions'))
report=dict(scope=__doc__,same_metadata_and_input_tokens=True,stages=[],sources={})
for name in ('ffn_input','ffn_topk_ids','ffn_topk_weights','ffn_routed','ffn_out'):
    records=[json.loads((p/f'{stem}-{name}.json').read_text()) for p in paths]
    values=[]
    for p,r in zip(paths,records):
        file=p/r['values_file']
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest==r['file_sha256']
        report['sources'][str(file)]=digest
        values.append(torch.load(file,weights_only=True,map_location='cpu'))
    changed=(values[0]['row_hashes']!=values[1]['row_hashes']).any(1).nonzero().flatten().tolist()
    item=dict(stage=name,full_equal=records[0]['sha256']==records[1]['sha256'],changed_rows=changed)
    if changed:
        item['sampled_numeric']=[]
        for row in changed:
            if row not in metas[0]['metadata']['sampled_rows']:continue
            i=metas[0]['metadata']['sampled_rows'].index(row)
            a,b=[v['samples'][i].view(torch.bfloat16).float() for v in values]
            columns=(a!=b).nonzero().flatten()
            item['sampled_numeric'].append(dict(row=row,position=int(metas[0]['positions'][row]),
                columns=columns.tolist(),old=a[columns].tolist(),new=b[columns].tolist(),
                max_abs=float((a-b).abs().max())))
    report['stages'].append(item)
target=root/'historical.json';assert not target.exists()
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['stages'],indent=2))
