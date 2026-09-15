"""Exact input witnesses and comparison with in-trial and prior native controls."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
waves={};manifests=[];sources={}
for arm in ['A1','B','A2']:
    manifest=json.loads((root/arm/'inputs.json').read_text());manifests.append(manifest)
    for rep in range(2):
        p=root/arm/f'quality-{rep}.json';rows=json.loads(p.read_text())
        byid={r['meta_info']['id']:r for r in rows}
        assert len(rows)==16 and set(byid)=={f'owner-{arm}-{rep}-{i}' for i in range(16)}
        ordered=[byid[f'owner-{arm}-{rep}-{i}'] for i in range(16)]
        for req,row in zip(manifest['requests'],ordered,strict=True):
            assert row['prompt_token_ids']==req['input_ids'] and row['meta_info']['cached_tokens']==0
            assert len(row['output_ids'])==row['meta_info']['completion_tokens']==128
        waves[f'{arm}.{rep}']=ordered;sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
assert manifests[0]==manifests[1]==manifests[2]
old=[]
for arm in ['A1','A2']:
    prior=root.parent/'dsv4_c16_owner_producer_perf_20260915'/arm
    assert json.loads((prior/'inputs.json').read_text())==manifests[0]
    for rep in range(2):
        rows=json.loads((prior/f'quality-{rep}.json').read_text())
        old.append(sorted(rows,key=lambda r:int(r['meta_info']['id'].rsplit('-',1)[-1])))
comparisons={}
for a,b in [('A1.0','A1.1'),('A2.0','A2.1'),('A1.0','A2.0'),('B.0','B.1'),('A1.0','B.0')]:
    comparisons[f'{a} vs {b}']=sum(x['output_ids']==y['output_ids'] for x,y in zip(waves[a],waves[b]))
controls=[waves[k] for k in ['A1.0','A1.1','A2.0','A2.1']]+old
candidates=[]
for rep in range(2):
    for i,row in enumerate(waves[f'B.{rep}']):
        matches=[j for j,c in enumerate(controls) if row['output_ids']==c[i]['output_ids']]
        candidates.append(dict(wave=rep,case=i,control_matches=matches,text=row['text']))
result=dict(exact_quality_echoes=96,comparisons=comparisons,candidates=candidates,sources=sources,
    matching_control=sum(bool(r['control_matches']) for r in candidates),
    caveat='Matching controls is a numerical/trajectory witness, not certified factual correctness of code analysis.')
out=root/'quality-review.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(comparisons,'matching control',result['matching_control'])
for row in candidates:
    if not row['control_matches']:print('NEW CANDIDATE TEXT',row['wave'],row['case'],row['text'])
