"""Exact input/output witnesses, plus comparison with previously read candidates."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
prior=root.parent/'dsv4_c16_owner_producer_service_20260915/B-check'
waves={};manifests=[];sources={}
for arm in ['A1','B','A2']:
    manifest=json.loads((root/arm/'inputs.json').read_text());manifests.append(manifest)
    for rep in range(2):
        p=root/arm/f'quality-{rep}.json'
        rows=json.loads(p.read_text());byid={r['meta_info']['id']:r for r in rows}
        assert set(byid)=={f'owner-{arm}-{rep}-{i}' for i in range(16)}
        ordered=[byid[f'owner-{arm}-{rep}-{i}'] for i in range(16)]
        for req,row in zip(manifest['requests'],ordered,strict=True):
            assert row['prompt_token_ids']==req['input_ids'] and row['meta_info']['cached_tokens']==0
            assert len(row['output_ids'])==row['meta_info']['completion_tokens']==128
        waves[f'{arm}.{rep}']=ordered;sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
assert manifests[0]==manifests[1]==manifests[2]==json.loads((prior/'inputs.json').read_text())
old=[]
for rep in range(2):
    rows=json.loads((prior/f'quality-{rep}.json').read_text())
    old.append(sorted(rows,key=lambda r:int(r['meta_info']['id'].rsplit('-',1)[-1])))
    assert all(r['prompt_token_ids']==q['input_ids'] for r,q in zip(old[-1],manifests[0]['requests']))
comparisons={}
for a,b in [('A1.0','A1.1'),('A2.0','A2.1'),('A1.0','A2.0'),('B.0','B.1'),('A1.0','B.0')]:
    comparisons[f'{a} vs {b}']=sum(x['output_ids']==y['output_ids'] for x,y in zip(waves[a],waves[b]))
candidates=[]
for rep in range(2):
    for i,row in enumerate(waves[f'B.{rep}']):
        matches=[j for j in range(2) if row['output_ids']==old[j][i]['output_ids'][:128]]
        candidates.append(dict(wave=rep,case=i,prior_candidate_matches=matches,text=row['text']))
result=dict(exact_quality_echoes=96,comparisons=comparisons,candidates=candidates,sources=sources,
    prior_candidate_exact=sum(bool(r['prior_candidate_matches']) for r in candidates),
    caveat='Prior matching text was read for coherence, not certified factual code analysis. No global determinism claim.')
out=root/'quality-review.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(comparisons,'prior candidate exact',result['prior_candidate_exact'])
for row in candidates:
    if not row['prior_candidate_matches']:print('NEW CANDIDATE TEXT',row['wave'],row['case'],row['text'])
