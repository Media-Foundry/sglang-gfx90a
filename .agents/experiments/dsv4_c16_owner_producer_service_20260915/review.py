"""Input/output identity evidence only; does not certify generated code claims."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
waves={};sources={};manifests=[]
for arm in ['A-quality','B-check']:
    manifest=json.loads((root/arm/'inputs.json').read_text());manifests.append(manifest)
    assert json.loads((root/arm/'complete.json').read_text())['input_echo_exact']==32
    for rep in range(2):
        p=root/arm/f'quality-{rep}.json'
        rows=json.loads(p.read_text())
        rows=sorted(rows,key=lambda row:int(row['meta_info']['id'].rsplit('-',1)[-1]))
        assert len(rows)==16
        for i,(req,row) in enumerate(zip(manifest['requests'],rows,strict=True)):
            assert row['meta_info']['id']==f'producer-{arm}-{rep}-{i}'
            assert row['prompt_token_ids']==req['input_ids']
            assert row['meta_info']['cached_tokens']==0
            assert len(row['output_ids'])==row['meta_info']['completion_tokens']==256
        waves[f'{arm}.{rep}']=rows
        sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
assert manifests[0]==manifests[1]
def prefix(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
result=dict(identical_inputs=True,exact_echoes=64,sources=sources,comparisons={},candidates=[],
            manual_review_completed=False,semantic_caveat='Coherence is not factual verification of generated bug claims.')
for left,right in [('A-quality.0','A-quality.1'),('B-check.0','B-check.1'),
                   ('A-quality.0','B-check.0'),('A-quality.0','B-check.1')]:
    common=[prefix(a['output_ids'],b['output_ids']) for a,b in zip(waves[left],waves[right])]
    result['comparisons'][f'{left} vs {right}']=dict(exact=sum(n==256 for n in common),prefixes=common)
for rep in range(2):
    for i,row in enumerate(waves[f'B-check.{rep}']):
        matches=[control for control in ['A-quality.0','A-quality.1']
                 if row['output_ids']==waves[control][i]['output_ids']]
        result['candidates'].append(dict(wave=rep,case=i,matching_controls=matches,text=row['text']))
path=root/'quality-review.json';assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['comparisons'],indent=2))
for row in result['candidates']:
    if not row['matching_controls']:
        print('UNMATCHED CANDIDATE',row['wave'],row['case'],row['text'])
