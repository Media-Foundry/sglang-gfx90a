"""Exact-output comparison; explicitly not an automatic semantic verifier."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
output=root/'quality-review.json';assert not output.exists()
waves={};sources={};manifests=[]
for arm in ('A1','B','A2'):
    manifests.append(json.loads((root/arm/'inputs.json').read_text()))
    for rep in range(2):
        path=root/arm/f'quality-{rep}.json'
        sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        byid={r['meta_info']['id']:r for r in json.loads(path.read_text())}
        waves[f'{arm}.{rep}']=[byid[f'owner-{arm}-{rep}-{i}'] for i in range(16)]
        for req,r in zip(manifests[-1]['requests'],waves[f'{arm}.{rep}']):
            assert req['input_ids']==r['prompt_token_ids'] and r['meta_info']['cached_tokens']==0
            assert len(r['output_ids'])==r['meta_info']['completion_tokens']==128
assert manifests[0]==manifests[1]==manifests[2]
controls=('A1.0','A1.1','A2.0','A2.1');candidates=[]
for wave in ('B.0','B.1'):
    for i,row in enumerate(waves[wave]):
        matches=[c for c in controls if row['output_ids']==waves[c][i]['output_ids']]
        prefixes={c:next((j for j,(a,b) in enumerate(zip(row['output_ids'],waves[c][i]['output_ids'])) if a!=b),128) for c in controls}
        candidates.append(dict(wave=wave,case=i,matching_controls=matches,common_prefix_tokens=prefixes,text=row['text']))
result=dict(identical_inputs=True,exact_echoes=96,candidates=candidates,
    novel_candidate_cases=[dict(wave=c['wave'],case=c['case']) for c in candidates if not c['matching_controls']],
    sources=sources,manual_review_completed=False,
    caveat='Output equality/France do not prove all model numerics deterministic or generated code-bug claims true.')
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('candidates','sources')},indent=2))
for c in candidates:
    if not c['matching_controls']:print(c['wave'],c['case'],c['text'])
