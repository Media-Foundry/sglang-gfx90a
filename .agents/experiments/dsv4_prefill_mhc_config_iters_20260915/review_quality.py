"""Prepare bounded review evidence; require an explicit manual-review assertion."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--confirm-bounded-review',action='store_true',
               help='Only after reading all candidate texts and unique control alternatives.')
args=p.parse_args()
root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
waves={};hashes={}
for arm in ('A1','B','A2'):
    manifest=json.loads((root/arm/'inputs.json').read_text())
    for rep in (0,1):
        path=root/arm/f'quality-{rep}.json'
        responses=json.loads(path.read_text())
        by_id={x['meta_info']['id']:x for x in responses}
        assert len(responses)==len(by_id)==16
        ordered=[]
        for i,request in enumerate(manifest['requests']):
            response=by_id[f'mhc20-{arm}-{rep}-{i}']
            assert response['prompt_token_ids']==request['input_ids']
            assert response['meta_info']['completion_tokens']==128 and len(response['output_ids'])>=128
            ordered.append(dict(tokens=response['output_ids'][-128:],text=response['text']))
        waves[f'{arm}.{rep}']=ordered
        hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
controls=('A1.0','A1.1','A2.0','A2.1')
candidates=[]
for name in ('B.0','B.1'):
    for i,row in enumerate(waves[name]):
        matches=[c for c in controls if waves[c][i]['tokens']==row['tokens']]
        prefixes={c:next((j for j,(x,y) in enumerate(zip(row['tokens'],waves[c][i]['tokens'],strict=True)) if x!=y),128) for c in controls}
        candidates.append(dict(wave=name,case=i,matching_controls=matches,
                               common_prefix_tokens=prefixes,text=row['text']))
result=dict(bounded_coherence_pass=args.confirm_bounded_review,
    scope='Manual coherence/topic/collapse review only. Not complete-answer factual accuracy, global determinism, or proof that iteration policy resolves all observed drift.',
    input_echo_exact=sum(v['input_echo_exact'] for v in summary['quality'].values()),
    quality=summary['quality'],source_sha256=hashes,
    candidate_novel_cases=[dict(wave=x['wave'],case=x['case']) for x in candidates if not x['matching_controls']],
    candidates=candidates)
(root/'quality-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
