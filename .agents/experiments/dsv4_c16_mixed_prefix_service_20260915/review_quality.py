"""Collect quality evidence; manual bounded review remains an explicit action."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--confirm-bounded-review',action='store_true',
    help='Only after reading all candidate texts and unique control alternatives.')
args=p.parse_args()
summary=json.loads((ROOT/'summary.json').read_text())
waves={};sources={}
for arm in ('A1','B','A2'):
    path=ROOT/arm/'quality.json'
    raw=path.read_bytes();data=json.loads(raw)
    assert data['status']=='complete' and len(data['rounds'])==2
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    for rep,wave in enumerate(data['rounds']):
        assert wave['cached_tokens']==summary['actual_cached_tokens']
        assert len(wave['responses'])==16
        ordered=[]
        for i,row in enumerate(wave['responses']):
            assert row['case']==i
            response=row['response'];assert len(response['output_ids'])==128
            assert response['output_ids']==summary['quality'][arm]['answers'][rep][i]
            ordered.append(response)
        waves[f'{arm}.{rep}']=ordered
controls=('A1.0','A1.1','A2.0','A2.1')
candidates=[]
for name in ('B.0','B.1'):
    for case,row in enumerate(waves[name]):
        matches=[control for control in controls if waves[control][case]['output_ids']==row['output_ids']]
        prefixes={control:next((i for i,(a,b) in enumerate(zip(row['output_ids'],waves[control][case]['output_ids'],strict=True)) if a!=b),128)
                  for control in controls}
        candidates.append(dict(wave=name,case=case,matching_controls=matches,
            common_prefix_tokens=prefixes,text=row['text']))
result=dict(bounded_review_completed=args.confirm_bounded_review,
    scope='Manual coherence/topic/repetition review of128-token excerpts, not factual accuracy or global determinism. Retain observed answer limitations in manual-review.md.',
    input_echo_exact=sum(r['input_echo_exact'] for r in summary['quality'].values()),
    quality_counts={k:{f:v for f,v in r.items() if f!='answers'} for k,r in summary['quality'].items()},
    candidate_novel_cases=[dict(wave=r['wave'],case=r['case']) for r in candidates if not r['matching_controls']],
    candidates=candidates,source_sha256=sources)
(ROOT/'quality-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
