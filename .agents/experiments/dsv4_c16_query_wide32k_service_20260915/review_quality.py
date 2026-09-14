"""Summarize bounded output review; no automatic claim of semantic correctness."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--confirm-bounded-review',action='store_true',
               help='Use only after manually inspecting both candidate waves and control differences.')
args=p.parse_args()
root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
waves={};hashes={}
for arm in ('A1','B','A2'):
    manifest=json.loads((root/arm/'inputs.json').read_text())
    for rep in (0,1):
        path=root/arm/f'quality-{rep}.json'
        rows=json.loads(path.read_text())
        by_id={r['meta_info']['id']:r for r in rows}
        ordered=[]
        for case,request in enumerate(manifest['requests']):
            response=by_id[f'wide32k-{arm}-{rep}-{case}']
            assert response['prompt_token_ids']==request['input_ids']
            assert response['meta_info']['completion_tokens']==128
            tokens=response['output_ids'][-128:]
            assert len(tokens)==128
            ordered.append(dict(tokens=tokens,text=response['text']))
        waves[f'{arm}.{rep}']=ordered
        hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
controls=['A1.0','A1.1','A2.0','A2.1']
candidate=[]
for name in ('B.0','B.1'):
    for case,row in enumerate(waves[name]):
        matches=[c for c in controls if waves[c][case]['tokens']==row['tokens']]
        prefix={}
        for c in controls:
            prefix[c]=next((i for i,(a,b) in enumerate(zip(row['tokens'],waves[c][case]['tokens'],strict=True)) if a!=b),128)
        candidate.append(dict(wave=name,case=case,matching_control_waves=matches,
                              common_prefix_tokens=prefix,text=row['text']))
result=dict(bounded_semantic_smoke_pass=args.confirm_bounded_review,
    scope='Manual coherence/topic/repetition smoke review of 32 candidate 128-token excerpts. Not validation of every asserted code bug, complete answers, model accuracy or global bitwise determinism.',
    input_echo_exact=sum(x['input_echo_exact'] for x in summary['quality'].values()),
    quality_counts=summary['quality'],source_sha256=hashes,
    candidate_novel_cases=[dict(wave=x['wave'],case=x['case']) for x in candidate if not x['matching_control_waves']],
    candidates=candidate)
(root/'quality-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
