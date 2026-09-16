"""Audit the failed historical comparison from raw records without GPU work."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'32k-B/inputs.json').read_text())
def load(path,arm):
    data={r['meta_info']['id']:r for r in json.loads(path.read_text())}
    assert set(data)=={f'teacher-{arm}-{i}' for i in range(16)}
    return [data[f'teacher-{arm}-{i}'] for i in range(16)]
current={arm:load(root/('32k-'+arm)/'teacher-forced.json',arm) for arm in ('A1','B','A2')}
old_path=root.parent/'dsv4_owner_k32_20260916/B/teacher-forced.json'
old=load(old_path,'B')
archive=json.loads((old_path.parent.parent/'archive-manifest.json').read_text())
entry=next(r for r in archive['files'] if r['path']=='B/teacher-forced.json')
assert hashlib.sha256(old_path.read_bytes()).hexdigest()==entry['sha256']
checks=[]
for i,request in enumerate(manifest['requests']):
    base=request['input_ids'];rows=[current[arm][i] for arm in ('A1','B','A2')]
    assert all(r['prompt_token_ids'][:-64]==base for r in [*rows,old[i]])
    assert all(r['prompt_token_ids']==rows[0]['prompt_token_ids'] for r in rows)
    for field in ('input_token_logprobs','input_top_logprobs'):
        assert len(rows[0]['meta_info'][field])==64
        assert all(r['meta_info'][field]==rows[0]['meta_info'][field] for r in rows)
    a,b=old[i]['prompt_token_ids'],rows[0]['prompt_token_ids']
    first=next((j for j,(x,y) in enumerate(zip(a,b,strict=True)) if x!=y),None)
    assert first is not None and first>=len(base)
    checks.append(dict(request=i,base_tokens=len(base),original_prompt_equal=True,
        current_arms_prompt_and_logprobs_exact=True,historical_continuation_equal=False,
        first_continuation_difference=first-len(base)))
print(json.dumps(dict(status='teacher_lineage_mismatch_confirmed',checks=checks,
    historical_teacher_sha256=entry['sha256'],
    correction='Replay candidate on historical teacher prompt IDs; do not compare differing continuations or alter scored waves'),indent=2))
