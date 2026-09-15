"""Compare new long-input replies with previously reviewed same-input wide runs."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent;out=root/'service'
results=[]
for length,prior_name in ((16,'dsv4_c16_query_wide_service_20260915'),(32,'dsv4_c16_query_wide32k_service_20260915')):
    manifest=json.loads((out/f'inputs{length}k.json').read_text())
    old=[]
    for arm in ('A1','B','A2'):
        prior=root.parent/prior_name/arm
        assert json.loads((prior/'inputs.json').read_text())==manifest
        for rep in range(2):
            rows=json.loads((prior/f'quality-{rep}.json').read_text())
            old.append(sorted(rows,key=lambda r:int(r['meta_info']['id'].rsplit('-',1)[-1])))
    waves=[]
    for rep in range(2):
        rows=json.loads((out/f'{length}k-quality-{rep}.json').read_text())
        byid={r['meta_info']['id']:r for r in rows}
        assert len(rows)==16 and set(byid)=={f'prewarm-{length}-{rep}-{i}' for i in range(16)}
        ordered=[byid[f'prewarm-{length}-{rep}-{i}'] for i in range(16)];waves.append(ordered)
        for i,(request,row) in enumerate(zip(manifest['requests'],ordered,strict=True)):
            assert row['prompt_token_ids']==request['input_ids'] and row['meta_info']['cached_tokens']==0
            assert len(row['output_ids'])==row['meta_info']['completion_tokens']==128
            matches=[j for j,previous in enumerate(old) if previous[i]['output_ids']==row['output_ids']]
            result=dict(length=length,wave=rep,case=i,prior_matches=matches,text=row['text'])
            results.append(result)
            if not matches:print('NEW TEXT',length,rep,i,row['text'])
    print('REPEAT',length,sum(a['output_ids']==b['output_ids'] for a,b in zip(*waves)))
target=root/'quality-review.json';assert not target.exists()
target.write_text(json.dumps(dict(results=results,scope=__doc__,exact_echoes=64),indent=2)+'\n')
print('Matching prior',sum(bool(r['prior_matches']) for r in results),'of',len(results))
