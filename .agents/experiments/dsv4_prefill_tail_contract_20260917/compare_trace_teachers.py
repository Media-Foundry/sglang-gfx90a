"""Compare trace arms on identical full prompts; no performance conclusions."""
import json
import math
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent;target=root/'trace-teacher-comparison.json'
assert not target.exists()
results=[]
for version in ('v2','v3'):
    arms=[]
    for arm in ('mixed','serial'):
        d=root/f'trace-{version}-{arm}'
        assert json.loads((d/'complete.json').read_text())['status']=='diagnostic_complete'
        assert not json.loads((d/f'P32k-tail-policy-trace-{version}-{arm}.stop.json').read_text())['remaining']
        data={r['meta_info']['id']:r for r in json.loads((d/'teacher-forced.json').read_text())}
        assert set(data)=={f'teacher-B-{i}' for i in range(16)}
        arms.append(data)
    delta=[];top1=0;requests=[]
    for i in range(16):
        a,b=[data[f'teacher-B-{i}'] for data in arms]
        assert a['prompt_token_ids']==b['prompt_token_ids']
        assert a['meta_info']['cached_tokens']==b['meta_info']['cached_tokens']==0
        x,y=[r['meta_info']['input_token_logprobs'] for r in (a,b)]
        assert len(x)==len(y)==64 and x[0][0] is y[0][0] is None
        assert [q[1] for q in x]==[q[1] for q in y]==a['prompt_token_ids'][-64:]
        ds=[abs(u[0]-v[0]) for u,v in zip(x[1:],y[1:],strict=True)]
        assert all(math.isfinite(q) for q in ds)
        tx,ty=[r['meta_info']['input_top_logprobs'] for r in (a,b)]
        assert len(tx)==len(ty)==64 and tx[0] is ty[0] is None
        assert all(len(q)==5 for q in [*tx[1:],*ty[1:]])
        same=sum(u[0][1]==v[0][1] for u,v in zip(tx[1:],ty[1:],strict=True))
        top1+=same;delta.extend(ds)
        requests.append(dict(request=i,top1_same=same,max_abs=max(ds),exact=sum(v==0 for v in ds)))
    assert len(delta)==1008
    results.append(dict(version=version,positions=len(delta),top1_same=top1,
                        max_abs=max(delta),mean_abs=statistics.mean(delta),requests=requests))
target.write_text(json.dumps(dict(status='complete',results=results,performance_scored=False),indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='requests'} for r in results],indent=2))
