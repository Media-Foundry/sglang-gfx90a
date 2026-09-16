"""Compare identical teacher prompts across tail-policy/admission interventions.

Report residual differences without declaring universal batch invariance or
silently treating the historical mixed-precision checkpoint as exact reference.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'service-analysis.json';assert not target.exists()
arms={};plans={};layouts={};completions={}
for arm in ('baseline','mixed','serial'):
    d=root/arm
    done=json.loads((d/'complete.json').read_text())
    assert done['status']=='diagnostic_complete' and done['positions']==1008
    assert done['france_passed'] and not done['scored_performance'] and done['route_ranks']==[]
    assert done['tail_ranks']==([] if arm=='baseline' else list(map(str,range(8))))
    assert not json.loads((d/f'P32k-tail-policy-{arm}.stop.json').read_text())['remaining']
    plans[arm]=json.loads((d/'plan.json').read_text())
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plans[arm]['sources'].items())
    data={r['meta_info']['id']:r for r in json.loads((d/'teacher-forced.json').read_text())}
    assert set(data)=={f'teacher-B-{i}' for i in range(16)}
    arms[arm]=[data[f'teacher-B-{i}'] for i in range(16)]
    log=(d/f'P32k-tail-policy-{arm}.service.log').read_text()
    layouts[arm]=[dict(requests=int(n),rows=int(m)) for n,m in
        re.findall(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)',log)]
    completions[arm]=done
assert plans['baseline']['sources']==plans['mixed']['sources']==plans['serial']['sources']
assert max(v['requests'] for v in layouts['serial'])==1
assert any(v['requests']>1 for v in layouts['mixed']),'No mixed-shape intervention actually occurred'

old=root.parent/'dsv4_ck_route_producer_regression_20260917/32k-prior-control/teacher-forced.json'
prior={r['meta_info']['id']:r for r in json.loads(old.read_text())}
arms['prior_fresh_control']=[prior[f'teacher-B-{i}'] for i in range(16)]
comparisons=[]
for lhs,rhs in [('prior_fresh_control','baseline'),('baseline','mixed'),('mixed','serial')]:
    differences=[];top1=0;top5=0;requests=[]
    for i,(a,b) in enumerate(zip(arms[lhs],arms[rhs],strict=True)):
        assert a['prompt_token_ids']==b['prompt_token_ids']
        assert a['meta_info']['cached_tokens']==b['meta_info']['cached_tokens']==0
        x,y=a['meta_info']['input_token_logprobs'],b['meta_info']['input_token_logprobs']
        assert len(x)==len(y)==64 and x[0][0] is None and y[0][0] is None
        assert [q[1] for q in x]==[q[1] for q in y]==a['prompt_token_ids'][-64:]
        delta=[abs(v[0]-w[0]) for v,w in zip(x[1:],y[1:],strict=True)]
        assert all(math.isfinite(v) for v in delta)
        differences.extend(delta)
        tx,ty=a['meta_info']['input_top_logprobs'],b['meta_info']['input_top_logprobs']
        assert len(tx)==len(ty)==64 and tx[0] is None and ty[0] is None
        assert all(v and w for v,w in zip(tx[1:],ty[1:],strict=True))
        same=sum(v[0][1]==w[0][1] for v,w in zip(tx[1:],ty[1:],strict=True))
        exact=sum(v==w for v,w in zip(tx[1:],ty[1:],strict=True))
        top1+=same;top5+=exact
        requests.append(dict(request=i,max_abs_logprob=max(delta),mean_abs_logprob=statistics.mean(delta),
            exact_logprobs=sum(v==0 for v in delta),top1_same=same,top5_exact=exact))
    assert len(differences)==1008
    comparisons.append(dict(lhs=lhs,rhs=rhs,positions=1008,
        max_abs_logprob=max(differences),mean_abs_logprob=statistics.mean(differences),
        exact_logprobs=sum(v==0 for v in differences),top1_same=top1,top5_exact=top5,requests=requests))
result=dict(status='diagnostic_complete',scored_performance=False,
    universal_batch_invariance_claimed=False,comparisons=comparisons,
    layouts=layouts,arm_historical_comparisons=completions,
    same_prompt_fixture_sha256=hashlib.sha256((root/'baseline/teacher-prompts.json').read_bytes()).hexdigest())
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
