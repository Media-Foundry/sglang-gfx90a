"""Recompute ABBA from client timestamps; report quality differences, not equivalence."""
import hashlib
import json
import math
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'summary.json';assert not target.exists()
results={};plans=[];answers={};manifests=[];shapes={}
for arm in ('A1','B','A2'):
    d=root/arm
    complete=json.loads((d/'complete.json').read_text())
    assert complete['status']=='complete' and complete['france_passed']
    assert not json.loads((d/f'P8-first-mhc-{arm}.stop.json').read_text())['remaining']
    plan=json.loads((d/'plan.json').read_text());plans.append(plan)
    assert plan['candidate']==(arm=='B') and plan['kv_tokens']==1048576 and plan['route_producer']
    assert not plan['diagnostic']
    assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest for name,digest in plan['sources'].items())
    manifest=json.loads((d/'inputs.json').read_text());manifests.append(manifest)
    for leg in complete['progress']:
        if leg['leg']=='warmup':continue
        data=json.loads((d/(leg['leg']+'.json')).read_text())
        assert len(data['rounds'])==3
        rates=[]
        for wave in data['rounds']:
            assert wave['request_count']==16 and wave['total_prompt_tokens']==131069
            assert wave['input_echo_exact'] and wave['cached_tokens']==[0]*16
            assert wave['completion_lengths']==[1]*16
            assert len(wave['raw_times'])==16
            seconds=max(t['first'] for t in wave['raw_times'])-min(t['begin'] for t in wave['raw_times'])
            assert seconds>0 and math.isclose(seconds,wave['prefill_wall_s'],abs_tol=1e-9)
            rate=131069/seconds
            assert math.isclose(rate,wave['aggregate_input_tok_s'],rel_tol=1e-12)
            rates.append(rate)
        assert rates==leg['rates'] and math.isclose(statistics.median(rates),leg['median'])
        results[leg['leg']]=dict(median=statistics.median(rates),rates=rates,
            range_pct=100*(max(rates)/min(rates)-1))
    quality=json.loads((d/'quality.json').read_text())
    byid={r['meta_info']['id']:r for r in quality}
    assert set(byid)=={f'quality-{arm}-{i}' for i in range(16)}
    ordered=[byid[f'quality-{arm}-{i}'] for i in range(16)]
    for i,response in enumerate(ordered):
        assert response['prompt_token_ids']==manifest['requests'][i]['input_ids']
        assert response['meta_info']['cached_tokens']==0
        assert isinstance(response.get('output_ids'),list) and len(response['output_ids'])==128
        assert isinstance(response.get('text'),str) and response['text'].strip()
    answers[arm]=ordered
    shapes[arm]=json.loads((d/'timing-shapes.json').read_text())
assert all(m==manifests[0] for m in manifests)
assert all(p['sources']==plans[0]['sources'] for p in plans)
assert set(results)=={'A1','B1','B2','A2'}
comparisons=[]
for left,right in (('A1','A2'),('A1','B')):
    rows=[]
    for i,(a,b) in enumerate(zip(answers[left],answers[right],strict=True)):
        x,y=a['output_ids'],b['output_ids']
        prefix=next((j for j in range(128) if x[j]!=y[j]),128)
        rows.append(dict(request=i,exact=x==y,common_prefix=prefix))
    comparisons.append(dict(lhs=left,rhs=right,exact_out_of16=sum(r['exact'] for r in rows),requests=rows))
control=statistics.mean(results[leg]['median'] for leg in ('A1','A2'))
candidate=statistics.mean(results[leg]['median'] for leg in ('B1','B2'))
summary=dict(status='complete',scope='Original V4 TP8 C16x8K,1M pool,32K budget,route producer ON',
    policy='A:accepted mixed MHC; B:small-tail plus unfused first-boundary FP32 fixes',
    control_input_tok_s=control,candidate_input_tok_s=candidate,gain_pct=100*(candidate/control-1),
    control_drift_pct=100*(results['A2']['median']/results['A1']['median']-1),legs=results,
    input_tokens=131069,shapes=shapes,quality_comparisons=comparisons,
    numerical_equivalence_claimed=False,default_changed=False,quality_manual_review_pending=True)
target.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('shapes','quality_comparisons')},indent=2))
