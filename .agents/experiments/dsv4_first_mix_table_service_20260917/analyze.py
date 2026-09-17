"""Validate table ABBA against corrected FP32 control; reject empty quality data."""
import hashlib
import json
import math
from pathlib import Path
import statistics
root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'summary.json';assert not target.exists()
check=json.loads((root/'check/complete.json').read_text())
assert check['diagnostic'] and check['live_exact_checks']>=40
plans=[];manifests=[];results={};answers={};teachers={}
for arm in ('A1','B','A2'):
    d=root/arm;done=json.loads((d/'complete.json').read_text())
    assert done['status']=='complete' and done['france_passed'] and not done['diagnostic']
    assert not json.loads((d/f'P8-first-table-{arm}.stop.json').read_text())['remaining']
    plan=json.loads((d/'plan.json').read_text());plans.append(plan)
    assert plan['kv_tokens']==1048576 and plan['candidate']==(arm=='B')
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['sources'].items())
    manifests.append(json.loads((d/'inputs.json').read_text()))
    for leg in done['progress']:
        if leg['leg']=='warmup':continue
        measured=json.loads((d/(leg['leg']+'.json')).read_text());assert len(measured['rounds'])==3
        rates=[]
        for wave in measured['rounds']:
            assert wave['input_echo_exact'] and wave['total_prompt_tokens']==131069
            assert wave['cached_tokens']==[0]*16 and wave['completion_lengths']==[1]*16
            assert len(wave['raw_times'])==16
            seconds=max(t['first'] for t in wave['raw_times'])-min(t['begin'] for t in wave['raw_times'])
            assert math.isclose(seconds,wave['prefill_wall_s'],abs_tol=1e-9)
            rate=131069/seconds;assert math.isclose(rate,wave['aggregate_input_tok_s'],rel_tol=1e-12)
            rates.append(rate)
        assert rates==leg['rates'] and statistics.median(rates)==leg['median']
        results[leg['leg']]=dict(median=leg['median'],rates=rates,range_pct=100*(max(rates)/min(rates)-1))
    waves=[]
    for rep in range(2):
        values=json.loads((d/f'quality-{rep}.json').read_text())
        byid={r['meta_info']['id']:r for r in values};ids=[f'quality-{arm}-{rep}-{i}' for i in range(16)]
        assert set(byid)==set(ids);ordered=[byid[rid] for rid in ids]
        for i,row in enumerate(ordered):
            assert row['prompt_token_ids']==manifests[-1]['requests'][i]['input_ids']
            assert row['meta_info']['cached_tokens']==0 and isinstance(row.get('output_ids'),list) and len(row['output_ids'])==128
        waves.append(ordered)
    answers[arm]=waves
    values=json.loads((d/'teacher-forced.json').read_text());byid={r['meta_info']['id']:r for r in values}
    assert set(byid)=={f'teacher-{arm}-{i}' for i in range(16)}
    teachers[arm]=[byid[f'teacher-{arm}-{i}'] for i in range(16)]
assert all(p['sources']==plans[0]['sources'] for p in plans)
assert all(m==manifests[0] for m in manifests)
assert set(results)=={'A1','B1','B2','A2'}
teacher_checks=[]
for left,right in (('A1','A2'),('A1','B')):
    delta=[];top1=0;top5=0
    for a,b in zip(teachers[left],teachers[right],strict=True):
        assert a['prompt_token_ids']==b['prompt_token_ids']
        assert a['meta_info']['cached_tokens']==b['meta_info']['cached_tokens']==0
        x,y=[r['meta_info']['input_token_logprobs'] for r in (a,b)]
        assert len(x)==len(y)==64 and x[0][0] is y[0][0] is None
        assert [v[1] for v in x]==[v[1] for v in y]==a['prompt_token_ids'][-64:]
        diffs=[abs(v[0]-w[0]) for v,w in zip(x[1:],y[1:],strict=True)]
        assert all(math.isfinite(v) for v in diffs);delta.extend(diffs)
        tx,ty=[r['meta_info']['input_top_logprobs'] for r in (a,b)]
        assert len(tx)==len(ty)==64 and tx[0] is ty[0] is None
        assert all(len(v)==5 for v in [*tx[1:],*ty[1:]])
        top1+=sum(v[0][1]==w[0][1] for v,w in zip(tx[1:],ty[1:],strict=True))
        top5+=sum(v==w for v,w in zip(tx[1:],ty[1:],strict=True))
    assert len(delta)==1008
    teacher_checks.append(dict(lhs=left,rhs=right,positions=1008,max_abs=max(delta),mean_abs=statistics.mean(delta),top1_same=top1,top5_exact=top5))
control=statistics.mean(results[k]['median'] for k in ('A1','A2'))
candidate=statistics.mean(results[k]['median'] for k in ('B1','B2'))
cross=[]
for i in range(16):
    sequences=[answers[a][rep][i]['output_ids'] for a in ('A1','B','A2') for rep in range(2)]
    cross.append(dict(request=i,distinct_outputs=len({tuple(v) for v in sequences})))
summary=dict(status='complete',scope='Original V4 TP8 C16x8K FP32-first MHC reference vs exact table,route ON,1M KV',
    control_input_tok_s=control,candidate_input_tok_s=candidate,gain_pct=100*(candidate/control-1),
    control_drift_pct=100*(results['A2']['median']/results['A1']['median']-1),legs=results,
    live_exact_checks=check['live_exact_checks'],table_builds=json.loads((root/'B/complete.json').read_text())['built'],
    teacher_checks=teacher_checks,quality_cross=cross,universal_batch_invariance_claimed=False,default_changed=False)
target.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
