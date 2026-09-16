"""Validate C16 real 16K wide-owner ABBA and model-output evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
target=root/'summary.json'
assert not target.exists()
check=json.loads((root/'check/complete.json').read_text())
assert check['live_comparisons']>0 and check['diagnostic']
plans=[];results={};answers={};teachers={};quality={}

def tokens(response):
    ids=response.get('output_ids')
    assert isinstance(ids,list) and len(ids)==128, response.keys()
    return ids

for arm in ('A1','B','A2'):
    directory=root/arm
    done=json.loads((directory/'complete.json').read_text())
    assert done['france_passed'] and not done['diagnostic']
    assert not json.loads((directory/f'P16-wide-owner-{arm}.stop.json').read_text())['remaining']
    plans.append(json.loads((directory/'plan.json').read_text()))
    log=(directory/f'P16-wide-owner-{arm}.service.log').read_text()
    assert 'max_total_num_tokens=1048576' in log
    for rank in range(8):
        assert f'[TP{rank}] pre-mix owner selected:' in log
        assert f' TP{rank}] DSV4 direct-row CK dequant selected:' in log
    manifest=json.loads((directory/'inputs.json').read_text())
    waves=[]
    for rep in range(4):
        data=json.loads((directory/f'quality-{rep}.json').read_text())
        byid={r['meta_info']['id']:r for r in data}
        ordered=[byid[f'corrected-{arm}-{rep}-{i}'] for i in range(16)]
        assert all(r['prompt_token_ids']==q['input_ids'] and r['meta_info']['cached_tokens']==0
                   for q,r in zip(manifest['requests'],ordered,strict=True))
        waves.append([tokens(r) for r in ordered])
    answers[arm]=waves
    repeats=[sum(a==b for a,b in zip(waves[0],w,strict=True)) for w in waves[1:]]
    assert repeats==done['quality_repeat_exact']
    quality[arm]=dict(repeat_exact_out_of16=repeats)
    for leg in done['progress']:
        if leg['leg']!='warmup':
            measured=json.loads((directory/(leg['leg']+'.json')).read_text())
            assert len(measured['rounds'])==3
            for wave in measured['rounds']:
                assert wave['total_prompt_tokens']==262141
                assert wave['cached_tokens']==[0]*16 and wave['input_echo_exact']
                times=wave['raw_times']
                seconds=max(t['first'] for t in times)-min(t['begin'] for t in times)
                assert math.isclose(seconds,wave['prefill_wall_s'],abs_tol=1e-9)
                assert math.isclose(262141/seconds,wave['aggregate_input_tok_s'],rel_tol=1e-12)
            assert leg['rates']==[w['aggregate_input_tok_s'] for w in measured['rounds']]
            assert len(leg['rates'])==3
            assert math.isclose(leg['median'],statistics.median(leg['rates']))
            results[leg['leg']]=leg
    ts=json.loads((directory/'teacher-forced.json').read_text())
    byid={r['meta_info']['id']:r for r in ts}
    teachers[arm]=[byid[f'teacher-{arm}-{i}'] for i in range(16)]
assert all(p['sources']==plans[0]['sources'] for p in plans)
assert len({hashlib.sha256((root/a/'inputs.json').read_bytes()).hexdigest() for a in ('A1','B','A2')})==1
cross=[]
for i in range(16):
    sequences=[answers[a][r][i] for a in ('A1','B','A2') for r in range(4)]
    prefix=next((j for j in range(128) if len({v[j] for v in sequences})>1),128)
    cross.append(dict(request=i,common_prefix=prefix,distinct_outputs=len({tuple(t) for t in sequences})))
teacher_checks=[]
for lhs,rhs in [('A1','A2'),('A1','B')]:
    differences=[];same=0;top_records_exact=0
    for i,(a,b) in enumerate(zip(teachers[lhs],teachers[rhs],strict=True)):
        assert a['prompt_token_ids']==b['prompt_token_ids']
        x,y=a['meta_info']['input_token_logprobs'],b['meta_info']['input_token_logprobs']
        assert len(x)==len(y)==64
        assert [v[1] for v in x]==[v[1] for v in y]==a['prompt_token_ids'][-64:]
        assert x[0][0] is None and y[0][0] is None
        for v,w in zip(x[1:],y[1:],strict=True):
            assert math.isfinite(v[0]) and math.isfinite(w[0])
            differences.append(abs(v[0]-w[0]))
        tx,ty=a['meta_info']['input_top_logprobs'],b['meta_info']['input_top_logprobs']
        assert len(tx)==len(ty)==64 and tx[0] is None and ty[0] is None
        assert all(v and w for v,w in zip(tx[1:],ty[1:],strict=True))
        same+=sum(v[0][1]==w[0][1] for v,w in zip(tx[1:],ty[1:],strict=True))
        top_records_exact+=sum(v==w for v,w in zip(tx[1:],ty[1:],strict=True))
    assert len(differences)==1008
    teacher_checks.append(dict(lhs=lhs,rhs=rhs,max_abs_logprob=max(differences),
        mean_abs_logprob=statistics.mean(differences),top1_same=same,top5_records_exact=top_records_exact,
        positions=1008,excluded_leading_nulls=16))
control=statistics.mean(results[n]['median'] for n in ('A1','A2'))
candidate=statistics.mean(results[n]['median'] for n in ('B1','B2'))
summary=dict(status='complete',scope='Original V4 TP8 C16x16K, wide-owner versus full wide-query16; exact 10.206k 8K checkpoint common, native AR',
    control_input_tok_s=control,candidate_input_tok_s=candidate,gain_pct=100*(candidate/control-1),
    control_drift_pct=100*(results['A2']['median']/results['A1']['median']-1),legs=results,
    kv_tokens=1048576,original_weights=True,live_comparisons=check['live_comparisons'],quality=quality,
    cross_arm_continuations=cross,teacher_forced=teacher_checks,
    universal_batch_invariance_claimed=False)
target.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
