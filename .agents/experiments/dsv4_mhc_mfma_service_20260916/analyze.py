"""Audit service ABBA, repeated continuations and fixed-prefix logprobs separately."""
import collections
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent;target=root/'summary.json';assert not target.exists()
spec=importlib.util.spec_from_file_location('life_analysis',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
check=json.loads((root/'check/complete.json').read_text())
assert check['live_mix_passed'] and check['live_comparisons']==2720
log=(root/'check/P16-mix-mfma-check.service.log').read_text()
entries=re.findall(r'TP(\d+)\] DSV4 MFMA live mix: rows=(\d+) max_abs=([^ ]+) rel_l2=([^\n]+)',log)
assert len(entries)==2720
assert collections.Counter(int(e[0]) for e in entries)=={i:340 for i in range(8)}
assert all(8192<=int(e[1])<=65536 and math.isfinite(float(e[2])) and
           math.isfinite(float(e[3])) and float(e[3])<=1.e-5 for e in entries)
results={};plans=[];answers={};teachers={};quality={}
for arm in ('A1','B','A2'):
    directory=root/arm;done=json.loads((directory/'complete.json').read_text())
    service_log=(directory/f'P16-mix-mfma-{arm}.service.log').read_text()
    assert 'max_total_num_tokens=1048576' in service_log
    selected='DSV4 cooperative FP32 pre-mix selected:'
    if arm=='B':
        for rank in range(8):assert f' TP{rank}] '+selected in service_log
    else:assert selected not in service_log
    assert done['france_passed'] and done['unique_ck'] and not done['diagnostic']
    assert not json.loads((directory/f'P16-mix-mfma-{arm}.stop.json').read_text())['remaining']
    plans.append(json.loads((directory/'plan.json').read_text()))
    waves=[]
    for rep in range(4):
        data=json.loads((directory/f'quality-{rep}.json').read_text())
        byid={r['meta_info']['id']:r for r in data}
        waves.append([life.completion_ids(byid[f'corrected-{arm}-{rep}-{i}']) for i in range(16)])
        assert all(len(v)==128 for v in waves[-1])
    answers[arm]=waves
    repeats=[sum(a==b for a,b in zip(waves[0],w,strict=True)) for w in waves[1:]]
    assert repeats==done['quality_repeat_exact']
    quality[arm]=dict(repeat_exact_out_of16=repeats)
    for leg in done['progress']:
        if leg['leg']!='warmup':
            assert len(leg['rates'])==3
            results[leg['leg']]=leg
    ts=json.loads((directory/'teacher-forced.json').read_text())
    byid={r['meta_info']['id']:r for r in ts}
    teachers[arm]=[byid[f'teacher-{arm}-{i}'] for i in range(16)]
assert all(p['sources']==plans[0]['sources'] and p['input_sha256']==plans[0]['input_sha256'] for p in plans)
cross=[]
for i in range(16):
    tokens=[answers[a][r][i] for a in ('A1','B','A2') for r in range(4)]
    common=next((j for j in range(128) if len({v[j] for v in tokens})>1),128)
    cross.append(dict(request=i,common_prefix=common,distinct_outputs=len({tuple(t) for t in tokens})))
teacher_checks=[]
for lhs,rhs in [('A1','A2'),('A1','B')]:
    differences=[];top1_same=top1_count=0;records=[]
    for i,(a,b) in enumerate(zip(teachers[lhs],teachers[rhs],strict=True)):
        assert a['prompt_token_ids']==b['prompt_token_ids']
        x=a['meta_info']['input_token_logprobs'];y=b['meta_info']['input_token_logprobs']
        assert len(x)==len(y)==64,(i,len(x),len(y))
        assert [t[1] for t in x]==[t[1] for t in y]==a['prompt_token_ids'][-64:]
        ds=[]
        for j,(v,w) in enumerate(zip(x,y,strict=True)):
            if v[0] is None or w[0] is None:
                assert j==0 and v[0] is None and w[0] is None
                continue
            assert math.isfinite(v[0]) and math.isfinite(w[0])
            ds.append(abs(v[0]-w[0]))
        differences.extend(ds)
        tx=a['meta_info']['input_top_logprobs'];ty=b['meta_info']['input_top_logprobs']
        assert len(tx)==len(ty)==64
        assert tx[0] is None and ty[0] is None and len(ds)==63
        assert all(v and w for v,w in zip(tx[1:],ty[1:],strict=True))
        same=sum(v[0][1]==w[0][1] for v,w in zip(tx[1:],ty[1:],strict=True))
        top1_same+=same;top1_count+=63
        records.append(dict(request=i,max_abs_logprob=max(ds),mean_abs_logprob=statistics.mean(ds),top1_same=same,positions=63,excluded_leading_null=1))
    teacher_checks.append(dict(lhs=lhs,rhs=rhs,max_abs_logprob=max(differences),mean_abs_logprob=statistics.mean(differences),
        top1_same=top1_same,positions=top1_count,records=records))
control=statistics.mean(results[n]['median'] for n in ['A1','A2'])
candidate=statistics.mean(results[n]['median'] for n in ['B1','B2'])
summary=dict(status='measured-pending-quality-review',scope='Original V4 TP8 C16x8K, FP32 MFMA split16 opt-in vs accepted exact paired premix',
    control_input_tok_s=control,candidate_input_tok_s=candidate,gain_pct=100*(candidate/control-1),
    control_drift_pct=100*(results['A2']['median']/results['A1']['median']-1),legs=results,
    kv_tokens=1048576,original_weights=True,live_mix_check=check,quality=quality,cross_arm_continuations=cross,
    teacher_forced=teacher_checks,production_bit_exact=False,universal_batch_invariance_claimed=False)
target.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
