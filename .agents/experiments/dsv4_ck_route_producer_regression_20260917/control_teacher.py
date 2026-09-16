"""Fresh control teacher replay to isolate route-producer versus admission-history drift."""
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path
import re
import sys
import time

root=Path(__file__).resolve().parent
repo=root.parents[2]
prior=root/'32k-B'
for arm in ('A1','B','A2'):
    d=root/('32k-'+arm)
    assert (d/'complete.json').is_file()
    assert not json.loads((d/f'P32k-route-producer-regression-{arm}.stop.json').read_text())['remaining']
out=root/'32k-prior-control';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('prior_teacher_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life);life.ROOT=life.OLD=out
manifest=json.loads((prior/'inputs.json').read_text())
reference_path=root/'32k-prior-bridge/teacher-forced.json'
reference=json.loads(reference_path.read_text());byid={r['meta_info']['id']:r for r in reference}
assert set(byid)=={f'teacher-B-{i}' for i in range(16)}
reference=[byid[f'teacher-B-{i}'] for i in range(16)]
prompts=[r['prompt_token_ids'] for r in reference]
for request,prompt in zip(manifest['requests'],prompts,strict=True):
    assert prompt[:-64]==request['input_ids'] and len(prompt)==len(request['input_ids'])+64
life.save('inputs.json',manifest)
life.save('teacher-prompts.json',dict(input_ids=prompts,
    reference_path=str(reference_path),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()))
launcher=(prior/'start-ar-matrix.sh').read_text()
needle='export SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER=1'
assert launcher.count(needle)==1
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,'export SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER=0'))
paths=set(json.loads((prior/'plan.json').read_text())['sources'])|{str(Path(__file__).relative_to(repo))}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
assert all(sources[p]==h for p,h in json.loads((prior/'plan.json').read_text())['sources'].items())
life.save('plan.json',dict(sources=sources,candidate=False,scored_performance=False,
    kv_tokens=1048576,reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()))
label='P32k-route-prior-control';state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==32768
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER']=='0'
    for key in ('SGLANG_DSV4_C4_PREFILL_OWNER_K32',
                'SGLANG_DSV4_PREFILL_MHC_COMMON_FP32'):
        assert env[key]=='1'
    assert env['SGLANG_DSV4_DEBUG_CK_ROUTE_CHECK']=='0'
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'prior-bridge-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    rids=[f'teacher-B-{i}' for i in range(16)]
    response=life.post(life.URL+'/generate',dict(input_ids=prompts,rid=rids,
        cache_salt=[f'prior-bridge-{i}-{time.time_ns()}' for i in range(16)],
        sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True,
        return_logprob=True,logprob_start_len=[len(r['input_ids']) for r in manifest['requests']],
        top_logprobs_num=5),1800)
    life.save('teacher-forced.json',response)
    got={r['meta_info']['id']:r for r in response};assert set(got)==set(rids)
    positions=0;differences=[];top1_same=0;top5_exact=0;requests=[]
    for i,rid in enumerate(rids):
        a,b=reference[i],got[rid]
        assert b['prompt_token_ids']==prompts[i] and b['meta_info']['cached_tokens']==0
        x,y=a['meta_info']['input_token_logprobs'],b['meta_info']['input_token_logprobs']
        assert len(x)==len(y)==64 and x[0][0] is None and y[0][0] is None
        assert [v[1] for v in x]==[v[1] for v in y]==prompts[i][-64:]
        delta=[abs(v[0]-w[0]) for v,w in zip(x[1:],y[1:],strict=True)]
        assert all(math.isfinite(d) for d in delta)
        differences.extend(delta)
        tx,ty=a['meta_info']['input_top_logprobs'],b['meta_info']['input_top_logprobs']
        assert len(tx)==len(ty)==64 and tx[0] is None and ty[0] is None
        assert all(v and w for v,w in zip(tx[1:],ty[1:],strict=True))
        top1_same+=sum(v[0][1]==w[0][1] for v,w in zip(tx[1:],ty[1:],strict=True))
        top5_exact+=sum(v==w for v,w in zip(tx[1:],ty[1:],strict=True))
        requests.append(dict(request=i,max_abs_logprob=max(delta),exact_positions=sum(d==0 for d in delta)))
        positions+=63
    logs=Path(state['log']).read_text()
    ranks=sorted(set(re.findall(r'route-producer CK selected: rank=(\d+)',logs)))
    assert ranks==[]
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    result=dict(status='diagnostic_complete',positions=positions,
        comparison='fresh control versus fresh route candidate on identical historical teacher prompts',
        teacher_exact=max(differences)==0 and top5_exact==1008,
        max_abs_logprob=max(differences),mean_abs_logprob=statistics.mean(differences),
        top1_same=top1_same,top5_exact=top5_exact,requests=requests,
        route_ranks=ranks,france_passed=True,scored_performance=False)
    life.save('complete.json',result)
    print('FRESH CONTROL COMPARISON',json.dumps(result),flush=True)
finally:
    life.stop(state)
