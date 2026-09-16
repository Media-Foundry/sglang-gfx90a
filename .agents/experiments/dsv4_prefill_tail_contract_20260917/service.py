"""Fresh32K teacher intervention: legacy tails versus common FP32/20 tails."""
import hashlib
import argparse
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
reg=root.parent/'dsv4_ck_route_producer_regression_20260917'
prior=reg/'32k-B'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('baseline','mixed','serial'),required=True)
args=parser.parse_args()
small=args.arm!='baseline'
max_requests=1 if args.arm=='serial' else 16
oracle=json.loads((root/'integrated-v1.json').read_text())
assert oracle['status']=='complete' and len(oracle['cases'])==21
assert all(c['selector_exact'] for c in oracle['cases'])
for arm in ('A1','B','A2'):
    d=reg/('32k-'+arm)
    assert (d/'complete.json').is_file()
    assert not json.loads((d/f'P32k-route-producer-regression-{arm}.stop.json').read_text())['remaining']
out=root/args.arm;out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('prior_teacher_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life);life.ROOT=life.OLD=out
manifest=json.loads((prior/'inputs.json').read_text())
reference_path=root.parent/'dsv4_owner_k32_20260916/B/teacher-forced.json'
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
launcher=launcher.replace(needle,'export SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER=0')
entry='exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(entry)==1
launcher=launcher.replace(entry,f'export SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL={int(small)}\nexport PREFILL_MAX_REQUESTS={max_requests}\n'+entry)
(out/'start-ar-matrix.sh').write_text(launcher)
paths=set(json.loads((prior/'plan.json').read_text())['sources'])|{str(Path(__file__).relative_to(repo)),
    'python/sglang/srt/layers/dsv4_prefill_tail_policy.py',
    'python/sglang/srt/model_executor/runner/eager_runner.py'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
changed={p for p,h in json.loads((prior/'plan.json').read_text())['sources'].items() if sources[p]!=h}
assert changed <= {'python/sglang/kernels/ops/layernorm/gfx90a_mhc_prefill_policy.py',
                   'python/sglang/srt/model_executor/runner/eager_runner.py'},changed
life.save('plan.json',dict(sources=sources,candidate=False,scored_performance=False,
    kv_tokens=1048576,small_policy=small,max_prefill_requests=max_requests,changed_since_prior=sorted(changed),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()))
label='P32k-tail-policy-'+args.arm;state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==32768
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER']=='0'
    assert env['SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL']==str(int(small))
    assert env['PREFILL_MAX_REQUESTS']==str(max_requests)
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
    tail_hits=sorted(set(re.findall(r' TP(\d+)\] DSV4 common FP32/20 prefill MHC selected:.*small_tail=True',logs)))
    assert tail_hits==(list(map(str,range(8))) if small else []),tail_hits
    if max_requests==1:
        batches=[int(x) for x in re.findall(r'Prefill batch, #new-seq: (\d+)',logs)]
        assert batches and max(batches)==1,batches
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    result=dict(status='diagnostic_complete',positions=positions,
        comparison='current arm versus historical K32 teacher on identical prompts',
        arm=args.arm,small_policy=small,max_prefill_requests=max_requests,tail_ranks=tail_hits,
        teacher_exact=max(differences)==0 and top5_exact==1008,
        max_abs_logprob=max(differences),mean_abs_logprob=statistics.mean(differences),
        top1_same=top1_same,top5_exact=top5_exact,requests=requests,
        route_ranks=ranks,france_passed=True,scored_performance=False)
    life.save('complete.json',result)
    print('FRESH CONTROL COMPARISON',json.dumps(result),flush=True)
finally:
    life.stop(state)
