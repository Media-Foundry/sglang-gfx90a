"""Repair32K historical teacher lineage; no scored performance waves.

Use the exact archived teacher prompt IDs,not a different free-running answer.
Keep measured32K-B launch/runtime untouched and preserve the original failed
analysis. New process independently compares1008 positions to historical B.
"""
import hashlib
import importlib.util
import json
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
out=root/'32k-prior-bridge';out.mkdir(exist_ok=False)
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
(out/'start-ar-matrix.sh').write_text((prior/'start-ar-matrix.sh').read_text())
paths=set(json.loads((prior/'plan.json').read_text())['sources'])|{str(Path(__file__).relative_to(repo))}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
assert all(sources[p]==h for p,h in json.loads((prior/'plan.json').read_text())['sources'].items())
life.save('plan.json',dict(sources=sources,candidate=True,scored_performance=False,
    kv_tokens=1048576,reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()))
label='P32k-route-prior-bridge';state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==32768
    env=life.owned(state).environ()
    for key in ('SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER','SGLANG_DSV4_C4_PREFILL_OWNER_K32',
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
    positions=0
    for i,rid in enumerate(rids):
        a,b=reference[i],got[rid]
        assert b['prompt_token_ids']==prompts[i] and b['meta_info']['cached_tokens']==0
        x,y=a['meta_info']['input_token_logprobs'],b['meta_info']['input_token_logprobs']
        assert len(x)==len(y)==64 and x[0][0] is None and y[0][0] is None
        assert [v[1] for v in x]==[v[1] for v in y]==prompts[i][-64:]
        assert x==y,('historical teacher logprob mismatch',i)
        tx,ty=a['meta_info']['input_top_logprobs'],b['meta_info']['input_top_logprobs']
        assert len(tx)==len(ty)==64 and tx[0] is None and ty[0] is None
        assert all(v and w for v,w in zip(tx[1:],ty[1:],strict=True))
        assert tx==ty,('historical teacher Top5 mismatch',i)
        positions+=63
    logs=Path(state['log']).read_text()
    ranks=sorted(set(re.findall(r'route-producer CK selected: rank=(\d+)',logs)))
    assert ranks==list(map(str,range(8)))
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(status='complete',positions=positions,
        teacher_exact=True,route_ranks=ranks,france_passed=True,scored_performance=False))
    print('PRIOR TEACHER BRIDGE EXACT',positions,flush=True)
finally:
    life.stop(state)
