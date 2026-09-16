"""Exact-reference versus row-owned pre-mix: live check, ABBA, fixed continuation."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('check','A1','B','A2'),required=True)
args=parser.parse_args()
checking=args.arm=='check'
candidate=args.arm in ('check','B')
oracle=json.loads((root/'oracle.json').read_text())
assert oracle['status']=='complete'
assert all(c['timing']['eager']['saved_ms']>2 for c in oracle['cases'] if c['rows'] in (32767,32768))
if not checking:
    assert json.loads((root/'check/complete.json').read_text())['live_comparisons']==2720
out=root/args.arm
out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('owner_perf_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life)
life.ROOT=life.OLD=out
prior=root.parent/'dsv4_mhc_mfma_service_20260916/A2'
manifest=json.loads((prior/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
life.save('inputs.json',manifest)
launcher=(root.parent/'dsv4_mhc_post_tiles_20260916/validated-launcher.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle)==1
flags=('export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA=0\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA_CHECK=0\n'
       'unset SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_DIR\n'
       f'export SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER={int(candidate)}\n'
       f'export SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER_CHECK={int(checking)}\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
paths=set(json.loads((prior/'plan.json').read_text())['sources']) | {
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_owner.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_pair.py',
    'python/sglang/kernels/ops/debug/dsv4_premix_owner_audit.py'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(sources=sources,candidate=candidate,diagnostic=checking,
    kv_tokens=1048576,prefill_budget=32768,original_weights=True))
label='P16-mix-owner-'+args.arm
state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['max_total_tokens']==1048576 and info['ep_size']==1
    env=life.owned(state).environ()
    for key,value in [('SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER','0'),
                      ('SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT','1'),
                      ('SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA','0'),
                      ('SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE','1'),
                      ('SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER',str(int(candidate)))]:
        assert env[key]==value
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'owner-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    assert 'pre-mix owner selected:' not in Path(state['log']).read_text()
    client=root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
    legs=[('check',1)] if checking else [('warmup',1),*[(n,3) for n in (('B1','B2') if candidate else (args.arm,))]]
    progress=[]
    for name,rounds in legs:
        life.resources(name+'-before',life.owned(state))
        command=[sys.executable,str(client),'--base-url',life.URL,'--inputs',str(out/'inputs.json'),
            '--request-count','16','--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        data=json.loads((out/(name+'.json')).read_text())
        assert all(r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16 for r in data['rounds'])
        progress.append(dict(leg=name,median=data['median_input_tok_s'],rates=[r['aggregate_input_tok_s'] for r in data['rounds']]))
        life.save('progress.json',progress);print('RESULT',args.arm,progress[-1],flush=True)
    answers=[]
    if not checking:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
        for rep in range(4):
            rids=[f'corrected-{args.arm}-{rep}-{i}' for i in range(16)]
            payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
                cache_salt=[f'owner-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
                sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),return_prompt_token_ids=True)
            responses=life.post(life.URL+'/generate',payload,1800)
            life.save(f'quality-{rep}.json',responses)
            byid={r['meta_info']['id']:r for r in responses};assert set(byid)==set(rids)
            ordered=[byid[rid] for rid in rids]
            for request,response in zip(manifest['requests'],ordered,strict=True):
                ids=life.completion_ids(response)
                assert response['prompt_token_ids']==request['input_ids']
                assert response['meta_info']['cached_tokens']==0 and len(ids)==128
                assert tokenizer.decode(ids,skip_special_tokens=False)==response['text']
            answers.append(ordered);print('QUALITY COMPLETE',args.arm,rep,flush=True)
        reference=json.loads((root.parent/'dsv4_mhc_post_tiles_20260916/A2/quality-0.json').read_text())
        refbyid={r['meta_info']['id']:r for r in reference}
        prompts=[r['input_ids']+life.completion_ids(refbyid[f'corrected-A2-0-{i}'])[:64] for i,r in enumerate(manifest['requests'])]
        rids=[f'teacher-{args.arm}-{i}' for i in range(16)]
        responses=life.post(life.URL+'/generate',dict(input_ids=prompts,rid=rids,
            cache_salt=[f'owner-teacher-{args.arm}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True,return_logprob=True,
            logprob_start_len=[len(r['input_ids']) for r in manifest['requests']],top_logprobs_num=5),1800)
        life.save('teacher-forced.json',responses)
        byid={r['meta_info']['id']:r for r in responses};assert set(byid)==set(rids)
        for i,rid in enumerate(rids):
            response=byid[rid]
            assert response['prompt_token_ids']==prompts[i] and response['meta_info']['cached_tokens']==0
            assert response['meta_info']['input_token_logprobs']
        print('TEACHER FORCED COMPLETE',args.arm,flush=True)
    logs=Path(state['log']).read_text()
    exact=re.findall(r'\[TP(\d+)\] pre-mix owner full-reference exact: rows=(\d+)',logs)
    if checking:
        assert Counter(rank for rank,rows in exact)==Counter({str(r):340 for r in range(8)})
    else:
        assert not exact
    for rank in range(8):
        assert f'unique-slot CK selected: rank={rank} ' in logs
        assert f'[TP{rank}] H16 peer selected:' in logs
        assert (f'[TP{rank}] pre-mix owner selected:' in logs)==candidate
    assert 'DSV4 cooperative FP32 pre-mix selected:' not in logs
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(progress=progress,live_comparisons=len(exact),diagnostic=checking,
        france_passed=True,quality_repeat_exact=[sum(life.completion_ids(a)==life.completion_ids(b)
            for a,b in zip(answers[0],wave,strict=True)) for wave in answers[1:]]))
    print('ARM COMPLETE',args.arm,flush=True)
finally:
    life.stop(state)
