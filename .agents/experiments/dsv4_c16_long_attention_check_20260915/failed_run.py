"""Same-live-input long-prefill stages1/stages2 equality; no speed measurement."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent;repo=root.parents[2]
out=root/'service';out.mkdir(exist_ok=False)
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('wide_prewarm_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life);life.ROOT=life.OLD=out
prior=root.parent/'dsv4_c16_attention_stage1_service_20260915'
assert json.loads((prior/'acceptance.json').read_text())['status']=='accepted_scoped_profile'
fixtures=root.parent/'dsv4_c16_query_wide_20260915'
validation=json.loads((fixtures/'input-validation.json').read_text())
manifests={}
for length,total in ((16,262141),(32,524286)):
    path=fixtures/f'inputs{length}k/prefill.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==validation[f'inputs{length}k']['manifest_sha256']
    assert validation[f'inputs{length}k']['official_chat_encoding_exact']==16
    data=json.loads(path.read_text())
    assert len(data['requests'])==16 and sum(len(r['input_ids']) for r in data['requests'])==total
    manifests[length]=data;life.save(f'inputs{length}k.json',data)

# Explicit path shared by primer and all workers; do not clear existing cache.
cache=Path('/home/pc/.cache/sglang/triton')
life.resources('before-prime')
env=dict(os.environ,HIP_VISIBLE_DEVICES='4',TRITON_CACHE_DIR=str(cache))
command=[sys.executable,str(repo/'scripts/rocm/prewarm_dsv4_indexer.py'),
    '--signature','4096:64:64','--signature','8192:128:128','--output',str(out/'prewarm.json')]
with (out/'prewarm.log').open('x') as log:
    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
primed=json.loads((out/'prewarm.json').read_text())
assert primed['compile_only'] and primed['cache_dir_env']==str(cache)
artifacts={r['artifact'] for r in primed['records']};assert len(artifacts)==2
life.resources('after-prime')

launcher=(prior/'B/start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
flags=('export SGLANG_DSV4_C4_PREFILL_QUERY_WIDE=1\n'
       'export SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER=0\n'
       'export SGLANG_DSV4_PREFILL_ATTN_STAGE1=1\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_ATTN_STAGE1_CHECK=1\n'
       'export SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES=1\n'
       f'export TRITON_CACHE_DIR={shlex.quote(str(cache))}\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
paths=set(json.loads((prior/'B/plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_prewarm.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_query_reuse.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_runtime_m.py',
    'scripts/rocm/prewarm_dsv4_indexer.py'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
life.save('plan.json',dict(sources=sources,cache=str(cache),scope=__doc__,
    producer=False,stage1=True,wide=True,cold_cache_control=False,diagnostic=True))
state=life.start('P16-long-attn-check',0)
try:
    life.ready(state)
    info=json.loads((out/'P16-long-attn-check.server-info.json').read_text())
    assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768 and info['ep_size']==1
    env=life.owned(state).environ()
    expected={'SGLANG_DSV4_C4_PREFILL_QUERY_WIDE':'1','SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER':'0',
        'SGLANG_DSV4_PREFILL_ATTN_STAGE1':'1','SGLANG_DSV4_DEBUG_PREFILL_ATTN_STAGE1_CHECK':'1',
        'SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE':'16','SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M':'1',
        'SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES':'1','TRITON_CACHE_DIR':str(cache)}
    assert all(env[k]==v for k,v in expected.items())
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'prewarm-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    progress=[]
    for length,width in ((16,4096),(32,8192)):
        for rep in range(2):
            rids=[f'prewarm-{length}-{rep}-{i}' for i in range(16)]
            payload=dict(input_ids=[r['input_ids'] for r in manifests[length]['requests']],rid=rids,
                cache_salt=[f'prewarm-quality-{length}-{rep}-{i}-{time.time_ns()}' for i in range(16)],
                sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),return_prompt_token_ids=True)
            before=Path(state['log']).stat().st_size
            answers=life.post(life.URL+'/generate',payload,1800);life.save(f'{length}k-quality-{rep}.json',answers)
            byid={r['meta_info']['id']:r for r in answers};assert len(answers)==16 and set(byid)==set(rids)
            for request,rid in zip(manifests[length]['requests'],rids,strict=True):
                response=byid[rid];ids=life.completion_ids(response)
                assert response['prompt_token_ids']==request['input_ids'] and response['meta_info']['cached_tokens']==0
                assert len(ids)==128 and tokenizer.decode(ids,skip_special_tokens=False)==response['text']
            segment=Path(state['log']).read_bytes()[before:].decode(errors='replace')
            checks=[tuple(map(int,m)) for m in re.findall(
                r'\[TP(\d+)\] prefill attention-stage1 checked: layer=(\d+) ratio=(\d+) rows=(\d+) exact=1',segment)]
            assert {r[0] for r in checks}==set(range(8))
            for rank in range(8):
                assert {r[1] for r in checks if r[0]==rank}==set(range(43)),(length,rank,checks)
                assert {r[2] for r in checks if r[0]==rank}=={4,128}
            progress.append(dict(length=length,rep=rep,checks=checks))
            life.save('progress.json',progress)
            print('QUALITY',length,rep,'16 exact input echoes',len(checks),'exact attention checks',flush=True)
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(progress=progress,quality_echoes=64,first_warm_echoes=0,cold_cache_control=False,diagnostic=True))
finally:life.stop(state)
