"""Owned native TP8 C16 prefill pilot/ABBA, one process per requested arm."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('pilot','A1','B','A2'),required=True)
args=parser.parse_args()
out=root/args.arm;out.mkdir(exist_ok=False)
candidate=args.arm in ('pilot','B');budget=65536 if candidate else 32768
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('life64',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
old=repo/'.agents/experiments/dsv4_tp8_c16_prefill_20260914/B'
manifest=json.loads((old/'inputs.json').read_text())
assert len(manifest['requests'])==16
assert sum(len(x['input_ids']) for x in manifest['requests'])==131069
life.save('inputs.json',manifest)
launcher=(old/'start-ar-matrix.sh').read_text()
line='export CHUNKED_PREFILL_SIZE=32768 MAX_PREFILL_TOKENS=32768 PREFILL_MAX_REQUESTS=16'
assert launcher.count(line)==1
launcher=launcher.replace(line,f'export CHUNKED_PREFILL_SIZE={budget} MAX_PREFILL_TOKENS={budget} PREFILL_MAX_REQUESTS=16')
flags=f'export SGLANG_DSV4_DEBUG_TP8_CK_64K={int(candidate)}\n'
for flag in ('PREFILL_QKV_STABLE','PREFILL_WQB_STABLE','PREFILL_WOA_STABLE',
             'PREFILL_ATTN_AR_FP32','PREFILL_FFN_AR_FP32','PREPARE_DUMP'):
    flags+=f'export SGLANG_DSV4_DEBUG_{flag}=0\n'
flags+='unset SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
flags+='export SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT=0\n'
launcher=launcher.replace('exec bash scripts/rocm_dsv4_flash.sh serve',flags+'exec bash scripts/rocm_dsv4_flash.sh serve')
(out/'start-ar-matrix.sh').write_text(launcher)
life.save('plan.json',dict(budget=budget,original_weight=True,kv_tokens=1048576,
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256((repo/'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py').read_bytes()).hexdigest()))
state=life.start('P16-'+args.arm,0)
try:
    life.ready(state)
    info=json.loads((out/('P16-'+args.arm+'.server-info.json')).read_text())
    assert info['chunked_prefill_size']==budget and info['max_prefill_tokens']==budget
    assert info['max_total_tokens']==1048576 and info['speculative_algorithm'] is None
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    result=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'64k-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',result)
    assert 'paris' in result['text'].lower()
    legs=['B1','B2'] if args.arm=='B' else [args.arm]
    progress=[]
    for name,rounds in [('warmup',1),*[(leg,3) for leg in legs]]:
        life.resources(name+'-before',life.owned(state))
        begin=Path(state['log']).stat().st_size
        command=[sys.executable,str(repo/'scripts/rocm/bench_dsv4_prefill_diverse_concurrent.py'),
                 '--base-url',life.URL,'--inputs',str(out/'inputs.json'),'--request-count','16',
                 '--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        data=json.loads((out/(name+'.json')).read_text())
        assert all(r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16 for r in data['rounds'])
        record=dict(leg=name,median=data['median_input_tok_s'],log_start=begin,
                    log_end=Path(state['log']).stat().st_size,
                    rates=[r['aggregate_input_tok_s'] for r in data['rounds']])
        progress.append(record);life.save('progress.json',progress);print('RESULT',record,flush=True)
    if candidate:
        assert 'experimental TP8 CK large-prefill selected' in Path(state['log']).read_text()
    # Separate input/quality check; echo payload is not added to timed TTFT tests.
    answers=[]
    for rep in range(2):
        life.resources('quality-'+str(rep),life.owned(state))
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],
            cache_salt=[f'64k-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),
            return_prompt_token_ids=True)
        result=life.post(life.URL+'/generate',payload,1800)
        life.save('quality-'+str(rep)+'.json',result)
        assert len(result)==16
        for request,response in zip(manifest['requests'],result,strict=True):
            assert response['prompt_token_ids']==request['input_ids']
            assert response['meta_info']['cached_tokens']==0
            assert len(life.completion_ids(response))==128
        answers.append(result)
    life.save('complete.json',dict(budget=budget,progress=progress,input_echo_exact=32,
        quality_repeat_exact=sum(life.completion_ids(a)==life.completion_ids(b)
                                 for a,b in zip(*answers,strict=True))))
finally:
    life.stop(state)
