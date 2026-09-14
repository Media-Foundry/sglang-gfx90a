"""C16 original-V4 prefill empty-tile ABBA, fresh owned service per arm."""
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
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('A1','B','A2'),required=True)
args=p.parse_args()
out=root/args.arm;out.mkdir(exist_ok=False)
candidate=args.arm=='B'
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('life_empty',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
old=repo/'.agents/experiments/dsv4_tp8_c16_prefill_20260914/B'
manifest=json.loads((old/'inputs.json').read_text())
assert len(manifest['requests'])==16
assert sum(len(x['input_ids']) for x in manifest['requests'])==131069
life.save('inputs.json',manifest)
launcher=(old/'start-ar-matrix.sh').read_text()
assert 'CHUNKED_PREFILL_SIZE=32768 MAX_PREFILL_TOKENS=32768' in launcher
flags=f'export SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP={int(candidate)}\n'
for flag in ('PREFILL_QKV_STABLE','PREFILL_WQB_STABLE','PREFILL_WOA_STABLE','PREFILL_WOB_STABLE',
             'PREFILL_SHARED_STABLE','PREFILL_ATTN_AR_FP32','PREFILL_FFN_AR_FP32','PREPARE_DUMP',
             'COMPRESSOR_DUMP','PREFILL_CORE_COMPRESSOR_STABLE','TP8_CK_64K'):
    flags+=f'export SGLANG_DSV4_DEBUG_{flag}=0\n'
flags+='unset SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
flags+='export SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT=0\n'
launcher=launcher.replace('exec bash scripts/rocm_dsv4_flash.sh serve',flags+'exec bash scripts/rocm_dsv4_flash.sh serve')
(out/'start-ar-matrix.sh').write_text(launcher)
paths=('python/sglang/srt/layers/attention/dsv4/indexer.py',
       'python/sglang/srt/layers/attention/dsv4/compressor.py',
       'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py')
life.save('plan.json',dict(candidate=candidate,budget=32768,original_weight=True,kv_tokens=1048576,
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),
    sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}))
state=life.start('P16-empty-'+args.arm,0)
try:
    life.ready(state)
    info=json.loads((out/('P16-empty-'+args.arm+'.server-info.json')).read_text())
    assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['max_total_tokens']==1048576 and info['speculative_algorithm'] is None
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP']==str(int(candidate))
    assert env['SGLANG_DSV4_DEBUG_PREFILL_CORE_COMPRESSOR_STABLE']=='0'
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    result=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'empty-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',result);assert 'paris' in result['text'].lower()
    legs=['B1','B2'] if candidate else [args.arm]
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
        if name=='warmup':
            hit='prefill empty tiles selected' in Path(state['log']).read_text()
            assert hit==candidate,('unexpected selector hit',hit,candidate)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    answers=[]
    for rep in range(2):
        life.resources('quality-'+str(rep),life.owned(state))
        rids=[f'empty-{args.arm}-{rep}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'empty-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),
            return_prompt_token_ids=True)
        result=life.post(life.URL+'/generate',payload,1800)
        life.save('quality-'+str(rep)+'.json',result);assert len(result)==16
        by_id={r['meta_info']['id']:r for r in result};assert set(by_id)==set(rids)
        ordered=[by_id[r] for r in rids]
        for request,response in zip(manifest['requests'],ordered,strict=True):
            assert response['prompt_token_ids']==request['input_ids']
            assert response['meta_info']['cached_tokens']==0
            ids=life.completion_ids(response);assert len(ids)==128
            assert tokenizer.decode(ids,skip_special_tokens=False)==response['text']
        answers.append(ordered)
    life.save('complete.json',dict(progress=progress,input_echo_exact=32,
        quality_repeat_exact=sum(life.completion_ids(a)==life.completion_ids(b)
                                 for a,b in zip(*answers,strict=True))))
finally:
    life.stop(state)
