"""Owner-Q producer A1/B1/B2/A2: original131069-token C16 performance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent;repo=root.parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('A1','B','A2'),required=True)
args=p.parse_args();out=root/args.arm;out.mkdir(exist_ok=False)
candidate=args.arm=='B';diagnostic=False
gate=json.loads((root.parent/'dsv4_c16_owner_producer_teacher_20260915/B-check/complete.json').read_text())
assert gate['full_q_check'] and gate['scored_tokens']==4096
quality_gate=json.loads((root.parent/'dsv4_c16_owner_producer_service_20260915/B-check/complete.json').read_text())
assert quality_gate['dual_check'] and quality_gate['input_echo_exact']==32
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('life_owner',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
old=root.parent/'dsv4_c16_indexer_owner_service_20260915/B'
manifest=json.loads((old/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
life.save('inputs.json',manifest)
launcher=(old/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER={int(candidate)}\n'
       'export SGLANG_DSV4_C4_PREFILL_QUERY_OWNER=1\n'
       'export SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK=0\n'
       'unset SGLANG_DSV4_DEBUG_OWNER_PRODUCER_DIR SGLANG_DSV4_DEBUG_OWNER_PRODUCER_BLAS\n'
       'unset SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR SGLANG_DSV4_DEBUG_FIRST_DIV_DIR SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
       f'export SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK={int(diagnostic)}\n'
       'unset SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR\n'
       'unset SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
paths=set(json.loads((old/'plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_producer.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_rocblas_linear.py',
    'python/sglang/kernels/ops/attention/dsv4/topk.py',
    'python/sglang/kernels/jit/csrc/deepseek_v4/topk_deterministic_hip.cuh',
    'python/sglang/srt/distributed/parallel_state.py'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(candidate=candidate,diagnostic=diagnostic,original_weights=True,
    kv_tokens=1048576,prefill_budget=32768,sources=sources,
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
label='P16-producer-perf-'+args.arm
state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['max_total_tokens']==1048576 and info['speculative_algorithm'] is None and info['ep_size']==1
    env=life.owned(state).environ()
    for name,value in {'SGLANG_DSV4_C4_PREFILL_QUERY_OWNER':'1',
        'SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER':str(int(candidate)),
        'SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK':'0',
        'SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK':str(int(diagnostic)),
        'SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS':'1','SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE':'8',
        'SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE':'16','SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M':'1',
        'SGLANG_DSV4_C4_PREFILL_QUERY_WIDE':'0','SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS':'0',
        'SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20':'0'}.items():assert env[name]==value,(name,env.get(name))
    for key in ('SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR','SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR',
                'SGLANG_DSV4_DEBUG_OWNER_PRODUCER_DIR','SGLANG_DSV4_DEBUG_FIRST_DIV_DIR','SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR'):
        assert not env.get(key),key
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    response=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],cache_salt=f'owner-france-{time.time_ns()}',
        sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',response);assert 'paris' in response['text'].lower()
    assert 'prefill query-owner selected:' not in Path(state['log']).read_text()
    assert 'prefill query-producer selected:' not in Path(state['log']).read_text()
    progress=[]
    legs=[] if diagnostic else [('warmup',1),*[(n,3) for n in (['B1','B2'] if candidate else [args.arm])]]
    client=root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
    for name,rounds in legs:
        assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
        life.resources(name+'-before',life.owned(state))
        begin=Path(state['log']).stat().st_size
        command=[sys.executable,str(client),'--base-url',life.URL,'--inputs',str(out/'inputs.json'),
            '--request-count','16','--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        data=json.loads((out/(name+'.json')).read_text())
        assert all(r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16 for r in data['rounds'])
        record=dict(leg=name,median=data['median_input_tok_s'],log_start=begin,
            log_end=Path(state['log']).stat().st_size,rates=[r['aggregate_input_tok_s'] for r in data['rounds']])
        progress.append(record);life.save('progress.json',progress);print('RESULT',record,flush=True)
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    answers=[]
    for rep in range(1 if diagnostic else 2):
        life.resources('quality-'+str(rep),life.owned(state))
        rids=[f'owner-{args.arm}-{rep}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'owner-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),return_prompt_token_ids=True)
        result=life.post(life.URL+'/generate',payload,1800)
        life.save('quality-'+str(rep)+'.json',result)
        by_id={r['meta_info']['id']:r for r in result};assert set(by_id)==set(rids)
        ordered=[by_id[rid] for rid in rids]
        for request,response in zip(manifest['requests'],ordered,strict=True):
            assert response['prompt_token_ids']==request['input_ids'] and response['meta_info']['cached_tokens']==0
            ids=life.completion_ids(response);assert len(ids)==128
            assert tokenizer.decode(ids,skip_special_tokens=False)==response['text']
        answers.append(ordered)
    logs=Path(state['log']).read_text()
    path_name='producer' if candidate else 'owner'
    for rank in range(8):
        assert any(f'TP{rank}]' in line and f'prefill query-{path_name} selected:' in line
                   and 'check=0' in line for line in logs.splitlines()),rank
    if not candidate:assert 'prefill query-producer selected:' not in logs
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(progress=progress,diagnostic=diagnostic,input_echo_exact=16*len(answers),
        producer_hit=candidate,quality_repeat_exact=None if diagnostic else
        sum(life.completion_ids(a)==life.completion_ids(b) for a,b in zip(*answers,strict=True))))
    print('ARM COMPLETE',args.arm,flush=True)
finally:
    life.stop(state)
