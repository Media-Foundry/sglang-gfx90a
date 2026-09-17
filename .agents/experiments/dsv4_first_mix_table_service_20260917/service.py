"""TP8 first-mix table against the corrected FP32 reference, never legacy BF16 Fn."""
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

root=Path(__file__).resolve().parent;repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('check','A1','B','A2'),required=True)
args=parser.parse_args();checking=args.arm=='check';candidate=args.arm in ('check','B')
prior=root.parent/'dsv4_first_mhc_service_20260917/B'
oracle=json.loads((root.parent/'dsv4_first_mix_table_20260917/oracle.json').read_text())
assert oracle['status']=='complete' and oracle['whole_vocab_native_variants_exact'] and oracle['service_mixes_exact']
if not checking:
    check=json.loads((root/'check/complete.json').read_text())
    assert check['live_exact_checks']>=40
out=root/args.arm;out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('first_table_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life);life.ROOT=life.OLD=out
manifest=json.loads((prior/'inputs.json').read_text());life.save('inputs.json',manifest)
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
launcher=(prior/'start-ar-matrix.sh').read_text()
entry='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(entry)==1
flags=f'export SGLANG_DSV4_PREFILL_FIRST_MIX_TABLE={int(candidate)}\nexport SGLANG_DSV4_DEBUG_FIRST_MIX_TABLE_CHECK={int(checking)}\n'
(out/'start-ar-matrix.sh').write_text(launcher.replace(entry,flags+entry))
client=root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
paths=set(json.loads((prior/'plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/layernorm/gfx90a_first_mix_table.py',
    'python/sglang/srt/model_executor/model_runner_components/weight_updater.py'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(sources=sources,candidate=candidate,diagnostic=checking,kv_tokens=1048576,
    first_mhc='original FP32 Fn on both arms',route_producer=True))
label='P8-first-table-'+args.arm;state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==32768
    env=life.owned(state).environ()
    for flag in ('SGLANG_DSV4_PREFILL_MHC_COMMON_FP32','SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL',
                 'SGLANG_DSV4_PREFILL_MHC_COMMON_UNFUSED','SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER'):
        assert env[flag]=='1',flag
    assert env['SGLANG_DSV4_PREFILL_FIRST_MIX_TABLE']==str(int(candidate))
    assert env['SGLANG_DSV4_DEBUG_FIRST_MIX_TABLE_CHECK']==str(int(checking))
    assert not env.get('SGLANG_DSV4_DEBUG_FIRST_DIV_DIR')
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'first-table-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    progress=[]
    legs=[('check',1)] if checking else [('warmup',1),*[(n,3) for n in (('B1','B2') if candidate else (args.arm,))]]
    for name,rounds in legs:
        life.resources(name+'-before',life.owned(state))
        command=[sys.executable,str(client),'--base-url',life.URL,'--inputs',str(out/'inputs.json'),
                 '--request-count','16','--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        data=json.loads((out/(name+'.json')).read_text())
        assert all(r['input_echo_exact'] and r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16 for r in data['rounds'])
        progress.append(dict(leg=name,median=data['median_input_tok_s'],rates=[r['aggregate_input_tok_s'] for r in data['rounds']]))
        life.save('progress.json',progress);print('RESULT',args.arm,progress[-1],flush=True)
    # Complete quality output and identical teacher fixture in all arms.
    quality=[]
    for rep in range(2):
        rids=[f'quality-{args.arm}-{rep}-{i}' for i in range(16)]
        result=life.post(life.URL+'/generate',dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'table-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],return_prompt_token_ids=True,
            sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True)),1800)
        life.save(f'quality-{rep}.json',result)
        byid={r['meta_info']['id']:r for r in result};assert set(byid)==set(rids)
        rows=[byid[rid] for rid in rids]
        for i,row in enumerate(rows):
            assert row['prompt_token_ids']==manifest['requests'][i]['input_ids']
            assert row['meta_info']['cached_tokens']==0 and len(life.completion_ids(row))==128
        quality.append(rows)
    teacher_seed=json.loads((prior/'quality.json').read_text())
    seed={r['meta_info']['id']:r for r in teacher_seed}
    prompts=[r['input_ids']+life.completion_ids(seed[f'quality-B-{i}'])[:64] for i,r in enumerate(manifest['requests'])]
    rids=[f'teacher-{args.arm}-{i}' for i in range(16)]
    teacher=life.post(life.URL+'/generate',dict(input_ids=prompts,rid=rids,
        cache_salt=[f'table-teacher-{i}-{time.time_ns()}' for i in range(16)],return_prompt_token_ids=True,
        sampling_params=dict(temperature=0,max_new_tokens=1),return_logprob=True,
        logprob_start_len=[len(r['input_ids']) for r in manifest['requests']],top_logprobs_num=5),1800)
    life.save('teacher-forced.json',teacher)
    byid={r['meta_info']['id']:r for r in teacher};assert set(byid)==set(rids)
    for i,rid in enumerate(rids):
        row=byid[rid];assert row['prompt_token_ids']==prompts[i] and row['meta_info']['cached_tokens']==0
        assert len(row['meta_info']['input_token_logprobs'])==64
    logs=Path(state['log']).read_text()
    built=re.findall(r'first-mix table built: rank=(\d+) local_vocab=(\d+) padded_vocab=(\d+) bytes=(\d+)',logs)
    exact=re.findall(r'first-mix table exact: rank=(\d+) rows=(\d+)',logs)
    consumed=re.findall(r'first-mix table consumed: rank=(\d+) rows=(\d+)',logs)
    if candidate:
        assert len(built)==8 and {r[0] for r in built}==set(map(str,range(8)))
        assert all(int(r[3])==int(r[2])*24*4 and int(r[2])>=129280 for r in built)
    else:assert not built
    if checking:
        assert Counter(exact)==Counter(consumed) and len(exact)>=40
        for rank in map(str,range(8)):
            assert [m for r,m in exact if r==rank]==[m for r,m in consumed if r==rank]
        assert set(r[0] for r in exact)==set(map(str,range(8)))
        assert len(set(Counter(r[0] for r in exact).values()))==1
        assert any(int(r[1])>=8192 for r in exact) and any(int(r[1])<8192 for r in exact)
    else:assert not exact and not consumed
    for rank in range(8):
        assert f'route-producer CK selected: rank={rank} ' in logs
        assert f'[TP{rank}] owner-K32 logits selected:' in logs
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(status='complete',diagnostic=checking,progress=progress,france_passed=True,
        built=built,live_exact_checks=len(exact),quality_repeat_exact=sum(life.completion_ids(a)==life.completion_ids(b)
            for a,b in zip(*quality,strict=True))))
finally:
    life.stop(state)
