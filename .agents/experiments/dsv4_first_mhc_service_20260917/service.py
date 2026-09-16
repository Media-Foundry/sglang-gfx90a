"""Uninstrumented C16x8K ABBA: accepted MHC policy vs complete FP32 policy.

Route producer stays ON, KV1M,32K budget,original checkpoint. B enables both
the small-tail and unfused-first-boundary fixes; no claim of global invariance.
"""
import argparse
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
parser.add_argument('--arm',choices=('A1','B','A2'),required=True)
args=parser.parse_args();candidate=args.arm=='B'
prior=root.parent/'dsv4_ck_route_producer_service_20260917/B'
proof=root.parent/'dsv4_prefill_tail_contract_20260917'
assert json.loads((proof/'first-mhc-integrated.json').read_text())['status']=='complete'
trace=json.loads((proof/'trace-v3-analysis.json').read_text())
for row in trace['results']:
    if row['layer']==0:
        stages={s['stage']:s for s in row['stages']}
        for name in ('unfused_mhc_mixes','attn_input_post','attn_input_comb','attn_norm'):
            assert stages[name]['different_rows']==0
out=root/args.arm;out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('mhc_perf_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life);life.ROOT=life.OLD=out
manifest=json.loads((prior/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
life.save('inputs.json',manifest)
launcher=(prior/'start-ar-matrix.sh').read_text()
entry='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(entry)==1
flags=(f'export SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL={int(candidate)}\n'
       f'export SGLANG_DSV4_PREFILL_MHC_COMMON_UNFUSED={int(candidate)}\n'
       'export SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM=1\n'
       'unset SGLANG_DSV4_DEBUG_FIRST_DIV_DIR SGLANG_DSV4_DEBUG_PREPARE_DUMP\n'
       'unset SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
       'unset SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(entry,flags+entry))
client=root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
paths=set(json.loads((prior/'plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),str(client.relative_to(repo)),
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_pre_mix.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_prefill_policy.py',
    'python/sglang/srt/layers/dsv4_prefill_tail_policy.py',
    'python/sglang/srt/model_executor/runner/eager_runner.py'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(sources=sources,candidate=candidate,kv_tokens=1048576,
    route_producer=True,small_tail_fp32=candidate,unfused_fn_fp32=candidate,
    diagnostic=False,global_batch_invariance_claimed=False))
label='P8-first-mhc-'+args.arm;state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576
    assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['ep_size']==1
    env=life.owned(state).environ()
    for key,value in {'SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER':'1',
        'SGLANG_DSV4_PREFILL_MHC_COMMON_FP32':'1',
        'SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL':str(int(candidate)),
        'SGLANG_DSV4_PREFILL_MHC_COMMON_UNFUSED':str(int(candidate)),
        'SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM':'1',
        'SGLANG_DSV4_DEBUG_CK_ROUTE_CHECK':'0'}.items():assert env[key]==value,(key,env.get(key))
    assert not any(env.get(k) for k in ('SGLANG_DSV4_DEBUG_FIRST_DIV_DIR',
        'SGLANG_DSV4_DEBUG_PREPARE_DUMP','SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR'))
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'first-mhc-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    progress=[]
    for name,rounds in [('warmup',1),*[(n,3) for n in (('B1','B2') if candidate else (args.arm,))]]:
        life.resources(name+'-before',life.owned(state))
        command=[sys.executable,str(client),'--base-url',life.URL,'--inputs',str(out/'inputs.json'),
                 '--request-count','16','--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        data=json.loads((out/(name+'.json')).read_text())
        assert all(r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16
                   and r['input_echo_exact'] for r in data['rounds'])
        progress.append(dict(leg=name,median=data['median_input_tok_s'],
                             rates=[r['aggregate_input_tok_s'] for r in data['rounds']]))
        life.save('progress.json',progress);print('RESULT',args.arm,progress[-1],flush=True)
    logs=Path(state['log']).read_text()
    for pattern in (r'route-producer CK selected: rank=(\d+)',
                    r'\[TP(\d+)\] pre-mix owner selected:',r'\[TP(\d+)\] owner-K32 logits selected:'):
        assert sorted(set(re.findall(pattern,logs)))==list(map(str,range(8))),pattern
    assert 'first-div layer completed:' not in logs
    assert 'DSV4 native TP8 prefill splitk selected:' not in logs
    life.save('timing-shapes.json',re.findall(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)',logs))
    # One unscored long-output wave; numerical identity is not presumed.
    rids=[f'quality-{args.arm}-{i}' for i in range(16)]
    quality=life.post(life.URL+'/generate',dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
        cache_salt=[f'mhc-quality-{i}-{time.time_ns()}' for i in range(16)],return_prompt_token_ids=True,
        sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True)),1800)
    life.save('quality.json',quality)
    byid={r['meta_info']['id']:r for r in quality};assert set(byid)==set(rids)
    for i,rid in enumerate(rids):
        row=byid[rid];assert row['prompt_token_ids']==manifest['requests'][i]['input_ids']
        assert row['meta_info']['cached_tokens']==0 and len(life.completion_ids(row))==128
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(status='complete',progress=progress,france_passed=True,
        quality_requests=16,quality_tokens=128,numerical_equivalence_claimed=False))
finally:
    life.stop(state)
