"""16K/32K route-producer regressions; current K32 profile and1M KV fixed."""
import argparse
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
parser.add_argument('--arm',choices=('A1','B','A2'),required=True)
parser.add_argument('--length',choices=('16k','32k'),required=True)
args=parser.parse_args()
checking=False
candidate=args.arm=='B'
accepted=json.loads((root.parent/'dsv4_ck_route_producer_service_20260917/acceptance.json').read_text())
assert accepted['status']=='accepted_explicit_8k_profile' and accepted['numerical_exact_on_tested_inputs']
expected_tokens=262141 if args.length=='16k' else 524286
out=root/(args.length+'-'+args.arm)
out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('owner_perf_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life)
life.ROOT=life.OLD=out
prior=(root.parent/'dsv4_owner_k32_regression_20260916/16k-B' if args.length=='16k'
       else root.parent/'dsv4_owner_k32_20260916/B')
manifest_path=prior/'inputs.json'
archive=json.loads((prior.parent/'archive-manifest.json').read_text())
entry=next(f for f in archive['files'] if f['path']==str(Path(prior.name)/'inputs.json'))
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==entry['sha256']
manifest=json.loads(manifest_path.read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==expected_tokens
life.save('inputs.json',manifest)
launcher=(prior/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle)==1
flags=('export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA=0\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA_CHECK=0\n'
       'unset SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_DIR\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER=1\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER_CHECK=0\n'
       'export SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT=1\n'
       'export SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT_CHECK=0\n'
       'export SGLANG_DSV4_C4_PREFILL_QUERY_WIDE=1\n'
       'export SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE=1\n'
       'export SGLANG_DSV4_PREFILL_MHC_COMMON_FP32=1\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK=0\n'
       'export SGLANG_DSV4_C4_PREFILL_OWNER_K32=1\n'
       f'export SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER={int(candidate)}\n'
       f'export SGLANG_DSV4_DEBUG_CK_ROUTE_CHECK={int(checking)}\n'
       f'export SGLANG_DSV4_DEBUG_CK_ROUTE_MANIFEST={root.parent}/dsv4_ck_route_producer_20260916/build-v1/route/manifest.json\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
paths=set(json.loads((prior/'plan.json').read_text())['sources']) | {
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_owner.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_pair.py',
    'python/sglang/kernels/ops/debug/dsv4_premix_owner_audit.py',
    'python/sglang/kernels/ops/moe/gfx90a_bf16_direct_rows.py',
    'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_bf16_direct_rows.cuh',
    'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_query_reuse.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_prefill_policy.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner_k32.py',
    'python/sglang/kernels/ops/moe/gfx90a_ck_route_producer.py',
    'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_ck_route_producer.cuh',
    '.agents/experiments/dsv4_ck_route_producer_20260916/build-v1/route/manifest.json'}
sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(sources=sources,candidate=candidate,diagnostic=checking,
    kv_tokens=1048576,prefill_budget=32768,original_weights=True))
label='P'+args.length+'-route-producer-regression-'+args.arm
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
                      ('SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER','1'),
                      ('SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT','1'),
                      ('SGLANG_DSV4_C4_PREFILL_QUERY_WIDE','1'),
                      ('SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE','1'),
                      ('SGLANG_DSV4_PREFILL_MHC_COMMON_FP32','1'),
                      ('SGLANG_DSV4_C4_PREFILL_OWNER_K32','1'),
                      ('SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER',str(int(candidate))),
                      ('SGLANG_DSV4_DEBUG_CK_ROUTE_CHECK',str(int(checking)))]:
        assert env[key]==value
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'owner-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    assert 'pre-mix owner selected:' not in Path(state['log']).read_text()
    assert 'owner-K32 logits selected:' not in Path(state['log']).read_text()
    assert 'route-producer CK selected:' not in Path(state['log']).read_text()
    client=root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
    legs=[('warmup',1),*[(n,3) for n in (('B1','B2') if candidate else (args.arm,))]]
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
    # Freeze timed paths before teacher forcing can introduce mixed chunks.
    timing_log=Path(state['log']).read_text()
    legacy=re.findall(r'TP(\d+)\] DSV4 native TP8 prefill splitk selected: path=(\w+) rows=(\d+) batch=(\d+) weight_dtype=([^;]+)',timing_log)
    mix_owner_ranks=sorted(set(re.findall(r'\[TP(\d+)\] pre-mix owner selected:',timing_log)))
    k32_ranks=sorted(set(re.findall(r'\[TP(\d+)\] owner-K32 logits selected:',timing_log)))
    route_ranks=sorted(set(re.findall(r'route-producer CK selected: rank=(\d+)',timing_log)))
    life.save('timing-paths.json',dict(legacy_splitk=legacy,premix_owner_ranks=mix_owner_ranks,k32_ranks=k32_ranks,route_ranks=route_ranks))
    assert k32_ranks==list(map(str,range(8))),k32_ranks
    assert route_ranks==(list(map(str,range(8))) if candidate else []),route_ranks
    assert not legacy,legacy
    assert mix_owner_ranks==list(map(str,range(8))),mix_owner_ranks
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
        reference_dir=(root.parent/'dsv4_owner_k32_regression_20260916/16k-A1' if args.length=='16k'
                       else root.parent/'dsv4_owner_k32_20260916/A1')
        reference=json.loads((reference_dir/'quality-0.json').read_text())
        refbyid={r['meta_info']['id']:r for r in reference}
        prompts=[r['input_ids']+life.completion_ids(refbyid[f'corrected-A1-0-{i}'])[:64] for i,r in enumerate(manifest['requests'])]
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
    exact=re.findall(r'\[TP(\d+)\] route-producer exact: rows=(\d+) output=byte_exact',logs)
    # The1376 live comparisons belong to the prior8K diagnostic only.
    assert not exact
    for rank in range(8):
        assert f'unique-slot CK selected: rank={rank} ' in logs
        assert f'[TP{rank}] H16 peer selected:' in logs
        assert f' TP{rank}] DSV4 direct-row CK dequant selected:' in logs
        hits=re.findall(rf'\[TP{rank}\] prefill query-owner selected: rows=\d+ local_rows=\d+ width=(\d+)',timing_log)
        assert any(int(w)==(4096 if args.length=='16k' else 8192) for w in hits),(rank,hits)
        assert f' TP{rank}] DSV4 common FP32/20 prefill MHC selected:' in logs
    assert 'DSV4 cooperative FP32 pre-mix selected:' not in logs
    assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(progress=progress,live_comparisons=len(exact),diagnostic=checking,
        france_passed=True,quality_repeat_exact=[sum(life.completion_ids(a)==life.completion_ids(b)
            for a,b in zip(answers[0],wave,strict=True)) for wave in answers[1:]]))
    print('ARM COMPLETE',args.arm,flush=True)
finally:
    life.stop(state)
