"""Wide-query mixed-prefix ABBA; isolate cache work before accepting speed."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
BASE=ROOT.parent/'dsv4_c16_query_wide32k_service_20260915'
FIX=ROOT.parent/'dsv4_c16_query_wide_20260915'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',choices=('A1','B','A2'),required=True)
p.add_argument('--validate-only',action='store_true')
args=p.parse_args()
candidate=args.arm=='B'
manifest=json.loads((FIX/'inputs32k/prefill.json').read_text())
validation=json.loads((FIX/'input-validation.json').read_text())['inputs32k']
assert validation['official_chat_encoding_exact']==16
assert validation['manifest_sha256']==hashlib.sha256((FIX/'inputs32k/prefill.json').read_bytes()).hexdigest()
assert sum(len(r['input_ids']) for r in manifest['requests'])==524286
old=json.loads((BASE/'summary.json').read_text())
assert old['throughput_gain_percent']>2
assert json.loads((BASE/'quality-review.json').read_text())['bounded_semantic_smoke_pass']
launcher=(BASE/'start-base.sh').read_text()
flags=dict(SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4='1',
    SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE='16',
    SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M='1',
    SGLANG_DSV4_C4_PREFILL_QUERY_WIDE=str(int(candidate)),
    SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE='8',
    SGLANG_DSV4_PREFILL_MIX_REUSE4='1',
    SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS='0',
    SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20='0',
    SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES='0')
launcher=launcher.replace('exec bash scripts/rocm_dsv4_flash.sh serve',
    ''.join(f'export {k}={v}\n' for k,v in flags.items())+'exec bash scripts/rocm_dsv4_flash.sh serve')
paths=[
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_reuse.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_reuse8.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_runtime_m.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_query_reuse.py',
    'python/sglang/kernels/ops/layernorm/mhc.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_fused4.py',
    'python/sglang/srt/layers/dsv4_prefill_experiments.py',
    'python/sglang/srt/layers/dsv4_prefill_mhc_policy.py',
    'python/sglang/srt/model_executor/runner/eager_runner.py',
    'python/sglang/srt/layers/attention/dsv4/indexer.py',
    'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py',
    'scripts/rocm_dsv4_flash.sh',str((FIX/'bench_prefix.py').relative_to(REPO)),
    str(Path(__file__).resolve().relative_to(REPO))]
def hashes():return {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths}
sources=hashes()
if args.arm!='A1':
    previous=json.loads((ROOT/'A1/plan.json').read_text())
    assert sources==previous['sources']
    assert (ROOT/'A1/complete.json').exists()
    for k,v in flags.items():
        if k!='SGLANG_DSV4_C4_PREFILL_QUERY_WIDE':assert previous['flags'][k]==v
if args.validate_only:
    print(json.dumps(dict(validated=True,arm=args.arm,flags=flags,sources=sources)));sys.exit(0)
out=ROOT/args.arm;out.mkdir(exist_ok=False)
helper=REPO/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('prefix_lifecycle',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
life.save('inputs.json',manifest)
(out/'start-ar-matrix.sh').write_text(launcher)
life.save('plan.json',dict(flags=flags,sources=sources,original_weight=True,
    scope='Original V4 TP8 EP1 native AR C16x32K mixed0/25/50/75% prefixes;1M KV,32K budget.'))
label='P16-prefix-'+args.arm
state=life.start(label,0)
try:
    life.ready(state)
    env=life.owned(state).environ()
    assert all(env.get(k)==v for k,v in flags.items())
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['ep_size']==1 and info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert not info['enable_mixed_chunk']
    france=json.loads((REPO/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'prefix-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    progress=[]
    legs=['B1','B2'] if candidate else [args.arm]
    for name,rounds,tokens in [('warmup',1,1),*[(n,3,1) for n in legs],('quality',2,128)]:
        assert hashes()==sources
        life.resources(name+'-before',life.owned(state))
        begin=Path(state['log']).stat().st_size
        command=[sys.executable,str(FIX/'bench_prefix.py'),'--inputs',str(out/'inputs.json'),
            '--base-url',life.URL,'--rounds',str(rounds),'--tokens',str(tokens),
            '--output',str(out/(name+'.json'))]
        reference=(ROOT/'A1/A1.json') if args.arm!='A1' else out/'warmup.json'
        if reference.exists():command+=['--cache-reference',str(reference)]
        with (out/(name+'.client.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        data=json.loads((out/(name+'.json')).read_text());assert data['status']=='complete'
        rates=[r['newly_computed_tok_s'] for r in data['rounds']]
        record=dict(leg=name,tokens=tokens,rates=rates,median=statistics.median(rates),
            log_start=begin,log_end=Path(state['log']).stat().st_size)
        progress.append(record);life.save('progress.json',progress);print('RESULT',json.dumps(record),flush=True)
        if name=='warmup':
            text=Path(state['log']).read_text()
            assert 'MHC Sinkhorn policy selected' not in text
            if candidate:
                for rank in range(8):
                    assert any(f'TP{rank}]' in line and 'prefill wide-query-reuse selected:' in line
                        and 'query_group=16' in line and 'runtime_m=1' in line for line in text.splitlines())
            else:assert 'prefill wide-query-reuse selected:' not in text
    life.save('complete.json',dict(progress=progress))
finally:
    life.stop(state)
