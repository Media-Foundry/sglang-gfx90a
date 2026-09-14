"""Fresh-service ABBA for prefill config20 only; wide query reuse stays on."""
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
p.add_argument('--arm',required=True,choices=('A1','B','A2'))
args=p.parse_args()
candidate=args.arm=='B'
previous=root.parent/'dsv4_c16_query_wide32k_service_20260915'
assert json.loads((previous/'summary.json').read_text())['throughput_gain_percent']>2
assert all(json.loads((previous/a/f'P16-wide32k-{a}.stop.json').read_text())['remaining']==[] for a in ('A1','B','A2'))
for name in ('screen.json','large.json'):
    component=json.loads((root/name).read_text())
    assert component['status']=='complete'
    assert all(x['reference_exact'] and x['row_permutation_exact'] for x in component['results'])
out=root/args.arm;out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('mhc20_life',repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
manifest=json.loads((previous/'A1/inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==524286
life.save('inputs.json',manifest)
base=(previous/'start-base.sh').read_text()
flags={
    'SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4':'1',
    'SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE':'16',
    'SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M':'1',
    'SGLANG_DSV4_C4_PREFILL_QUERY_WIDE':'1',
    'SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP':'1',
    'SGLANG_DSV4_PREFILL_POST_FUSED4':'1',
    'SGLANG_DSV4_PREFILL_MIX_REUSE4':'1',
    'SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE':'8',
    'SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES':'0',
    'SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS':'8',
    'SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS':str(int(candidate)),
}
needle='exec bash scripts/rocm_dsv4_flash.sh serve'
assert base.count(needle)==1
base=base.replace(needle,'\n'.join(f'export {k}={v}' for k,v in flags.items())+
                  '\nunset SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR\n'+needle)
(out/'start-ar-matrix.sh').write_text(base)
paths=set(json.loads((previous/'B/plan.json').read_text())['sources'])
paths.update(('python/sglang/srt/layers/dsv4_prefill_mhc_policy.py',
              'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_pre.py'))
def hashes():return {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
sources=hashes()
life.save('plan.json',dict(candidate=candidate,flags=flags,sources=sources,
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),
    scope='Only native TP8 ordinary prefill Sinkhorn policy; all arms keep wide query reuse, original checkpoint, 1M KV, legacy FP16 Fn and native AR decode.'))
label='P16-mhc20-'+args.arm
state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['tp_size']==8 and info['ep_size']==1 and info['speculative_algorithm'] is None
    assert info['max_total_tokens']==1048576
    assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['enable_mixed_chunk'] is False
    env=life.owned(state).environ()
    assert all(env[k]==v for k,v in flags.items())
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    response=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'mhc20-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',response);assert 'paris' in response['text'].lower()
    progress=[]
    legs=['B1','B2'] if candidate else [args.arm]
    for name,rounds in [('warmup',1),*[(leg,3) for leg in legs]]:
        assert sources==hashes()
        life.resources(name+'-before',life.owned(state))
        begin=Path(state['log']).stat().st_size
        command=[sys.executable,str(repo/'scripts/rocm/bench_dsv4_prefill_diverse_concurrent.py'),
                 '--base-url',life.URL,'--inputs',str(out/'inputs.json'),'--request-count','16',
                 '--rounds',str(rounds),'--tokens','1','--output',str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        data=json.loads((out/(name+'.json')).read_text())
        assert all(r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16 for r in data['rounds'])
        progress.append(dict(leg=name,median=data['median_input_tok_s'],
            rates=[r['aggregate_input_tok_s'] for r in data['rounds']],
            log_start=begin,log_end=Path(state['log']).stat().st_size))
        life.save('progress.json',progress);print('RESULT',progress[-1],flush=True)
        if name=='warmup':
            lines=Path(state['log']).read_text().splitlines()
            for rank in range(8):
                assert any(f'TP{rank}]' in s and 'prefill wide-query-reuse selected:' in s and 'C4_capacity=8192' in s for s in lines)
                assert any(f'TP{rank}]' in s and 'prefill splitk selected:' in s and 'weight_dtype=torch.float16' in s for s in lines)
                policy_hit=any(f'TP{rank}]' in s and 'MHC Sinkhorn policy selected: legacy=8 config=20' in s for s in lines)
                assert policy_hit==candidate,(rank,'policy hit',policy_hit)
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    waves=[]
    for rep in range(2):
        life.resources('quality-'+str(rep),life.owned(state))
        rids=[f'mhc20-{args.arm}-{rep}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'mhc20-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),return_prompt_token_ids=True)
        responses=life.post(life.URL+'/generate',payload,1800)
        life.save('quality-'+str(rep)+'.json',responses)
        by_id={r['meta_info']['id']:r for r in responses};assert len(responses)==16 and set(by_id)==set(rids)
        ordered=[by_id[rid] for rid in rids]
        for request,response in zip(manifest['requests'],ordered,strict=True):
            assert response['prompt_token_ids']==request['input_ids']
            assert response['meta_info']['cached_tokens']==0
            tokens=life.completion_ids(response)
            assert len(tokens)==128 and tokenizer.decode(tokens,skip_special_tokens=False)==response['text']
        waves.append(ordered)
    life.save('complete.json',dict(progress=progress,input_echo_exact=32,
        quality_repeat_exact=sum(life.completion_ids(a)==life.completion_ids(b) for a,b in zip(*waves,strict=True))))
finally:
    life.stop(state)
