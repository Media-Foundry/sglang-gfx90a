"""Fresh integrated owner producer quality gate with exact same-backend dual check."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

parser=argparse.ArgumentParser()
parser.add_argument('--arm',choices=['A','B'],required=True)
parser.add_argument('--check',action='store_true')
parser.add_argument('--name',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent;repo=root.parents[2]
assert Path(args.name).name==args.name
out=root/args.name;out.mkdir(exist_ok=False)
source=root.parent/'dsv4_c16_indexer_owner_service_20260915/B'
manifest=json.loads((source/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('producer_lifecycle',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life);life.ROOT=life.OLD=out
candidate=args.arm=='B'
launcher=(source/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER={int(candidate)}\n'
       f'export SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK={int(args.check)}\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK=0\n'
       'unset SGLANG_DSV4_DEBUG_OWNER_PRODUCER_DIR SGLANG_DSV4_DEBUG_OWNER_PRODUCER_BLAS\n'
       'unset SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR SGLANG_DSV4_DEBUG_FIRST_DIV_DIR\n'
       'unset SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
life.save('inputs.json',manifest)
paths=set(json.loads((source/'plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_producer.py',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_rocblas_linear.py',
    'python/sglang/srt/layers/attention/dsv4/indexer.py'}
hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(quality_only=True,arm=args.arm,check=args.check,sources=hashes,
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
state=life.start('P16-producer-'+args.name,0)
try:
    life.ready(state)
    info=json.loads((out/('P16-producer-'+args.name+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['ep_size']==1 and info['speculative_algorithm'] is None
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER']==str(int(candidate))
    assert env['SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK']==str(int(args.check))
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],cache_salt=f'producer-france-{time.time_ns()}',
        sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    assert 'prefill query-producer selected:' not in Path(state['log']).read_text()
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    for rep in range(2):
        rids=[f'producer-{args.name}-{rep}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'producer-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=256,ignore_eos=True),return_prompt_token_ids=True)
        life.save(f'sent-{rep}.json',payload)
        result=life.post(life.URL+'/generate',payload,2400);life.save(f'quality-{rep}.json',result)
        byid={r['meta_info']['id']:r for r in result};assert set(byid)==set(rids)
        for req,rid in zip(manifest['requests'],rids,strict=True):
            response=byid[rid]
            assert response['prompt_token_ids']==req['input_ids'] and response['meta_info']['cached_tokens']==0
            ids=life.completion_ids(response);assert len(ids)==256
            assert tokenizer.decode(ids,skip_special_tokens=False)==response['text']
        assert hashes=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in hashes}
        print('QUALITY COMPLETE',rep,flush=True)
    logs=Path(state['log']).read_text()
    if candidate:
        for rank in range(8):
            assert any(f'TP{rank}]' in line and 'prefill query-producer selected:' in line
                       and f'check={int(args.check)}' in line for line in logs.splitlines())
    else:assert 'prefill query-producer selected:' not in logs
    life.save('complete.json',dict(input_echo_exact=32,arm=args.arm,dual_check=args.check,quality_only=True))
    print('SERVICE QUALITY COMPLETE',args.name,flush=True)
finally:
    life.stop(state)
