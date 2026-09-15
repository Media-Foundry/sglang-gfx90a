"""Read-only cross-rank indexer audit on current accepted C16 prefill; not timing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import argparse
import re

root=Path(__file__).resolve().parent;repo=root.parents[2]
p=argparse.ArgumentParser();p.add_argument('--label',default='capture')
args=p.parse_args();assert re.fullmatch(r'[A-Za-z0-9-]+',args.label)
out=root/args.label;out.mkdir(exist_ok=False)
data=out/'data';data.mkdir()
source=root.parent/'dsv4_c16_premix_pair_service_20260915/B'
manifest=json.loads((source/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('owner_audit_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
launcher=(source/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR={data}\n'
       'unset SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
life.save('inputs.json',manifest)
paths=set(json.loads((source/'plan.json').read_text())['sources'])|{
    'python/sglang/kernels/ops/debug/dsv4_indexer_owner_capture.py',
    '.agents/experiments/dsv4_c16_indexer_owner_20260915/capture.py'}
digests={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(diagnostic_only=True,sources=digests,layers=[2,20,42],
    scope='Capture first large forward only; compare logical keys, not raw physical page IDs.',
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
state=life.start('P16-owner-audit',0)
try:
    life.ready(state)
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR']==str(data)
    assert env['SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS']=='1'
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE']=='16'
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M']=='1'
    info=json.loads((out/'P16-owner-audit.server-info.json').read_text())
    assert info['ep_size']==1 and info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['max_total_tokens']==1048576 and info['speculative_algorithm'] is None
    life.resources('before-wave',life.owned(state))
    rids=[f'owner-audit-{i}' for i in range(16)]
    payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
        cache_salt=[f'owner-audit-{i}-{time.time_ns()}' for i in range(16)],
        sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
    life.save('sent.json',payload)
    result=life.post(life.URL+'/generate',payload,1800)
    life.save('responses.json',result)
    assert len(result)==16
    by_id={r['meta_info']['id']:r for r in result};assert set(by_id)==set(rids)
    for request,rid in zip(manifest['requests'],rids,strict=True):
        response=by_id[rid]
        assert response['prompt_token_ids']==request['input_ids']
        assert response['meta_info']['cached_tokens']==0
        assert len(life.completion_ids(response))==1
    for rank in range(8):
        for layer in (2,20,42):
            assert (data/f'rank-{rank}-layer-{layer}.json').is_file()
    assert (data/'layer20-rank0-full.pt').is_file()
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==s for p,s in digests.items())
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'owner-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    life.save('complete.json',dict(input_echo_exact=16,records=24,diagnostic_only=True))
    print('CAPTURE COMPLETE',flush=True)
finally:
    life.stop(state)
