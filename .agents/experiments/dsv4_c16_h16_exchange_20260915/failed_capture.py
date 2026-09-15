"""One fresh production-profile capture, then owned shutdown before replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent; repo=root.parents[2]
out=root/'capture'; out.mkdir(exist_ok=False); data=out/'fixture'
source=root.parent/'dsv4_c16_attention_stage1_service_20260915/B'
manifest=json.loads((source/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('real_ck_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life);life.ROOT=life.OLD=out
launcher=(source/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve'; assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_DEBUG_ATTN_PEER_CAPTURE_DIR={data}\n'
       'unset SGLANG_DSV4_DEBUG_FIRST_DIV_DIR SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR\n'
       'unset SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK=0\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
life.save('inputs.json',manifest)
paths=set(json.loads((source/'plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/debug/dsv4_attention_peer_capture.py',
    'python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py',
    'python/sglang/srt/models/deepseek_v4.py'}
hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(diagnostic_only=True,sources=hashes,
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
label='P16-real-peer-fixture';state=life.start(label,0)
try:
    life.ready(state)
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_OWNER']=='1'
    assert env['SGLANG_DSV4_DEBUG_ATTN_PEER_CAPTURE_DIR']==str(data)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['ep_size']==1 and info['speculative_algorithm'] is None
    rids=[f'real-peer-{i}' for i in range(16)]
    payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
        cache_salt=[f'real-peer-{i}-{time.time_ns()}' for i in range(16)],
        sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
    life.save('sent.json',payload)
    result=life.post(life.URL+'/generate',payload,2400);life.save('responses.json',result)
    byid={r['meta_info']['id']:r for r in result};assert set(byid)==set(rids)
    for req,rid in zip(manifest['requests'],rids):
        response=byid[rid]
        assert response['prompt_token_ids']==req['input_ids'] and response['meta_info']['cached_tokens']==0
        assert len(life.completion_ids(response))==1
    for rank in range(8):assert (data/f'rank-{rank}.json').exists()
    assert hashes=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in hashes}
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],cache_salt=f'real-peer-france-{time.time_ns()}',
        sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    life.save('complete.json',dict(input_echo_exact=16,diagnostic_only=True))
    print('CAPTURE COMPLETE',flush=True)
finally:
    import psutil
    try:life.stop(state)
    except psutil.NoSuchProcess:
        life.save('already-exited.json',dict(pid=state.get('pid'),reason='owned process exited before cleanup'))
