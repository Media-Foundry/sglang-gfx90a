"""One fresh production-profile capture, then owned shutdown before replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent; repo=root.parents[2]
out=root/'capture'; out.mkdir(exist_ok=False); data=out/'fixture'
source=root.parent/'dsv4_c16_attention_stage1_service_20260915/B'
manifest=json.loads((root.parent/'dsv4_cached_chunk_drift_20260915/A/inputs.json').read_text())
assert len(manifest['requests'])==1 and len(manifest['requests'][0]['input_ids'])==16384
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('real_ck_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life);life.ROOT=life.OLD=out
launcher=(source/'start-ar-matrix.sh').read_text().replace('CHUNKED_PREFILL_SIZE=32768 MAX_PREFILL_TOKENS=32768', 'CHUNKED_PREFILL_SIZE=8192 MAX_PREFILL_TOKENS=8192')
needle='exec bash scripts/rocm_dsv4_flash.sh serve'; assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR={data}\n'
       'export SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_LAYER=1\n'
       'export SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_RANK=6\n'
       'export SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_MIN_PREFIX=8192\n'
       'unset SGLANG_DSV4_DEBUG_ATTN_PEER_CAPTURE_DIR\n'
       'unset SGLANG_DSV4_DEBUG_FIRST_DIV_DIR SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR\n'
       'unset SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
       'export SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK=0\n')
script=(repo/'scripts/rocm_dsv4_flash.sh').read_text()
serve_line='exec "${server_prefix[@]}" "${PYTHON_BIN}" -m sglang.launch_server "${server_args[@]}"'
assert script.count(serve_line)==1
diagnostic=out/'diagnostic-launcher.sh'
diagnostic.write_text(script.replace(serve_line,serve_line+' --watchdog-timeout 1800'))
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+f'exec bash {diagnostic} serve'))
life.save('inputs.json',manifest)
paths=set(json.loads((source/'plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/debug/dsv4_ck_stage_capture.py',
    'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py',
    'python/sglang/srt/models/deepseek_v4.py'}
hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(diagnostic_only=True,sources=hashes,
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
label='C1-cached-ck-fixture';state=life.start(label,0)
try:
    life.ready(state)
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_OWNER']=='1'
    assert env['SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR']==str(data)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==info['max_prefill_tokens']==8192
    assert info['ep_size']==1 and info['speculative_algorithm'] is None
    rids=[f'real-ck-{i}' for i in range(1)]
    payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
        cache_salt=[f'real-ck-{i}-{time.time_ns()}' for i in range(1)],
        sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
    life.save('sent.json',payload)
    result=life.post(life.URL+'/generate',payload,2400);life.save('responses.json',result)
    byid={r['meta_info']['id']:r for r in result};assert set(byid)==set(rids)
    for req,rid in zip(manifest['requests'],rids):
        response=byid[rid]
        assert response['prompt_token_ids']==req['input_ids'] and response['meta_info']['cached_tokens']==0
        assert len(life.completion_ids(response))==1
    meta=json.loads((data/'manifest.json').read_text())
    assert meta['provenance']['layer']==1 and meta['provenance']['rank']==6
    assert meta['provenance']['prefix_lens']==[8192] and meta['provenance']['extend_lens']==[8192]
    assert hashes=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in hashes}
    france=json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer=life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],cache_salt=f'real-ck-france-{time.time_ns()}',
        sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer);assert 'paris' in answer['text'].lower()
    life.save('complete.json',dict(input_echo_exact=1,diagnostic_only=True))
    print('CAPTURE COMPLETE',flush=True)
finally:
    import psutil
    try:life.stop(state)
    except psutil.NoSuchProcess:
        life.save('already-exited.json',dict(pid=state.get('pid'),reason='owned process exited before cleanup'))
