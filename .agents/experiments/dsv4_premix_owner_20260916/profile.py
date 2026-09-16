"""Non-scoring same-rank envelope profile of the 9.975k exact-owner checkpoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parent;repo=root.parents[2]
out=root/'profile';out.mkdir(exist_ok=False)
summary=json.loads((root/'summary.json').read_text())
assert summary['gain_pct']>10 and all(c['distinct_outputs']==1 for c in summary['cross_arm_continuations'])
directory=out/'markers';directory.mkdir()
spec=importlib.util.spec_from_file_location('owner_profile_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life);life.ROOT=life.OLD=out
source=root/'B';manifest=json.loads((source/'inputs.json').read_text())
launcher=(source/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR={directory}\n'
       'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY=0\n'
       'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY=0\n'
       'unset SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
life.save('inputs.json',manifest)
paths=set(json.loads((source/'plan.json').read_text())['sources'])|{
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/debug/dsv4_prefill_markers.py',
    'python/sglang/kernels/ops/debug/gfx90a_realtime_marker.py',
    'python/sglang/kernels/jit/csrc/debug/gfx90a_realtime_marker.cuh'}
hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
life.save('plan.json',dict(diagnostic_only=True,sources=hashes,reference_summary=summary))
label='P16-markers-owner';state=life.start(label,0)
try:
    life.ready(state)
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==32768
    env=life.owned(state).environ()
    for key,value in [('SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT','1'),
                      ('SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE','1'),
                      ('SGLANG_DSV4_DEBUG_CK_REDUCE_VEC4','1'),
                      ('SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER','1'),
                      ('SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER_CHECK','0'),
                      ('SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA','0')]:
        assert env[key]==value
    for wave,name in enumerate(('warmup','trace1','trace2','trace3')):
        assert hashes=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
        life.resources(name+'-before',life.owned(state))
        rids=[f'owner-markers-{name}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'owner-markers-{name}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
        life.save(name+'-sent.json',payload)
        start=time.perf_counter();responses=life.post(life.URL+'/generate',payload,1800)
        wall=time.perf_counter()-start;life.save(name+'-response.json',responses)
        byid={r['meta_info']['id']:r for r in responses};assert set(byid)==set(rids)
        for request,rid in zip(manifest['requests'],rids,strict=True):
            response=byid[rid]
            assert response['prompt_token_ids']==request['input_ids'] and response['meta_info']['cached_tokens']==0
            assert len(life.completion_ids(response))==1
        expected=32*(wave+1);deadline=time.monotonic()+30
        while len(list(directory.glob('rank-*-frame-*.json')))<expected and time.monotonic()<deadline:time.sleep(.1)
        assert len(list(directory.glob('rank-*-frame-*.json')))==expected
        life.save(name+'-summary.json',dict(input_echo_exact=16,frames=expected,request_wall_s=wall,diagnostic_only=True))
        print('MARKER WAVE',name,expected,wall,flush=True)
    logs=Path(state['log']).read_text()
    for rank in range(8):
        assert f'[TP{rank}] pre-mix owner selected:' in logs
        assert f'[TP{rank}] H16 peer selected:' in logs
        assert f'unique-slot CK selected: rank={rank} ' in logs
    life.save('complete.json',dict(input_echo_exact=64,frames=128,diagnostic_only=True))
finally:
    life.stop(state)
