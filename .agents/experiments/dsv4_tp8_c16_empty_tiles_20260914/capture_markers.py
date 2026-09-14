"""Owned C16 marker run without Kineto; async readback, full input witnesses."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent;repo=root.parents[2]
source=root/'B';assert (source/'complete.json').exists()
out=root/'markers-B';out.mkdir(exist_ok=False)
directory=out/'markers';directory.mkdir()
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('marker_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
launcher=(source/'start-ar-matrix.sh').read_text()
launcher=launcher.replace('export SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP=1',
                          'unset SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP')
flags=(f'export SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR={directory}\n'
       'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY=0\n'
       'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY=0\n')
launcher=launcher.replace('exec bash scripts/rocm_dsv4_flash.sh serve',flags+'exec bash scripts/rocm_dsv4_flash.sh serve')
(out/'start-ar-matrix.sh').write_text(launcher)
manifest=json.loads((source/'inputs.json').read_text());life.save('inputs.json',manifest)
paths=('python/sglang/srt/model_executor/runner/eager_runner.py',
       'python/sglang/kernels/ops/debug/dsv4_prefill_markers.py',
       'python/sglang/kernels/ops/debug/gfx90a_realtime_marker.py',
       'python/sglang/kernels/jit/csrc/debug/gfx90a_realtime_marker.cuh')
life.save('plan.json',dict(diagnostic_only=True,kineto=False,
    sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths},
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
state=life.start('P16-markers-B',0)
try:
    life.ready(state)
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP']=='1'
    assert env['SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR']==str(directory)
    life.save('runtime-contract.json',dict(default_resolved='1',marker_dir=str(directory)))
    for wave,name in enumerate(('warmup','trace1','trace2','trace3')):
        life.resources(name+'-before',life.owned(state))
        rids=[f'markers-{name}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'markers-{name}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
        life.save(name+'-sent.json',payload)
        start=time.perf_counter()
        responses=life.post(life.URL+'/generate',payload,1800)
        wall=time.perf_counter()-start;life.save(name+'-response.json',responses)
        assert len(responses)==16
        by_id={r['meta_info']['id']:r for r in responses};assert set(by_id)==set(rids)
        for request,rid in zip(manifest['requests'],rids,strict=True):
            response=by_id[rid]
            assert response['prompt_token_ids']==request['input_ids']
            assert response['meta_info']['cached_tokens']==0
            assert len(life.completion_ids(response))==1
        deadline=time.monotonic()+30
        expected=8*4*(wave+1)
        while len(list(directory.glob('rank-*-frame-*.json')))<expected and time.monotonic()<deadline:
            time.sleep(.1)
        files=sorted(directory.glob('rank-*-frame-*.json'))
        assert len(files)==expected,('missing frame snapshots',len(files),expected)
        life.save(name+'-summary.json',dict(input_echo_exact=16,frames=len(files),
            request_wall_s=wall,diagnostic_only=True))
        print('MARKER WAVE',name,len(files),wall,flush=True)
    life.save('complete.json',dict(input_echo_exact=64,frames=128,diagnostic_only=True))
finally:
    life.stop(state)
