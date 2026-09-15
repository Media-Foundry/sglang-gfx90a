"""Fresh post-query-owner C16x8K profile, same-rank envelope accounting."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import argparse

root=Path(__file__).resolve().parent;repo=root.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--producer',action='store_true',help='Profile default-off, non-bitwise owner-Q candidate')
args=parser.parse_args()
source=root.parent/('dsv4_c16_owner_producer_perf_20260915/B' if args.producer else 'dsv4_c16_indexer_owner_service_20260915/B')
accept=json.loads((source.parent/('analysis.json' if args.producer else 'acceptance.json')).read_text())
if args.producer:
    assert accept['candidate_center']>8200 and accept['formal_echoes']==192
    assert json.loads((source/'P16-producer-perf-B.stop.json').read_text())['remaining']==[]
else:
    assert accept['status']=='accepted_scoped_profile' and accept['candidate']>7900
historical=json.loads((source/'plan.json').read_text())
manifest=json.loads((source/'inputs.json').read_text())
assert len(manifest['requests'])==16 and sum(len(r['input_ids']) for r in manifest['requests'])==131069
paths=set(historical['sources'])|{
    'python/sglang/srt/models/deepseek_v4.py',
    'python/sglang/kernels/ops/debug/dsv4_prefill_markers.py',
    'python/sglang/kernels/ops/debug/gfx90a_realtime_marker.py',
    'python/sglang/kernels/jit/csrc/debug/gfx90a_realtime_marker.cuh',
    str(Path(__file__).relative_to(repo))}
hashes={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in sorted(paths)}
differences={name:dict(historical=digest,current=hashes[name])
             for name,digest in historical['sources'].items() if digest!=hashes[name]}
# Since acceptance: launcher promotion, width safety guard, and default-off probes.
assert set(differences)<={
    'scripts/rocm_dsv4_flash.sh',
    'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py',
    'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py',
    'python/sglang/srt/models/deepseek_v4.py',
    'python/sglang/srt/layers/attention/dsv4/indexer.py'},differences
out=root/('capture-producer-current' if args.producer else 'capture-owner-current');out.mkdir(exist_ok=False)
directory=out/'markers';directory.mkdir()
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('owner_marker_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life);life.ROOT=life.OLD=out
launcher=(source/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR={directory}\n'
       'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY=0\n'
       'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY=0\n'
       'unset SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES\n'
       'unset SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR SGLANG_DSV4_DEBUG_FIRST_DIV_DIR\n'
       'unset SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR\n'
       'unset SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR\n'
       f'export SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER={int(args.producer)}\n'
       'export SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK=0\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
life.save('inputs.json',manifest)
life.save('plan.json',dict(diagnostic_only=True,accepted_performance=accept,
    sources=hashes,source_differences=differences,producer_candidate=args.producer,
    input_sha256=hashlib.sha256((source/'inputs.json').read_bytes()).hexdigest()))
state=life.start('P16-markers-B',0)
try:
    life.ready(state)
    env=life.owned(state).environ()
    expected={'PREFILL_MIX_PAIR_COLUMNS':'1','PREFILL_MIX_GROUP_SIZE':'8',
              'C4_PREFILL_QUERY_GROUP_SIZE':'16','C4_PREFILL_QUERY_RUNTIME_M':'1',
              'C4_PREFILL_QUERY_OWNER':'1','C4_PREFILL_QUERY_WIDE':'0',
              'PREFILL_MHC_CONFIG_ITERS':'0','PREFILL_MHC_COMB_REFINE20':'0',
              'C4_PREFILL_EMPTY_TILE_SKIP':'1','PREFILL_POST_FUSED4':'1',
              'PREFILL_MIX_REUSE4':'1','C4_PREFILL_QUERY_REUSE4':'1',
              'DEBUG_PREFILL_OWNER_CHECK':'0','GFX90A_BF16_CK_FIXED_SLOT':'0'}
    expected['C4_PREFILL_QUERY_PRODUCER']=str(int(args.producer))
    expected['DEBUG_OWNER_PRODUCER_CHECK']='0'
    assert all(env['SGLANG_DSV4_'+k]==v for k,v in expected.items())
    assert env['SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR']==str(directory)
    info=json.loads((out/'P16-markers-B.server-info.json').read_text())
    assert info['tp_size']==8 and info['ep_size']==1 and info['speculative_algorithm'] is None
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    life.save('runtime-contract.json',dict(expected=expected,diagnostic_only=True))
    for wave,name in enumerate(('warmup','trace1','trace2','trace3')):
        assert hashes=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in hashes}
        life.resources(name+'-before',life.owned(state))
        rids=[f'owner-markers-{name}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'owner-markers-{name}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
        life.save(name+'-sent.json',payload)
        start=time.perf_counter();responses=life.post(life.URL+'/generate',payload,1800)
        wall=time.perf_counter()-start;life.save(name+'-response.json',responses)
        byid={r['meta_info']['id']:r for r in responses};assert len(responses)==16 and set(byid)==set(rids)
        for request,rid in zip(manifest['requests'],rids,strict=True):
            response=byid[rid]
            assert response['prompt_token_ids']==request['input_ids']
            assert response['meta_info']['cached_tokens']==0 and len(life.completion_ids(response))==1
        expected_frames=32*(wave+1);deadline=time.monotonic()+30
        while len(list(directory.glob('rank-*-frame-*.json')))<expected_frames and time.monotonic()<deadline:
            time.sleep(.1)
        frames=len(list(directory.glob('rank-*-frame-*.json')));assert frames==expected_frames
        life.save(name+'-summary.json',dict(input_echo_exact=16,frames=frames,request_wall_s=wall,diagnostic_only=True))
        print('MARKER WAVE',name,frames,wall,flush=True)
        if wave==0:
            text=Path(state['log']).read_text()
            for rank in range(8):
                mode='producer' if args.producer else 'owner'
                assert any(f'TP{rank}]' in line and f'prefill query-{mode} selected:' in line and 'width=2048' in line for line in text.splitlines())
                assert any(f'TP{rank}]' in line and 'prefill mix-pair selected:' in line and 'group=8 columns=2' in line for line in text.splitlines())
    life.save('complete.json',dict(input_echo_exact=64,frames=128,diagnostic_only=True))
finally:life.stop(state)
