"""Profile the completed runtime-M ABBA configuration, not an old launcher."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parent
repo = root.parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-arm', choices=('A1', 'B', 'A2'), default='B')
args = p.parse_args()
trial = root.parent / 'dsv4_c16_indexer_runtime_service_20260915'
summary = json.loads((trial / 'summary.json').read_text())
assert summary['identical_timed_forward_shape_counts']
assert [leg['name'] for leg in summary['legs']] == ['A1', 'B1', 'B2', 'A2']
source = trial / args.source_arm
plan = json.loads((source / 'plan.json').read_text())
assert (source / 'complete.json').exists()
assert json.loads((source / f'P16-qrunm-{args.source_arm}.stop.json').read_text())['remaining'] == []
assert plan['query_group_size'] == 16
assert plan['runtime_m'] == (args.source_arm == 'B')
if args.source_arm == 'B':
    assert summary['throughput_gain_percent'] > -1, 'Review runtime-M regression before profiling'
for name, digest in plan['sources'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name

out = root / 'capture'
out.mkdir(exist_ok=False)
directory = out / 'markers'
directory.mkdir()
helper = repo / '.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec = importlib.util.spec_from_file_location('final_marker_life', helper)
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)
life.ROOT = life.OLD = out
launcher = (source / 'start-ar-matrix.sh').read_text()
flags = (f'export SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR={directory}\n'
         'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY=0\n'
         'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY=0\n'
         'unset SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES\n')
needle = 'exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle) == 1
launcher = launcher.replace(needle, flags + needle)
(out / 'start-ar-matrix.sh').write_text(launcher)
manifest = json.loads((source / 'inputs.json').read_text())
assert len(manifest['requests']) == 16
assert sum(len(r['input_ids']) for r in manifest['requests']) == 131069
life.save('inputs.json', manifest)
marker_sources = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in (
    'python/sglang/kernels/ops/debug/dsv4_prefill_markers.py',
    'python/sglang/kernels/ops/debug/gfx90a_realtime_marker.py',
    'python/sglang/kernels/jit/csrc/debug/gfx90a_realtime_marker.cuh')}
life.save('plan.json', dict(diagnostic_only=True, source_arm=args.source_arm,
    source_plan=plan, marker_sources=marker_sources,
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
state = life.start('P16-markers-B', 0)
try:
    life.ready(state)
    env = life.owned(state).environ()
    for name in ('C4_PREFILL_EMPTY_TILE_SKIP', 'PREFILL_POST_FUSED4',
                 'PREFILL_MIX_REUSE4', 'C4_PREFILL_QUERY_REUSE4'):
        assert env[f'SGLANG_DSV4_{name}'] == '1'
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE'] == '16'
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M'] == str(int(plan['runtime_m']))
    assert env['SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR'] == str(directory)
    assert not env.get('SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES')
    info = json.loads((out/'P16-markers-B.server-info.json').read_text())
    assert info['tp_size'] == 8 and info['ep_size'] == 1
    assert info['model_path'] == '/home/pc/models/modelscope'
    assert info['speculative_algorithm'] is None
    assert info['max_total_tokens'] == 1048576
    assert info['chunked_prefill_size'] == info['max_prefill_tokens'] == 32768
    life.save('runtime-contract.json', dict(query_group=16, runtime_m=plan['runtime_m'],
        marker_dir=str(directory), diagnostic_only=True))
    for wave, name in enumerate(('warmup', 'trace1', 'trace2', 'trace3')):
        for path, digest in (plan['sources'] | marker_sources).items():
            assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == digest, path
        life.resources(name+'-before', life.owned(state))
        rids = [f'final-markers-{name}-{i}' for i in range(16)]
        payload = dict(input_ids=[r['input_ids'] for r in manifest['requests']], rid=rids,
            cache_salt=[f'final-markers-{name}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0, max_new_tokens=1), return_prompt_token_ids=True)
        life.save(name+'-sent.json', payload)
        start = time.perf_counter()
        responses = life.post(life.URL+'/generate', payload, 1800)
        wall = time.perf_counter()-start
        life.save(name+'-response.json', responses)
        assert len(responses) == 16
        by_id = {r['meta_info']['id']:r for r in responses}
        assert set(by_id) == set(rids)
        for request, rid in zip(manifest['requests'], rids, strict=True):
            response = by_id[rid]
            assert response['prompt_token_ids'] == request['input_ids']
            assert response['meta_info']['cached_tokens'] == 0
            assert len(life.completion_ids(response)) == 1
        expected = 8*4*(wave+1)
        deadline = time.monotonic()+30
        while len(list(directory.glob('rank-*-frame-*.json'))) < expected and time.monotonic() < deadline:
            time.sleep(.1)
        frames = len(list(directory.glob('rank-*-frame-*.json')))
        assert frames == expected, (frames, expected)
        life.save(name+'-summary.json', dict(input_echo_exact=16, frames=frames,
            request_wall_s=wall, diagnostic_only=True))
        print('MARKER WAVE', name, frames, wall, flush=True)
        if name == 'warmup':
            lines = Path(state['log']).read_text().splitlines()
            for rank in range(8):
                for hit in ('prefill empty tiles selected:', 'prefill post-fused4 selected:',
                            'prefill mix-reuse4 selected:'):
                    assert any(f'TP{rank}]' in line and hit in line for line in lines), (rank, hit)
                assert any(f'TP{rank}]' in line and 'prefill query-reuse4 selected' in line
                    and 'query_group=16' in line and f'runtime_m={int(plan["runtime_m"])}' in line
                    for line in lines), rank
    life.save('complete.json', dict(input_echo_exact=64, frames=128, diagnostic_only=True))
finally:
    life.stop(state)
