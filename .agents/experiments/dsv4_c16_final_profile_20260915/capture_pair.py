"""Fresh diagnostic of the accepted paired-column C16x8K configuration."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
from path_checks import check_paths

root = Path(__file__).resolve().parent
repo = root.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', default='capture-pair-current')
parser.add_argument('--validate-only', action='store_true')
args = parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9-]+', args.label)
trial = root.parent/'dsv4_c16_premix_pair_service_20260915'
summary = json.loads((trial/'summary.json').read_text())
review = json.loads((trial/'quality-review.json').read_text())
assert summary['gain_pct'] > 5 and review['manual_review_completed']
assert not review['novel_candidate_cases']
source = trial/'B'
assert json.loads((source/'P16-mix-pair-B.stop.json').read_text())['remaining'] == []
historical = json.loads((source/'plan.json').read_text())
assert historical['pair_columns'] and historical['mix_group_size'] == 8
assert historical['query_group_size'] == 16 and historical['runtime_m']
manifest = json.loads((source/'inputs.json').read_text())
assert len(manifest['requests']) == 16
assert sum(len(r['input_ids']) for r in manifest['requests']) == 131069
assert hashlib.sha256((source/'inputs.json').read_bytes()).hexdigest() == historical['input_sha256']
paths = set(historical['sources']) | {
    'python/sglang/srt/layers/dsv4_prefill_mhc_policy.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_pre.py',
    'python/sglang/srt/models/deepseek_v4.py',
    'python/sglang/kernels/ops/debug/dsv4_prefill_markers.py',
    'python/sglang/kernels/ops/debug/gfx90a_realtime_marker.py',
    'python/sglang/kernels/jit/csrc/debug/gfx90a_realtime_marker.cuh',
    str(Path(__file__).resolve().relative_to(repo)),
}
digests = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in sorted(paths)}
differences = {name: {'historical': digest, 'current': digests[name]}
               for name,digest in historical['sources'].items() if digest != digests[name]}
assert set(differences) <= {'scripts/rocm_dsv4_flash.sh'}, differences
if args.validate_only:
    print(json.dumps(dict(status='validated', input_tokens=131069, source_differences=differences),indent=2))
    raise SystemExit(0)

out = root/args.label
out.mkdir(exist_ok=False)
directory = out/'markers'
directory.mkdir()
helper = repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec = importlib.util.spec_from_file_location('pair_marker_life', helper)
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)
life.ROOT = life.OLD = out
launcher = (source/'start-ar-matrix.sh').read_text()
needle = 'exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle) == 1
flags = (f'export SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR={directory}\n'
         'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY=0\n'
         'export SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY=0\n'
         'unset SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle, flags+needle))
life.save('inputs.json', manifest)
life.save('plan.json', dict(diagnostic_only=True, historical_source_plan=historical,
    sources=digests, source_differences=differences, pair_columns=True,
    input_sha256=historical['input_sha256']))
state = life.start('P16-markers-B', 0)
try:
    life.ready(state)
    env = life.owned(state).environ()
    expected = {'PREFILL_MIX_PAIR_COLUMNS':'1', 'PREFILL_MIX_GROUP_SIZE':'8',
                'C4_PREFILL_QUERY_GROUP_SIZE':'16', 'C4_PREFILL_QUERY_RUNTIME_M':'1',
                'C4_PREFILL_QUERY_WIDE':'0', 'PREFILL_MHC_CONFIG_ITERS':'0',
                'PREFILL_MHC_COMB_REFINE20':'0', 'C4_PREFILL_EMPTY_TILE_SKIP':'1',
                'PREFILL_POST_FUSED4':'1', 'PREFILL_MIX_REUSE4':'1',
                'C4_PREFILL_QUERY_REUSE4':'1'}
    assert all(env['SGLANG_DSV4_'+k] == v for k,v in expected.items())
    assert env['SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR'] == str(directory)
    assert not env.get('SGLANG_DSV4_DEBUG_INDEXER_COMPILE_SHAPES')
    info = json.loads((out/'P16-markers-B.server-info.json').read_text())
    assert info['tp_size'] == 8 and info['ep_size'] == 1
    assert info['chunked_prefill_size'] == info['max_prefill_tokens'] == 32768
    assert info['max_total_tokens'] == 1048576 and info['speculative_algorithm'] is None
    life.save('runtime-contract.json', dict(expected=expected, diagnostic_only=True))
    for wave,name in enumerate(('warmup','trace1','trace2','trace3')):
        assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest() == digest for p,digest in digests.items())
        life.resources(name+'-before', life.owned(state))
        rids = [f'pair-markers-{name}-{i}' for i in range(16)]
        payload = dict(input_ids=[r['input_ids'] for r in manifest['requests']], rid=rids,
            cache_salt=[f'pair-markers-{name}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
        life.save(name+'-sent.json',payload)
        start = time.perf_counter()
        responses = life.post(life.URL+'/generate',payload,1800)
        wall = time.perf_counter()-start
        life.save(name+'-response.json',responses)
        by_id = {r['meta_info']['id']:r for r in responses}
        assert len(responses) == 16 and set(by_id) == set(rids)
        for request,rid in zip(manifest['requests'],rids,strict=True):
            response=by_id[rid]
            assert response['prompt_token_ids'] == request['input_ids']
            assert response['meta_info']['cached_tokens'] == 0
            assert len(life.completion_ids(response)) == 1
        expected_frames=8*4*(wave+1)
        deadline=time.monotonic()+30
        while len(list(directory.glob('rank-*-frame-*.json'))) < expected_frames and time.monotonic()<deadline:
            time.sleep(.1)
        frames=len(list(directory.glob('rank-*-frame-*.json')))
        assert frames == expected_frames, (frames,expected_frames)
        life.save(name+'-summary.json',dict(input_echo_exact=16,frames=frames,
            request_wall_s=wall,diagnostic_only=True))
        print('MARKER WAVE',name,frames,wall,flush=True)
        if wave==0:
            text=Path(state['log']).read_text()
            check_paths(text,True)
            for rank in range(8):
                assert any(f'TP{rank}]' in line and 'prefill mix-pair selected:' in line
                           and 'group=8 columns=2' in line for line in text.splitlines())
    life.save('complete.json',dict(input_echo_exact=64,frames=128,diagnostic_only=True))
finally:
    life.stop(state)
