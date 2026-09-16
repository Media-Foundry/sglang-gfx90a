"""Observe exact-reference pre-mix across TP; no owner computation deployed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
repo = root.parents[2]
out = root/'audit'
out.mkdir(exist_ok=False)
helper = root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec = importlib.util.spec_from_file_location('owner_life', helper)
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)
sys.path.insert(0, str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life)
life.ROOT = life.OLD = out
prior = root.parent/'dsv4_mhc_mfma_service_20260916/A2'
manifest = json.loads((prior/'inputs.json').read_text())
assert len(manifest['requests']) == 16
assert sum(len(r['input_ids']) for r in manifest['requests']) == 131069
life.save('inputs.json', manifest)
launcher = (root.parent/'dsv4_mhc_post_tiles_20260916/validated-launcher.sh').read_text()
needle = 'exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle) == 1
script = (repo/'scripts/rocm_dsv4_flash.sh').read_text()
serve = 'exec "${server_prefix[@]}" "${PYTHON_BIN}" -m sglang.launch_server "${server_args[@]}"'
assert script.count(serve) == 1
(out/'diagnostic-launcher.sh').write_text(script.replace(serve, serve+' --watchdog-timeout 1800'))
flags = ('export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA=0\n'
         'export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA_CHECK=0\n'
         f'export SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_DIR={out}/data\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle, flags+f'exec bash {out}/diagnostic-launcher.sh serve'))
paths = set(json.loads((prior/'plan.json').read_text())['sources']) | {
    str(Path(__file__).relative_to(repo)),
    'python/sglang/kernels/ops/debug/dsv4_premix_owner_audit.py',
    'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_pair.py'}
sources = {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json', dict(sources=sources, diagnostic_only=True, output_replaced=False,
    kv_tokens=1048576, prefill_budget=32768, original_weights=True))
label = 'P16-premix-owner-audit'
state = life.start(label, 0)
try:
    life.ready(state)
    info = json.loads((out/(label+'.server-info.json')).read_text())
    assert info['chunked_prefill_size'] == info['max_prefill_tokens'] == 32768
    assert info['max_total_tokens'] == 1048576 and info['ep_size'] == 1
    env = life.owned(state).environ()
    for name,value in [('SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER','0'),
                       ('SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT','1'),
                       ('SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA','0'),
                       ('SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE','1')]:
        assert env[name] == value
    rids = [f'owner-audit-{i}' for i in range(16)]
    payload = dict(input_ids=[r['input_ids'] for r in manifest['requests']], rid=rids,
        cache_salt=[f'owner-audit-{i}-{time.time_ns()}' for i in range(16)],
        sampling_params=dict(temperature=0,max_new_tokens=1), return_prompt_token_ids=True)
    life.save('sent.json', payload)
    print('REQUEST SENT: diagnostic C16 real-code wave', flush=True)
    responses = life.post(life.URL+'/generate', payload, 2400)
    life.save('responses.json', responses)
    byid = {r['meta_info']['id']: r for r in responses}
    assert set(byid) == set(rids)
    for request,rid in zip(manifest['requests'],rids,strict=True):
        answer=byid[rid]
        assert answer['prompt_token_ids'] == request['input_ids']
        assert answer['meta_info']['cached_tokens'] == 0
        assert len(life.completion_ids(answer)) == 1
    records = [json.loads(p.read_text()) for p in sorted((out/'data').glob('*.json'))]
    calls = [0,40,84,85,125,169,170,210,254,255,295,339]
    assert {(r['call'],r['rank']) for r in records} == {(c,r) for c in calls for r in range(8)}
    comparison = []
    for call in calls:
        group = sorted([r for r in records if r['call']==call], key=lambda r:r['rank'])
        assert len({(r['rows'],r['eps']) for r in group}) == 1
        identity = {name:len({json.dumps(r['tensors'][name],sort_keys=True) for r in group}) == 1
                    for name in ('residual','fn','rms','mix')}
        comparison.append(dict(call=call, rows=group[0]['rows'], rank_identity=identity,
                               local_slices_exact=all(r['local_byte_exact'] for r in group)))
    life.save('comparison.json', comparison)
    france = json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer = life.post(life.URL+'/generate',dict(input_ids=france['input_ids'],
        cache_salt=f'owner-france-{time.time_ns()}',sampling_params=dict(temperature=0,max_new_tokens=32)),600)
    life.save('France.json',answer)
    assert 'paris' in answer['text'].lower()
    assert sources == {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json',dict(diagnostic_only=True, input_echo_exact=16, france_passed=True,
        records=len(records), all_rank_inputs_equal=all(all(r['rank_identity'].values()) for r in comparison),
        all_local_slices_exact=all(r['local_slices_exact'] for r in comparison)))
    print('AUDIT COMPLETE', comparison, flush=True)
finally:
    life.stop(state)
