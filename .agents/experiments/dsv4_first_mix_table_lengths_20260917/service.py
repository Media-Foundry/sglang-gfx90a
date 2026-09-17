"""Long-context regression of the exact first-MHC table against FP32 reference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
repo = root.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--length', choices=('16k', '32k'), required=True)
parser.add_argument('--arm', choices=('A1', 'B', 'A2'), required=True)
args = parser.parse_args()
candidate = args.arm == 'B'
prior = root.parent/'dsv4_first_mix_table_service_20260917'
assert json.loads((prior/'acceptance.json').read_text())['status'] == 'accepted_for_measured_8k_profile'
out = root/args.length/args.arm
out.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('length_life', root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)
sys.path.insert(0, str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life)
life.ROOT = life.OLD = out

fixtures = root.parent/'dsv4_c16_query_wide_20260915'
manifest_path = fixtures/f'inputs{args.length}'/'prefill.json'
validation = json.loads((fixtures/'input-validation.json').read_text())[f'inputs{args.length}']
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == validation['manifest_sha256']
assert validation['official_chat_encoding_exact'] == 16
manifest = json.loads(manifest_path.read_text())
total = sum(len(r['input_ids']) for r in manifest['requests'])
assert len(manifest['requests']) == 16 and total == validation['total_tokens']
life.save('inputs.json', manifest)
launcher = (prior/'B/start-ar-matrix.sh').read_text()
needle = 'exec bash scripts/rocm_dsv4_flash.sh serve'
assert launcher.count(needle) == 1
flags = f'export SGLANG_DSV4_PREFILL_FIRST_MIX_TABLE={int(candidate)}\n'
flags += 'export SGLANG_DSV4_DEBUG_FIRST_MIX_TABLE_CHECK=0\n'
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle, flags+needle))
paths = set(json.loads((prior/'B/plan.json').read_text())['sources'])
paths.add(str(Path(__file__).relative_to(repo)))
sources = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json', dict(sources=sources, candidate=candidate, diagnostic=False,
    kv_tokens=1048576, length=args.length, total_tokens=total, chunk=32768,
    first_mhc='original FP32 Fn in both arms', route_producer=True))
label = f'P{args.length}-first-table-{args.arm}'
state = life.start(label, 0)
try:
    life.ready(state)
    info = json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens'] == 1048576
    assert info['chunked_prefill_size'] == info['max_prefill_tokens'] == 32768
    assert info['ep_size'] == 1
    env = life.owned(state).environ()
    required = ('SGLANG_DSV4_PREFILL_MHC_COMMON_FP32', 'SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL',
        'SGLANG_DSV4_PREFILL_MHC_COMMON_UNFUSED', 'SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER',
        'SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE', 'SGLANG_DSV4_C4_PREFILL_OWNER_K32')
    assert all(env[k] == '1' for k in required)
    assert env['SGLANG_DSV4_PREFILL_FIRST_MIX_TABLE'] == str(int(candidate))
    assert env['SGLANG_DSV4_DEBUG_FIRST_MIX_TABLE_CHECK'] == '0'
    assert not env.get('SGLANG_DSV4_DEBUG_FIRST_DIV_DIR')
    france = json.loads((repo/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
    answer = life.post(life.URL+'/generate', dict(input_ids=france['input_ids'],
        cache_salt=f'first-table-france-{time.time_ns()}',
        sampling_params=dict(temperature=0, max_new_tokens=32)), 600)
    life.save('France.json', answer)
    assert 'paris' in answer['text'].lower()
    client = root.parent/'dsv4_c16_premix_pair_service_20260915/client.py'
    progress = []
    legs = [('warmup', 1)] + [(n, 3) for n in (('B1', 'B2') if candidate else (args.arm,))]
    for name, rounds in legs:
        life.resources(name+'-before', life.owned(state))
        command = [sys.executable, str(client), '--base-url', life.URL, '--inputs', str(out/'inputs.json'),
            '--request-count', '16', '--rounds', str(rounds), '--tokens', '1', '--output', str(out/(name+'.json'))]
        with (out/(name+'.client.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
        data = json.loads((out/(name+'.json')).read_text())
        assert len(data['rounds']) == rounds
        for wave in data['rounds']:
            assert wave['input_echo_exact'] and wave['total_prompt_tokens'] == total
            assert wave['cached_tokens'] == [0]*16 and wave['completion_lengths'] == [1]*16
        progress.append(dict(leg=name, median=data['median_input_tok_s'],
            rates=[w['aggregate_input_tok_s'] for w in data['rounds']]))
        life.save('progress.json', progress)
        print('RESULT', args.length, args.arm, progress[-1], flush=True)
    timing_log = Path(state['log']).read_text()
    life.save('timing-log.json', dict(text=timing_log))
    for rank in range(8):
        assert f'route-producer CK selected: rank={rank} ' in timing_log
        assert f'[TP{rank}] owner-K32 logits selected:' in timing_log
        widths = re.findall(rf'\[TP{rank}\] prefill query-owner selected: rows=\d+ local_rows=\d+ width=(\d+)', timing_log)
        assert widths and max(map(int, widths)) > 2048
    quality = []
    for rep in range(2):
        rids = [f'quality-{args.arm}-{rep}-{i}' for i in range(16)]
        values = life.post(life.URL+'/generate', dict(input_ids=[r['input_ids'] for r in manifest['requests']],
            rid=rids, cache_salt=[f'length-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            return_prompt_token_ids=True,
            sampling_params=dict(temperature=0, max_new_tokens=128, ignore_eos=True)), 1800)
        life.save(f'quality-{rep}.json', values)
        byid = {r['meta_info']['id']: r for r in values}
        assert set(byid) == set(rids)
        rows = [byid[rid] for rid in rids]
        for i, row in enumerate(rows):
            assert row['prompt_token_ids'] == manifest['requests'][i]['input_ids']
            assert row['meta_info']['cached_tokens'] == 0 and len(life.completion_ids(row)) == 128
        quality.append(rows)
        print('QUALITY', args.length, args.arm, rep, flush=True)
    ref = {r['meta_info']['id']: r for r in json.loads((root/args.length/'A1/quality-0.json').read_text())}
    prompts = [r['input_ids']+life.completion_ids(ref[f'quality-A1-0-{i}'])[:64]
               for i, r in enumerate(manifest['requests'])]
    rids = [f'teacher-{args.arm}-{i}' for i in range(16)]
    teacher = life.post(life.URL+'/generate', dict(input_ids=prompts, rid=rids,
        cache_salt=[f'length-teacher-{i}-{time.time_ns()}' for i in range(16)], return_prompt_token_ids=True,
        sampling_params=dict(temperature=0, max_new_tokens=1), return_logprob=True,
        logprob_start_len=[len(r['input_ids']) for r in manifest['requests']], top_logprobs_num=5), 1800)
    life.save('teacher-forced.json', teacher)
    byid = {r['meta_info']['id']: r for r in teacher}
    assert set(byid) == set(rids)
    for i, rid in enumerate(rids):
        row = byid[rid]
        assert row['prompt_token_ids'] == prompts[i] and row['meta_info']['cached_tokens'] == 0
        assert len(row['meta_info']['input_token_logprobs']) == 64
    logs = Path(state['log']).read_text()
    built = re.findall(r'first-mix table built: rank=(\d+) local_vocab=(\d+) padded_vocab=(\d+) bytes=(\d+)', logs)
    assert 'first-mix table exact:' not in logs and 'first-mix table consumed:' not in logs
    if candidate:
        assert len(built) == 8 and {r[0] for r in built} == set(map(str, range(8)))
        assert all(int(r[3]) == 12410880 for r in built)
    else:
        assert not built
    assert sources == {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
    life.save('complete.json', dict(status='complete', diagnostic=False, progress=progress,
        france_passed=True, built=built, quality_repeat_exact=sum(life.completion_ids(a) == life.completion_ids(b)
            for a, b in zip(*quality, strict=True))))
finally:
    life.stop(state)
