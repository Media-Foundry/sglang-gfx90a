"""Compare original and clean-rebuilt binaries, with identical model arithmetic."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
target = root / 'summary.json'
assert not target.exists()
spec = importlib.util.spec_from_file_location('analysis_life', root.parent / 'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)
check = json.loads((root / 'check/complete.json').read_text())
assert check['all_layers_exact'] and check['unique_ck']
logpath = root / 'check/P16-rebuilt-extras-check.service.log'
entries = re.findall(r'\[TP(\d+)\] H16 peer output exact: layer=(\d+) rows=(\d+)', logpath.read_text())
assert len(entries) == 1312
assert {(int(r), int(l)) for r, l, m in entries} == {(r, l) for r in range(8) for l in range(2, 43)}
assert all(n == 4 for n in collections.Counter((r, l) for r, l, m in entries).values())
results, plans, answers, quality = {}, [], {}, {}
for arm in ('A1', 'B', 'A2'):
    directory = root / arm
    done = json.loads((directory / 'complete.json').read_text())
    assert done['france_passed'] and done['unique_ck'] and not done['diagnostic']
    plans.append(json.loads((directory / 'plan.json').read_text()))
    assert not json.loads((directory / f'P16-rebuilt-extras-{arm}.stop.json').read_text())['remaining']
    for leg in done['progress']:
        if leg['leg'] != 'warmup':
            assert len(leg['rates']) == 3
            results[leg['leg']] = leg
    manifest = json.loads((directory / 'inputs.json').read_text())['requests']
    waves = []
    for rep in range(4):
        data = json.loads((directory / f'quality-{rep}.json').read_text())
        byid = {r['meta_info']['id']: r for r in data}
        ordered = [byid[f'corrected-{arm}-{rep}-{i}'] for i in range(16)]
        assert all(r['prompt_token_ids'] == m['input_ids'] for r, m in zip(ordered, manifest, strict=True))
        assert all(len(life.completion_ids(r)) == 128 for r in ordered)
        waves.append(ordered)
    answers[arm] = waves
    equal = [sum(life.completion_ids(a) == life.completion_ids(b) for a, b in zip(waves[0], w, strict=True)) for w in waves[1:]]
    assert equal == done['quality_repeat_exact']
    quality[arm] = dict(repeated_exact_out_of16=equal)
assert plans[0]['sources'] == plans[1]['sources'] == plans[2]['sources']
assert plans[0]['input_sha256'] == plans[1]['input_sha256'] == plans[2]['input_sha256']
cross = []
for i in range(16):
    tokens = [life.completion_ids(answers[a][r][i]) for a in ('A1', 'B', 'A2') for r in range(4)]
    common = next((j for j in range(128) if len({t[j] for t in tokens}) > 1), 128)
    cross.append(dict(request=i, all_twelve_exact=len({tuple(t) for t in tokens}) == 1,
                      common_prefix_tokens=common, distinct_outputs=len({tuple(t) for t in tokens})))
control = statistics.mean(results[n]['median'] for n in ('A1', 'A2'))
candidate = statistics.mean(results[n]['median'] for n in ('B1', 'B2'))
summary = dict(scope='Original versus clean-rebuilt unique CK and direct-HIP IPC; corrected sinks and H16 on both arms; original V4 TP8 C16x8K',
    control_input_tok_s=control, candidate_input_tok_s=candidate, gain_pct=100*(candidate/control-1),
    control_drift_pct=100*(results['A2']['median']/results['A1']['median']-1), legs=results,
    original_weights=True, kv_tokens=1048576, prefill_budget=32768, attention_exact_comparisons=1312,
    quality=quality, cross_arm_continuations=cross, all_model_batch_invariance_claimed=False,
    diagnostic_log_sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
target.write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
