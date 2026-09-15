"""Close completed paired-column ABBA from raw timestamps; never assert semantic quality."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
out = root / 'summary.json'
assert not out.exists(), 'Do not overwrite a completed analysis'
plans = [json.loads((root/a/'plan.json').read_text()) for a in ('A1', 'B', 'A2')]
assert all(p['sources'] == plans[0]['sources'] for p in plans)
assert all(p['input_sha256'] == plans[0]['input_sha256'] for p in plans)
for arm, plan in zip(('A1', 'B', 'A2'), plans, strict=True):
    assert plan['candidate'] == plan['pair_columns'] == (arm == 'B')
    assert plan['budget'] == 32768 and plan['kv_tokens'] == 1048576
    assert plan['mix_group_size'] == 8 and plan['query_group_size'] == 16
    assert plan['runtime_m'] and plan['original_weight']
    assert hashlib.sha256((root/arm/'inputs.json').read_bytes()).hexdigest() == plan['input_sha256']
results = {}
quality = {}
artifact_hashes = {}
echoes = 0
def read(path):
    artifact_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
for arm in ('A1', 'B', 'A2'):
    done = read(root/arm/'complete.json')
    stop = read(root/arm/f'P16-mix-pair-{arm}.stop.json')
    assert stop['remaining'] == []
    manifest = read(root/arm/'inputs.json')
    token_count = sum(len(q['input_ids']) for q in manifest['requests'])
    assert token_count == 131069
    for leg in (('B1', 'B2') if arm == 'B' else (arm,)):
        data = read(root/arm/(leg+'.json'))
        assert len(data['rounds']) == 3
        rates = []
        for wave in data['rounds']:
            assert wave['input_echo_exact'] and wave['cached_tokens'] == [0]*16
            assert wave['completion_lengths'] == [1]*16
            assert wave['total_prompt_tokens'] == token_count
            raw = wave['raw_times']
            assert len(raw) == 16 and all(r['begin'] <= r['first'] <= r['end'] for r in raw)
            wall = max(r['first'] for r in raw)-min(r['begin'] for r in raw)
            assert math.isclose(wall, wave['prefill_wall_s'], abs_tol=1e-10)
            rate = token_count / wall
            assert math.isclose(rate, wave['aggregate_input_tok_s'], rel_tol=1e-12)
            rates.append(rate)
            echoes += 16
        assert math.isclose(median(rates), data['median_input_tok_s'], rel_tol=1e-12)
        results[leg] = {'rates': rates, 'median': median(rates)}
    answers = []
    for rep in (0, 1):
        rows = read(root/arm/f'quality-{rep}.json')
        by_id = {r['meta_info']['id']: r for r in rows}
        ordered = [by_id[f'mix-pair-{arm}-{rep}-{i}'] for i in range(16)]
        for request, response in zip(manifest['requests'], ordered, strict=True):
            assert request['input_ids'] == response['prompt_token_ids']
            assert response['meta_info']['cached_tokens'] == 0
            assert response['meta_info']['completion_tokens'] == 128
            ids = response['output_ids']
            assert len(ids) == 128
        answers.append([r['output_ids'] for r in ordered])
    quality[arm] = {
        'within_arm_exact_outputs': sum(a == b for a,b in zip(*answers, strict=True)),
        'within_arm_exact_first_tokens': sum(a[0] == b[0] for a,b in zip(*answers, strict=True)),
        'manual_semantic_review': 'pending; not implied by hashes or France',
    }
    assert quality[arm]['within_arm_exact_outputs'] == done['quality_repeat_exact']
control = (results['A1']['median']+results['A2']['median'])/2
candidate = (results['B1']['median']+results['B2']['median'])/2
summary = dict(status='timing_closed_quality_review_pending', metric='new input tokens / wave TTFT; zero cache',
    legs=results, control=control, candidate=candidate, gain_pct=100*(candidate/control-1),
    formal_input_echoes=echoes, quality=quality, artifact_sha256=artifact_hashes,
    sources=plans[0]['sources'], default_enabled=False)
out.write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('artifact_sha256','sources')}, indent=2))
