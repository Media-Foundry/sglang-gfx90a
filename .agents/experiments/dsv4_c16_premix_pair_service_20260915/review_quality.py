"""Compare exact frozen inputs and output branches; not a semantic-quality verifier."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--confirm-bounded-review', action='store_true',
                    help='Only after reading all candidate texts and unique control alternatives.')
args = parser.parse_args()
output = root/'quality-review.json'
assert not output.exists()
waves = {}
inputs = []
sources = {}
for arm in ('A1', 'B', 'A2'):
    manifest = json.loads((root/arm/'inputs.json').read_text())
    inputs.append(manifest)
    for rep in (0, 1):
        path = root/arm/f'quality-{rep}.json'
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = json.loads(path.read_text())
        by_id = {r['meta_info']['id']: r for r in raw}
        rows = [by_id[f'mix-pair-{arm}-{rep}-{i}'] for i in range(16)]
        for request, row in zip(manifest['requests'], rows, strict=True):
            assert request['input_ids'] == row['prompt_token_ids']
            assert row['meta_info']['cached_tokens'] == 0
            assert len(row['output_ids']) == row['meta_info']['completion_tokens'] == 128
        waves[f'{arm}.{rep}'] = rows
assert inputs[0] == inputs[1] == inputs[2]
controls = ('A1.0', 'A1.1', 'A2.0', 'A2.1')
candidates = []
for name in ('B.0', 'B.1'):
    for i, row in enumerate(waves[name]):
        matches = [c for c in controls if row['output_ids'] == waves[c][i]['output_ids']]
        prefixes = {c: next((j for j,(a,b) in enumerate(zip(row['output_ids'], waves[c][i]['output_ids'], strict=True)) if a != b),128) for c in controls}
        candidates.append(dict(wave=name, case=i, matching_controls=matches,
                               common_prefix_tokens=prefixes, text=row['text']))
result = dict(identical_full_input_manifests=True, exact_echoes=96,
    actual_cached_tokens=[0]*16, candidates=candidates,
    novel_candidate_cases=[dict(wave=c['wave'],case=c['case']) for c in candidates if not c['matching_controls']],
    manual_review_completed=args.confirm_bounded_review,
    caveat='Matching IDs and cache counts do not establish numerical state identity. Read manual-review.md separately; generated code-bug claims are not verified.',
    sources=sources)
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'candidates'},indent=2))
