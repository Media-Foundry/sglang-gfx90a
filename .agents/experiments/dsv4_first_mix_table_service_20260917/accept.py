"""Fail closed: table performance does not excuse any observed numerical drift."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
target=root/'acceptance.json';assert not target.exists()
summary=json.loads((root/'summary.json').read_text())
assert summary['status']=='complete' and summary['live_exact_checks']>=40
assert len(summary['teacher_checks'])==2
assert all(t['positions']==1008 and t['max_abs']==0 and t['top5_exact']==1008 for t in summary['teacher_checks'])
assert all(q['distinct_outputs']==1 for q in summary['quality_cross'])
assert summary['gain_pct']>0
prior=root.parent/'dsv4_first_mhc_service_20260917/B/quality.json'
old={r['meta_info']['id']:r for r in json.loads(prior.read_text())}
new={r['meta_info']['id']:r for r in json.loads((root/'B/quality-0.json').read_text())}
for i in range(16):
    a,b=old[f'quality-B-{i}'],new[f'quality-B-0-{i}']
    assert len(a['output_ids'])==len(b['output_ids'])==128
    assert a['prompt_token_ids']==b['prompt_token_ids'] and a['output_ids']==b['output_ids']
record=dict(status='accepted_for_measured_8k_profile',candidate_input_tok_s=summary['candidate_input_tok_s'],
    gain_pct=summary['gain_pct'],corrected_reference_outputs_exact=16,
    summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
    launcher=str(root/'B/start-ar-matrix.sh'),launcher_sha256=hashlib.sha256((root/'B/start-ar-matrix.sh').read_bytes()).hexdigest(),
    limitations=['Not universal batch invariance','Longer-length regression pending',
                 'Static original-checkpoint experimental profile; online weight updates rejected',
                 'Global launcher default unchanged'])
target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
