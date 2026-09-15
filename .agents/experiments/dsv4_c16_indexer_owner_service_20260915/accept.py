"""Close source/lifecycle/exact-output evidence; manual prose is a separate gate."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent;repo=root.parents[2]
out=root/'acceptance.json';assert not out.exists()
summary=json.loads((root/'summary.json').read_text())
assert summary['gain_pct']>5 and summary['formal_input_echoes']==192
assert (root/'manual-review.md').exists()
for arm in ('check','A1','B','A2'):
    assert (root/arm/'complete.json').exists()
    assert json.loads((root/arm/f'P16-owner-{arm}.stop.json').read_text())['remaining']==[]
gpu=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in gpu for p in g.get('process_list',[]))
(root/'final-gpu.json').write_text(json.dumps(gpu,indent=2)+'\n')
tested=summary['sources']
helper='python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py'
assert hashlib.sha256((root/'tested_helper.py').read_bytes()).hexdigest()==tested[helper]
current={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in tested}
changes={p:dict(tested=tested[p],current=current[p]) for p in tested if current[p]!=tested[p]}
assert set(changes)=={helper,'scripts/rocm_dsv4_flash.sh'}
shapes=json.loads((root.parent/'dsv4_c16_indexer_owner_20260915/integrated-shapes.json').read_text())
assert shapes['status']=='complete' and shapes['invalid_truncated_score_width_rejected']
review=json.loads((root/'quality-review.json').read_text())
assert review['novel_candidate_cases']==[dict(wave='B.1',case=8)]
new=next(r for r in json.loads((root/'B/quality-1.json').read_text()) if r['meta_info']['id']=='owner-B-1-8')
historical=root.parent/'dsv4_c16_premix_pair_service_20260915/A1/quality-1.json'
old=next(r for r in json.loads(historical.read_text()) if r['prompt_token_ids']==new['prompt_token_ids'])
assert old['output_ids']==new['output_ids']
result=dict(status='accepted_scoped_profile',control=summary['control'],candidate=summary['candidate'],
    gain_pct=summary['gain_pct'],original_weights=True,logical_kv_tokens=1048576,
    current_control_output_matches=31,historical_control_match_for_remaining=True,
    historical_sha256=hashlib.sha256(historical.read_bytes()).hexdigest(),
    manual_review_sha256=hashlib.sha256((root/'manual-review.md').read_bytes()).hexdigest(),
    post_measurement_changes=changes,
    post_measurement_validation='14 CPU tests, bash syntax, AST equivalence except CPU width guard, and eight-rank shape checks passed.',
    global_drift_solved=False,all_gpus_released=True)
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
