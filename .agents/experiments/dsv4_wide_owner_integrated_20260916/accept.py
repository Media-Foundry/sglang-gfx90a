"""Fail closed on formal service/numerical gates; no global default changes."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
repo=root.parents[2]
target=root/'acceptance.json'
assert not target.exists()
summary=json.loads((root/'summary.json').read_text())
assert summary['status']=='complete' and summary['gain_pct']>=2
assert abs(summary['control_drift_pct'])<1
assert summary['live_comparisons']==21*8*8
assert all(q['repeat_exact_out_of16']==[16]*3 for q in summary['quality'].values())
assert len(summary['cross_arm_continuations'])==16
assert all(q['common_prefix']==128 and q['distinct_outputs']==1
           for q in summary['cross_arm_continuations'])
assert len(summary['teacher_forced'])==2
assert all(t['positions']==1008 and t['max_abs_logprob']==0
           and t['top1_same']==t['top5_records_exact']==1008
           for t in summary['teacher_forced'])
for arm in ('check','A1','B','A2'):
    plan=json.loads((root/arm/'plan.json').read_text())
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h
               for p,h in plan['sources'].items())
    assert not json.loads((root/arm/f'P16-wide-owner-{arm}.stop.json').read_text())['remaining']
record=dict(status='accepted_explicit_16k_profile',
    summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
    launcher_sha256=hashlib.sha256((root/'validated-launcher.sh').read_bytes()).hexdigest(),
    scope='Original V4 TP8 C16x16K, original checkpoint/native AR, 1M logical KV',
    default_promoted=False, service_32k_validated=False,
    universal_batch_invariance_claimed=False)
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
