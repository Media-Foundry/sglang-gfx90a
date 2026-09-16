"""Validate longer-length ABBA without claiming new live-reference checks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

root=Path(__file__).resolve().parent
repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--length',choices=('16k','32k'),required=True)
args=parser.parse_args()
target=root/('acceptance-'+args.length+'.json')
assert not target.exists()
validator=root.parent/'dsv4_ck_route_producer_service_20260917/accept.py'
spec=importlib.util.spec_from_file_location('k32_acceptance',validator)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
summary_path=root/('summary-'+args.length+'.json')
summary=json.loads(summary_path.read_text())
# Reuse the strict finite-fixture numerical/performance guard. The reference
# comparison count is explicitly the prior8K check, not new16K/32K diagnostics.
normalized={**summary,'live_comparisons':summary['prior_8k_live_comparisons']}
screen=module.validate_summary(normalized)
if args.length=='32k':
    directory=root/'32k-prior-bridge'
    done=json.loads((directory/'complete.json').read_text())
    assert done['positions']==1008 and done['teacher_exact'] and not done['scored_performance']
    assert done['route_ranks']==list(map(str,range(8)))
    assert summary['historical_teacher_bridge']==done
    comparison=next(q for q in summary['teacher_forced'] if q['lhs']=='prior_accepted')
    assert comparison['rhs_artifact']=='32k-prior-bridge/teacher-forced.json'
    plan=json.loads((directory/'plan.json').read_text())
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['sources'].items())
    original=json.loads((root/'32k-B/plan.json').read_text())
    assert all(plan['sources'][p]==h for p,h in original['sources'].items())
    assert (directory/'start-ar-matrix.sh').read_bytes()==(root/'32k-B/start-ar-matrix.sh').read_bytes()
    assert not json.loads((directory/'P32k-route-prior-bridge.stop.json').read_text())['remaining']
    archive=json.loads((root.parent/'dsv4_owner_k32_20260916/archive-manifest.json').read_text())
    entry=next(q for q in archive['files'] if q['path']=='B/teacher-forced.json')
    reference=root.parent/'dsv4_owner_k32_20260916/B/teacher-forced.json'
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==entry['sha256']==plan['reference_sha256']
for arm in ('A1','B','A2'):
    directory=root/(args.length+'-'+arm)
    plan=json.loads((directory/'plan.json').read_text())
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h
               for p,h in plan['sources'].items())
    assert not json.loads((directory/f'P{args.length}-route-producer-regression-{arm}.stop.json').read_text())['remaining']
record=dict(status='accepted_explicit_'+args.length+'_profile',scope=summary['scope'],
    numerical_exact_on_tested_inputs=True,default_promoted=False,
    universal_batch_invariance_claimed=False,new_live_reference_checks=0,
    prior_8k_live_comparisons=summary['prior_8k_live_comparisons'],screening=screen,
    historical_teacher_bridge=(args.length=='32k'),
    summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest())
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
