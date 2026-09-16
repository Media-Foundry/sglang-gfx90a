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
    summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest())
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
