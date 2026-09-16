"""Fail-closed shorter-input regression acceptance; not a new speedup claim."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
repo=root.parents[2]
target=root/'acceptance.json'
assert not target.exists()
reports={}
for length in ('8k','16k'):
    summary=json.loads((root/f'summary-{length}.json').read_text())
    assert summary['status']=='complete'
    assert summary['gain_pct']>=-2 and abs(summary['control_drift_pct'])<1
    assert len(summary['cross_arm_continuations'])==16
    assert all(v['common_prefix']==128 and v['distinct_outputs']==1 for v in summary['cross_arm_continuations'])
    assert all(v['repeat_exact_out_of16']==[16]*3 for v in summary['quality'].values())
    assert len(summary['teacher_forced'])==2
    for t in summary['teacher_forced']:
        assert t['positions']==t['top1_same']==t['top5_records_exact']==1008
        assert t['max_abs_logprob']==t['mean_abs_logprob']==0
        assert t['excluded_leading_nulls']==16
    for arm in ('A1','B','A2'):
        folder=root/(length+'-'+arm)
        plan=json.loads((folder/'plan.json').read_text())
        assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['sources'].items())
        assert not json.loads((folder/f'P{length}-common-regression-{arm}.stop.json').read_text())['remaining']
        assert not summary['timing_paths'][arm]['legacy_splitk']
        assert summary['timing_paths'][arm]['premix_owner_ranks']==list(map(str,range(8)))
    a=(root/(length+'-A1')/'start-ar-matrix.sh').read_text()
    b=(root/(length+'-B')/'start-ar-matrix.sh').read_text()
    assert a==(root/(length+'-A2')/'start-ar-matrix.sh').read_text()
    assert a.replace('export SGLANG_DSV4_PREFILL_MHC_COMMON_FP32=0',
                     'export SGLANG_DSV4_PREFILL_MHC_COMMON_FP32=1')==b
    reports[length]=dict(candidate_input_tok_s=summary['candidate_input_tok_s'],
        control_input_tok_s=summary['control_input_tok_s'],delta_pct=summary['gain_pct'],
        control_drift_pct=summary['control_drift_pct'],
        summary_sha256=hashlib.sha256((root/f'summary-{length}.json').read_bytes()).hexdigest())
assert (root/'8k-B/start-ar-matrix.sh').read_bytes()==(root/'16k-B/start-ar-matrix.sh').read_bytes()
record=dict(status='passed_shorter_input_common_mhc_regression',reports=reports,
    scope='Original V4 TP8 C16x8K/16K native AR, original weights,1M KV,32K chunk',
    default_promoted=False,universal_batch_invariance_claimed=False,
    prior_32k_checkpoint='34fa7b0e9c',
    caveat='Regression only; does not certify arbitrary batch shapes, prefixes or answer correctness.')
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
