"""Aggregate formal H16 service ABBA; keep diagnostic timing out of performance."""
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
target=root/'summary.json';assert not target.exists()
check=json.loads((root/'check/validated.json').read_text())
assert check['all_eligible_layers_exact'] and check['exact_comparisons']==1312
plans=[];results={};timings={}
for arm in ('A1','B','A2'):
    directory=root/arm
    complete=json.loads((directory/'complete.json').read_text())
    assert not complete['diagnostic'] and complete['france_passed']
    plans.append(json.loads((directory/'plan.json').read_text()))
    label='P16-h16-'+arm
    assert not json.loads((directory/(label+'.stop.json')).read_text())['remaining']
    for leg in complete['progress']:
        if leg['leg']=='warmup':continue
        assert len(leg['rates'])==3
        data=json.loads((directory/(leg['leg']+'.json')).read_text())
        assert data['request_count']==16 and data['tokens']==1
        assert all(r['input_echo_exact'] and r['cached_tokens']==[0]*16 and
            r['total_prompt_tokens']==131069 for r in data['rounds'])
        results[leg['leg']]=leg
        timings[leg['leg']]=[r['prefill_wall_s'] for r in data['rounds']]
assert plans[0]['sources']==plans[1]['sources']==plans[2]['sources']
assert plans[0]['input_sha256']==plans[1]['input_sha256']==plans[2]['input_sha256']
control=statistics.mean(results[n]['median'] for n in ('A1','A2'))
candidate=statistics.mean(results[n]['median'] for n in ('B1','B2'))
output=dict(scope='Original V4, TP8/EP1 native AR, C16 x8K, zero prefix hit, one output token; prefill wave throughput',
    original_weights=True,kv_tokens=1048576,prefill_budget=32768,
    control_input_tok_s=control,candidate_input_tok_s=candidate,gain_pct=100*(candidate/control-1),
    control_drift_pct=100*(results['A2']['median']/results['A1']['median']-1),
    legs=results,wave_times_s=timings,attention_exact_checks=1312,
    whole_model_batch_invariance_claimed=False,default_enabled=False,
    check_sha256=hashlib.sha256((root/'check/validated.json').read_bytes()).hexdigest())
target.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
