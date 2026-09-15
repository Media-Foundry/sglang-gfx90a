"""Recompute ABBA wave throughput from raw timestamps; quality stays separate."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

root=Path(__file__).resolve().parent
out=root/'summary.json';assert not out.exists()
plans=[json.loads((root/a/'plan.json').read_text()) for a in ('A1','B','A2')]
assert all(p['sources']==plans[0]['sources'] and p['input_sha256']==plans[0]['input_sha256'] for p in plans)
results={};quality={};hashes={};echoes=0
def read(path):
    hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
for arm,p in zip(('A1','B','A2'),plans):
    assert p['candidate']==(arm=='B') and not p['diagnostic']
    assert p['kv_tokens']==1048576 and p['prefill_budget']==32768 and p['original_weights']
    done=read(root/arm/'complete.json')
    assert read(root/arm/f'P16-owner-{arm}.stop.json')['remaining']==[]
    manifest=read(root/arm/'inputs.json')
    assert hashlib.sha256((root/arm/'inputs.json').read_bytes()).hexdigest()==p['input_sha256']
    n=sum(len(r['input_ids']) for r in manifest['requests']);assert n==131069
    for leg in (('B1','B2') if arm=='B' else (arm,)):
        data=read(root/arm/(leg+'.json'));assert len(data['rounds'])==3
        rates=[]
        for wave in data['rounds']:
            assert wave['input_echo_exact'] and wave['cached_tokens']==[0]*16 and wave['completion_lengths']==[1]*16
            assert wave['total_prompt_tokens']==n
            times=wave['raw_times'];assert len(times)==16
            assert all(t['begin']<=t['first']<=t['end'] for t in times)
            wall=max(t['first'] for t in times)-min(t['begin'] for t in times)
            assert math.isclose(wall,wave['prefill_wall_s'],abs_tol=1e-10)
            rate=n/wall;assert math.isclose(rate,wave['aggregate_input_tok_s'],rel_tol=1e-12)
            rates.append(rate);echoes+=16
        assert math.isclose(median(rates),data['median_input_tok_s'],rel_tol=1e-12)
        results[leg]=dict(rates=rates,median=median(rates))
    answers=[]
    for rep in range(2):
        byid={r['meta_info']['id']:r for r in read(root/arm/f'quality-{rep}.json')}
        ordered=[byid[f'owner-{arm}-{rep}-{i}'] for i in range(16)]
        for req,r in zip(manifest['requests'],ordered):
            assert req['input_ids']==r['prompt_token_ids'] and r['meta_info']['cached_tokens']==0
            assert len(r['output_ids'])==r['meta_info']['completion_tokens']==128
        answers.append(ordered)
    quality[arm]=dict(repeat_exact=sum(a['output_ids']==b['output_ids'] for a,b in zip(*answers)))
    assert quality[arm]['repeat_exact']==done['quality_repeat_exact']
control=(results['A1']['median']+results['A2']['median'])/2
candidate=(results['B1']['median']+results['B2']['median'])/2
result=dict(status='timing_closed_quality_review_pending',control=control,candidate=candidate,
    gain_pct=100*(candidate/control-1),legs=results,quality=quality,formal_input_echoes=echoes,
    artifact_sha256=hashes,sources=plans[0]['sources'],default_enabled=False)
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifact_sha256')},indent=2))
