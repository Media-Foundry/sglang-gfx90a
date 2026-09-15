"""Check request identity and compare tokens; repetition is only a review aid."""
import importlib.util
import json
from pathlib import Path

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
results={};table={}
for arm in ('A1','B'):
    directory=root/('quality-'+arm)
    results[arm]=json.loads((directory/'complete.json').read_text())
    manifest=json.loads((directory/'inputs.json').read_text())['requests']
    waves=[]
    for rep in range(2):
        responses=json.loads((directory/f'answers-{rep}.json').read_text())
        byid={r['meta_info']['id']:r for r in responses}
        ordered=[byid[f'h16-{arm}-{rep}-{i}'] for i in range(16)]
        assert all(a['prompt_token_ids']==b['input_ids'] for a,b in zip(ordered,manifest,strict=True))
        waves.append(ordered)
    table[arm]=waves
comparisons=[]
for i,(a,b) in enumerate(zip(table['A1'][1],table['B'][1],strict=True)):
    assert a['prompt_token_ids']==b['prompt_token_ids']
    x,y=life.completion_ids(a),life.completion_ids(b)
    prefix=next((j for j,(u,v) in enumerate(zip(x,y,strict=True)) if u!=v),len(x))
    comparisons.append(dict(request=i,exact=x==y,shared_prefix_tokens=prefix,
        control_text=a['text'],candidate_text=b['text']))
summary=dict(within_arm=results,between_arm_exact=sum(c['exact'] for c in comparisons),
    comparisons=comparisons,scope='128-token continuation checks, not arbitrary batch invariance or semantic oracle')
target=root/'quality-summary.json';assert not target.exists()
target.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2))
