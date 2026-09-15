"""Audit completed timing waves without treating an in-progress arm as complete."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
root=Path(__file__).resolve().parent
waves=[];sources={};manifest=None
for arm,names in [('A1',['warmup','A1']),('B',['warmup','B1','B2']),('A2',['warmup','A2'])]:
    mpath=root/arm/'inputs.json'
    if not mpath.exists():continue
    current=json.loads(mpath.read_text())
    if manifest is None:manifest=current
    assert current==manifest
    for name in names:
        path=root/arm/(name+'.json')
        if not path.exists():continue
        # A concurrent incomplete JSON write is a sampling failure, not evidence
        # of empty outputs or a stopped service. Re-run with a new output path.
        raw=path.read_bytes();data=json.loads(raw)
        sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
        for rep,row in enumerate(data['rounds']):
            records=row['responses'];assert len(records)==16
            answers=[]
            for case,(request,response) in enumerate(zip(manifest['requests'],records,strict=True)):
                assert response['case']==case
                value=response['response'];meta=value['meta_info']
                assert value['prompt_token_ids']==request['input_ids']
                assert meta['cached_tokens']==row['cached_tokens'][case]
                assert meta['completion_tokens']==len(value['output_ids'])==1
                answers.append(dict(token=value['output_ids'][0],text=value['text']))
            waves.append(dict(arm=arm,leg=name,round=rep,cached_tokens=row['cached_tokens'],
                input_echo_exact=16,answers=answers,newly_computed_tok_s=row['newly_computed_tok_s']))
assert waves
reference=waves[0];comparisons=[]
for wave in waves[1:]:
    assert wave['cached_tokens']==reference['cached_tokens']
    changes=[dict(case=i,cached_tokens=wave['cached_tokens'][i],reference=a,current=b)
             for i,(a,b) in enumerate(zip(reference['answers'],wave['answers'],strict=True))
             if a['token']!=b['token']]
    comparisons.append(dict(arm=wave['arm'],leg=wave['leg'],round=wave['round'],
        first_exact=16-len(changes),changes=changes))
result=dict(scope='Partial completed timing waves only. Same full input IDs and cache counts, not cache-value equality or numerical cause attribution.',
    complete_abba=False,waves=waves,comparisons_to_first_warmup=comparisons,source_sha256=sources)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(waves=len(waves),comparisons=comparisons),indent=2))
