"""Coverage of same-live-input attention equality; never a throughput result."""
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent;out=root/'service-v2'
done=json.loads((out/'complete.json').read_text())
assert done['diagnostic'] and done['quality_echoes']==64 and done['first_warm_echoes']==0
assert json.loads((out/'P16-long-attn-check.stop.json').read_text())['remaining']==[]
config=json.loads(Path('/home/pc/models/modelscope/config.json').read_text())
ratios=config['compress_ratios'][:config['num_hidden_layers']]
assert len(ratios)==43
report=[];total=0
for item in done['progress']:
    length,rep,checks=item['length'],item['rep'],item['checks']
    expected_forwards={16:8,32:16}[length]
    assert len(checks)==expected_forwards*43*8
    for rank in range(8):
        for layer,ratio in enumerate(ratios):
            subset=[r for r in checks if r[0]==rank and r[1]==layer]
            assert len(subset)==expected_forwards and all(r[2]==ratio and 8192<=r[3]<=65536 for r in subset)
    total+=len(checks)
    report.append(dict(length=length,rep=rep,exact_checks=len(checks),all_ranks_all_layers=True,
                       actual_rows=sorted({r[3] for r in checks})))
assert total==16512 and len(report)==4
log=(out/'P16-long-attn-check.service.log').read_text()
assert len(re.findall('prefill attention-stage1 checked:',log))==total
prior=root.parent/'dsv4_c16_wide_prewarm_service_20260915/service'
comparisons=[];hashes={}
for length in (16,32):
    manifest=json.loads((out/f'inputs{length}k.json').read_text())
    assert manifest==json.loads((prior/f'inputs{length}k.json').read_text())
    waves=[];old=[]
    for rep in range(2):
        path=out/f'{length}k-quality-{rep}.json';rows=json.loads(path.read_text())
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=sorted(rows,key=lambda r:int(r['meta_info']['id'].rsplit('-',1)[-1]));waves.append(rows)
        previous=json.loads((prior/f'{length}k-quality-{rep}.json').read_text())
        old.append(sorted(previous,key=lambda r:int(r['meta_info']['id'].rsplit('-',1)[-1])))
        assert len(rows)==16
        for request,response in zip(manifest['requests'],rows,strict=True):
            assert response['prompt_token_ids']==request['input_ids'] and response['meta_info']['cached_tokens']==0
            assert len(response['output_ids'])==response['meta_info']['completion_tokens']==128
    equal=lambda a,b:sum(x['output_ids']==y['output_ids'] for x,y in zip(a,b))
    comparisons.append(dict(length=length,diagnostic_repeat=equal(*waves),previous_repeat=equal(*old),
                            cross_run_matches=[[equal(a,b) for b in old] for a in waves]))
result=dict(scope=__doc__,exact_attention_calls=total,input_echoes=64,coverage=report,
            comparisons=comparisons,quality_hashes=hashes,global_drift_solved=False)
target=root/'analysis.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
