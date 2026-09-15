"""Compare unique-store cached EXTEND to the verified fixed-slot reference."""
import hashlib,json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent;target=root/'check-analysis.json';assert not target.exists()
reference=root.parent/'dsv4_cached_fixed_drift_20260916/A'
candidate=root/'check'
for directory in (reference,candidate):
    assert json.loads((directory/'complete.json').read_text())['input_echo_exact']==1
a=reference/'data';b=candidate/'data';records=[]
plan_a=json.loads((reference/'plan.json').read_text());plan_b=json.loads((candidate/'plan.json').read_text())
changed_sources={p:(plan_a['sources'][p],plan_b['sources'][p]) for p in plan_a['sources'].keys() & plan_b['sources'].keys()
                 if plan_a['sources'][p]!=plan_b['sources'][p]}
assert set(changed_sources) <= {'python/sglang/kernels/ops/moe/gfx90a_ck_fixed_slot.py'}
assert plan_a['input_sha256']==plan_b['input_sha256']
for layer in range(43):
    for rank in range(8):
        stem=f'layer-{layer}-rank-{rank}'
        x,y=[torch.load(d/(stem+'-metadata.pt'),map_location='cpu',weights_only=True) for d in (a,b)]
        assert x['metadata']==y['metadata']
        for key in ('input_ids','positions'):assert torch.equal(x[key],y[key])
        assert x['metadata']['rows']==8192 and x['metadata']['prefix_lens']==[8192]
        for stage in plan_a['stages']:
            x,y=[json.loads((d/(stem+'-'+stage+'.json')).read_text()) for d in (a,b)]
            for key in ('is_none','shape','dtype','row_tensor'):assert x.get(key)==y.get(key)
            equal=x.get('sha256')==y.get('sha256')
            records.append(dict(layer=layer,rank=rank,stage=stage,equal=equal))
result=dict(diagnostic_only=True,comparisons=len(records),mismatches=[r for r in records if not r['equal']],
            source_differences=changed_sources,records=records,all_stage_equal=all(r['equal'] for r in records))
target.write_text(json.dumps(result,indent=2)+'\n')
print('comparisons',len(records),'mismatches',len(result['mismatches']))
assert result['all_stage_equal']
