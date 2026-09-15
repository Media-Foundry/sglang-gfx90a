"""First observed stage divergence, requiring identical complete batch inputs."""
import hashlib
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
ARMS=('A','B')
target=root/'analysis.json';assert not target.exists()
read=lambda p:json.loads(p.read_text())
plans=[read(root/arm/'plan.json') for arm in ARMS]
assert plans[0]['sources']==plans[1]['sources']
assert plans[0]['input_sha256']==plans[1]['input_sha256']
for arm in ARMS:
    assert read(root/arm/'complete.json')['input_echo_exact']==1
    assert read(root/arm/f'C1-cached-layer-{arm}.stop.json')['remaining']==[]

records=[];meta_differences=[];anchors={};cross_rank=[]
for layer in range(43):
    for rank in range(8):
        stem=f'layer-{layer}-rank-{rank}'
        dirs=[root/arm/'data' for arm in ARMS]
        metas=[torch.load(d/(stem+'-metadata.pt'),weights_only=True,map_location='cpu') for d in dirs]
        for arm,meta in zip(ARMS,metas,strict=True):
            if arm not in anchors:anchors[arm]=meta
            anchor=anchors[arm]
            assert meta['metadata']['rows']==8192 and meta['metadata']['prefix_lens']==[8192]
            assert torch.equal(meta['positions'],torch.arange(8192,16384,dtype=meta['positions'].dtype))
            for key in ('input_ids','positions'):
                assert torch.equal(meta[key],anchor[key]),(arm,layer,rank,key)
            for key in ('rows','extend_lens','prefix_lens','input_sha256','position_sha256'):
                assert meta['metadata'][key]==anchor['metadata'][key],(arm,layer,rank,key)
        if (metas[0]['metadata']!=metas[1]['metadata'] or
            any(not torch.equal(metas[0][k],metas[1][k]) for k in ('input_ids','positions'))):
            meta_differences.append(dict(layer=layer,rank=rank,
                A=metas[0]['metadata'],B=metas[1]['metadata']))
            continue
        for name in plans[0]['stages']:
            a,b=[read(d/f'{stem}-{name}.json') for d in dirs]
            for key in ('stage','sequence','is_none','shape','dtype','row_tensor'):
                assert a.get(key)==b.get(key),(stem,name,key)
            item=dict(layer=layer,rank=rank,stage=name,sequence=a['sequence'],
                      equal=a.get('sha256')==b.get('sha256'),is_none=a['is_none'])
            if a.get('row_tensor'):
                values=[]
                for d,record in zip(dirs,(a,b),strict=True):
                    p=d/record['values_file']
                    assert hashlib.sha256(p.read_bytes()).hexdigest()==record['file_sha256']
                    values.append(torch.load(p,weights_only=True,map_location='cpu'))
                mask=torch.any(values[0]['row_hashes']!=values[1]['row_hashes'],dim=1)
                changed=mask.nonzero().flatten().tolist()
                assert (not changed)==item['equal']
                item.update(changed_rows=len(changed),first_changed_rows=changed[:32])
                # Numeric samples are not full-tensor maxima.
                dtype=getattr(torch,a['dtype'].removeprefix('torch.'))
                x,y=[v['samples'].contiguous().view(dtype).float() for v in values]
                item['sample_max_abs']=float((x-y).abs().max())
            records.append(item)

records.sort(key=lambda r:(r['layer'],r['sequence'],r['rank']))
different=[r for r in records if not r['equal']]
for arm in ARMS:
    for layer in range(43):
        for stage in ('attn_norm','ffn_input','ffn_out'):
            hashes=[read(root/arm/'data'/f'layer-{layer}-rank-{rank}-{stage}.json').get('sha256') for rank in range(8)]
            if len(set(hashes))!=1:cross_rank.append(dict(arm=arm,layer=layer,stage=stage,hashes=hashes))
result=dict(scope=__doc__,same_full_inputs_positions_and_layout=not meta_differences,
    metadata_differences=meta_differences,records=records,
    first_observed_differences=different[:24],different_stage_rank_count=len(different),
    global_drift_solved=False,cross_rank_replicated_boundary_mismatches=cross_rank,
    limitation='Second8192-token prefill chunk, prefix8192; not cached decode; stage hashes locate a frontier, not an unobserved operation or final logits.')
target.write_text(json.dumps(result,indent=2)+'\n')
print('metadata mismatches',len(meta_differences),'differing stage/rank records',len(different))
for r in different[:20]:print(r)
