"""Full-hash frontier plus sampled errors; never call sampled equality exact."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sglang.kernels.ops.debug.dsv4_indexer_owner_capture import digest,logical_rows

root=Path(__file__).resolve().parent;output=root/'analysis.json';assert not output.exists()
plans=[json.loads((root/a/'plan.json').read_text()) for a in ('A','B')]
assert plans[0]['sources']==plans[1]['sources'] and plans[0]['input_sha256']==plans[1]['input_sha256']
layers=plans[0]['layers'];assert layers==plans[1]['layers']
records={};samples={}
for arm in ('A','B'):
    assert json.loads((root/arm/'complete.json').read_text())['records']==8*len(layers)
    assert json.loads((root/arm/f'P16-late-drift-{arm}.stop.json').read_text())['remaining']==[]
    for layer in layers:
        for rank in range(8):
            directory=root/arm/'data';r=json.loads((directory/f'rank-{rank}-layer-{layer}.json').read_text())
            assert r['logical_decoder_version']==2 and r['layer']==layer and r['rank']==rank
            for field,sha in [('sample_file','sample_sha256'),('packed_file','packed_sha256')]:
                assert hashlib.sha256((directory/r[field]).read_bytes()).hexdigest()==r[sha]
            pages=torch.load(directory/r['packed_file'],weights_only=True)
            logical=[]
            for i,request in enumerate(pages['requests']):
                ids=np.searchsorted(pages['used_physical_pages'],request['page_ids'])
                k,s=logical_rows(pages['pages'].numpy()[ids],request['count'],r['preshuffle_tile'])
                logical.append(dict(request=i,rows=request['count'],keys_sha256=digest(k),scale_bytes_sha256=digest(s)))
            assert logical==r['logical_kv']
            records[arm,layer,rank]=r
            samples[arm,layer,rank]=torch.load(directory/r['sample_file'],weights_only=True)

result=dict(scope='Two fresh accepted-owner services; first large forward only.',layers={},first_differing_checkpoint=None)
identity=('rows','extend_lens','prefix_lens','input_ids_sha256','positions_sha256','seq_lens_sha256','sample_rows')
reference=records['A',layers[0],0]
assert all(all(r[k]==reference[k] for k in identity) for r in records.values()),'Input/row identity mismatch; do not attribute as same-input drift'
for layer in layers:
    item=dict(within_arm={},cross_arm={})
    for arm in ('A','B'):
        base=records[arm,layer,0]
        item['within_arm'][arm]=dict(
            tensors={name:all(records[arm,layer,rank]['tensors'][name]['sha256']==base['tensors'][name]['sha256'] for rank in range(8)) for name in ('x','q_lora','q','weights')},
            logical_kv_exact=all(records[arm,layer,rank]['logical_kv']==base['logical_kv'] for rank in range(8)))
    for name in ('x','q_lora','q','weights'):
        exact=[];errors={}
        for rank in range(8):
            a,b=records['A',layer,rank],records['B',layer,rank]
            exact.append(a['tensors'][name]['sha256']==b['tensors'][name]['sha256'])
            dtype=getattr(torch,a['tensors'][name]['dtype'].split('.')[-1])
            v=[samples[arm,layer,rank]['samples'][name].view(dtype).float() for arm in ('A','B')]
            assert all(bool(torch.isfinite(t).all()) for t in v)
            delta=v[1]-v[0]
            errors[rank]=dict(sampled_max_abs=float(delta.abs().max()),
                sampled_relative_l2=float(delta.norm()/v[0].norm().clamp_min(1e-30)),
                sampled_changed=int((v[0]!=v[1]).sum()))
        item['cross_arm'][name]=dict(exact_ranks=sum(exact),sample_metrics=errors)
    item['cross_arm']['logical_kv_exact_ranks']=sum(records['A',layer,r]['logical_kv']==records['B',layer,r]['logical_kv'] for r in range(8))
    changed=any(item['cross_arm'][name]['exact_ranks']!=8 for name in ('x','q_lora','q','weights')) or item['cross_arm']['logical_kv_exact_ranks']!=8
    if changed and result['first_differing_checkpoint'] is None:result['first_differing_checkpoint']=layer
    result['layers'][layer]=item
result['identical_all_captured_input_metadata']=True
manifest=json.loads((root/'A/inputs.json').read_text());ids=samples['A',layers[0],0]['input_ids'].tolist();offset=0;cases=[]
for n,p in zip(reference['extend_lens'],reference['prefix_lens']):
    matches=[i for i,r in enumerate(manifest['requests']) if r['input_ids'][p:p+n]==ids[offset:offset+n]]
    assert len(matches)==1;cases.append(matches[0]);offset+=n
result['captured_cases']=cases
result['limitation']='Checkpoint frontier is not yet a first-operator attribution; sample metrics are not full-tensor bounds. Diagnostic synchronization can alter scheduling.'
output.write_text(json.dumps(result,indent=2)+'\n')
print('first differing checkpoint',result['first_differing_checkpoint'],'cases',cases,flush=True)
for layer,item in result['layers'].items():
    print(layer,{n:v['exact_ranks'] if isinstance(v,dict) else v for n,v in item['cross_arm'].items()},flush=True)
