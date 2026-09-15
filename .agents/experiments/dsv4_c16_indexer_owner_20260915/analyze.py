"""Gate query-owner work on actual replicated inputs, not assumed TP semantics."""
import hashlib
import json
from pathlib import Path
import torch
import argparse
import numpy as np
from sglang.kernels.ops.debug.dsv4_indexer_owner_capture import logical_rows,digest

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--capture',default='capture');p.add_argument('--output',default='analysis.json')
args=p.parse_args()
capture=root/args.capture;data=capture/'data'
output=root/args.output;assert not output.exists()
assert json.loads((capture/'complete.json').read_text())['records']==24
assert json.loads((capture/'P16-owner-audit.stop.json').read_text())['remaining']==[]
records={(rank,layer):json.loads((data/f'rank-{rank}-layer-{layer}.json').read_text())
         for layer in (2,20,42) for rank in range(8)}
samples={}
for key,r in records.items():
    assert (r['rank'],r['layer'])==key
    assert r['logical_decoder_version']==2
    packed_path=data/r['packed_file']
    assert hashlib.sha256(packed_path.read_bytes()).hexdigest()==r['packed_sha256']
    packed=torch.load(packed_path,weights_only=True)
    logical=[]
    for i,req in enumerate(packed['requests']):
        indices=np.searchsorted(packed['used_physical_pages'],req['page_ids'])
        keys,scales=logical_rows(packed['pages'].numpy()[indices],req['count'],r['preshuffle_tile'])
        logical.append(dict(request=i,rows=req['count'],keys_sha256=digest(keys),scale_bytes_sha256=digest(scales)))
    assert logical==r['logical_kv']
    path=data/r['sample_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sample_sha256']
    samples[key]=torch.load(path,weights_only=True)
def decoded_sample(key,name):
    info=records[key]['tensors'][name]
    dtype=getattr(torch,info['dtype'].split('.')[-1])
    raw=samples[key]['samples'][name]
    return raw.view(dtype).reshape(len(records[key]['sample_rows']),*info['shape'][1:]).float()

result={'diagnostic_only':True,'layers':{},'sample_metric_scope':'Subset only; zero sampled differences cannot override a differing full-tensor hash.'}
for layer in (2,20,42):
    ref=records[0,layer]
    metadata_fields=('rows','extend_lens','prefix_lens','input_ids_sha256','positions_sha256','seq_lens_sha256','sample_rows')
    metadata_exact=all(all(r[k]==ref[k] for k in metadata_fields) for (rank,l),r in records.items() if l==layer)
    assert metadata_exact, 'Cross-rank query row identity differs; investigate admission first'
    item={'metadata_exact':metadata_exact,
          'logical_kv_exact':all(records[rank,layer]['logical_kv']==ref['logical_kv'] for rank in range(8)),
          'physical_page_ids_equal':all(records[rank,layer]['used_physical_pages']==ref['used_physical_pages'] for rank in range(8)),
          'tensors':{}}
    for name in ('x','q_lora','q','weights'):
        groups={}
        for rank in range(8):groups.setdefault(records[rank,layer]['tensors'][name]['sha256'],[]).append(rank)
        metrics={}
        base=decoded_sample((0,layer),name)
        assert bool(torch.isfinite(base).all()), (layer,name,0)
        for rank in range(1,8):
            other=decoded_sample((rank,layer),name)
            assert bool(torch.isfinite(other).all()), (layer,name,rank)
            diff=other-base
            metrics[rank]=dict(max_abs=float(diff.abs().max()),
                relative_l2=float(diff.norm()/base.norm().clamp_min(1e-30)),
                numeric_mismatches=int((other!=base).sum()))
        item['tensors'][name]=dict(hash_groups=list(groups.values()),all_rank_full_hash_exact=len(groups)==1,sampled_metrics=metrics)
    result['layers'][layer]=item
ids=samples[0,2]['input_ids'].tolist()
manifest=json.loads((capture/'inputs.json').read_text())
offset=0;matched=[]
for length,prefix in zip(records[0,2]['extend_lens'],records[0,2]['prefix_lens'],strict=True):
    part=ids[offset:offset+length];offset+=length
    matches=[i for i,r in enumerate(manifest['requests']) if r['input_ids'][prefix:prefix+length]==part]
    assert len(matches)==1
    matched.append(matches[0])
assert offset==len(ids)
result['captured_request_cases']=matched
result['same_input_ids_across_layers']=len({r['input_ids_sha256'] for r in records.values()})==1
result['all_checked_replicated_values_exact']=all(l['logical_kv_exact'] and all(t['all_rank_full_hash_exact'] for t in l['tensors'].values()) for l in result['layers'].values())
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='layers'},indent=2))
for layer,r in result['layers'].items():
    print(layer,'logical_kv_exact',r['logical_kv_exact'],'tensor_groups',{k:v['hash_groups'] for k,v in r['tensors'].items()})
