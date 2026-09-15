"""Find differing VALID logical cache rows from saved full rank0 pages."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import argparse
from sglang.kernels.ops.debug.dsv4_indexer_owner_capture import logical_rows

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',default='kv-rows.json')
args=parser.parse_args();out=root/args.output;assert not out.exists()
analysis=json.loads((root/'analysis.json').read_text());result={}
for layer in map(int,analysis['layers']):
    records=[json.loads((root/arm/'data'/f'rank-0-layer-{layer}.json').read_text()) for arm in ('A','B')]
    decoded=[]
    for arm,r in zip(('A','B'),records):
        path=root/arm/'data'/r['packed_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['packed_sha256']
        p=torch.load(path,weights_only=True);values=[]
        for req in p['requests']:
            ids=np.searchsorted(p['used_physical_pages'],req['page_ids'])
            values.append(logical_rows(p['pages'].numpy()[ids],req['count'],r['preshuffle_tile']))
        decoded.append(values)
    requests=[]
    for i,((ka,sa),(kb,sb)) in enumerate(zip(*decoded)):
        assert ka.shape==kb.shape and sa.shape==sb.shape
        changed=np.flatnonzero((ka!=kb).any(axis=1)|(sa!=sb).any(axis=1))
        info=dict(case=analysis['captured_cases'][i],valid_rows=len(ka),changed_rows=len(changed),
                  first_changed_logical_ids=changed[:32].tolist())
        if len(changed):
            info['changed_min']=int(changed[0]);info['changed_max']=int(changed[-1])
            info['changed_contiguous']=bool(np.all(np.diff(changed)==1))
            dtype=torch.float8_e4m3fnuz if records[0]['fp8_fnuz'] else torch.float8_e4m3fn
            a=torch.from_numpy(ka).view(dtype).float()*torch.from_numpy(sa.copy().view(np.float32))
            b=torch.from_numpy(kb).view(dtype).float()*torch.from_numpy(sb.copy().view(np.float32))
            assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            info['full_dequantized_max_abs']=float((a-b).abs().max())
            info['full_dequantized_relative_l2']=float((a-b).norm()/a.norm().clamp_min(1e-30))
        requests.append(info)
    result[layer]=requests
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k<=26},indent=2))
