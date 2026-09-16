"""CPU address-contract audit for a future stage1 route-major producer.

This is not a kernel implementation, numerical GPU oracle, or speed result.
Only sorter identities and destination bounds are modeled here.
"""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root=Path(__file__).resolve().parent
target=root/'producer-address-audit.json'
assert not target.exists()
header=Path('/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp')
source=header.read_text()
assert 'IsInputGemm ? problem.NumTokens * problem.TopK : problem.NumTokens,' in source
assert 'token_offset = token_offset * problem.TopK + (fused_token >> 24);' in source
report=dict(scope=__doc__,status='running',gpu_tests=False,
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (header,Path(__file__))},cases=[])
for location in ('dsv4_cached_ck_drift_20260915/capture',
                 'dsv4_c16_real_ck_replay_20260915/capture-v2'):
    fixture=root.parent/location/'fixture'
    meta=json.loads((fixture/'manifest.json').read_text())
    values={}
    for name in ('sorted_token_ids','num_valid_ids'):
        path=fixture/(name+'.pt')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
        values[name]=torch.load(path,map_location='cpu',weights_only=True)
    m,t,i=meta['stage2_input']['shape']
    assert t==6 and i==256
    capacity=values['sorted_token_ids'].numel()
    valid=int(values['num_valid_ids'][0])
    assert valid%64==0 and valid<=capacity
    checks=[]
    for reverse in (False,True):
        ids=values['sorted_token_ids'][:valid].long()
        if reverse:
            ids=ids.view(-1,64).flip(0).flatten()
        token=ids&0xffffff
        slot=ids>>24
        live=(token<m)&(slot>=0)&(slot<t)
        route=torch.arange(valid)[live]
        virtual=token[live]*t+slot[live]
        assert torch.equal(torch.sort(virtual).values,torch.arange(m*t))
        inverse=torch.full((m*t,),-1,dtype=torch.long)
        inverse[virtual]=route
        assert bool((inverse>=0).all())
        assert torch.equal(inverse[virtual],route)
        # A producer must write route*I, not (token*TopK+slot)*I.
        # The descriptor must also admit sorter capacity, not only M*TopK.
        assert int(route.max())<capacity
        last_byte=(route*i+i-1)*2
        assert int(last_byte.max())<capacity*i*2
        stage2_ids=torch.full((capacity,),capacity,dtype=torch.long)
        stage2_ids[route]=route
        assert torch.equal(stage2_ids[inverse],inverse)
        checks.append(dict(reverse_expert_blocks=reverse,assignments=int(live.sum()),
                           exact_inverse=True,last_live_route=int(route.max()),
                           live_routes_outside_old_descriptor=int((route>=m*t).sum())))
    report['cases'].append(dict(m=m,capacity=capacity,valid_padded_rows=valid,
        old_intermediate_bytes=m*t*i*2,new_intermediate_bytes=capacity*i*2,
        temporary_pack_removed_bytes=capacity*i*2,checks=checks))
report['status']='complete'
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
