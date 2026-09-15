"""Real layer20 C4 order-preserving pair-reuse work-count bound, not timing."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent;repo=root.parents[2]
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in owners for p in g.get('process_list',[]))
import numpy as np
import torch
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_query_reuse import prefill_query_reuse4
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512

source=root.parent/'dsv4_c16_indexer_owner_20260915/capture-v2/data'
record=json.loads((source/'rank-0-layer-20.json').read_text());path=source/record['full_file']
assert record['logical_decoder_version']==2 and hashlib.sha256(path.read_bytes()).hexdigest()==record['full_sha256']
f=torch.load(path,weights_only=True,map_location='cpu');meta=f['metadata'];m=meta['rows']
assert meta['prefix_lens']==[0]*4 and m==32767
q=f['q'].cuda().view(getattr(torch,meta['tensors']['q']['dtype'].split('.')[-1])).reshape(m,1,64,128)
w=f['weights'].cuda().view(torch.float32).reshape(m,64)
lengths=f['seq_lens'].reshape(-1).to(device='cuda',dtype=torch.int32)
pages=f['page_table'].to(device='cuda',dtype=torch.int32);cache=f['cache'].cuda()
scores=prefill_query_reuse4(q,cache,w,lengths,pages,int(lengths.max()),block_s=16,
    preshuffle_tile=meta['preshuffle_tile'],dot_fp16=meta['dot_fp16'],fp8_fnuz=meta['fp8_fnuz'],
    query_group_size=16,runtime_m=True)
assert scores is not None
physical=torch.empty((m,512),dtype=torch.int32,device='cuda');logical=torch.empty_like(physical)
topk_transform_512(scores,lengths,pages,physical,64,logical)
saved=logical.clone()
for _ in range(3):
    topk_transform_512(scores,lengths,pages,physical,64,logical)
    assert torch.equal(saved,logical)
ids=physical.cpu().numpy();lens=np.minimum(lengths.cpu().numpy(),512)
req=np.repeat(np.arange(4),meta['extend_lens']);pos=f['positions'].numpy().reshape(-1)
pairs=m//2;a=ids[:2*pairs:2].reshape(pairs,32,16);b=ids[1:2*pairs:2].reshape(pairs,32,16)
same=req[:2*pairs:2]==req[1:2*pairs:2]
blocks=(lens+15)//16;both=np.arange(32)[None,:]<np.minimum(blocks[:2*pairs:2],blocks[1:2*pairs:2])[:,None]
equal=np.all(a==b,axis=-1)&both&same[:,None]
compatible=np.all((a==b)|(a<0)|(b<0),axis=-1)&both&same[:,None]
base_prefix=int(blocks.sum());swa_lens=np.minimum(pos+1,128);base_extend=int(((swa_lens+15)//16).sum())
# Flat extend bank windows keep the same start only before the first SWA wrap.
# Per-row causal masks may differ, but K16 chunk boundaries are retained.
start=pos-swa_lens+1
extend_same=same&(start[:2*pairs:2]==start[1:2*pairs:2])
extend_saved=int(np.where(extend_same,np.minimum((swa_lens[:2*pairs:2]+15)//16,
                   (swa_lens[1:2*pairs:2]+15)//16),0).sum())
total=base_prefix+base_extend
result=dict(scope=__doc__,fixture_sha256=record['full_sha256'],rows=m,layer=20,requests=4,
    selection_replay_exact=3,non_cross_request_pairs=int(same.sum()),
    baseline_prefix_tiles=base_prefix,baseline_extend_tiles=base_extend,
    equal_prefix_tiles=int(equal.sum()),compatible_prefix_tiles=int(compatible.sum()),
    compatible_extend_tiles=extend_saved,
    whole_attention_tile_reduction_upper_fraction=(int(compatible.sum())+extend_saved)/total,
    prefix_equal_tile_reduction_fraction=int(equal.sum())/base_prefix,
    pairs_with_any_compatible_prefix=int(np.any(compatible,axis=1).sum()),
    compatible_prefix_tiles_per_pair_histogram={str(i):int(np.sum(compatible.sum(1)==i)) for i in range(33)},
    note='A shared tile replaces two original query tiles with one Q2xH8 tile. '
         'Counts assume free compatibility testing and no register/occupancy cost; not a speedup prediction. '
         'Only one real C4 layer/forward. Extend counts inferred from known flat causal SWA semantics, not a captured extend index tensor.')
for label,mask in [('trivial',np.maximum(lens[:2*pairs:2],lens[1:2*pairs:2])<512),
                   ('saturated',np.maximum(lens[:2*pairs:2],lens[1:2*pairs:2])==512)]:
    result[label]=dict(pairs=int(mask.sum()),compatible_prefix_tiles=int(compatible[mask].sum()),
        original_prefix_tiles=int((blocks[:2*pairs:2]+blocks[1:2*pairs:2])[mask].sum()))
torch.save(logical.cpu(),root/'real-layer20-logical-ids.pt')
result['logical_ids_file_sha256']=hashlib.sha256((root/'real-layer20-logical-ids.pt').read_bytes()).hexdigest()
(root/'overlap.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
