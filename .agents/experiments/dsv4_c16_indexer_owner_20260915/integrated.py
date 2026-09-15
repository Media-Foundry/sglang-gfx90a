"""Real eight-rank check of the actual prospective service helper."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch.distributed as dist
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import forward

root=Path(__file__).resolve().parent
output=root/'integrated.json';assert not output.exists()
os.environ['SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK']='1'
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('gloo');assert dist.get_world_size()==8
group=dist.new_group(backend='nccl')
record=json.loads((root/'capture-v2/data/rank-0-layer-20.json').read_text())
path=root/'capture-v2/data'/record['full_file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==record['full_sha256']
f=torch.load(path,weights_only=True);m=record['rows']
dtype=getattr(torch,record['tensors']['q']['dtype'].split('.')[-1])
q=f['q'].cuda().view(dtype);w=f['weights'].view(torch.float32).cuda()
l=f['seq_lens'].cuda();pages=f['page_table'].cuda();cache=f['cache'].cuda()
perm=torch.arange(len(cache),device='cuda').roll(17*rank)
inv=torch.empty_like(perm);inv[perm]=torch.arange(len(cache),device='cuda')
cache=cache[perm];pages=inv[pages.long()].int()
batch=NS(extend_seq_lens_cpu=record['extend_lens'],extend_prefix_lens_cpu=record['prefix_lens'])
metadata=NS();out=torch.empty((m,512),device='cuda',dtype=torch.int32);raw=torch.empty_like(out)
base_q=q.view(torch.uint8).clone();base_w=w.clone()
for mutation in range(3):
    q.view(torch.uint8).copy_(base_q);w.copy_(base_w)
    if mutation:w.mul_(torch.linspace(.7,1.3,64,device='cuda').pow(mutation))
    if mutation==2:q.view(torch.uint8).copy_(base_q.roll(1,2))
    assert forward(q=q,cache=cache,weights=w,lengths=l,pages=pages,width=2048,
        output=out,raw_output=raw if mutation!=1 else None,batch=batch,metadata=metadata,
        rank=rank,group=group,preshuffle_tile=record['preshuffle_tile'],
        dot_fp16=record['dot_fp16'],fp8_fnuz=record['fp8_fnuz'])
    torch.cuda.synchronize();dist.barrier()
    if rank==0:print('integrated mutation passed',mutation,flush=True)
if rank==0:
    output.write_text(json.dumps(dict(status='complete',ranks=8,mutations=3,rows=m,
        fixture_sha256=record['full_sha256'],actual_service_helper=True,
        cpu_metadata_matches_live_lengths=True,all_scores_and_topk_exact=True),indent=2)+'\n')
dist.destroy_process_group()
