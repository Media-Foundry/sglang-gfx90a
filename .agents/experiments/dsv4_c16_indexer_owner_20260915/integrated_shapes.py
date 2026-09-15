"""Additional owner ABI/metadata shapes using captured Q/K values, not E2E data."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch.distributed as dist
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import forward

root=Path(__file__).resolve().parent;out=root/'integrated-shapes.json';assert not out.exists()
os.environ['SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK']='1'
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('gloo');assert dist.get_world_size()==8
group=dist.new_group(backend='nccl')
record=json.loads((root/'capture-v2/data/rank-0-layer-20.json').read_text())
path=root/'capture-v2/data'/record['full_file']
assert hashlib.sha256(path.read_bytes()).hexdigest()==record['full_sha256']
f=torch.load(path,weights_only=True)
dtype=getattr(torch,record['tensors']['q']['dtype'].split('.')[-1])
baseq=f['q'].cuda();basew=f['weights'].view(torch.float32).cuda()
cache=f['cache'].cuda();basepages=f['page_table'].cuda()
perm=torch.arange(len(cache),device='cuda').roll(17*rank)
inv=torch.empty_like(perm);inv[perm]=torch.arange(len(cache),device='cuda')
cache=cache[perm];basepages=inv[basepages.long()].int()
result=dict(status='running',scope=__doc__,ranks=8,cases=[])
cases=[([8192],[0]),([8192,1],[0,4096]),([6144,6145],[0,2047]),
       ([8192]*8,[0]*8),([512]*16,[0]*16)]
for ext,pre in cases:
    m=sum(ext);idx=torch.arange(m,device='cuda')%len(baseq)
    q=baseq.index_select(0,idx).view(dtype);w=basew.index_select(0,idx)
    # Keep each request's page table fixed and sufficient for its causal range.
    pages=basepages[8191:8192].expand(m,-1).contiguous()
    causal=torch.cat([torch.arange(p+1,p+n+1,device='cuda',dtype=torch.int32) for n,p in zip(ext,pre)])
    lens=causal//4
    batch=NS(extend_seq_lens_cpu=ext,extend_prefix_lens_cpu=pre)
    metadata=NS();output=torch.full((m,512),-123,device='cuda',dtype=torch.int32)
    hit=forward(q=q,cache=cache,weights=w,lengths=lens,pages=pages,width=2048,
        output=output,raw_output=None,batch=batch,metadata=metadata,rank=rank,group=group,
        preshuffle_tile=record['preshuffle_tile'],dot_fp16=record['dot_fp16'],fp8_fnuz=record['fp8_fnuz'])
    expected_hit=max(n+p for n,p in zip(ext,pre))>=2052
    assert hit==expected_hit
    if not hit:assert bool((output==-123).all())
    torch.cuda.synchronize();dist.barrier()
    result['cases'].append(dict(rows=m,extend=ext,prefix=pre,hit=hit,dual_checks_passed=hit,
                                fallback_untouched=not hit))
    if rank==0:print(result['cases'][-1],flush=True)
result['status']='complete'
batch=NS(extend_seq_lens_cpu=[8192],extend_prefix_lens_cpu=[0])
output=torch.full((8192,512),-123,device='cuda',dtype=torch.int32)
assert not forward(q=baseq[:8192].view(dtype),cache=cache,weights=basew[:8192],
    lengths=torch.arange(1,8193,device='cuda',dtype=torch.int32)//4,pages=basepages[:8192],width=512,
    output=output,raw_output=None,batch=batch,metadata=NS(),rank=rank,group=group,
    preshuffle_tile=record['preshuffle_tile'],dot_fp16=record['dot_fp16'],fp8_fnuz=record['fp8_fnuz'])
assert bool((output==-123).all());dist.barrier()
result['invalid_truncated_score_width_rejected']=True
if rank==0:out.write_text(json.dumps(result,indent=2)+'\n')
dist.destroy_process_group()
