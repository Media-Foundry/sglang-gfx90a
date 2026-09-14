"""Independent wider-C4 query16 oracle; wide production admission is opt-in.

Synthetic paged causal/prefix metadata, not actual service captures. Compare
the current empty-tile score path plus deterministic Top-K to the unchanged
runtime-M query16 kernel, directly or through its opt-in production wrapper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from metadata import tile_stats

os.environ.setdefault('SGLANG_OPT_USE_TRITON_INDEXER_FULL','1')
os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER','3')
import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.dsv4 import indexer
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_runtime_m import reuse_runtime_m
from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_query_reuse import prefill_query_reuse4
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mutations',type=int,default=3)
p.add_argument('--replays',type=int,default=10)
p.add_argument('--cases',nargs='+')
p.add_argument('--integrated',action='store_true')
args=p.parse_args();assert not args.output.exists()
assert args.mutations > 0 and args.replays > 0
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
torch.manual_seed(20260915)
root=Path(__file__).resolve().parent
result=dict(status='running',scope=__doc__,integrated=args.integrated,cases=[],source_sha256={})
for name in ('python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_runtime_m.py',
             'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_query_reuse.py',
             'python/sglang/srt/layers/attention/dsv4/indexer.py'):
    path=root.parents[2]/name
    result['source_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
result['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
shuffle=indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE if indexer.aiter_can_use_preshuffle_paged_mqa() else 0
fp16=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_FP16_DOT.get()
fnuz=indexer.is_fp8_fnuz()
assert indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_BLOCK_S.get()==16
result['dtype_contract']=dict(fp8=str(indexer.FP8_DTYPE),shuffle=shuffle,fp16_dot=fp16)

def save():args.output.write_text(json.dumps(result,indent=2)+'\n')

cases=[('16k-causal',4096,[16384,16384],[0,0]),
       ('32k-causal',8192,[32768],[0]),
       ('mixed-prefix',8192,[8191,8193,8190,8194],[0,4096,16384,24574]),
       ('ragged-partial-tile',8201,[17],[32787]),
       ('16k-single',4096,[16384],[0]),
       ('16k-prefix-m8192',4096,[8192],[8192]),
       ('32k-prefix-m8192',8192,[8192],[24576]),
       ('32k-double-m65536',8192,[32768,32768],[0,0]),
       ('16k-ragged-m32763',4096,[16381,16382],[0,0]),
       ('32k-ragged-m32767',8192,[32767],[1])]
if args.cases:
    assert set(args.cases)<=set(c[0] for c in cases)
    cases=[c for c in cases if c[0] in args.cases]
if args.integrated:
    assert all(8192<=sum(c[2])<=65536 and 2048<c[1]<=8192 for c in cases)
save()
for name,width,counts,prefixes in cases:
    m=sum(counts);np=(width+63)//64;npages=np*len(counts)
    lengths_cpu=[];owners_cpu=[]
    for owner,(count,prefix) in enumerate(zip(counts,prefixes,strict=True)):
        lengths_cpu.extend((prefix+offset+1)//4 for offset in range(count))
        owners_cpu.extend([owner]*count)
    assert max(lengths_cpu)<=width
    geometry=tile_stats(lengths_cpu,owners_cpu,width)
    lengths=torch.tensor(lengths_cpu,device='cuda',dtype=torch.int32)
    owners=torch.tensor(owners_cpu,device='cuda',dtype=torch.int32)
    storage=torch.empty(m,np+3,device='cuda',dtype=torch.int32)
    pages=storage[:,:np]
    pages.copy_(owners[:,None]*np+torch.arange(np,device='cuda')[None,:])
    q=torch.randn(m,1,64,128,device='cuda').to(indexer.FP8_DTYPE)
    weights=torch.randn(m,64,device='cuda')
    cache=torch.empty(npages,8448,device='cuda',dtype=torch.uint8)
    values=torch.randn(npages*64,128,device='cuda')
    locations=torch.arange(npages*64,device='cuda',dtype=torch.int32)
    triton_fused_store_indexer(values,cache,locations,64)
    kv=cache.view(npages,64,1,132)
    assert prefill_query_reuse4(q,kv,weights,lengths,pages,width,block_s=16,
        preshuffle_tile=shuffle,dot_fp16=fp16,fp8_fnuz=fnuz,
        query_group_size=16,runtime_m=True) is None, 'Production guard must remain in force'
    resources=[]
    def stage(arm):
        if arm=='A':
            scores=indexer.fp8_paged_mqa_logits_torch(q,kv,weights,lengths,pages,
                None,width,False,skip_trivial_topk=512,skip_empty_tiles=True)
        elif args.integrated:
            scores=prefill_query_reuse4(q,kv,weights,lengths,pages,width,block_s=16,
                preshuffle_tile=shuffle,dot_fp16=fp16,fp8_fnuz=fnuz,
                query_group_size=16,runtime_m=True,allow_wide=True)
            assert scores is not None,(name,'integrated admission')
        else:
            scores=torch.empty(m,width,device='cuda',dtype=torch.float32)
            c=reuse_runtime_m[(triton.cdiv(m,16),triton.cdiv(width,16))](
                q.view(torch.uint8),cache,weights,lengths,pages,scores,m,width,np,
                pages.stride(0),16,16,shuffle,tl.float16 if fp16 else tl.bfloat16,
                tl.float8e4b8 if fnuz else tl.float8e4nv,num_warps=4)
            if c is not None and not resources:
                resources.append(dict(registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared,hash=c.hash))
        logical=torch.empty(m,512,device='cuda',dtype=torch.int32)
        physical=torch.empty_like(logical)
        topk_transform_512(scores,lengths,pages,physical,64,logical)
        return scores,logical,physical
    graphs={};outputs={}
    for arm in ('A','B'):
        stage(arm);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):outputs[arm]=stage(arm)
        graphs[arm]=graph
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(outputs['A'],outputs['B']))
    def check(label):
        assert torch.equal(outputs['A'][0].view(torch.int32),outputs['B'][0].view(torch.int32)),(name,label,'scores')
        assert torch.equal(outputs['A'][1],outputs['B'][1]),(name,label,'logical')
        assert torch.equal(outputs['A'][2],outputs['B'][2]),(name,label,'physical')
    for replay in range(args.replays):
        for g in graphs.values():g.replay()
        check(('replay',replay))
    samples={'A':[],'B':[]}
    for _ in range(3):
        for arm in ('A','B','B','A'):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):graphs[arm].replay()
            b.record();b.synchronize();samples[arm].append(a.elapsed_time(b)/5)
    for step in range(args.mutations):
        q.copy_(torch.randn(q.shape,device='cuda').to(indexer.FP8_DTYPE))
        weights.normal_();values.normal_()
        triton_fused_store_indexer(values,cache,locations,64)
        # Preserve causal-prefix lengths while changing physical ownership.
        maps=torch.stack([torch.randperm(np,device='cuda',dtype=torch.int32)+i*np for i in range(len(counts))])
        pages.copy_(maps[owners.long()])
        if name.startswith('ragged'):
            pages.random_(0,npages);lengths.random_(0,width+1)
        if step%3==0:q.zero_()
        for g in graphs.values():g.replay()
        check(('mutation',step))
        if step%3==0:
            # Membership favors small logical IDs; attention consumes them
            # in descending logical order (including the length<=K shortcut).
            offsets=torch.arange(512,device='cuda',dtype=torch.int32)[None,:]
            count=lengths.clamp(0,512)[:,None]
            expected=torch.where(offsets<count,count-1-offsets,-1)
            assert torch.equal(outputs['B'][1],expected),(name,'canonical tie IDs')
            safe=expected.clamp_min(0)
            expected_physical=torch.where(expected>=0,
                pages.gather(1,(safe//64).long())*64+safe%64,-1)
            assert torch.equal(outputs['B'][2],expected_physical),(name,'canonical tie physical IDs')
    item=dict(name=name,m=m,width=width,request_rows=counts,prefix_tokens=prefixes,
        score_bits_exact=True,logical_physical_exact=True,mutations=args.mutations,
        fixed_graph_replays=args.replays,samples_ms=samples,
        median_ms={a:median(v) for a,v in samples.items()},resources=resources,
        logits_bytes_per_arm=m*width*4,page_table_stride=pages.stride(0),initial_geometry=geometry)
    item['speedup']=item['median_ms']['A']/item['median_ms']['B']
    result['cases'].append(item);save();print(json.dumps(item),flush=True)
    del graphs,outputs,graph,q,weights,cache,kv,values,locations,storage,pages,lengths,owners,maps,expected,expected_physical,safe,offsets,count
    torch.cuda.empty_cache()
result['status']='complete';save()
