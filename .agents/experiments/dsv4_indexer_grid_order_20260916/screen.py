"""Single-GCD scheduling-only logits screen; synthetic inputs, not service throughput."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER','3')
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_runtime_m import reuse_runtime_m
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
from sglang.srt.layers.attention.dsv4 import indexer
from kernel import reuse_grouped_grid

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'screen.json';assert not target.exists()
torch.set_grad_enabled(False)
assert torch.cuda.mem_get_info()[0]>50*1024**3, 'Requires an idle GCD'
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
shuffle=indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE if indexer.aiter_can_use_preshuffle_paged_mqa() else 0
fp16=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_FP16_DOT.get();fnuz=indexer.is_fp8_fnuz()
report=dict(status='running',scope=__doc__,pci=pci.value.decode(),contract=dict(shuffle=shuffle,fp16=fp16,fnuz=fnuz),
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'kernel.py',
        repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_runtime_m.py']},cases=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    for name,width,ext,prefix,rank in [
        ('8k-owner0',2048,[8192]*4,[0]*4,0),
        ('32k-owner0',8192,[32768],[0],0),
        ('32k-ragged-owner7',8192,[32767],[1],7),
        ('mixed-prefix-owner7',8192,[8191,8193,8190,8194],[0,4096,16384,24574],7),
    ]:
        torch.manual_seed(20260916)
        lengths,rows,valid,inverse=host_plan(ext,prefix,rank)
        cap=len(rows);np=width//64
        lens=torch.from_numpy(lengths[rows]).cuda()*torch.from_numpy(valid).cuda()
        owner=torch.repeat_interleave(torch.arange(len(ext)),torch.tensor(ext))
        pages=(owner[torch.from_numpy(rows)][:,None]*np+torch.arange(np)[None,:]).to(torch.int32).cuda()
        q=torch.randn(cap,1,64,128,device='cuda').to(indexer.FP8_DTYPE)
        weights=torch.randn(cap,64,device='cuda')
        cache=torch.empty(np*len(ext),8448,device='cuda',dtype=torch.uint8)
        values=torch.randn(len(cache)*64,128,device='cuda')
        locations=torch.arange(len(cache)*64,device='cuda',dtype=torch.int32)
        triton_fused_store_indexer(values,cache,locations,64)
        perm=torch.arange(len(cache),device='cuda').roll(rank*17)
        inv=torch.empty_like(perm);inv[perm]=torch.arange(len(perm),device='cuda')
        cache=cache.index_select(0,perm);pages.copy_(inv[pages.long()])
        a=torch.empty((cap,width),device='cuda',dtype=torch.float32);b=torch.empty_like(a)
        raw_a=torch.empty((cap,512),device='cuda',dtype=torch.int32);raw_b=torch.empty_like(raw_a)
        phys_a=torch.empty_like(raw_a);phys_b=torch.empty_like(raw_a)
        def launch(dst,gq=0):
            args=(q.view(torch.uint8),cache,weights,lens,pages,dst,cap,width,np,pages.stride(0),16,16,shuffle,
                  tl.float16 if fp16 else tl.bfloat16,tl.float8e4b8 if fnuz else tl.float8e4nv)
            if gq==0:
                return reuse_runtime_m[(triton.cdiv(cap,16),triton.cdiv(width,16))](*args,num_warps=4)
            return reuse_grouped_grid[(triton.cdiv(cap,16*gq)*gq*triton.cdiv(width,16),)](*args,gq,num_warps=4)
        # Validate score bits and canonical logical/physical IDs for every order.
        checks=[]
        for mutation in range(3):
            if mutation==1:
                weights.mul_(torch.linspace(.7,1.3,64,device='cuda'))
                q.copy_(q.view(torch.uint8).roll(1,dims=2).view(q.dtype))
            if mutation==2:q.zero_()
            launch(a);topk_transform_512(a,lens,pages,phys_a,64,raw_a)
            assert bool(torch.isfinite(a).all())
            for gq in (1,4,8,16,32):
                b.fill_(float('nan'));launch(b,gq)
                assert torch.equal(a.view(torch.int32),b.view(torch.int32)),(name,gq,mutation,'score')
                topk_transform_512(b,lens,pages,phys_b,64,raw_b)
                assert torch.equal(raw_a,raw_b) and torch.equal(phys_a,phys_b),(name,gq,mutation,'IDs')
                checks.append(dict(mutation=mutation,gq=gq,score_bits_exact=True,logical_physical_exact=True))
        q.copy_(torch.randn(q.shape,device='cuda').to(q.dtype));launch(a)
        candidates=[]
        for gq in (1,4,8,16,32):
            artifact=launch(b,gq)
            assert torch.equal(a.view(torch.int32),b.view(torch.int32))
            for _ in range(3):launch(a);launch(b,gq)
            graphs={}
            for arm,dst,order in [('A',a,0),('B',b,gq)]:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):launch(dst,order)
                graphs[arm]=graph
            for _ in range(100):graphs['B'].replay()
            assert torch.equal(a.view(torch.int32),b.view(torch.int32))
            records=[]
            for cycle in range(3):
                for arm in ('A','B','B','A'):
                    torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(5):graphs[arm].replay()
                    end.record();end.synchronize()
                    records.append(dict(cycle=cycle,arm=arm,ms=start.elapsed_time(end)/5))
            med={arm:statistics.median(v['ms'] for v in records if v['arm']==arm) for arm in ('A','B')}
            candidates.append(dict(gq=gq,median_ms=med,speedup=med['A']/med['B'],samples=records,
                graph_replay100_exact=True,artifact=artifact.hash,n_regs=artifact.n_regs,n_spills=artifact.n_spills))
            print('GRID',name,gq,med,flush=True)
            del graphs,graph
        report['cases'].append(dict(name=name,cap=cap,width=width,ext=ext,prefix=prefix,rank=rank,checks=checks,candidates=candidates))
        save()
        del q,weights,cache,pages,lens,a,b,raw_a,raw_b,phys_a,phys_b,values,locations,perm,inv,owner
        torch.cuda.empty_cache()
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
