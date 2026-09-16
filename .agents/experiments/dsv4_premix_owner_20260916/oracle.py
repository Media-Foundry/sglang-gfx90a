"""Eight-rank exact paired pre-mix + RCCL mix allgather, full-chain oracle.

Synthetic occupancy extension of sampled layer0 activation; not a service score.
Run only after service_audit.py passes, without an overlapping service.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
import triton

from sglang.kernels.ops.debug.dsv4_premix_owner_audit import owned_rows
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_wave import post_wave_module

root = Path(__file__).resolve().parent
audit = json.loads((root/'audit/complete.json').read_text())
assert audit['all_rank_inputs_equal'] and audit['all_local_slices_exact']
rank = int(os.environ['LOCAL_RANK'])
assert os.environ['HIP_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
assert dist.get_world_size() == 8 and dist.get_rank() == rank
torch.set_grad_enabled(False)
torch.manual_seed(2026091613)
target = root/'oracle.json'
assert not target.exists()
fixture = root.parent/'dsv4_input_identity_20260914/trace-B1'
paths = [fixture/f'layer_0_rank_0_{name}.pt' for name in
         ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','hc_ffn_fn')]
seed = [torch.load(p,map_location='cuda',weights_only=True).contiguous() for p in paths]
postmod = post_wave_module()
report = dict(status='running', numerical_contract='existing paired kernel unchanged',
    measurement='Eight ranks, slowest rank per matching ABBA sample; full local compute + RCCL allgather',
    scope='Repeated sampled layer0 inputs for occupancy; not independent live full-M nor service',
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths]}, cases=[])


def save():
    if rank == 0:
        target.write_text(json.dumps(report,indent=2)+'\n')


try:
    for m in (8192,32767,32768,65536):
        x,res,post,comb = [s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed[:4]]
        fn = seed[4].clone()
        post,comb = post.view(m,4),comb.view(m,4,4)
        hidden = torch.empty_like(res)
        rms = torch.empty((m,64),device='cuda',dtype=torch.float32)
        reference = torch.empty((m,24),device='cuda',dtype=torch.float32)
        start,end = owned_rows(m,rank)
        n = end-start
        capacity = triton.cdiv(m,64)*8
        local = torch.zeros((capacity,24),device='cuda',dtype=torch.float32)
        gathered = torch.empty((capacity*8,24),device='cuda',dtype=torch.float32)
        # Same rank-ordered, 8-row-aligned contiguous regions; no gather/repack of H16384.
        xs,rs = hidden[start:end],rms[start:end]

        def prepare():
            postmod.run(x,res,post,comb,hidden,rms)

        def full():
            return premix8_pair[(12,triton.cdiv(m,8))](hidden,fn,rms,reference,m,1.e-6,num_warps=1)

        def owner():
            premix8_pair[(12,triton.cdiv(n,8))](xs,fn,rs,local,n,1.e-6,num_warps=1)
            dist.all_gather_into_tensor(gathered,local)

        def verify():
            exact = torch.equal(reference.view(torch.int32),gathered[:m].view(torch.int32))
            finite = bool(torch.isfinite(gathered[:m]).all())
            checks = [None]*8
            dist.all_gather_object(checks,dict(rank=rank,byte_exact=exact,finite=finite))
            assert all(c['byte_exact'] and c['finite'] for c in checks),checks
            return checks

        prepare()
        full();owner();verify()
        for mutation in range(10):
            fn.add_(torch.randn_like(fn)*1e-6)
            x.mul_(1.001)
            prepare();full();owner();verify()
        permutation = torch.randperm(m,device='cuda')
        permuted = hidden[permutation].contiguous()
        perm_rms = rms[permutation].contiguous()
        expected = reference[permutation].clone()
        hidden.copy_(permuted);rms.copy_(perm_rms)
        del permuted,perm_rms
        full();owner();verify()
        assert torch.equal(expected.view(torch.int32),reference.view(torch.int32))
        del expected,permutation
        # The expected graph replay result stays in reference; no output is used by a model.
        graphs = {}
        for name,call in [('A',full),('B',owner)]:
            for _ in range(3): call()
            torch.cuda.synchronize();dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            graphs[name] = graph
        for _ in range(1000):graphs['B'].replay()
        correctness = verify()
        timing = {}
        for mode in ('eager','graph'):
            samples=[]
            for cycle in range(3):
                for name in ('A','B','B','A'):
                    dist.barrier();torch.cuda.synchronize()
                    begin,finish = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    call = (full if name=='A' else owner) if mode=='eager' else graphs[name].replay
                    for _ in range(5):call()
                    finish.record();finish.synchronize()
                    values=[None]*8
                    dist.all_gather_object(values,begin.elapsed_time(finish)/5)
                    samples.append(dict(cycle=cycle,arm=name,rank_ms=values,rank_max_ms=max(values)))
            medians={name:statistics.median(s['rank_max_ms'] for s in samples if s['arm']==name) for name in ('A','B')}
            timing[mode]=dict(median_ms=medians,saved_ms=medians['A']-medians['B'],samples=samples)
        report['cases'].append(dict(rows=m,per_rank_capacity=capacity,global_mix_bytes=gathered.numel()*4,
            mutation10_exact=True,permutation_exact=True,replay1000=correctness,timing=timing))
        save()
        if rank==0:print('OWNER',m,{k:v['median_ms'] for k,v in timing.items()},flush=True)
        del graphs,graph,reference,gathered,local,x,res,post,comb,hidden,rms,xs,rs,fn
        torch.cuda.synchronize();torch.cuda.empty_cache()
    report['status']='complete';save()
finally:
    dist.destroy_process_group()
