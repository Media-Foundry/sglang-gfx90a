"""Peer-load/store H16 with full local Q copy and two RCCL fences versus H8.

Synthetic inputs with identical LOGICAL KV, independently permuted physical
banks. No real-model capture or capacity acceptance. Only original per-query
selections; no shared-index approximation. Uses accepted stages1 in both arms.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ['HIP_VISIBLE_DEVICES']=='4,5'
import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor
from peer_kernel import peer_prefill
import gc
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import _sparse_attn_v4_paged_prefill_kernel as kernel

rank=int(os.environ['LOCAL_RANK']);assert rank in (0,1)
torch.cuda.set_device(rank)
dist.init_process_group('gloo',timeout=datetime.timedelta(minutes=8))
assert dist.get_world_size()==2
comm=dist.new_group(backend='nccl',timeout=datetime.timedelta(minutes=8))
root=Path(__file__).resolve().parent;repo=root.parents[2]
result=dict(scope=__doc__,status='running',physical_gcds=[4,5],cases=[],sources={
    str(q.relative_to(repo)):hashlib.sha256(q.read_bytes()).hexdigest() for q in
    (Path(__file__),root/'peer_kernel.py',repo/'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py')})
if rank==0:result['topology']=json.loads(subprocess.check_output(['amd-smi','topology','--json']))
def save():
    if rank==0:args.output.write_text(json.dumps(result,indent=2)+'\n')
def exchange(send,receive):
    works=dist.batch_isend_irecv([dist.P2POp(dist.isend,send,1-rank,comm),
                                 dist.P2POp(dist.irecv,receive,1-rank,comm)])
    for work in works:work.wait()
try:
    for family,m,req_len in [('c128',8192,4096),('c4',8192,4096),
                              ('c128',32768,8192),('c4',32768,8192)]:
        half=m//2;ratio=128 if family=='c128' else 4
        rng=np.random.default_rng(20260915)
        pi=[];ei=[];pp=[0];ep=[0]
        for row in range(m):
            request,pos=divmod(row,req_len);history=(pos+1)//ratio
            chosen=np.sort(rng.choice(history,512,replace=False)) if history>512 else np.arange(history)
            pi.extend((chosen+request*(req_len//ratio)).tolist());pp.append(len(pi))
            ei.extend(range(row-min(pos,127),row+1));ep.append(len(ei))
        pi,pp,ei,ep=[torch.tensor(x,device='cuda',dtype=torch.int32) for x in (pi,pp,ei,ep)]
        torch.manual_seed(20260915)
        pk=torch.randn((m//ratio,512),device='cuda',dtype=torch.bfloat16).mul_(.5)
        ek=torch.randn((m,512),device='cuda',dtype=torch.bfloat16).mul_(.8)
        digest=hashlib.sha256()
        for t in (pk,ek,pi,pp,ei,ep):digest.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        hashes=[None]*2;dist.all_gather_object(hashes,digest.hexdigest());assert hashes[0]==hashes[1]
        # Keep physical IDs local. Kernel on each receiver uses its own banks.
        for bank,ids,shift in ((pk,pi,17),(ek,ei,31)):
            perm=torch.arange(len(bank),device='cuda').roll(shift*rank)
            inv=torch.empty_like(perm);inv[perm]=torch.arange(len(bank),device='cuda')
            bank.copy_(bank.index_select(0,perm));ids.copy_(inv[ids.long()].int())
        torch.manual_seed(20260915+rank)
        # Non-contiguous local head view forces send-side packing into timing.
        q=torch.randn((m,16,512),device='cuda',dtype=torch.bfloat16).mul_(.3)[:,:8,:]
        sink=torch.linspace(-4,4,16,device='cuda')[rank*8:(rank+1)*8].contiguous()
        sinks=torch.empty(16,device='cuda');dist.all_gather_into_tensor(sinks,sink,group=comm)
        ref=torch.empty_like(q);out=torch.empty_like(q)
        # Force a nonzero tensor offset; honor ALL PyTorch IPC offset fields.
        qbase=torch.empty(m*8*512+64,device='cuda',dtype=q.dtype)
        obase=torch.empty_like(qbase)
        qstage=qbase[64:].view(m,8,512)
        out=obase[64:].view(m,8,512)
        def share(tensor):
            rebuild,info=reduce_tensor(tensor)
            infos=[None]*2;dist.all_gather_object(infos,info)
            remote=list(infos[1-rank])
            assert remote[3]==64, remote[3]
            remote[6]=rank  # Open the remote allocation in the RECEIVER context.
            value=rebuild(*remote)
            assert value.shape==tensor.shape and value.device==tensor.device
            return value,dict(tensor_offset=int(info[3]),storage_offset_bytes=int(info[9]))
        peerq,qipc=share(qstage);peero,oipc=share(out)
        fence=torch.zeros(1,device='cuda')
        own=slice(rank*half,(rank+1)*half);other=slice((1-rank)*half,(2-rank)*half)
        resources={}
        def launch(query,output,ss,begin):
            c=kernel[(len(query),1)](query,pk,pi,pp[begin:],ek,ei,ep[begin:],ss,output,
                *query.stride(),*pk.stride(),*ek.stride(),*output.stride(),query.shape[1],512,
                512**-.5,BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1,num_stages=1)
            if str(query.shape[1]) not in resources:
                resources[str(query.shape[1])]=dict(hash=c.hash,registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared)
        def baseline():launch(q,ref,sink,0)
        def candidate():
            qstage.copy_(q)
            dist.all_reduce(fence,group=comm)
            q0,q1=(qstage,peerq) if rank==0 else (peerq,qstage)
            o0,o1=(out,peero) if rank==0 else (peero,out)
            c=peer_prefill[(half,1)](q0,pk,pi,pp,ek,ei,ep,sinks,o0,q1,o1,rank*half,
                *qstage.stride(),*pk.stride(),*ek.stride(),*out.stride(),16,512,
                512**-.5,BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1,num_stages=1)
            if 'peer' not in resources:
                resources['peer']=dict(hash=c.hash,registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared)
            dist.all_reduce(fence,group=comm)
        checks=[]
        for mutation in range(5):
            if mutation:
                q.mul_(1.001);pk.mul_(.999);ek.mul_(1.0001)
                sink.add_(.001);dist.all_gather_into_tensor(sinks,sink,group=comm)
            if mutation==2:pi[::97]=-1;ei[::113]=-1
            baseline();out.fill_(float('nan'));candidate()
            check=dict(mutation=mutation,bits_exact=torch.equal(ref.view(torch.uint8),out.view(torch.uint8)),
                max_abs=float((ref.float()-out.float()).abs().max()),finite=bool(torch.isfinite(out).all()))
            assert check['bits_exact'] and check['finite'],(rank,family,m,check)
            checks.append(check)
        for _ in range(3):baseline();candidate()
        samples={'A':[],'B':[]}
        for _ in range(3):
            for arm,fn in [('A',baseline),('B',candidate),('B',candidate),('A',baseline)]:
                torch.cuda.synchronize();dist.barrier()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):fn()
                end.record();end.synchronize();samples[arm].append(start.elapsed_time(end)/5)
        gathered=[None]*2;dist.all_gather_object(gathered,dict(samples=samples,checks=checks,resources=resources))
        slowest={arm:[max(r['samples'][arm][i] for r in gathered) for i in range(6)] for arm in samples}
        center={k:statistics.median(v) for k,v in slowest.items()}
        item=dict(family=family,m=m,request_len=req_len,logical_fixture_hashes=hashes,
            q_stride=q.stride(),per_rank=gathered,rankmax_samples_ms=slowest,median_ms=center,
            speedup=center['A']/center['B'],send_bytes_per_rank=half*8*512*2*2,
            extra_static_buffers_bytes=qbase.numel()*2+obase.numel()*2,ipc_offsets=dict(q=qipc,o=oipc),
            sink_exchange_once_outside_timing=True)
        result['cases'].append(item);save()
        if rank==0:print(family,m,center,'speedup',item['speedup'],flush=True)
        torch.cuda.synchronize();dist.barrier()
        del peerq,peero
        gc.collect();dist.barrier()
    result['status']='complete';save()
finally:dist.destroy_process_group()
