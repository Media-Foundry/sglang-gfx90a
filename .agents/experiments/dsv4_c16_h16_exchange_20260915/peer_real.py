"""Four concurrent pairs: peer-load/store H16 with full local Q copy and two RCCL fences versus H8.

Real layer20 M32767 captured from original V4 TP8 C16 prefill. Full local Q
staging plus two pair fences included. Not service or capacity acceptance.
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
assert os.environ['HIP_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor
from peer_kernel import peer_prefill
import gc
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import _sparse_attn_v4_paged_prefill_kernel as kernel

rank=int(os.environ['LOCAL_RANK']);assert rank in range(8)
parity=rank%2
torch.cuda.set_device(rank)
dist.init_process_group('gloo',timeout=datetime.timedelta(minutes=8))
assert dist.get_world_size()==8
for base in range(0,8,2):
    cpu_group=dist.new_group(ranks=[base,base+1],backend='gloo',timeout=datetime.timedelta(minutes=8))
    gpu_group=dist.new_group(ranks=[base,base+1],backend='nccl',timeout=datetime.timedelta(minutes=8))
    if rank in (base,base+1):pair_cpu,comm=cpu_group,gpu_group
root=Path(__file__).resolve().parent;repo=root.parents[2]
result=dict(scope=__doc__,status='running',physical_gcds=list(range(8)),cases=[],sources={
    str(q.relative_to(repo)):hashlib.sha256(q.read_bytes()).hexdigest() for q in
    (Path(__file__),root/'peer_kernel.py',repo/'python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py')})
if rank==0:result['topology']=json.loads(subprocess.check_output(['amd-smi','topology','--json']))
def save():
    if rank==0:args.output.write_text(json.dumps(result,indent=2)+'\n')
try:
    fixture=root/'capture-v2/fixture'
    metadata=[json.loads((fixture/f'rank-{i}.json').read_text()) for i in range(8)]
    for key in ('unified_kv','kv_extend','kv_indices_prefix','kv_indices_extend','kv_indptr_prefix','kv_indptr_extend','input_ids','positions'):
        assert len({r['hashes'][key] for r in metadata})==1,key
    path=fixture/f'rank-{rank}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata[rank]['file_sha256']
    real=torch.load(path,map_location='cpu',weights_only=True)
    for key,t in real.items():
        assert hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==metadata[rank]['hashes'][key],key
    for family,m,req_len in [('real-layer20-c4',metadata[rank]['rows'],8192)]:
        half=(m+1)//2;local_rows=half if parity==0 else m-half
        pk,ek,pi,pp,ei,ep=[real[k].cuda() for k in ('unified_kv','kv_extend','kv_indices_prefix','kv_indptr_prefix','kv_indices_extend','kv_indptr_extend')]
        q=torch.empty_strided(real['q'].shape,metadata[rank]['original_q_stride'],dtype=torch.bfloat16,device='cuda')
        q.copy_(real['q']);sink=real['attn_sink'].cuda()
        captured=real['output'].cuda()
        hashes=[r['file_sha256'] for r in metadata]
        scale=metadata[rank]['softmax_scale']
        sinks=torch.empty(16,device='cuda');dist.all_gather_into_tensor(sinks,sink,group=comm)
        ref=torch.empty_like(q);out=torch.empty_like(q)
        # Force a nonzero tensor offset; honor ALL PyTorch IPC offset fields.
        qbase=torch.empty(m*8*512+64,device='cuda',dtype=q.dtype)
        obase=torch.empty_like(qbase)
        qstage=qbase[64:].view(m,8,512)
        out=obase[64:].view(m,8,512)
        def share(tensor):
            rebuild,info=reduce_tensor(tensor)
            infos=[None]*2;dist.all_gather_object(infos,info,group=pair_cpu)
            remote=list(infos[1-parity])
            assert remote[3]==64, remote[3]
            remote[6]=rank  # Open the remote allocation in the RECEIVER context.
            value=rebuild(*remote)
            assert value.shape==tensor.shape and value.device==tensor.device
            return value,dict(tensor_offset=int(info[3]),storage_offset_bytes=int(info[9]))
        peerq,qipc=share(qstage);peero,oipc=share(out)
        fence=torch.zeros(1,device='cuda')
        resources={}
        def launch(query,output,ss,begin):
            c=kernel[(len(query),1)](query,pk,pi,pp[begin:],ek,ei,ep[begin:],ss,output,
                *query.stride(),*pk.stride(),*ek.stride(),*output.stride(),query.shape[1],512,
                scale,BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1,num_stages=1)
            if str(query.shape[1]) not in resources:
                resources[str(query.shape[1])]=dict(hash=c.hash,registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared)
        def baseline():launch(q,ref,sink,0)
        def candidate():
            qstage.copy_(q)
            dist.all_reduce(fence,group=comm)
            q0,q1=(qstage,peerq) if parity==0 else (peerq,qstage)
            o0,o1=(out,peero) if parity==0 else (peero,out)
            c=peer_prefill[(local_rows,1)](q0,pk,pi,pp,ek,ei,ep,sinks,o0,q1,o1,parity*half,
                *qstage.stride(),*pk.stride(),*ek.stride(),*out.stride(),16,512,
                scale,BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,num_warps=1,num_stages=1)
            if 'peer' not in resources:
                resources['peer']=dict(hash=c.hash,registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared)
            dist.all_reduce(fence,group=comm)
        checks=[]
        for mutation in range(5):
            if mutation:
                q.mul_(1.001);pk.mul_(.999);ek.mul_(1.0001)
                sink.add_(.001);dist.all_gather_into_tensor(sinks,sink,group=comm)
            if mutation==2:pi[::97]=-1;ei[::113]=-1
            baseline()
            if mutation==0:assert torch.equal(ref.view(torch.uint8),captured.view(torch.uint8)), 'baseline differs from captured model output'
            out.fill_(float('nan'));candidate()
            check=dict(mutation=mutation,bits_exact=torch.equal(ref.view(torch.uint8),out.view(torch.uint8)),
                max_abs=float((ref.float()-out.float()).abs().max()),finite=bool(torch.isfinite(out).all()))
            assert check['bits_exact'] and check['finite'],(rank,family,m,check)
            checks.append(check)
        for replay in range(100):
            out.fill_(float('nan'));candidate()
            assert torch.equal(ref.view(torch.uint8),out.view(torch.uint8)), ('stale replay',rank,replay)
        # Restore unmodified real inputs for timing; mutations are correctness-only.
        q.copy_(real['q']);pk.copy_(real['unified_kv']);ek.copy_(real['kv_extend'])
        pi.copy_(real['kv_indices_prefix']);ei.copy_(real['kv_indices_extend'])
        sink.copy_(real['attn_sink']);dist.all_gather_into_tensor(sinks,sink,group=comm)
        baseline();candidate();assert torch.equal(ref.view(torch.uint8),out.view(torch.uint8))
        for _ in range(3):baseline();candidate()
        samples={'A':[],'B':[]}
        for _ in range(3):
            for arm,fn in [('A',baseline),('B',candidate),('B',candidate),('A',baseline)]:
                torch.cuda.synchronize();dist.barrier()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):fn()
                end.record();end.synchronize();samples[arm].append(start.elapsed_time(end)/5)
        gathered=[None]*8;dist.all_gather_object(gathered,dict(samples=samples,checks=checks,resources=resources,ipc_offsets=dict(q=qipc,o=oipc)))
        slowest={arm:[max(r['samples'][arm][i] for r in gathered) for i in range(6)] for arm in samples}
        center={k:statistics.median(v) for k,v in slowest.items()}
        item=dict(family=family,m=m,request_len=req_len,logical_fixture_hashes=hashes,
            q_stride=q.stride(),per_rank=gathered,rankmax_samples_ms=slowest,median_ms=center,
            speedup=center['A']/center['B'],send_bytes_per_rank=half*8*512*2*2,
            extra_static_buffers_bytes=qbase.numel()*2+obase.numel()*2,ipc_offsets=dict(q=qipc,o=oipc),
            sink_exchange_once_outside_timing=True,poisoned_eager_replays=100,captured_baseline_byte_exact=True)
        result['cases'].append(item);save()
        if rank==0:print(family,m,center,'speedup',item['speedup'],flush=True)
        torch.cuda.synchronize();dist.barrier()
        del peerq,peero
        gc.collect();dist.barrier()
        del qstage,out,qbase,obase
        gc.collect();dist.barrier()
        torch.cuda.ipc_collect();dist.barrier()
    result['status']='complete';save()
finally:dist.destroy_process_group()
