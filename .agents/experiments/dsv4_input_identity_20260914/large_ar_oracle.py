"""8-rank same-value/shifted-row collective oracle using saved real wo_b partials."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--sweep',action='store_true')
    parser.add_argument('--chunks',action='store_true')
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='0,1,2,3,4,5,6,7'
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    cpu=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    class CheckedAR(CustomAllreduce):
        def _get_ipc_meta(self,inp):
            # The stock adapter assumes zero IPC offset. Fail closed if this
            # fresh large allocation does not satisfy that existing contract.
            shared=inp.untyped_storage()._share_cuda_()
            assert shared[3]==0 and inp.storage_offset()==0
            return super()._get_ipc_meta(inp)
    ca=CheckedAR(cpu,torch.device('cuda',rank),max_size=288*1024*1024)
    assert not ca.disabled and ca.world_size==8
    if args.sweep:
        import aiter
        from sglang.kernels.ops.debug.gfx90a_tp8_prefill_ar_oracle import module
        tuned=module()
        assert tuned.signal_bytes()==aiter.meta_size()
    partials=[]
    for r in range(8):
        prefix=f'layer_0_rank_{r}_'
        sample=torch.load(args.trace/(prefix+'sample_rows.pt'),weights_only=True)
        value=torch.load(args.trace/(prefix+'wo_b_partial.pt'),weights_only=True)
        at=(sample==24576).nonzero().flatten()
        assert at.numel()==1
        partials.append(value[int(at[0])].clone())
    reference=sum(v.double() for v in partials).bfloat16().cuda()
    source=partials[rank].cuda()
    saved={};records=[]
    for m,row in [(32768,24576),(32767,24575)]:
        x=torch.zeros((m,4096),dtype=torch.bfloat16,device='cuda')
        x[row].copy_(source)
        out=torch.empty_like(x)
        work=torch.empty_like(x,dtype=torch.float32)
        assert ca.should_custom_ar(x)
        def rccl_bf16():
            out.copy_(x);dist.all_reduce(out)
        def rccl_fp32():
            work.copy_(x);dist.all_reduce(work);out.copy_(work)
        def aiter_new():ca.all_reduce(x,out=out,use_new=True,registered=False)
        def aiter_legacy():ca.all_reduce(x,out=out,use_new=False,registered=False)
        choices=[('rccl-bf16',rccl_bf16),('rccl-fp32',rccl_fp32),
                 ('aiter-new',aiter_new),('aiter-legacy',aiter_legacy)]
        if args.chunks:
            def make_chunked(rows, use_new):
                def run():
                    for first in range(0,m,rows):
                        ca.all_reduce(x[first:first+rows],out=out[first:first+rows],
                                      use_new=use_new,registered=False)
                return run
            choices += [(f'chunk-{rows}-{kind}',make_chunked(rows,use_new))
                        for rows in (1024,2048,4096,8192)
                        for kind,use_new in [('new',True),('legacy',False)]]
        if args.sweep:
            registered=ca.buffer[:x.numel()*2].view(torch.bfloat16).view(m,4096)
            def make_tuned(blocks):
                def run():
                    registered.copy_(x)
                    tuned.run(ca._ptr,registered,out,blocks,ca.max_size)
                return run
            choices += [(f'legacy-{blocks}',make_tuned(blocks)) for blocks in (8,16,32,48,64,80)]
        for name,fn in choices:
            fn();fn();torch.cuda.synchronize();dist.barrier(group=cpu)
            first=out[row].clone();times=[];repeat=[]
            for _ in range(5):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();fn();end.record();end.synchronize()
                times.append(start.elapsed_time(end))
                repeat.append(torch.equal(out[row].view(torch.int16),first.view(torch.int16)))
            # No stale writes anywhere outside the one active row.
            assert int(torch.count_nonzero(out))==int(torch.count_nonzero(out[row]))
            observed=out[row].clone()
            shifted=saved.get(name)
            record=dict(m=m,row=row,backend=name,rank=rank,times_ms=times,
                repeat_exact=repeat,output_sha256=sha(observed),input_sha256=sha(source),
                changed_from_fp64=int(torch.count_nonzero(observed!=reference)),
                max_abs_from_fp64=float((observed.float()-reference.float()).abs().max()),
                shifted_equal=None if shifted is None else torch.equal(observed.view(torch.int16),shifted.view(torch.int16)),
                shifted_changed=None if shifted is None else int(torch.count_nonzero(observed!=shifted)))
            saved[name]=observed
            records.append(record)
            if rank==0:print('COLLECTIVE',m,name,record['changed_from_fp64'],record['shifted_changed'],flush=True)
        del x,out,work
    gathered=[None]*8
    dist.all_gather_object(gathered,records,group=cpu)
    if rank==0:
        summary=[]
        for j,r in enumerate(records):
            per_rank=[g[j] for g in gathered]
            summary.append(dict(m=r['m'],backend=r['backend'],
                rank_max_median_ms=max(statistics.median(v['times_ms']) for v in per_rank),
                all_ranks_agree=len({v['output_sha256'] for v in per_rank})==1,
                changed_from_fp64=r['changed_from_fp64'],shifted_changed=r['shifted_changed']))
        assert not args.output.exists()
        args.output.write_text(json.dumps(dict(summary=summary,ranks=gathered),indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    dist.barrier(group=cpu)
    ca.close()
    dist.destroy_process_group()


if __name__=='__main__':main()
