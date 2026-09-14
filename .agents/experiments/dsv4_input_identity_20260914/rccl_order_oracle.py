"""Frozen real FFN partials: RCCL algorithm/order screen, eight ranks only."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='0,1,2,3,4,5,6,7'
    assert not args.output.exists()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    cpu=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
    assert dist.get_world_size()==8
    values=[]
    for r in range(8):
        prefix=f'layer_0_rank_{r}_'
        rows=torch.load(args.trace/(prefix+'sample_rows.pt'),weights_only=True)
        pos=(rows==24576).nonzero().flatten()
        assert pos.numel()==1
        values.append(torch.load(args.trace/(prefix+'ffn_partial.pt'),weights_only=True)[int(pos.item())])
    records=[]
    for mutation in range(3):
        parts=[(v.float()*(1 if mutation==0 else (-1 if r%2 else 1) if mutation==1 else 2.0**(r-4))).bfloat16()
               for r,v in enumerate(values)]
        ref=sum(v.double() for v in parts).bfloat16().cuda()
        saved={}
        for m,row in [(32768,24576),(32767,24575)]:
            x=torch.zeros((m,4096),device='cuda',dtype=torch.bfloat16)
            x[row].copy_(parts[rank]);out=torch.empty_like(x)
            for dtype in (torch.bfloat16,torch.float32):
                work=torch.empty_like(x,dtype=dtype)
                def run():
                    work.copy_(x);dist.all_reduce(work);out.copy_(work)
                run();run();torch.cuda.synchronize();dist.barrier(group=cpu)
                first=out[row].clone();times=[]
                for _ in range(5):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();run();b.record();b.synchronize();times.append(a.elapsed_time(b))
                    assert torch.equal(first,out[row])
                assert int(torch.count_nonzero(out))==int(torch.count_nonzero(out[row]))
                prior=saved.get(str(dtype))
                item=dict(rank=rank,mutation=mutation,m=m,row=row,dtype=str(dtype),times_ms=times,
                    shifted_changed=None if prior is None else int(torch.count_nonzero(first!=prior)),
                    changed_from_fp64=int(torch.count_nonzero(first!=ref)),
                    max_abs_from_fp64=float((first.float()-ref.float()).abs().max()),
                    sha256=hashlib.sha256(first.cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
                records.append(item);saved[str(dtype)]=first
                if rank==0:print('RCCL',mutation,m,str(dtype),item['shifted_changed'],item['changed_from_fp64'],flush=True)
                del work
            del x,out
    gathered=[None]*8;dist.all_gather_object(gathered,records,group=cpu)
    if rank==0:
        summary=[]
        for i,r in enumerate(records):
            row={k:v for k,v in r.items() if k not in ('rank','times_ms','sha256')}
            row['rank_max_median_ms']=max(statistics.median(g[i]['times_ms']) for g in gathered)
            row['all_ranks_agree']=len({g[i]['sha256'] for g in gathered})==1
            summary.append(row)
        args.output.write_text(json.dumps(dict(summary=summary,ranks=gathered,
            env={k:os.environ.get(k) for k in ('NCCL_ALGO','NCCL_PROTO','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS')},
            torch=torch.__version__,hip=torch.version.hip),indent=2)+'\n')
    dist.barrier(group=cpu);dist.destroy_process_group()


if __name__=='__main__':main()
