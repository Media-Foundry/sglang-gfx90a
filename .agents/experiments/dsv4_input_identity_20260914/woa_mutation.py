"""100 real-input mutations and shifted-row checks for the best fixed-order tile."""
import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
import torch
from woa_tiles import project


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='4' and not args.output.exists()
    results=[]
    with safe_open('/home/pc/models/modelscope/model-00002-of-00048.safetensors',framework='pt',device='cpu') as f:
        raw=f.get_tensor('layers.0.attn.wo_a.weight')
        scales=f.get_tensor('layers.0.attn.wo_a.scale')
    for rank in (5,6):
        w=(raw[rank*1024:(rank+1)*1024].float().view(8,128,32,128)
           *scales[rank*8:(rank+1)*8].float()[:,None,:,None]).reshape(1024,4096).bfloat16().cuda()
        wd=w.double()
        samples=torch.load(args.trace/f'layer_0_rank_{rank}_attn_inverse_rope.pt',weights_only=True).flatten(1).cuda()
        xa=torch.zeros((32768,4096),device='cuda',dtype=torch.bfloat16)
        xb=torch.zeros((32767,4096),device='cuda',dtype=torch.bfloat16)
        olda=oldb=0;records=[]
        for i in range(100):
            source=(samples[i%len(samples)].float()*(1+(i%9-4)/64)).bfloat16()
            rowa=24576+(i%8)*128
            rowb=rowa-[1,2,63,64,65,127,128,129][i%8]
            xa[olda].zero_();xb[oldb].zero_()
            xa[rowa].copy_(source);xb[rowb].copy_(source)
            ya=project(xa,w,(128,128,128,8))[rowa].clone()
            yb=project(xb,w,(128,128,128,8))[rowb].clone()
            exact=torch.equal(ya,yb)
            assert exact,(rank,i)
            ref=(wd@source.double()).bfloat16()
            records.append(dict(iteration=i,sample=i%len(samples),row_a=rowa,row_b=rowb,
                shifted_exact=exact,changed_vs_fp64=int(torch.count_nonzero(ya!=ref)),
                max_abs_vs_fp64=float((ya.float()-ref.float()).abs().max())))
            olda,oldb=rowa,rowb
        results.append(dict(rank=rank,trials=records,shifted_exact=sum(r['shifted_exact'] for r in records)))
        print('MUTATIONS',rank,results[-1]['shifted_exact'],
              'max_abs_fp64',max(r['max_abs_vs_fp64'] for r in records),flush=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
