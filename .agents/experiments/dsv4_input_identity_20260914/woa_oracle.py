"""Frozen real wo_a input: separate GEMM shape choice from request identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

from safetensors import safe_open
import torch
import torch.nn.functional as F
from triton.runtime.errors import OutOfResources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/home/pc/models/modelscope'))
    p.add_argument('--library-variants',action='store_true')
    p.add_argument('--tiles',action='store_true')
    p.add_argument('--tiles2',action='store_true')
    args=p.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
    assert not args.output.exists()
    from sglang.kernels.ops.quantization.gfx90a_row_stable_linear import row_stable_linear
    records=[]
    with safe_open(args.model/'model-00002-of-00048.safetensors',framework='pt',device='cpu') as f:
        raw=f.get_tensor('layers.0.attn.wo_a.weight')
        scales=f.get_tensor('layers.0.attn.wo_a.scale')
    assert raw.shape==(8192,4096) and scales.shape==(64,32)
    assert raw.dtype==torch.float8_e4m3fn and scales.dtype==torch.float8_e8m0fnu
    for rank in (5,6):
        # Same block-128 FP8 -> BF16 math as model _dequant_fp8, then TP8 N shard.
        weight=(raw[rank*1024:(rank+1)*1024].float().view(8,128,32,128)
                *scales[rank*8:(rank+1)*8].float()[:,None,:,None]).reshape(1024,4096).bfloat16().cuda()
        prefix=f'layer_0_rank_{rank}_'
        sample=torch.load(args.trace/(prefix+'sample_rows.pt'),weights_only=True)
        at=int((sample==24576).nonzero().flatten().item())
        source=torch.load(args.trace/(prefix+'attn_inverse_rope.pt'),weights_only=True)[at].flatten().cuda()
        captured=torch.load(args.trace/(prefix+'wo_a.pt'),weights_only=True)[at].flatten().cuda()
        assert source.numel()==4096 and captured.numel()==1024
        saved={}
        for m,row in [(32768,24576),(32767,24575)]:
            x=torch.zeros((m,4096),device='cuda',dtype=torch.bfloat16)
            x[row].copy_(source)
            def einsum():return torch.einsum('tgd,grd->tgr',x[:,None,:],weight[None,:,:]).flatten(1)
            def padded():
                # Candidate keeps the M geometry at a 128-row bucket; no change to K.
                xp=F.pad(x,(0,0,0,(-m)%128)) if m%128 else x
                return torch.einsum('tgd,grd->tgr',xp[:,None,:],weight[None,:,:]).flatten(1)[:m]
            choices=[('einsum',einsum),('pad128',padded),('fixed-k64',lambda:row_stable_linear(x,weight))]
            if args.tiles or args.tiles2:
                from woa_tiles import project,TILES,TILES2
                tiles=TILES2 if args.tiles2 else TILES
                choices += [(f'tile-{bm}-{bn}-{bk}-{warps}',
                             lambda tile=(bm,bn,bk,warps):project(x,weight,tile))
                            for bm,bn,bk,warps in tiles]
            if args.library_variants:
                weight_t=weight.t().contiguous()
                def no_reduced():
                    old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                    try:
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
                        return einsum()
                    finally:
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old
                choices += [('linear',lambda:F.linear(x,weight)),
                            ('mm-contiguous-wt',lambda:x@weight_t),
                            ('transposed-mm',lambda:(weight@x.t()).t().contiguous()),
                            ('no-reduced',no_reduced)]
            for name,fn in choices:
                try:
                    fn();out=fn();torch.cuda.synchronize()
                except OutOfResources as exc:
                    records.append(dict(rank=rank,m=m,row=row,backend=name,
                                        status='out-of-resources',error=str(exc)))
                    print('RESOURCE REJECT',rank,m,name,str(exc),flush=True)
                    continue
                first=out[row].clone();times=[]
                for _ in range(7):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();out=fn();b.record();b.synchronize()
                    times.append(a.elapsed_time(b))
                    assert torch.equal(out[row],first)
                assert int(torch.count_nonzero(out))==int(torch.count_nonzero(out[row]))
                ref=saved.get(name)
                record=dict(rank=rank,m=m,row=row,backend=name,times_ms=times,
                    median_ms=statistics.median(times),
                    changed_vs_captured_a1=int(torch.count_nonzero(first!=captured)),
                    max_abs_vs_captured_a1=float((first.float()-captured.float()).abs().max()),
                    shifted_changed=None if ref is None else int(torch.count_nonzero(first!=ref)),
                    source_sha256=hashlib.sha256(source.cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
                records.append(record);saved[name]=first
                print(rank,m,name,record['changed_vs_captured_a1'],record['shifted_changed'],record['median_ms'],flush=True)
        del weight,x
    args.output.write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
