"""Fixed routed-stage input, fresh sorter output on every replay."""
import argparse
import torch
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_down_mfma32, gfx90a_fp4_expert_gate_up_mfma32,
)
from sglang.kernels.ops.quantization.gfx90a_int8_quant import gfx90a_int8_group32_quant

p = argparse.ArgumentParser()
p.add_argument('--rounds', type=int, default=20)
p.add_argument('--assignments', type=int, default=64)
p.add_argument('--hidden', type=int, default=512)
p.add_argument('--output-dim', type=int, default=64)
args = p.parse_args()
torch.manual_seed(12345)
e,m,t,h,i,n = 256,2304,6,args.hidden,512,args.output_dim
w13=torch.randint(0,256,(e,2*i,h//2),device='cuda',dtype=torch.uint8)
w2=torch.randint(0,256,(e,n,i//2),device='cuda',dtype=torch.uint8)
s13=shuffle_scale_a16w4(torch.randint(115,120,(e*2*i,h//32),device='cuda',dtype=torch.uint8),e,True)
s2=shuffle_scale_a16w4(torch.randint(115,120,(e*n,i//32),device='cuda',dtype=torch.uint8),e,False)
x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)
ids=torch.rand((m,e),device='cuda').topk(t,dim=1).indices.int()
weights=torch.rand((m,t),device='cuda')
reference=None
for rep in range(args.rounds):
    si,_,se,valid,_=moe_sorting(ids,weights,e,h,torch.bfloat16,block_size=args.assignments)
    xq,xs=gfx90a_int8_group32_quant(x)
    mid=gfx90a_fp4_expert_gate_up_mfma32(xq,xs,w13,s13,si,se,valid,t,10.,blocks=32,split=4,broadcast_scales=1,assignments=args.assignments)
    iq,isc=gfx90a_int8_group32_quant(mid)
    out=gfx90a_fp4_expert_down_mfma32(iq,isc,w2,s2,si,se,valid,weights,blocks=32,split=2,broadcast_scales=1,assignments=args.assignments)
    torch.cuda.synchronize()
    current=[xq,xs,mid,iq,isc,out]
    if reference is None:
        reference=[v.clone() for v in current]
    print(rep, {name:dict(exact=torch.equal(a,b),max_abs=(a.float()-b.float()).abs().max().item()) for name,a,b in zip(('xq','xs','mid','iq','isc','out'),reference,current)},flush=True)
