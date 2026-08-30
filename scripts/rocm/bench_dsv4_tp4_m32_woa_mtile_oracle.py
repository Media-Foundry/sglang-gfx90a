#!/usr/bin/env python3
"""M32 wo_a cross-token weight-reuse VALU oracle."""
from __future__ import annotations
import argparse,os,statistics,torch
import torch.distributed as dist
from sglang.kernels.jit.utils import cache_once,load_jit,make_cpp_args
from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import _jit_gfx90a_bf16_grouped_gemv_module

M,G,N,K=32,2,1024,4096
@cache_once
def mod(mt,rows=1,unroll=2,waves=4):
 args=make_cpp_args(M,G,N,K,mt,rows,unroll,waves)
 return load_jit('gfx90a_bf16_grouped_gemv_mtile_oracle',*args,
  cuda_files=['gemm/gfx90a_bf16_grouped_gemv_mtile_oracle.cuh'],
  cuda_wrappers=[('run',f'sglang::Gfx90aBf16GroupedGemvMtileOracle<{args}>::run')],extra_cuda_cflags=['-O3'])
def cap(fn):
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g): o=fn()
 return g,o
def rankmax(g,iters,world):
 dist.barrier();a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
 for _ in range(iters):g.replay()
 b.record();b.synchronize();u=a.elapsed_time(b)*1000/iters;v=[None]*world;dist.all_gather_object(v,u);return max(map(float,v))
def trim(v):return statistics.mean(sorted(v)[1:-1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--dump',default='/tmp/dsv4_ffn_dump.f3ZQ89');p.add_argument('--mutations',type=int,default=100);p.add_argument('--rounds',type=int,default=7);p.add_argument('--iters',type=int,default=300);a=p.parse_args()
 lr=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(lr);dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size()
 ps=[os.path.join(a.dump,f'layer_20_rank_{2*rank+i}_attn_inverse_rope.pt') for i in (0,1)]
 x=torch.cat([torch.load(q,map_location='cpu',weights_only=False) for q in ps],1).reshape(M,G,K).cuda().contiguous();base=x.clone();mut=torch.linspace(-1,1,x.numel(),device='cuda').view_as(x).to(torch.bfloat16)
 gen=torch.Generator(device='cuda').manual_seed(55000+rank);w=torch.randn((G,N,K),generator=gen,device='cuda',dtype=torch.bfloat16).mul_(1/64)
 outs={k:torch.empty((M,G,N),device='cuda',dtype=torch.bfloat16) for k in ('R','M2','M4')};raw=_jit_gfx90a_bf16_grouped_gemv_module(M);mods={'M2':mod(2),'M4':mod(4)}
 def prod():return torch.einsum('mgk,gnk->mgn',x,w)
 def rr():raw.run(x,w,outs['R']);return outs['R']
 def mk(k):mods[k].run(x,w,outs[k]);return outs[k]
 f={'P':prod,'R':rr,'M2':lambda:mk('M2'),'M4':lambda:mk('M4')};graphs={};out={}
 for k in f:graphs[k],out[k]=cap(f[k])
 exact_raw={'M2':True,'M4':True};exact_prod={'M2':True,'M4':True};maxprod={'M2':0.,'M4':0.}
 for i in range(a.mutations):
  x.copy_(base).add_(mut,alpha=((i*1543+17)%2047-1023)/32768.)
  for k in f:graphs[k].replay()
  torch.cuda.synchronize()
  for k in ('M2','M4'):
   exact_raw[k]&=torch.equal(out[k],out['R']);exact_prod[k]&=torch.equal(out[k],out['P']);maxprod[k]=max(maxprod[k],float((out[k].float()-out['P'].float()).abs().max()))
 if rank==0:print(f'CORRECTNESS mutations={a.mutations} mtile2_vs_raw_exact={exact_raw["M2"]} mtile4_vs_raw_exact={exact_raw["M4"]} mtile2_vs_prod_exact={exact_prod["M2"]} mtile4_vs_prod_exact={exact_prod["M4"]} max_abs_prod_m2={maxprod["M2"]} max_abs_prod_m4={maxprod["M4"]}')
 vals={k:[] for k in f}
 for _ in range(a.rounds):
  for k in ('P','R','M2','M4','M4','M2','R','P'):vals[k].append(rankmax(graphs[k],a.iters,world))
 if rank==0:
  t={k:trim(v) for k,v in vals.items()}
  for k in f:print(f'RESULT profile={k} samples_us={",".join(f"{z:.3f}" for z in vals[k])} trimmed_rankmax_us={t[k]:.3f}')
  for k in ('M2','M4'):print(f'DECISION profile={k} prod_us={t["P"]:.3f} candidate_us={t[k]:.3f} gain_pct={(t["P"]/t[k]-1)*100:.3f} passes_33us={t[k]<33}')
if __name__=='__main__':main()
