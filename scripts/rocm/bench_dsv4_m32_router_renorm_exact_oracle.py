#!/usr/bin/env python3
"""Check generic moe_fused_gate renorm order against HIP router oracle."""
import torch
from sglang.kernels.jit.utils import cache_once,load_jit
from sglang.srt.layers.moe.topk import biased_topk_jit_kernel_impl

@cache_once
def mod():
 return load_jit('gfx90a_sqrt_router_generic_renorm_oracle',cuda_files=['moe/gfx90a_grouped_router.cuh'],cuda_wrappers=[('run','sglang::Gfx90aSqrtSoftplusRouterFp32BiasKernel::run')],extra_cuda_cflags=['-O3','-DSGLANG_GFX90A_ROUTER_GENERIC_RENORM_ORACLE'])
def ref(x,b):return biased_topk_jit_kernel_impl(torch.empty(32,1,device='cuda'),x,b,6,True,'sqrtsoftplus',routed_scaling_factor=1.5)
def cand(x,b,o=None):
 if o is None:o=(torch.empty(32,6,device='cuda'),torch.empty(32,6,device='cuda',dtype=torch.int32))
 for r in range(32):mod().run(x[r:r+1],b,o[0][r:r+1],o[1][r:r+1],1.5,False)
 return o
def main():
 torch.manual_seed(20260830);x=torch.empty(32,256,device='cuda');b=torch.randn(256,device='cuda')
 for i in range(1000):
  x.uniform_(-12,12)
  if i%4==0:
   # Exact and adjacent ties exercise lower-expert-id ordering.
   x[:,1]=x[:,0];x[:,3]=torch.nextafter(x[:,2],torch.full_like(x[:,2],float('inf')))
  a=ref(x,b);c=cand(x,b);torch.cuda.synchronize()
  if not torch.equal(a[1],c[1]) or not torch.equal(a[0],c[0]):
   print('FAIL',i,'ids_mismatch',int((a[1]!=c[1]).sum()),'w_max',float((a[0]-c[0]).abs().max()));return
 print('CORRECTNESS mutations=1000 ids_exact=True weights_exact=True')
if __name__=='__main__':main()
