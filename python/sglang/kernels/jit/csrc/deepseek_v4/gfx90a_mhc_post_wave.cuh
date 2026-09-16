#pragma once
#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {
template<int Control,int Rows=15>
__device__ __forceinline__ float post_dpp(float x) {
  return __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(x),__float_as_uint(x),Control,Rows,15,true));
}
__device__ __forceinline__ float post_reference_wave_sum(float rounded) {
  // Reproduce the measured four-wave Triton reference's scalar DPP tree,
  // including its first fused square/add. All lanes execute this function.
  float v=rounded*rounded;
  v=__builtin_fmaf(rounded,rounded,post_dpp<0x118>(v));
  v=post_dpp<0x114>(v)+v;
  v=post_dpp<0x112>(v)+v;
  v=post_dpp<0x111>(v)+v;
  v=post_dpp<0x142,10>(v)+v;
  v=post_dpp<0x143>(v)+v;
  return __shfl(v,63,64);
}
__global__ void post_exact_wave_kernel(const bf16_t* x,const bf16_t* residual,
    const float* post,const float* comb,bf16_t* out,float* partials) {
  const int token=blockIdx.x,wave=threadIdx.x/64,lane=threadIdx.x%64;
  const int block=blockIdx.y*4+wave;
  const int h=block*256+lane;
  float xv[4],r0[4],r1[4],r2[4],r3[4];
  #pragma unroll
  for(int q=0;q<4;++q) {
    xv[q]=static_cast<float>(x[token*4096+h+q*64]);
    const size_t base=static_cast<size_t>(token)*16384+h+q*64;
    r0[q]=static_cast<float>(residual[base]);
    r1[q]=static_cast<float>(residual[base+4096]);
    r2[q]=static_cast<float>(residual[base+8192]);
    r3[q]=static_cast<float>(residual[base+12288]);
  }
  #pragma unroll
  for(int hc=0;hc<4;++hc) {
    const float pv=post[token*4+hc];
    const float c0=comb[token*16+hc],c1=comb[token*16+4+hc];
    const float c2=comb[token*16+8+hc],c3=comb[token*16+12+hc];
    float sums[4];
    #pragma unroll
    for(int q=0;q<4;++q) {
      const float p0=c0*r0[q];
      float acc=__builtin_fmaf(c1,r1[q],p0);
      acc=acc+c2*r2[q];
      acc=acc+c3*r3[q];
      acc=pv*xv[q]+acc;
      const bf16_t y=cast<bf16_t>(acc);
      out[static_cast<size_t>(token)*16384+hc*4096+h+q*64]=y;
      sums[q]=post_reference_wave_sum(static_cast<float>(y));
    }
    if(lane==0) partials[(token*4+hc)*16+block]=(sums[0]+sums[2])+(sums[1]+sums[3]);
  }
}
struct Gfx90aMhcPostWave {
  static void run(const tvm::ffi::TensorView x,const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post,const tvm::ffi::TensorView comb,
      const tvm::ffi::TensorView out,const tvm::ffi::TensorView partials) {
    using namespace host;
    const int64_t m=x.size(0);RuntimeCheck(m>0 && m<=65536,"invalid rows");
    auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();
    TensorMatcher({m,4096}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({m,4,4096}).with_dtype<bf16_t>().with_device(dev).verify(residual);
    TensorMatcher({m,4}).with_dtype<float>().with_device(dev).verify(post);
    TensorMatcher({m,4,4}).with_dtype<float>().with_device(dev).verify(comb);
    TensorMatcher({m,4,4096}).with_dtype<bf16_t>().with_device(dev).verify(out);
    TensorMatcher({m,64}).with_dtype<float>().with_device(dev).verify(partials);
    LaunchKernel(dim3(m,4),256,x.device())(post_exact_wave_kernel,
      static_cast<const bf16_t*>(x.data_ptr()),static_cast<const bf16_t*>(residual.data_ptr()),
      static_cast<const float*>(post.data_ptr()),static_cast<const float*>(comb.data_ptr()),
      static_cast<bf16_t*>(out.data_ptr()),static_cast<float*>(partials.data_ptr()));
  }
};
}
