#pragma once
#include "premix.cuh"
namespace sglang {
template<int Split>
__global__ void mhc_fp32_split_kernel(const bf16_t* x,const float* fn,float* out,int m) {
  constexpr int K=16384,N=24,U=8;
  const int lane=threadIdx.x%64,wave=threadIdx.x/64;
  const int mbase=(blockIdx.x*4+wave)*16,nbase=blockIdx.y*16;
  const int ar=mbase+lane%16,bn=nbase+lane%16,ki=lane/16;
  const int begin=blockIdx.z*(K/Split),end=begin+K/Split;
  MhcFloat4 acc={0,0,0,0};
  for(int start=begin;start<end;start+=4*U) {
    float a[U],b[U];
    #pragma unroll
    for(int u=0;u<U;++u) {
      const int kk=start+4*u+ki;
      a[u]=ar<m?static_cast<float>(x[static_cast<size_t>(ar)*K+kk]):0.0f;
      b[u]=bn<N?fn[static_cast<size_t>(bn)*K+kk]:0.0f;
    }
    #pragma unroll
    for(int u=0;u<U;++u)
      acc=__builtin_amdgcn_mfma_f32_16x16x4f32(a[u],b[u],acc,0,0,0);
  }
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int row=mbase+(lane/16)*4+i,col=nbase+lane%16;
    if(row<m && col<N) out[(static_cast<size_t>(blockIdx.z)*m+row)*N+col]=acc[i];
  }
}
struct MhcSplitScreen {
  template<int Split>
  static void run(const tvm::ffi::TensorView x,const tvm::ffi::TensorView fn,const tvm::ffi::TensorView out) {
    using namespace host;
    const int64_t m=x.size(0);
    RuntimeCheck(m>0 && m<=65536,"invalid M");
    auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();
    TensorMatcher({m,16384}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({24,16384}).with_dtype<float>().with_device(dev).verify(fn);
    TensorMatcher({Split,m,24}).with_dtype<float>().with_device(dev).verify(out);
    LaunchKernel(dim3((m+63)/64,2,Split),256,x.device())(mhc_fp32_split_kernel<Split>,
      static_cast<const bf16_t*>(x.data_ptr()),static_cast<const float*>(fn.data_ptr()),static_cast<float*>(out.data_ptr()),static_cast<int>(m));
  }
  static void s4(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<4>(a,b,c);}
  static void s16(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<16>(a,b,c);}
  static void s32(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<32>(a,b,c);}
};
}
