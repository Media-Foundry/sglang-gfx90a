#pragma once
#include "deepseek_v4/gfx90a_fp4_expert_gemv.cuh"

namespace sglang {
using MhcFloat4 = float __attribute__((ext_vector_type(4)));
template<int Mapping,int Unroll>
__global__ void mhc_fp32_mfma_kernel(const bf16_t* x,const float* fn,float* out,int m,int n,int k) {
  const int lane=threadIdx.x%64,wave=threadIdx.x/64;
  const int mbase=(blockIdx.x*4+wave)*16,nbase=blockIdx.y*16;
  const int ar=mbase+lane%16,bn=nbase+lane%16,ki=lane/16;
  MhcFloat4 acc={0,0,0,0};
  for(int start=0;start<k;start+=4*Unroll) {
    float a[Unroll],b[Unroll];
    #pragma unroll
    for(int u=0;u<Unroll;++u) {
      const int kk=start+4*u+ki;
      a[u]=(ar<m && kk<k)?static_cast<float>(x[static_cast<size_t>(ar)*k+kk]):0.0f;
      b[u]=(bn<n && kk<k)?fn[static_cast<size_t>(bn)*k+kk]:0.0f;
    }
    #pragma unroll
    for(int u=0;u<Unroll;++u)
      acc=__builtin_amdgcn_mfma_f32_16x16x4f32(a[u],b[u],acc,0,0,0);
  }
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int row=mbase+(Mapping==0?(lane/16)*4+i:lane%16);
    const int col=nbase+(Mapping==0?lane%16:(lane/16)*4+i);
    if(row<m && col<n) out[static_cast<size_t>(row)*n+col]=acc[i];
  }
}
struct MhcMfmaScreen {
  template<int Mapping,int Unroll>
  static void run(const tvm::ffi::TensorView x,const tvm::ffi::TensorView fn,const tvm::ffi::TensorView out) {
    using namespace host;
    const int64_t m=x.size(0),k=x.size(1),n=fn.size(0);
    RuntimeCheck(m>0 && m<=65536 && k>0 && k<=16384 && k%4==0,"invalid M/K");
    RuntimeCheck(n==16 || n==24 || n==32,"invalid N");
    auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();
    TensorMatcher({m,k}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({n,k}).with_dtype<float>().with_device(dev).verify(fn);
    TensorMatcher({m,n}).with_dtype<float>().with_device(dev).verify(out);
    LaunchKernel(dim3((m+63)/64,(n+15)/16),256,x.device())(mhc_fp32_mfma_kernel<Mapping,Unroll>,
      static_cast<const bf16_t*>(x.data_ptr()),static_cast<const float*>(fn.data_ptr()),static_cast<float*>(out.data_ptr()),
      static_cast<int>(m),static_cast<int>(n),static_cast<int>(k));
  }
  static void map0(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<0,1>(a,b,c);}
  static void map1(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<1,1>(a,b,c);}
  static void u4(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<0,4>(a,b,c);}
  static void u8(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<0,8>(a,b,c);}
};
}
