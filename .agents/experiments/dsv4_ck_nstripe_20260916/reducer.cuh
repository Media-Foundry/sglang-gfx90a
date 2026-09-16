#pragma once
#include "deepseek_v4/gfx90a_ck_fixed_slot.cuh"

namespace sglang {
template<class Out>
__global__ void stripe_reduce(const float* partial,Out* out,int rows,int width,int offset) {
  int vectors=rows*(width/4);
  for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<vectors;j+=blockDim.x*gridDim.x) {
    const int token=j/(width/4),col=(j%(width/4))*4;
    const size_t base=static_cast<size_t>(token)*6*width+col;
    auto sum=*reinterpret_cast<const CkSlotVec4<float>*>(partial+base);
    #pragma unroll
    for(int k=1;k<6;++k) {
      auto x=*reinterpret_cast<const CkSlotVec4<float>*>(partial+base+k*width);
      #pragma unroll
      for(int c=0;c<4;++c)sum.v[c]=sum.v[c]+x.v[c];
    }
    CkSlotVec4<Out> y;
    #pragma unroll
    for(int c=0;c<4;++c)y.v[c]=cast<Out>(sum.v[c]);
    *reinterpret_cast<CkSlotVec4<Out>*>(out+static_cast<size_t>(token)*4096+offset+col)=y;
  }
}
struct StripeReducer {
  template<class Out>
  static void run(tvm::ffi::TensorView p,tvm::ffi::TensorView o,int64_t offset) {
    using namespace host;
    int64_t m=o.size(0),n=p.size(2);
    RuntimeCheck(m>0 && m<=36864 && n>=128 && n<=4096 && n%128==0,"shape");
    RuntimeCheck(offset>=0 && offset+n<=4096 && offset%128==0,"offset");
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({m,6,n}).with_dtype<float>().with_device(device).verify(p);
    TensorMatcher({m,4096}).with_dtype<Out>().with_device(device).verify(o);
    RuntimeCheck(reinterpret_cast<uintptr_t>(p.data_ptr())%16==0,"alignment");
    LaunchKernel(1664,256,o.device())(stripe_reduce<Out>,static_cast<const float*>(p.data_ptr()),
      static_cast<Out*>(o.data_ptr()),int(m),int(n),int(offset));
  }
  static void bf16(tvm::ffi::TensorView p,tvm::ffi::TensorView o,int64_t n) {run<bf16_t>(p,o,n);}
  static void fp32(tvm::ffi::TensorView p,tvm::ffi::TensorView o,int64_t n) {run<float>(p,o,n);}
};
}
