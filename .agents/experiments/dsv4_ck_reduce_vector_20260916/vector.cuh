#pragma once
#include "deepseek_v4/gfx90a_ck_fixed_slot.cuh"

namespace sglang {
template<class T> struct alignas(sizeof(T)*4) ReduceVec4 { T v[4]; };

template<class Out>
__global__ void ck_slot_reduce_vec4(const float* partial, Out* out, int vectors) {
  for (int j=blockIdx.x*blockDim.x+threadIdx.x; j<vectors;
       j+=blockDim.x*gridDim.x) {
    const int token=j/1024, col=(j%1024)*4;
    const size_t base=static_cast<size_t>(token)*6*4096+col;
    ReduceVec4<float> sum=*reinterpret_cast<const ReduceVec4<float>*>(partial+base);
    #pragma unroll
    for(int k=1;k<6;++k) {
      const auto x=*reinterpret_cast<const ReduceVec4<float>*>(partial+base+k*4096);
      #pragma unroll
      for(int c=0;c<4;++c) sum.v[c]=sum.v[c]+x.v[c];
    }
    ReduceVec4<Out> y;
    #pragma unroll
    for(int c=0;c<4;++c) y.v[c]=cast<Out>(sum.v[c]);
    reinterpret_cast<ReduceVec4<Out>*>(out)[j]=y;
  }
}

struct ReduceVectorScreen {
  template<class Out>
  static void run(const tvm::ffi::TensorView partial, const tvm::ffi::TensorView out,
                  int64_t blocks) {
    using namespace host;
    const int64_t rows=out.size(0);
    RuntimeCheck(rows>0 && rows<=36864, "unsupported rows");
    RuntimeCheck(blocks==416 || blocks==832 || blocks==1664, "unsupported grid");
    auto device=SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({rows,6,4096}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({rows,4096}).with_dtype<Out>().with_device(device).verify(out);
    RuntimeCheck(reinterpret_cast<uintptr_t>(partial.data_ptr())%16==0, "unaligned input");
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr())%(4*sizeof(Out))==0, "unaligned output");
    LaunchKernel(blocks,256,out.device())(ck_slot_reduce_vec4<Out>,
      static_cast<const float*>(partial.data_ptr()),static_cast<Out*>(out.data_ptr()),
      static_cast<int>(rows*1024));
  }
  static void bf16(const tvm::ffi::TensorView p,const tvm::ffi::TensorView o,int64_t b) {run<bf16_t>(p,o,b);}
  static void fp32(const tvm::ffi::TensorView p,const tvm::ffi::TensorView o,int64_t b) {run<float>(p,o,b);}
};
}
