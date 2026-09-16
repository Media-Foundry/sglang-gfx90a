#pragma once
#include "premix.cuh"
namespace sglang {
template<int BK,bool Pad=false,int Split=1>
__global__ void mhc_cooperative_kernel(const bf16_t* x,const float* fn,float* out,int m) {
  constexpr int K=16384,N=24;
  // Only a bounded CTA tile: activations remain BF16 in LDS.
  constexpr int AS=BK+(Pad?8:0),BS=BK+(Pad?4:0);
  __shared__ bf16_t as[64*AS];
  __shared__ float bs[32*BS];
  const int tid=threadIdx.x,lane=tid%64,wave=tid/64;
  const int mb=blockIdx.x*64;
  MhcFloat4 c0={0,0,0,0},c1={0,0,0,0};
  for(int kb=blockIdx.z*(K/Split);kb<(blockIdx.z+1)*(K/Split);kb+=BK) {
    // Aligned contiguous vectors: no per-element scalar global loads.
    #pragma unroll
    for(int j=tid;j<64*BK/8;j+=256) {
      const int row=j/(BK/8),col=(j%(BK/8))*8;
      uint4 v={0,0,0,0};
      if(mb+row<m) v=*reinterpret_cast<const uint4*>(x+static_cast<size_t>(mb+row)*K+kb+col);
      *reinterpret_cast<uint4*>(as+row*AS+col)=v;
    }
    #pragma unroll
    for(int j=tid;j<32*BK/4;j+=256) {
      const int row=j/(BK/4),col=(j%(BK/4))*4;
      float4 v={0,0,0,0};
      if(row<N) v=*reinterpret_cast<const float4*>(fn+row*K+kb+col);
      *reinterpret_cast<float4*>(bs+row*BS+col)=v;
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<BK;k+=4) {
      const int kk=k+lane/16;
      const float a=static_cast<float>(as[(wave*16+lane%16)*AS+kk]);
      const float b0=bs[(lane%16)*BS+kk];
      const float b1=bs[(16+lane%16)*BS+kk];
      c0=__builtin_amdgcn_mfma_f32_16x16x4f32(a,b0,c0,0,0,0);
      c1=__builtin_amdgcn_mfma_f32_16x16x4f32(a,b1,c1,0,0,0);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int row=mb+wave*16+(lane/16)*4+i,col=lane%16;
    if(row<m) {
      out[(static_cast<size_t>(blockIdx.z)*m+row)*N+col]=c0[i];
      if(col+16<N) out[(static_cast<size_t>(blockIdx.z)*m+row)*N+col+16]=c1[i];
    }
  }
}
struct MhcCooperativeScreen {
  template<int BK,bool Pad=false,int Split=1>
  static void run(const tvm::ffi::TensorView x,const tvm::ffi::TensorView fn,const tvm::ffi::TensorView out) {
    using namespace host;
    const int64_t m=x.size(0);RuntimeCheck(m>0 && m<=65536,"invalid M");
    auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();
    TensorMatcher({m,16384}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({24,16384}).with_dtype<float>().with_device(dev).verify(fn);
    if constexpr(Split==1) TensorMatcher({m,24}).with_dtype<float>().with_device(dev).verify(out);
    else TensorMatcher({Split,m,24}).with_dtype<float>().with_device(dev).verify(out);
    LaunchKernel(dim3((m+63)/64,1,Split),256,x.device())(mhc_cooperative_kernel<BK,Pad,Split>,
      static_cast<const bf16_t*>(x.data_ptr()),static_cast<const float*>(fn.data_ptr()),static_cast<float*>(out.data_ptr()),static_cast<int>(m));
  }
  static void k32(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<32>(a,b,c);}
  static void k64(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<64>(a,b,c);}
  static void k128(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<128>(a,b,c);}
  static void p32(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<32,true>(a,b,c);}
  static void p64(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<64,true>(a,b,c);}
  static void p128(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<128,true>(a,b,c);}
  static void s4(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<64,true,4>(a,b,c);}
  static void s16(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c){run<64,true,16>(a,b,c);}
};
}
