#pragma once
#include "gfx90a_fp4_bf16_dequant_oracle.cuh"

namespace sglang {
struct alignas(16) DirectRowsBf16x8 { bf16_t values[8]; };

template<uint32_t E,uint32_t N,uint32_t K>
__global__ void fp4_bf16_direct_rows(bf16_t* __restrict__ out,const uint8_t* __restrict__ w,
                                    const uint8_t* __restrict__ scales) {
  constexpr uint32_t N0=N/16,K0=K/32,TILES=E*N0*K0;
  const uint32_t lane=threadIdx.x%64,wave=threadIdx.x/64,nlane=lane%16;
  for(uint32_t tile=(blockIdx.x*4+wave)*4+lane/16;tile<TILES;tile+=gridDim.x*16) {
    const uint32_t expert=tile/(N0*K0),inner=tile%(N0*K0),nr=inner/K0,kr=inner%K0;
    const size_t row=static_cast<size_t>(expert)*N+nr*16+nlane;
    const uint4 packed=*reinterpret_cast<const uint4*>(w+row*(K/2)+kr*16);
    const float s=gfx90a_e8m0_value(scales[row*(K/32)+kr])*0.5f;
    #pragma unroll
    for(uint32_t klane=0;klane<4;++klane) {
      const uint32_t word=klane==0 ? packed.x : klane==1 ? packed.y : klane==2 ? packed.z : packed.w;
      DirectRowsBf16x8 decoded;
      #pragma unroll
      for(uint32_t j=0;j<8;++j)
        decoded.values[j]=cast<bf16_t>(static_cast<float>(gfx90a_fp4_i8_code((word>>(j*4))&15))*s);
      *reinterpret_cast<DirectRowsBf16x8*>(out+static_cast<size_t>(tile)*512+klane*128+nlane*8)=decoded;
    }
  }
}

template<uint32_t N,uint32_t K,uint32_t Blocks>
struct Gfx90aFp4Bf16DirectRows {
  static void run(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView out) {
    using namespace host;
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({256,N,K/2}).with_dtype<uint8_t>().with_device(device).verify(w);
    TensorMatcher({256,N,K/32}).with_dtype<uint8_t>().with_device(device).verify(s);
    TensorMatcher({256,N,K}).with_dtype<bf16_t>().with_device(device).verify(out);
    RuntimeCheck(reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"unaligned packed row");
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr())%16==0,"unaligned BF16 output");
    LaunchKernel(Blocks,256,w.device())(fp4_bf16_direct_rows<256,N,K>,static_cast<bf16_t*>(out.data_ptr()),
      static_cast<const uint8_t*>(w.data_ptr()),static_cast<const uint8_t*>(s.data_ptr()));
  }
};
}
