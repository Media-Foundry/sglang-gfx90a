#pragma once
#include "deepseek_v4/gfx90a_fp4_bf16_dequant_oracle.cuh"

namespace sglang {
struct alignas(16) DequantBf16x8 { bf16_t values[8]; };
struct alignas(16) DequantBitsx8 { uint16_t values[8]; };

template<uint32_t E,uint32_t N,uint32_t K>
__global__ void row_preshuffle(bf16_t* __restrict__ out,const uint8_t* __restrict__ w,
                               const uint8_t* __restrict__ scales) {
  constexpr uint32_t N0=N/16,K0=K/32,TILES=E*N0*K0;
  const uint32_t lane=threadIdx.x%64,wave=threadIdx.x/64,nlane=lane%16;
  for(uint32_t tile=(blockIdx.x*4+wave)*4+lane/16;tile<TILES;tile+=gridDim.x*16) {
    const uint32_t expert=tile/(N0*K0),inner=tile%(N0*K0),nr=inner/K0,kr=inner%K0;
    const size_t row=static_cast<size_t>(expert)*N+nr*16+nlane;
    const uint4 packed=*reinterpret_cast<const uint4*>(w+row*(K/2)+kr*16);
    const uint32_t exponent=scales[row*(K/32)+kr];
    const float s=gfx90a_e8m0_value(exponent)*0.5f;
    #pragma unroll
    for(uint32_t klane=0;klane<4;++klane) {
      const uint32_t word=klane==0 ? packed.x : klane==1 ? packed.y : klane==2 ? packed.z : packed.w;
      DequantBf16x8 decoded;
      #pragma unroll
      for(uint32_t j=0;j<8;++j)
        decoded.values[j]=cast<bf16_t>(static_cast<float>(gfx90a_fp4_i8_code((word>>(j*4))&15))*s);
      *reinterpret_cast<DequantBf16x8*>(out+static_cast<size_t>(tile)*512+klane*128+nlane*8)=decoded;
    }
  }
}

template<uint32_t E,uint32_t N,uint32_t K,bool Bits=false>
__global__ void direct_preshuffle(bf16_t* __restrict__ out,const uint8_t* __restrict__ w,
                                  const uint8_t* __restrict__ scales) {
  constexpr uint32_t N0=N/16,K0=K/32,TILES=E*N0*K0;
  const uint32_t lane=threadIdx.x%64,wave=threadIdx.x/64;
  for(uint32_t tile=blockIdx.x*4+wave;tile<TILES;tile+=gridDim.x*4) {
    const uint32_t expert=tile/(N0*K0),inner=tile%(N0*K0);
    const uint32_t nr=inner/K0,kr=inner%K0;
    // Destination tile is [klane4,nlane16,kpack8], all contiguous.
    const uint32_t nlane=lane%16,klane=lane/16;
    const size_t row=static_cast<size_t>(expert)*N+nr*16+nlane;
    const uint32_t packed=*reinterpret_cast<const uint32_t*>(w+row*(K/2)+kr*16+klane*4);
    const uint32_t exponent=scales[row*(K/32)+kr];
    if constexpr(Bits) {
      // Normal BF16 products are exactly representable. Preserve the original
      // multiply/cast for underflow, overflow and exceptional scale encodings.
      if(exponent>=2 && exponent<=252) {
        DequantBitsx8 decoded;
        #pragma unroll
        for(uint32_t j=0;j<8;++j) {
          const uint32_t code=(packed>>(j*4))&15,mag=code&7;
          decoded.values[j]=mag==0 ? 0 : ((code&8)<<12) |
            ((exponent+(mag>>1)-1)<<7) | ((mag>1 ? mag&1 : 0)<<6);
        }
        *reinterpret_cast<DequantBitsx8*>(out+static_cast<size_t>(tile)*512+lane*8)=decoded;
        continue;
      }
    }
    const float s=gfx90a_e8m0_value(exponent)*0.5f;
    DequantBf16x8 decoded;
    #pragma unroll
    for(uint32_t j=0;j<8;++j)
      decoded.values[j]=cast<bf16_t>(static_cast<float>(gfx90a_fp4_i8_code((packed>>(j*4))&15))*s);
    *reinterpret_cast<DequantBf16x8*>(out+static_cast<size_t>(tile)*512+lane*8)=decoded;
  }
}

struct DirectDequant {
  template<uint32_t E,uint32_t N,uint32_t K,bool Bits=false,bool Rows=false>
  static void run(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView out,int64_t blocks) {
    using namespace host;
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({E,N,K/2}).with_dtype<uint8_t>().with_device(device).verify(w);
    TensorMatcher({E,N,K/32}).with_dtype<uint8_t>().with_device(device).verify(s);
    TensorMatcher({E,N,K}).with_dtype<bf16_t>().with_device(device).verify(out);
    RuntimeCheck(blocks==416 || blocks==832 || blocks==1664,"bounded grid");
    RuntimeCheck(reinterpret_cast<uintptr_t>(w.data_ptr())%4==0,"unaligned packed weights");
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr())%16==0,"unaligned BF16 output");
    if constexpr(Rows) {
      RuntimeCheck(reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"unaligned row vector");
      LaunchKernel(blocks,256,w.device())(row_preshuffle<E,N,K>,static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),static_cast<const uint8_t*>(s.data_ptr()));
    } else {
      LaunchKernel(blocks,256,w.device())(direct_preshuffle<E,N,K,Bits>,static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),static_cast<const uint8_t*>(s.data_ptr()));
    }
  }
  static void gate(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<256,512,4096>(w,s,o,b);}
  static void down(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<256,4096,256>(w,s,o,b);}
  static void tiny(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<2,16,32>(w,s,o,b);}
  static void gate_bits(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<256,512,4096,true>(w,s,o,b);}
  static void down_bits(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<256,4096,256,true>(w,s,o,b);}
  static void tiny_bits(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<2,16,32,true>(w,s,o,b);}
  static void gate_rows(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<256,512,4096,false,true>(w,s,o,b);}
  static void down_rows(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<256,4096,256,false,true>(w,s,o,b);}
  static void tiny_rows(tvm::ffi::TensorView w,tvm::ffi::TensorView s,tvm::ffi::TensorView o,int64_t b) {run<2,16,32,false,true>(w,s,o,b);}
};
}
