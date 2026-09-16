#pragma once
#include "deepseek_v4/gfx90a_ck_fixed_slot.cuh"

namespace sglang {
struct alignas(16) RouteBf16x8 { bf16_t v[8]; };

// Change storage ownership only: every real (token,slot) still appears once.
__global__ void route_pack(const bf16_t* inter, const int32_t* ids,
                           const int32_t* valid, bf16_t* packed,
                           int32_t* identity, int32_t* inverse, int rows, int capacity) {
  for (int j=blockIdx.x*blockDim.x+threadIdx.x; j<capacity*32;
       j+=blockDim.x*gridDim.x) {
    int route=j/32, col=(j%32)*8;
    uint32_t encoded=route<valid[0] ? static_cast<uint32_t>(ids[route]) : 0xffffffffu;
    uint32_t token=encoded&0xffffffu, slot=encoded>>24;
    bool live=token<rows && slot<6;
    RouteBf16x8 value{};
    if (live) value=*reinterpret_cast<const RouteBf16x8*>(inter+(size_t(token)*6+slot)*256+col);
    *reinterpret_cast<RouteBf16x8*>(packed+size_t(route)*256+col)=value;
    if (col==0) {
      identity[route]=live ? route : capacity;
      if (live) inverse[token*6+slot]=route;
    }
  }
}

template<class Out>
__global__ void route_reduce(const float* partial, const int32_t* inverse, Out* out, int rows) {
  for (int j=blockIdx.x*blockDim.x+threadIdx.x; j<rows*1024;
       j+=blockDim.x*gridDim.x) {
    int token=j/1024, col=(j%1024)*4;
    auto sum=*reinterpret_cast<const CkSlotVec4<float>*>(partial+size_t(inverse[token*6])*4096+col);
    #pragma unroll
    for (int slot=1;slot<6;++slot) {
      auto x=*reinterpret_cast<const CkSlotVec4<float>*>(partial+size_t(inverse[token*6+slot])*4096+col);
      #pragma unroll
      for (int c=0;c<4;++c) sum.v[c]=sum.v[c]+x.v[c];
    }
    CkSlotVec4<Out> y;
    #pragma unroll
    for (int c=0;c<4;++c) y.v[c]=cast<Out>(sum.v[c]);
    reinterpret_cast<CkSlotVec4<Out>*>(out)[j]=y;
  }
}

struct RouteMajor {
  static void pack(tvm::ffi::TensorView inter,tvm::ffi::TensorView ids,tvm::ffi::TensorView valid,
                   tvm::ffi::TensorView packed,tvm::ffi::TensorView identity,tvm::ffi::TensorView inverse) {
    using namespace host;
    int64_t m=inter.size(0),s=ids.size(0);
    RuntimeCheck(m>=8192 && m<=36864 && s>=m*6 && s<(1<<24),"route-major shape");
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({m,6,256}).with_dtype<bf16_t>().with_device(device).verify(inter);
    TensorMatcher({s}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(valid);
    TensorMatcher({s,256}).with_dtype<bf16_t>().with_device(device).verify(packed);
    TensorMatcher({s}).with_dtype<int32_t>().with_device(device).verify(identity);
    TensorMatcher({m,6}).with_dtype<int32_t>().with_device(device).verify(inverse);
    RuntimeCheck(reinterpret_cast<uintptr_t>(inter.data_ptr())%16==0 &&
                 reinterpret_cast<uintptr_t>(packed.data_ptr())%16==0,"route pack alignment");
    LaunchKernel(832,256,inter.device())(route_pack,static_cast<const bf16_t*>(inter.data_ptr()),
        static_cast<const int32_t*>(ids.data_ptr()),static_cast<const int32_t*>(valid.data_ptr()),
        static_cast<bf16_t*>(packed.data_ptr()),static_cast<int32_t*>(identity.data_ptr()),
        static_cast<int32_t*>(inverse.data_ptr()),int(m),int(s));
  }
  template<class Out>
  static void reduce(tvm::ffi::TensorView partial,tvm::ffi::TensorView inverse,tvm::ffi::TensorView out) {
    using namespace host;
    int64_t m=out.size(0),s=partial.size(0);
    RuntimeCheck(m>=8192 && m<=36864 && s>=m*6 && s<(1<<24),"route reduce shape");
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({s,4096}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({m,6}).with_dtype<int32_t>().with_device(device).verify(inverse);
    TensorMatcher({m,4096}).with_dtype<Out>().with_device(device).verify(out);
    RuntimeCheck(reinterpret_cast<uintptr_t>(partial.data_ptr())%16==0 &&
                 reinterpret_cast<uintptr_t>(out.data_ptr())%(4*sizeof(Out))==0,"route reduce alignment");
    LaunchKernel(1664,256,out.device())(route_reduce<Out>,static_cast<const float*>(partial.data_ptr()),
        static_cast<const int32_t*>(inverse.data_ptr()),static_cast<Out*>(out.data_ptr()),int(m));
  }
  static void bf16(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {reduce<bf16_t>(p,i,o);}
  static void fp32(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {reduce<float>(p,i,o);}
};
}


namespace sglang {
__global__ void route_metadata(const int32_t* ids,const int32_t* valid,
                               int32_t* identity,int32_t* inverse,int rows,int capacity) {
  for(int route=blockIdx.x*blockDim.x+threadIdx.x;route<capacity;route+=blockDim.x*gridDim.x) {
    uint32_t encoded=route<valid[0] ? static_cast<uint32_t>(ids[route]) : 0xffffffffu;
    uint32_t token=encoded&0xffffffu,slot=encoded>>24;
    bool live=token<rows && slot<6;
    identity[route]=live ? route : capacity;
    if(live) inverse[token*6+slot]=route;
  }
}
struct RouteMetadata {
  static void run(tvm::ffi::TensorView ids,tvm::ffi::TensorView valid,
                  tvm::ffi::TensorView identity,tvm::ffi::TensorView inverse) {
    using namespace host;
    int64_t m=inverse.size(0),s=ids.size(0);
    RuntimeCheck(m>=8192 && m<=36864 && s>=m*6 && s<(1<<24),"route metadata shape");
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({s}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(valid);
    TensorMatcher({s}).with_dtype<int32_t>().with_device(device).verify(identity);
    TensorMatcher({m,6}).with_dtype<int32_t>().with_device(device).verify(inverse);
    LaunchKernel(832,256,ids.device())(route_metadata,static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const int32_t*>(valid.data_ptr()),static_cast<int32_t*>(identity.data_ptr()),
      static_cast<int32_t*>(inverse.data_ptr()),int(m),int(s));
  }
};
}
