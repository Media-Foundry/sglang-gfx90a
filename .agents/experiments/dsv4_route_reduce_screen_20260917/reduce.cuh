#pragma once
#include "deepseek_v4/gfx90a_ck_route_producer.cuh"

namespace sglang {
template<class T, int V> struct alignas(sizeof(T)*V) RouteScreenVec { T v[V]; };

// Every wave covers one token:4096/V is a multiple of64, the block and
// persistent-grid stride are multiples of256, and all rows are fullH4096.
// The broadcast variant only changes address ownership, never slot order.
template<class Out, int V, bool Broadcast>
__global__ void route_reduce_screen(const float* partial, const int32_t* inverse,
                                    Out* out, int rows) {
  constexpr int vectors_per_row=4096/V;
  for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<rows*vectors_per_row;
      j+=blockDim.x*gridDim.x) {
    int token=j/vectors_per_row;
    if constexpr(Broadcast) token=__builtin_amdgcn_readfirstlane(token);
    int col=(j%vectors_per_row)*V;
    int route=inverse[token*6];
    if constexpr(Broadcast) route=__builtin_amdgcn_readfirstlane(route);
    auto sum=*reinterpret_cast<const RouteScreenVec<float,V>*>(partial+size_t(route)*4096+col);
    #pragma unroll
    for(int slot=1;slot<6;++slot) {
      route=inverse[token*6+slot];
      if constexpr(Broadcast) route=__builtin_amdgcn_readfirstlane(route);
      const auto x=*reinterpret_cast<const RouteScreenVec<float,V>*>(partial+size_t(route)*4096+col);
      #pragma unroll
      for(int c=0;c<V;++c) sum.v[c]=sum.v[c]+x.v[c];
    }
    RouteScreenVec<Out,V> y;
    #pragma unroll
    for(int c=0;c<V;++c) y.v[c]=cast<Out>(sum.v[c]);
    reinterpret_cast<RouteScreenVec<Out,V>*>(out)[j]=y;
  }
}

struct RouteReduceScreen {
  template<class Out, int V, bool Broadcast>
  static void run(tvm::ffi::TensorView partial,tvm::ffi::TensorView inverse,
                  tvm::ffi::TensorView out) {
    using namespace host;
    int64_t m=out.size(0),s=partial.size(0);
    RuntimeCheck(m>=8192 && m<=36864 && s>=m*6 && s<(1<<24),"route screen shape");
    auto device=SymbolicDevice{};device.set_options<kDLCUDA>();
    TensorMatcher({s,4096}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({m,6}).with_dtype<int32_t>().with_device(device).verify(inverse);
    TensorMatcher({m,4096}).with_dtype<Out>().with_device(device).verify(out);
    RuntimeCheck(reinterpret_cast<uintptr_t>(partial.data_ptr())%(sizeof(float)*V)==0 &&
                 reinterpret_cast<uintptr_t>(out.data_ptr())%(sizeof(Out)*V)==0,"route vector alignment");
    LaunchKernel(1664,256,out.device())(route_reduce_screen<Out,V,Broadcast>,
      static_cast<const float*>(partial.data_ptr()),static_cast<const int32_t*>(inverse.data_ptr()),
      static_cast<Out*>(out.data_ptr()),int(m));
  }
  static void v4wave(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {run<bf16_t,4,true>(p,i,o);}
  static void v8(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {run<bf16_t,8,false>(p,i,o);}
  static void v8wave(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {run<bf16_t,8,true>(p,i,o);}
  static void v4wave_f32(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {run<float,4,true>(p,i,o);}
  static void v8_f32(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {run<float,8,false>(p,i,o);}
  static void v8wave_f32(tvm::ffi::TensorView p,tvm::ffi::TensorView i,tvm::ffi::TensorView o) {run<float,8,true>(p,i,o);}
};
}
