#pragma once
#include "../dsv4_ck_route_major_20260916/route_major.cuh"

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
