#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <cstdint>

namespace sglang {

__global__ void gfx90a_realtime_marker_kernel(uint64_t* output, int slot) {
  if (threadIdx.x == 0) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    output[slot] = __builtin_amdgcn_s_memrealtime();
#else
    output[slot] = clock64();
#endif
  }
}

struct Gfx90aRealtimeMarkerKernel {
  static void run(tvm::ffi::TensorView output, int64_t slot) {
    using namespace host;
    RuntimeCheck(output.IsContiguous(), "output must be contiguous");
    RuntimeCheck(output.dtype().code == kDLUInt && output.dtype().bits == 64,
                 "output must be uint64");
    RuntimeCheck(slot >= 0 && slot < output.numel(), "slot out of bounds");
    const auto stream = LaunchKernel::resolve_device(output.device());
    LaunchKernel(1, 64, stream)(
        gfx90a_realtime_marker_kernel,
        static_cast<uint64_t*>(output.data_ptr()), static_cast<int>(slot));
  }
};

}  // namespace sglang
