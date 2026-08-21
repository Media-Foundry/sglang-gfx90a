#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

__global__ void gfx90a_system_fence_kernel() {
  if (threadIdx.x == 0) __threadfence_system();
}

struct Gfx90aSystemFenceKernel {
  static void run(const tvm::ffi::TensorView anchor) {
    using namespace host;
    auto n = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({n}).with_dtype<bf16_t>().with_device(device).verify(anchor);
    LaunchKernel(1, 1, device.unwrap())(gfx90a_system_fence_kernel);
  }
};

}  // namespace sglang
