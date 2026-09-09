#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <torch/all.h>
#include "custom_all_reduce.cuh"

namespace sglang {
struct Gfx90aTp8DsparkArOracle {
  static int64_t signal_bytes() { return sizeof(aiter::Signal); }
  static void run(int64_t handle, tvm::ffi::TensorView input,
                  tvm::ffi::TensorView output, int64_t blocks) {
    if (!handle || input.ndim() != 2 || output.ndim() != 2 ||
        !(input.size(0) == 64 || input.size(0) == 128) ||
        input.size(1) != 4096 || output.size(0) != input.size(0) ||
        output.size(1) != 4096 ||
        input.dtype().code != kDLBfloat || input.dtype().bits != 16 ||
        output.dtype().code != kDLBfloat || output.dtype().bits != 16 ||
        input.dtype().lanes != 1 || output.dtype().lanes != 1 ||
        input.device().device_id != output.device().device_id ||
        input.data_ptr() == output.data_ptr() || blocks < 1 || blocks > aiter::kMaxBlocks)
      throw std::runtime_error("TP8 DSpark AR oracle requires separate BF16 M64/M128 H4096 buffers");
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(handle);
    if (comm->world_size_ != 8 || !comm->full_nvlink_ || comm->rank_ < 0 || comm->rank_ >= 8)
      throw std::runtime_error("TP8 DSpark AR communicator mismatch");
    auto stream = sglang::host::LaunchKernel::resolve_device(input.device());
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, input.device().device_id);
    if (std::string(prop.gcnArchName).find("gfx90a") != 0)
      throw std::runtime_error("TP8 DSpark AR oracle requires gfx90a");
    // Exactly the new AIter two-stage kernel selected for these payloads.
    // Preserve registered peer addresses, pack width, rank order and barriers;
    // expose only its grid, without rebuilding or patching installed AIter.
    auto* peers = comm->get_buffer_RD(stream, input.data_ptr());
    const int packed_size = input.size(0) * 4096 / aiter::packed_t<__hip_bfloat16>::P::size;
    aiter::cross_device_reduce_2stage<__hip_bfloat16, 8>
        <<<static_cast<int>(blocks), 512, 0, stream>>>(
            peers, comm->sg_, comm->self_sg_,
            static_cast<__hip_bfloat16*>(output.data_ptr()), comm->rank_, packed_size);
    auto error = hipGetLastError();
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
  }
};
} // namespace sglang
