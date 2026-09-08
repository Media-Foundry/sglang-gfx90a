#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <torch/all.h>
#include "custom_all_reduce.cuh"

namespace sglang {
struct Gfx90aTp8ArGeometryOracle {
  static int64_t signal_bytes() { return sizeof(aiter::Signal); }
  static void run(int64_t handle, tvm::ffi::TensorView input,
                  tvm::ffi::TensorView output, int64_t blocks) {
    if (!handle || input.ndim() != 2 || output.ndim() != 2 ||
        input.size(0) != 32 || input.size(1) != 4096 ||
        output.size(0) != 32 || output.size(1) != 4096 ||
        input.dtype().code != kDLBfloat || input.dtype().bits != 16 ||
        output.dtype().code != kDLBfloat || output.dtype().bits != 16 ||
        input.dtype().lanes != 1 || output.dtype().lanes != 1 ||
        input.device().device_id != output.device().device_id ||
        input.data_ptr() == output.data_ptr() ||
        !(blocks == 4 || blocks == 8 || blocks == 12 || blocks == 16 || blocks == 24 || blocks == 32))
      throw std::runtime_error("TP8 AR oracle requires separate BF16 M32/H4096 buffers and validated grid");
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(handle);
    if (comm->world_size_ != 8 || !comm->full_nvlink_ ||
        comm->rank_ < 0 || comm->rank_ >= 8)
      throw std::runtime_error("TP8 AR communicator contract mismatch");
    auto stream = sglang::host::LaunchKernel::resolve_device(input.device());
    // Preserve AIter's registered-buffer lookup, two-stage choice, rank order,
    // pack width and synchronization. Only vary its existing block_limit arg.
    comm->allreduce<__hip_bfloat16>(stream,
        static_cast<__hip_bfloat16*>(input.data_ptr()),
        static_cast<__hip_bfloat16*>(output.data_ptr()), 32 * 4096,
        false, 512, static_cast<int>(blocks));
  }
};
}  // namespace sglang
