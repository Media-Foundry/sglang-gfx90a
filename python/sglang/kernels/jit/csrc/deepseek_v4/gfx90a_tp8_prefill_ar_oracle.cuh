#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>
#include <torch/all.h>
#include "custom_all_reduce.cuh"

namespace sglang {
struct Gfx90aTp8PrefillArOracle {
  static int64_t signal_bytes() { return sizeof(aiter::Signal); }
  static void run(int64_t handle, tvm::ffi::TensorView input,
                  tvm::ffi::TensorView output, int64_t blocks,
                  int64_t capacity_bytes) {
    if (input.ndim() != 2 || output.ndim() != 2)
      throw std::runtime_error("TP8 prefill AR requires rank-two tensors");
    const int64_t rows = input.size(0);
    if (!(input.device().device_type == kDLCUDA || input.device().device_type == kDLROCM) ||
        input.device().device_type != output.device().device_type)
      throw std::runtime_error("TP8 prefill AR requires matching GPU device types");
    if (!handle || !input.IsContiguous() || !output.IsContiguous() ||
        rows < 8192 || rows > 36864 || input.size(1) != 4096 ||
        output.size(0) != rows || output.size(1) != 4096 ||
        rows * 4096 * 2 > capacity_bytes ||
        input.dtype().code != kDLBfloat || input.dtype().bits != 16 ||
        output.dtype().code != kDLBfloat || output.dtype().bits != 16 ||
        input.dtype().lanes != 1 || output.dtype().lanes != 1 ||
        input.device().device_id != output.device().device_id ||
        input.data_ptr() == output.data_ptr() ||
        !(blocks == 8 || blocks == 16 || blocks == 32 || blocks == 48 || blocks == 64 || blocks == 80))
      throw std::runtime_error("TP8 prefill AR requires registered BF16 M8192..36864/H4096 and valid capacity/grid");
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(handle);
    if (comm->world_size_ != 8 || !comm->full_nvlink_ || comm->rank_ < 0 || comm->rank_ >= 8)
      throw std::runtime_error("TP8 AR communicator contract mismatch");
    auto stream = sglang::host::LaunchKernel::resolve_device(input.device());
    comm->allreduce<__hip_bfloat16>(stream,
        static_cast<__hip_bfloat16*>(input.data_ptr()),
        static_cast<__hip_bfloat16*>(output.data_ptr()), rows * 4096,
        false, 512, static_cast<int>(blocks));
  }
};
}  // namespace sglang
