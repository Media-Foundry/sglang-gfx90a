#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

namespace sglang {

__global__ void gfx90a_m32_a4_sorter_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ sorted_experts,
    int32_t* __restrict__ num_valid) {
  constexpr int kM = 32;
  constexpr int kTopK = 6;
  constexpr int kExperts = 256;
  constexpr int kAssignments = 4;
  __shared__ int32_t counts[kExperts];
  __shared__ int32_t scan[kExperts];
  __shared__ int32_t cursors[kExperts];

  const int tid = threadIdx.x;
  counts[tid] = 0;
  for (int index = tid; index < 768; index += blockDim.x) {
    sorted_ids[index] = (kTopK << 24) | kM;
  }
  __syncthreads();

  if (tid < kM * kTopK) {
    const int expert = topk_ids[tid];
    if (expert >= 0 && expert < kExperts) atomicAdd(counts + expert, 1);
  }
  __syncthreads();

  scan[tid] = (counts[tid] + kAssignments - 1) / kAssignments;
  __syncthreads();
  for (int offset = 1; offset < kExperts; offset <<= 1) {
    const int index = (tid + 1) * offset * 2 - 1;
    if (index < kExperts) scan[index] += scan[index - offset];
    __syncthreads();
  }
  if (tid == 0) scan[kExperts - 1] = 0;
  __syncthreads();
  for (int offset = kExperts / 2; offset >= 1; offset >>= 1) {
    const int index = (tid + 1) * offset * 2 - 1;
    if (index < kExperts) {
      const int left = scan[index - offset];
      scan[index - offset] = scan[index];
      scan[index] += left;
    }
    __syncthreads();
  }

  const int block_begin = scan[tid];
  const int blocks = (counts[tid] + kAssignments - 1) / kAssignments;
  cursors[tid] = block_begin * kAssignments;
  for (int block = 0; block < blocks; ++block) {
    sorted_experts[block_begin + block] = tid;
  }
  __syncthreads();

  if (tid < kM * kTopK) {
    const int expert = topk_ids[tid];
    if (expert >= 0 && expert < kExperts) {
      const int token = tid / kTopK;
      const int slot = tid - token * kTopK;
      const int write = atomicAdd(cursors + expert, 1);
      sorted_ids[write] = (slot << 24) | token;
    }
  }

  __syncthreads();
  if (tid == 0) {
    const int total_blocks = scan[kExperts - 1] +
                             (counts[kExperts - 1] + kAssignments - 1) /
                                 kAssignments;
    num_valid[0] = total_blocks * kAssignments;
    num_valid[1] = kM;
  }
}

struct Gfx90aM32A4Sorter {
  static void run(const tvm::ffi::TensorView topk_ids,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_experts,
                  const tvm::ffi::TensorView num_valid) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({32, 6}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({768}).with_dtype<int32_t>().with_device(device).verify(sorted_ids);
    TensorMatcher({192}).with_dtype<int32_t>().with_device(device).verify(sorted_experts);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid);
    LaunchKernel(1, 256, topk_ids.device())(
        gfx90a_m32_a4_sorter_kernel,
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<int32_t*>(sorted_ids.data_ptr()),
        static_cast<int32_t*>(sorted_experts.data_ptr()),
        static_cast<int32_t*>(num_valid.data_ptr()));
  }
};

}  // namespace sglang
