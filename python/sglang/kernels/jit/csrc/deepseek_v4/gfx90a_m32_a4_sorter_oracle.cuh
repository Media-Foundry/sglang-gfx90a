#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

__global__ void gfx90a_m32_a4_sorter_oracle_kernel(
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int32_t* __restrict__ sorted_ids,
    float* __restrict__ sorted_weights,
    int32_t* __restrict__ sorted_experts,
    int32_t* __restrict__ num_valid) {
  constexpr int kM = 32;
  constexpr int kTopK = 6;
  constexpr int kExperts = 256;
  constexpr int kAssignments = 4;
  constexpr int kSortedCapacity = kM * kTopK + kExperts * kAssignments - kTopK;
  __shared__ int32_t counts[kExperts];
  __shared__ int32_t scan[kExperts];
  const int tid = threadIdx.x;
  counts[tid] = 0;
  for (int index = tid; index < kSortedCapacity; index += blockDim.x) {
    sorted_ids[index] = (kTopK << 24) | kM;
    sorted_weights[index] = 0.0f;
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
  for (int block = 0; block < blocks; ++block) {
    sorted_experts[block_begin + block] = tid;
  }
  __syncthreads();
  if (tid < kM * kTopK) {
    const int expert = topk_ids[tid];
    if (expert >= 0 && expert < kExperts) {
      const int token = tid / kTopK;
      const int slot = tid - token * kTopK;
      // AIter's sorter is stable in flattened token/slot order. Avoid an
      // atomic cursor whose arbitration can permute assignments of one expert.
      int local_rank = 0;
      for (int previous = 0; previous < tid; ++previous) {
        local_rank += topk_ids[previous] == expert;
      }
      const int destination = scan[expert] * kAssignments + local_rank;
      sorted_ids[destination] = (slot << 24) | token;
      sorted_weights[destination] = topk_weights[tid];
    }
  }
  __syncthreads();
  if (tid == 0) {
    const int total_blocks = scan[kExperts - 1] +
        (counts[kExperts - 1] + kAssignments - 1) / kAssignments;
    num_valid[0] = total_blocks * kAssignments;
    num_valid[1] = kM;
  }
}

struct Gfx90aM32A4SorterOracle {
  static void run(const tvm::ffi::TensorView topk_ids,
                  const tvm::ffi::TensorView topk_weights,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_weights,
                  const tvm::ffi::TensorView sorted_experts,
                  const tvm::ffi::TensorView num_valid) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({32, 6}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({32, 6}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({1210}).with_dtype<int32_t>().with_device(device).verify(sorted_ids);
    TensorMatcher({1210}).with_dtype<float>().with_device(device).verify(sorted_weights);
    TensorMatcher({303}).with_dtype<int32_t>().with_device(device).verify(sorted_experts);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid);
    LaunchKernel(1, 256, topk_ids.device())(
        gfx90a_m32_a4_sorter_oracle_kernel,
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<int32_t*>(sorted_ids.data_ptr()),
        static_cast<float*>(sorted_weights.data_ptr()),
        static_cast<int32_t*>(sorted_experts.data_ptr()),
        static_cast<int32_t*>(num_valid.data_ptr()));
  }
};

}  // namespace sglang
