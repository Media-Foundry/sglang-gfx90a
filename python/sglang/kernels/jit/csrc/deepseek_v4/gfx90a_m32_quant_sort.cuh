#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

using namespace device;

template <int kM>
__global__ void gfx90a_m32_quant_sort_kernel(
    const bf16_t* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ sorted_experts,
    int32_t* __restrict__ num_valid) {
  constexpr int kTopK = 6;
  constexpr int kExperts = 256;
  constexpr int kAssignments = 4;
  constexpr int kWave = 64;
  constexpr int kSubgroup = 16;

  constexpr uint32_t kGroupsPerBlock = 16;
  constexpr uint32_t kQuantBlocks = kM * (4096 / 32) / kGroupsPerBlock;
  if (blockIdx.x < kQuantBlocks) {
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t wave = threadIdx.x / kWave;
    const uint32_t subgroup = lane / kSubgroup;
    const uint32_t subgroup_lane = lane & (kSubgroup - 1);
    const uint32_t group = blockIdx.x * kGroupsPerBlock +
                           wave * 4 + subgroup;
    const size_t base = static_cast<size_t>(group) * 32;
    const float x0 = cast<float>(input[base + subgroup_lane]);
    const float x1 = cast<float>(input[base + 16 + subgroup_lane]);
    float absmax = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
    for (uint32_t offset = 8; offset > 0; offset >>= 1) {
      absmax = fmaxf(absmax, __shfl_xor(absmax, offset, kSubgroup));
    }
    const float scale = fmaxf(absmax, 1.0e-10f) / 127.0f;
    output[base + subgroup_lane] = static_cast<int8_t>(
        fmaxf(-128.0f, fminf(127.0f, x0 / scale)));
    output[base + 16 + subgroup_lane] = static_cast<int8_t>(
        fmaxf(-128.0f, fminf(127.0f, x1 / scale)));
    if (subgroup_lane == 0) scales[group] = scale;
    return;
  }

  __shared__ int32_t counts[kExperts];
  __shared__ int32_t scan[kExperts];
  const int tid = threadIdx.x;
  counts[tid] = 0;
  for (int index = tid; index < kM * kTopK * kAssignments;
       index += blockDim.x) {
    sorted_ids[index] = (kTopK << 24) | kM;
  }
  __syncthreads();
  for (int assignment = tid; assignment < kM * kTopK;
       assignment += blockDim.x) {
    const int expert = topk_ids[assignment];
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
  // Preserve AIter's original token-major assignment order. Atomic cursor
  // insertion is race ordered and can silently permute same-expert rows.
  for (int assignment = tid; assignment < kM * kTopK;
       assignment += blockDim.x) {
    const int expert = topk_ids[assignment];
    if (expert >= 0 && expert < kExperts) {
      int local_rank = 0;
      for (int prior = 0; prior < assignment; ++prior) {
        local_rank += topk_ids[prior] == expert;
      }
      const int token = assignment / kTopK;
      const int slot = assignment - token * kTopK;
      sorted_ids[scan[expert] * kAssignments + local_rank] =
          (slot << 24) | token;
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

template <int kM>
struct Gfx90aQuantSort {
  static void run(const tvm::ffi::TensorView input,
                  const tvm::ffi::TensorView topk_ids,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView scales,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_experts,
                  const tvm::ffi::TensorView num_valid) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({kM, 4096}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({kM, 6}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({kM, 4096}).with_dtype<int8_t>().with_device(device).verify(output);
    TensorMatcher({kM, 128}).with_dtype<float>().with_device(device).verify(scales);
    TensorMatcher({kM * 6 * 4}).with_dtype<int32_t>().with_device(device).verify(sorted_ids);
    TensorMatcher({kM * 6}).with_dtype<int32_t>().with_device(device).verify(sorted_experts);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid);
    constexpr uint32_t kQuantBlocks = kM * (4096 / 32) / 16;
    LaunchKernel(kQuantBlocks + 1, 256, input.device())(
        gfx90a_m32_quant_sort_kernel<kM>,
        static_cast<const bf16_t*>(input.data_ptr()),
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<int8_t*>(output.data_ptr()),
        static_cast<float*>(scales.data_ptr()),
        static_cast<int32_t*>(sorted_ids.data_ptr()),
        static_cast<int32_t*>(sorted_experts.data_ptr()),
        static_cast<int32_t*>(num_valid.data_ptr()));
  }
};

using Gfx90aM32QuantSort = Gfx90aQuantSort<32>;
using Gfx90aM64QuantSort = Gfx90aQuantSort<64>;

}  // namespace sglang
