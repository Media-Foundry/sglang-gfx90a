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

constexpr uint32_t kM32QsGroups = 32 * (4096 / 32);
constexpr uint32_t kM32QsGroupsPerBlock = 16;
constexpr uint32_t kM32QsQuantBlocks =
    kM32QsGroups / kM32QsGroupsPerBlock;

__global__ void gfx90a_m32_quant_sort_kernel(
    const bf16_t* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ sorted_experts,
    int32_t* __restrict__ num_valid) {
  constexpr int kM = 32;
  constexpr int kTopK = 6;
  constexpr int kExperts = 256;
  constexpr int kAssignments = 4;
  constexpr int kWave = 64;
  constexpr int kSubgroup = 16;

  if (blockIdx.x < kM32QsQuantBlocks) {
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t wave = threadIdx.x / kWave;
    const uint32_t subgroup = lane / kSubgroup;
    const uint32_t subgroup_lane = lane & (kSubgroup - 1);
    const uint32_t group = blockIdx.x * kM32QsGroupsPerBlock +
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
      sorted_ids[atomicAdd(cursors + expert, 1)] = (slot << 24) | token;
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

struct Gfx90aM32QuantSort {
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
    TensorMatcher({32, 4096}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({32, 6}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({32, 4096}).with_dtype<int8_t>().with_device(device).verify(output);
    TensorMatcher({32, 128}).with_dtype<float>().with_device(device).verify(scales);
    TensorMatcher({768}).with_dtype<int32_t>().with_device(device).verify(sorted_ids);
    TensorMatcher({192}).with_dtype<int32_t>().with_device(device).verify(sorted_experts);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid);
    LaunchKernel(kM32QsQuantBlocks + 1, 256, input.device())(
        gfx90a_m32_quant_sort_kernel,
        static_cast<const bf16_t*>(input.data_ptr()),
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<int8_t*>(output.data_ptr()),
        static_cast<float*>(scales.data_ptr()),
        static_cast<int32_t*>(sorted_ids.data_ptr()),
        static_cast<int32_t*>(sorted_experts.data_ptr()),
        static_cast<int32_t*>(num_valid.data_ptr()));
  }
};

}  // namespace sglang
