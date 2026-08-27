#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

// Standalone oracle for singleton-expert down projection.  Every wave owns one
// expert/assignment and its eight 2-row subgroups advance together.  Ordering
// rows before experts makes adjacent waves in a CTA consume different A1
// experts without changing any dot-product or output-reduction order.
template <uint32_t kBlocks>
__global__ void __launch_bounds__(8 * kFp4ExpertWave)
    gfx90a_fp4_a1_wave_owned_down_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        const float* __restrict__ topk_weights) {
  constexpr uint32_t E = 256;
  constexpr uint32_t M = 32;
  constexpr uint32_t T = 6;
  constexpr uint32_t N = 4096;
  constexpr uint32_t K = 256;
  constexpr uint32_t kRows = 2;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kSubgroupWidth = kGroups;
  constexpr uint32_t kSubgroupsPerWave = kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kNumWaves = 8;
  constexpr uint32_t kRowChunks =
      (N + kRows * kSubgroupsPerWave - 1) /
      (kRows * kSubgroupsPerWave);

  __shared__ uint32_t pair_lut[256];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x & 63;
  const uint32_t wave = threadIdx.x >> 6;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  constexpr uint32_t total_waves = kBlocks * kNumWaves;
  const uint32_t valid = max(num_valid_ids[0], 0);

  for (uint32_t task = global_wave; task < valid * kRowChunks;
       task += total_waves) {
    // Expert-minor ordering deliberately assigns adjacent waves to different
    // singleton experts while every subgroup in a wave shares that expert.
    const uint32_t expert_block = task % valid;
    const uint32_t row_chunk = task / valid;
    const uint32_t row0 =
        (row_chunk * kSubgroupsPerWave + subgroup) * kRows;

    const int32_t expert_lane =
        lane == 0 ? sorted_expert_ids[expert_block] : 0;
    const uint32_t encoded_lane = lane == 0
        ? static_cast<uint32_t>(sorted_ids[expert_block])
        : 0;
    const int32_t expert_id = __shfl(expert_lane, 0, kFp4ExpertWave);
    const uint32_t encoded = __shfl(encoded_lane, 0, kFp4ExpertWave);
    const uint32_t token = encoded & 0x00ffffffu;
    const uint32_t slot = encoded >> 24;
    const bool valid_assignment =
        expert_id >= 0 && expert_id < static_cast<int32_t>(E) &&
        token < M && slot < T;
    if (!valid_assignment) continue;

    const uint32_t expert = static_cast<uint32_t>(expert_id);
    const size_t input_assignment = static_cast<size_t>(token) * T + slot;
    const float routed_lane = lane == 0 ? topk_weights[input_assignment] : 0.0f;
    const float routed_weight = __shfl(routed_lane, 0, kFp4ExpertWave);
    const uint32_t group = subgroup_lane;
    const uint32_t k0 = group * 32;
    float acc[kRows] = {};

#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const uint32_t row = row0 + r;
      if (row >= N) continue;
      const size_t weight_base =
          (static_cast<size_t>(expert) * N + row) * (K / 2) + group * 16;
      const float scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_down_scale_offset<E, N, K>(
              expert, row, group)]);
      int32_t weight_i8[8];
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        weight_i8[j] = gfx90a_fp4_pack4_i8_lds(
            *reinterpret_cast<const uint16_t*>(weight + weight_base + j * 2),
            pair_lut);
      }
      const size_t xq_group = input_assignment * kGroups + group;
      acc[r] = gfx90a_fp4_dot32_i8_prepacked(
          xq + input_assignment * K + k0, weight_i8,
          x_scale[xq_group] * scale * 0.5f);
    }

#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
      for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
        acc[r] += __shfl_down(acc[r], offset, kSubgroupWidth);
      }
      if (subgroup_lane == 0 && row0 + r < N) {
        partial[input_assignment * N + row0 + r] =
            acc[r] * routed_weight;
      }
    }
  }
}

template <uint32_t kBlocks>
struct Gfx90aFp4A1WaveOwnedDownOracle {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView topk_weights,
                  const tvm::ffi::TensorView partial) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({32, 6, 256}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({32, 6, 8}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({256, 4096, 128}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({256, 4096, 8}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({32, 6}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({32, 6, 4096}).with_dtype<float>().with_device(device).verify(partial);
    LaunchKernel(kBlocks, 8 * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_a1_wave_owned_down_kernel<kBlocks>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
