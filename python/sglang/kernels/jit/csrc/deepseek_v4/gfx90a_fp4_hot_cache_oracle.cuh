#pragma once

// Compiled after gfx90a_fp4_expert_gemv.cuh by a standalone oracle wrapper.
// This file intentionally has no production selector.

namespace sglang {

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kHotExperts>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_hot_cache_down_oracle_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ packed_weight,
        const int8_t* __restrict__ hot_weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ expert_to_cache,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        const float* __restrict__ topk_weights) {
  static_assert(K == 256, "TP8 w2 oracle expects K=256");
  static_assert(kAssignments == 4, "oracle preserves production A4 metadata");
  __shared__ uint32_t pair_lut[256];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kSubgroupWidth = kGroups;
  constexpr uint32_t kSubgroupsPerWave = kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kSubgroupsPerBlock = kNumWaves * kSubgroupsPerWave;
  constexpr uint32_t kTilesPerExpertBlock = (N + kRows - 1) / kRows;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t block_subgroup = wave * kSubgroupsPerWave + subgroup;
  const uint32_t global_subgroup =
      blockIdx.x * kSubgroupsPerBlock + block_subgroup;
  const uint32_t total_subgroups = gridDim.x * kSubgroupsPerBlock;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = global_subgroup;
       task < valid_blocks * kTilesPerExpertBlock;
       task += total_subgroups) {
    const uint32_t expert_block = task / kTilesPerExpertBlock;
    const uint32_t row0 = (task % kTilesPerExpertBlock) * kRows;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    const int32_t cache_id = expert_to_cache[expert];
    const bool is_hot = cache_id >= 0 && cache_id < static_cast<int32_t>(kHotExperts);

    uint32_t tokens[kAssignments];
    uint32_t slots[kAssignments];
    bool assignment_valid[kAssignments];
    float acc[kAssignments][kRows] = {};
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[expert_block * kAssignments + assignment]);
      tokens[assignment] = encoded & 0x00ffffffu;
      slots[assignment] = encoded >> 24;
      assignment_valid[assignment] =
          tokens[assignment] < M && slots[assignment] < T;
    }

    for (uint32_t group = subgroup_lane; group < kGroups;
         group += kSubgroupWidth) {
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t row = row0 + r;
        if (row >= N) continue;
        const float scale = gfx90a_e8m0_value(
            weight_scale[gfx90a_down_scale_offset<E, N, K>(
                expert, row, group)]);
        int32_t weight_i8[8];
        if (is_hot) {
          const size_t base =
              (static_cast<size_t>(cache_id) * N + row) * K + group * 32;
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            weight_i8[j] = *reinterpret_cast<const int32_t*>(
                hot_weight + base + j * 4);
          }
        } else {
          const size_t base =
              (static_cast<size_t>(expert) * N + row) * (K / 2) + group * 16;
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            weight_i8[j] = gfx90a_fp4_pack4_i8_lds(
                *reinterpret_cast<const uint16_t*>(
                    packed_weight + base + j * 2),
                pair_lut);
          }
        }
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const size_t input_assignment =
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + input_assignment * K + k0, weight_i8,
              x_scale[input_assignment * kGroups + group] * scale * 0.5f);
        }
      }
    }

#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      if (!assignment_valid[assignment]) continue;
      const size_t output_assignment =
          static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
      const float routed_weight = topk_weights[output_assignment];
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
        for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
          acc[assignment][r] +=
              __shfl_down(acc[assignment][r], offset, kSubgroupWidth);
        }
        if (subgroup_lane == 0 && row0 + r < N) {
          partial[output_assignment * N + row0 + r] =
              acc[assignment][r] * routed_weight;
        }
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kHotExperts>
struct Gfx90aFp4HotCacheDownOracleKernel {
  static void run_partial(const tvm::ffi::TensorView xq,
                          const tvm::ffi::TensorView x_scale,
                          const tvm::ffi::TensorView packed_weight,
                          const tvm::ffi::TensorView hot_weight,
                          const tvm::ffi::TensorView weight_scale,
                          const tvm::ffi::TensorView expert_to_cache,
                          const tvm::ffi::TensorView sorted_ids,
                          const tvm::ffi::TensorView sorted_expert_ids,
                          const tvm::ffi::TensorView num_valid_ids,
                          const tvm::ffi::TensorView topk_weights,
                          const tvm::ffi::TensorView partial) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(packed_weight);
    TensorMatcher({kHotExperts, N, K}).with_dtype<int8_t>().with_device(device).verify(hot_weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(expert_to_cache);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partial);
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_hot_cache_down_oracle_kernel<
            E, M, T, N, K, kAssignments, kRows, kNumWaves, kBlocks,
            kHotExperts>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(packed_weight.data_ptr()),
        static_cast<const int8_t*>(hot_weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(expert_to_cache.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
