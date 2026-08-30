#pragma once

// Oracle-only exact E2M1 signed-code repack.  This header is compiled after
// gfx90a_fp4_expert_gemv.cuh and is deliberately not reachable from a
// production selector.

namespace sglang {

__device__ __forceinline__ int32_t gfx90a_int5_expand4(uint32_t bits20) {
  // Spread four two's-complement int5 fields into four bytes.
  uint32_t bytes = (bits20 & 0x0000001fu) |
                   ((bits20 & 0x000003e0u) << 3) |
                   ((bits20 & 0x00007c00u) << 6) |
                   ((bits20 & 0x000f8000u) << 9);
  // Per-byte sign extension. Each marker is bit 4 of its byte; multiplying by
  // 15 produces bits 4..7 without cross-byte carries (0x10 * 15 == 0xf0).
  const uint32_t sign = bytes & 0x10101010u;
  return static_cast<int32_t>(bytes | sign * 15u);
}

template <uint32_t J>
__device__ __forceinline__ int32_t gfx90a_int5_load4(
    const uint32_t* __restrict__ words) {
  static_assert(J < 8);
  constexpr uint32_t bit = J * 20;
  constexpr uint32_t word = bit / 32;
  constexpr uint32_t shift = bit % 32;
  uint32_t packed = words[word] >> shift;
  if constexpr (shift > 12) packed |= words[word + 1] << (32 - shift);
  return gfx90a_int5_expand4(packed & 0x000fffffu);
}

template <uint32_t Half>
__device__ __forceinline__ int32_t gfx90a_int5_bitplane_expand4(
    uint32_t low_word, uint32_t high_plane, uint32_t quartet) {
  static_assert(Half < 2);
  const uint32_t low16 = (low_word >> (Half * 16)) & 0xffffu;
  const uint32_t low_bytes =
      (low16 & 0x000fu) | ((low16 & 0x00f0u) << 4) |
      ((low16 & 0x0f00u) << 8) | ((low16 & 0xf000u) << 12);
  const uint32_t signs = (high_plane >> (quartet * 4)) & 0xfu;
  const uint32_t markers = ((signs & 1u) << 4) | ((signs & 2u) << 11) |
                           ((signs & 4u) << 18) | ((signs & 8u) << 25);
  return static_cast<int32_t>(low_bytes | markers * 15u);
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, bool kBitplane>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_int5_down_oracle_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint32_t* __restrict__ int5_weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        const float* __restrict__ topk_weights) {
  static_assert(K % 32 == 0);
  static_assert(kAssignments == 4);
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kSubgroupWidth = kGroups < 16 ? kGroups : 16;
  constexpr uint32_t kSubgroupsPerWave = kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kTilesPerExpertBlock = (N + kRows - 1) / kRows;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t block_subgroup = wave * kSubgroupsPerWave + subgroup;
  const uint32_t subgroups_per_block = kNumWaves * kSubgroupsPerWave;
  const uint32_t global_subgroup = blockIdx.x * subgroups_per_block + block_subgroup;
  const uint32_t total_subgroups = gridDim.x * subgroups_per_block;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks = (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = global_subgroup;
       task < valid_blocks * kTilesPerExpertBlock; task += total_subgroups) {
    const uint32_t expert_block = task / kTilesPerExpertBlock;
    const uint32_t row0 = (task % kTilesPerExpertBlock) * kRows;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    uint32_t tokens[kAssignments], slots[kAssignments];
    bool assignment_valid[kAssignments];
    float acc[kAssignments][kRows] = {};
#pragma unroll
    for (uint32_t a = 0; a < kAssignments; ++a) {
      const uint32_t encoded =
          static_cast<uint32_t>(sorted_ids[expert_block * kAssignments + a]);
      tokens[a] = encoded & 0x00ffffffu;
      slots[a] = encoded >> 24;
      assignment_valid[a] = tokens[a] < M && slots[a] < T;
    }
    for (uint32_t group = subgroup_lane; group < kGroups;
         group += kSubgroupWidth) {
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t row = row0 + r;
        if (row >= N) continue;
        const size_t group_index =
            (static_cast<size_t>(expert) * N + row) * kGroups + group;
        const uint32_t* words = int5_weight + group_index * 5;
        int32_t w0, w1, w2, w3, w4, w5, w6, w7;
        if constexpr (kBitplane) {
          const uint32_t low0 = words[0], low1 = words[1];
          const uint32_t low2 = words[2], low3 = words[3];
          const uint32_t high = words[4];
          w0 = gfx90a_int5_bitplane_expand4<0>(low0, high, 0);
          w1 = gfx90a_int5_bitplane_expand4<1>(low0, high, 1);
          w2 = gfx90a_int5_bitplane_expand4<0>(low1, high, 2);
          w3 = gfx90a_int5_bitplane_expand4<1>(low1, high, 3);
          w4 = gfx90a_int5_bitplane_expand4<0>(low2, high, 4);
          w5 = gfx90a_int5_bitplane_expand4<1>(low2, high, 5);
          w6 = gfx90a_int5_bitplane_expand4<0>(low3, high, 6);
          w7 = gfx90a_int5_bitplane_expand4<1>(low3, high, 7);
        } else {
          w0 = gfx90a_int5_load4<0>(words); w1 = gfx90a_int5_load4<1>(words);
          w2 = gfx90a_int5_load4<2>(words); w3 = gfx90a_int5_load4<3>(words);
          w4 = gfx90a_int5_load4<4>(words); w5 = gfx90a_int5_load4<5>(words);
          w6 = gfx90a_int5_load4<6>(words); w7 = gfx90a_int5_load4<7>(words);
        }
        const float scale = gfx90a_e8m0_value(
            weight_scale[gfx90a_down_scale_offset<E, N, K>(expert, row, group)]);
#pragma unroll
        for (uint32_t a = 0; a < kAssignments; ++a) {
          if (!assignment_valid[a]) continue;
          const size_t input_assignment =
              static_cast<size_t>(tokens[a]) * T + slots[a];
          const int8_t* xv = xq + input_assignment * K + k0;
          int32_t dot = 0;
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 0), w0, dot, false);
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 4), w1, dot, false);
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 8), w2, dot, false);
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 12), w3, dot, false);
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 16), w4, dot, false);
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 20), w5, dot, false);
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 24), w6, dot, false);
          dot = __builtin_amdgcn_sdot4(*reinterpret_cast<const int32_t*>(xv + 28), w7, dot, false);
          const size_t xg = input_assignment * kGroups + group;
          acc[a][r] += static_cast<float>(dot) * x_scale[xg] * scale * 0.5f;
        }
      }
    }
#pragma unroll
    for (uint32_t a = 0; a < kAssignments; ++a) {
      if (!assignment_valid[a]) continue;
      const size_t oa = static_cast<size_t>(tokens[a]) * T + slots[a];
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
        for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1)
          acc[a][r] += __shfl_down(acc[a][r], offset, kSubgroupWidth);
        if (subgroup_lane == 0 && row0 + r < N)
          partial[oa * N + row0 + r] = acc[a][r] * topk_weights[oa];
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, bool kBitplane>
struct Gfx90aFp4Int5DownOracleKernel {
  static void run_partial(const tvm::ffi::TensorView xq,
                          const tvm::ffi::TensorView x_scale,
                          const tvm::ffi::TensorView int5_weight,
                          const tvm::ffi::TensorView weight_scale,
                          const tvm::ffi::TensorView sorted_ids,
                          const tvm::ffi::TensorView sorted_expert_ids,
                          const tvm::ffi::TensorView num_valid_ids,
                          const tvm::ffi::TensorView topk_weights,
                          const tvm::ffi::TensorView partial) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 32, 5}).with_dtype<uint32_t>().with_device(device).verify(int5_weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partial);
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_int5_down_oracle_kernel<E, M, T, N, K, kAssignments,
                                           kRows, kNumWaves, kBlocks, kBitplane>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint32_t*>(int5_weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
