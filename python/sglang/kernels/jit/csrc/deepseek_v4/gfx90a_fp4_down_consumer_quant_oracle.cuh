#pragma once

// This header is compiled after gfx90a_fp4_expert_gemv.cuh by the standalone
// oracle wrapper.  It intentionally has no production selector.

namespace sglang {

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kCtasPerExpert>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_down_consumer_quant_oracle_kernel(
        float* __restrict__ partial,
        const bf16_t* __restrict__ intermediate,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        const float* __restrict__ topk_weights) {
  static_assert(K == 256 || K == 512,
                "oracle supports the TP8 K256 and TP4 K512 shards");
  static_assert(kAssignments == 4, "oracle keeps production A4 metadata");
  static_assert(kRows == 2 && kNumWaves == 8,
                "oracle keeps production R2/W8 geometry");
  static_assert(kCtasPerExpert >= 1 && kCtasPerExpert <= 16,
                "unsupported CTA replication");
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kSubgroupWidth = kGroups;
  constexpr uint32_t kSubgroupsPerWave = kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kSubgroupsPerBlock =
      kNumWaves * kSubgroupsPerWave;
  constexpr uint32_t kRowTiles = N / kRows;

  __shared__ uint32_t pair_lut[256];
  __shared__ int8_t q_lds[kAssignments][K];
  __shared__ float scale_lds[kAssignments][kGroups];
  __shared__ uint32_t tokens[kAssignments];
  __shared__ uint32_t slots[kAssignments];
  __shared__ uint32_t assignment_valid[kAssignments];

  const uint32_t expert_block = blockIdx.x / kCtasPerExpert;
  const uint32_t cta_shard = blockIdx.x % kCtasPerExpert;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;
  if (expert_block >= valid_blocks) return;

  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  if (threadIdx.x < kAssignments) {
    const uint32_t encoded = static_cast<uint32_t>(
        sorted_ids[expert_block * kAssignments + threadIdx.x]);
    const uint32_t token = encoded & 0x00ffffffu;
    const uint32_t slot = encoded >> 24;
    tokens[threadIdx.x] = token;
    slots[threadIdx.x] = slot;
    assignment_valid[threadIdx.x] = token < M && slot < T;
  }
  // The 512-thread CTA contains exactly 32 independent 16-lane subgroups:
  // one subgroup for each A4 assignment x eight group-32 slices.  This is the
  // same reduction, division and int8 conversion sequence as the bit-exact
  // gfx90a_int8_group32_quant kernel, but the result stays in LDS.
  constexpr uint32_t kQuantSubgroup = 16;
  const uint32_t quant_subgroup = threadIdx.x / kQuantSubgroup;
  const uint32_t quant_lane = threadIdx.x % kQuantSubgroup;
  constexpr uint32_t kQuantSubgroups =
      (kNumWaves * kFp4ExpertWave) / kQuantSubgroup;
  for (uint32_t quant_task = quant_subgroup;
       quant_task < kAssignments * kGroups;
       quant_task += kQuantSubgroups) {
    const uint32_t quant_assignment = quant_task / kGroups;
    const uint32_t quant_group = quant_task % kGroups;
    const uint32_t quant_encoded = static_cast<uint32_t>(
        sorted_ids[expert_block * kAssignments + quant_assignment]);
    const uint32_t quant_token = quant_encoded & 0x00ffffffu;
    const uint32_t quant_slot = quant_encoded >> 24;
    const bool quant_valid = quant_token < M && quant_slot < T;
    float x0 = 0.0f;
    float x1 = 0.0f;
    if (quant_valid) {
      const size_t input_assignment =
          static_cast<size_t>(quant_token) * T + quant_slot;
      const size_t base = input_assignment * K + quant_group * 32;
      x0 = cast<float>(intermediate[base + quant_lane]);
      x1 = cast<float>(intermediate[base + 16 + quant_lane]);
    }
    float absmax = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
    for (uint32_t offset = 8; offset > 0; offset >>= 1) {
      absmax = fmaxf(absmax, __shfl_xor(absmax, offset, kQuantSubgroup));
    }
    const float scale = fmaxf(absmax, 1.0e-10f) / 127.0f;
    const float q0 = fmaxf(-128.0f, fminf(127.0f, x0 / scale));
    const float q1 = fmaxf(-128.0f, fminf(127.0f, x1 / scale));
    q_lds[quant_assignment][quant_group * 32 + quant_lane] =
        static_cast<int8_t>(q0);
    q_lds[quant_assignment][quant_group * 32 + 16 + quant_lane] =
        static_cast<int8_t>(q1);
    if (quant_lane == 0) {
      scale_lds[quant_assignment][quant_group] = scale;
    }
  }
  __syncthreads();

  const int32_t expert_id = sorted_expert_ids[expert_block];
  if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) return;
  const uint32_t expert = static_cast<uint32_t>(expert_id);
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t block_subgroup = wave * kSubgroupsPerWave + subgroup;

  for (uint32_t row_tile = cta_shard * kSubgroupsPerBlock + block_subgroup;
       row_tile < kRowTiles;
       row_tile += kCtasPerExpert * kSubgroupsPerBlock) {
    const uint32_t row0 = row_tile * kRows;
    float acc[kAssignments][kRows] = {};
    const uint32_t group = subgroup_lane;
    const uint32_t k0 = group * 32;
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const uint32_t row = row0 + r;
      const size_t weight_base =
          (static_cast<size_t>(expert) * N + row) * (K / 2) + group * 16;
      const float weight_group_scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_down_scale_offset<E, N, K>(
              expert, row, group)]);
      int32_t weight_i8[8];
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        weight_i8[j] = gfx90a_fp4_pack4_i8_lds(
            *reinterpret_cast<const uint16_t*>(weight + weight_base + j * 2),
            pair_lut);
      }
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        if (!assignment_valid[assignment]) continue;
        acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
            q_lds[assignment] + k0, weight_i8,
            scale_lds[assignment][group] * weight_group_scale * 0.5f);
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
        if (subgroup_lane == 0) {
          partial[output_assignment * N + row0 + r] =
              acc[assignment][r] * routed_weight;
        }
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kCtasPerExpert>
struct Gfx90aFp4DownConsumerQuantOracleKernel {
  static void run(const tvm::ffi::TensorView intermediate,
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
    auto expert_blocks = SymbolicSize{"expert_blocks"};
    TensorMatcher({M, T, K}).with_dtype<bf16_t>().with_device(device).verify(intermediate);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({expert_blocks}).with_dtype<int32_t>().with_device(device).verify(sorted_expert_ids);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partial);
    LaunchKernel(static_cast<uint32_t>(expert_blocks.unwrap()) * kCtasPerExpert,
                 kNumWaves * kFp4ExpertWave, intermediate.device())(
        gfx90a_fp4_down_consumer_quant_oracle_kernel<
            E, M, T, N, K, kAssignments, kRows, kNumWaves, kCtasPerExpert>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const bf16_t*>(intermediate.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
