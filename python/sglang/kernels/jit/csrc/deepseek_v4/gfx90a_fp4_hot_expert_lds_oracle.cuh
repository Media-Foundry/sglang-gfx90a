#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

union Gfx90aHotExpertPacked16 {
  uint4 vector;
  uint16_t halves[8];
};

// Oracle-only hot-expert gate/up.  One wave owns one (expert, output row),
// expands that row's packed gate/up weights into wave-private LDS once, then
// consumes every consecutive A4 assignment chunk for the expert.  The cold
// experts stay on the unmodified production A4 kernel in the Python oracle.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_hot_expert_lds_gate_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ active_experts,
        const int32_t* __restrict__ block_starts,
        const int32_t* __restrict__ block_counts,
        const int32_t* __restrict__ num_active, float limit) {
  static_assert(kAssignments == 4, "hot-expert gate preserves production A4");
  static_assert(K == 4096, "hot-expert gate is the DSV4 H4096 shape");
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kDecodedPerGroup = 8;

  __shared__ uint32_t pair_lut[256];
  __shared__ int32_t gate_lds[kNumWaves][K / 4];
  __shared__ int32_t up_lds[kNumWaves][K / 4];
  __shared__ uint8_t gate_scale_lds[kNumWaves][kGroups];
  __shared__ uint8_t up_scale_lds[kNumWaves][kGroups];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t active = static_cast<uint32_t>(max(num_active[0], 0));

  for (uint32_t task = global_wave; task < active * I; task += total_waves) {
    const uint32_t active_index = task / I;
    const uint32_t row = task % I;
    const int32_t expert_id = active_experts[active_index];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    const uint32_t block_start =
        static_cast<uint32_t>(max(block_starts[active_index], 0));
    const uint32_t block_count =
        static_cast<uint32_t>(max(block_counts[active_index], 0));

    // Each lane owns its LDS groups, so no cross-lane publication is needed.
    // LDS keeps the decoded row live while the wave walks every A4 chunk.
    for (uint32_t group = lane; group < kGroups; group += kFp4ExpertWave) {
      const size_t gate_base =
          (static_cast<size_t>(expert) * (2 * I) + row) * (K / 2) +
          group * 16;
      const size_t up_base =
          (static_cast<size_t>(expert) * (2 * I) + I + row) * (K / 2) +
          group * 16;
      Gfx90aHotExpertPacked16 gate_packed;
      Gfx90aHotExpertPacked16 up_packed;
      gate_packed.vector =
          *reinterpret_cast<const uint4*>(weight + gate_base);
      up_packed.vector = *reinterpret_cast<const uint4*>(weight + up_base);
#pragma unroll
      for (uint32_t j = 0; j < kDecodedPerGroup; ++j) {
        gate_lds[wave][group * kDecodedPerGroup + j] =
            gfx90a_fp4_pack4_i8_lds(gate_packed.halves[j], pair_lut);
        up_lds[wave][group * kDecodedPerGroup + j] =
            gfx90a_fp4_pack4_i8_lds(up_packed.halves[j], pair_lut);
      }
      gate_scale_lds[wave][group] = weight_scale[
          gfx90a_gate_up_scale_offset<E, I, K>(expert, row, group)];
      up_scale_lds[wave][group] = weight_scale[
          gfx90a_gate_up_scale_offset<E, I, K>(expert, I + row, group)];
    }

    for (uint32_t chunk = 0; chunk < block_count; ++chunk) {
      uint32_t tokens[kAssignments];
      uint32_t slots[kAssignments];
      bool assignment_valid[kAssignments];
      float gate_acc[kAssignments] = {};
      float up_acc[kAssignments] = {};
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[(block_start + chunk) * kAssignments + assignment]);
        tokens[assignment] = encoded & 0x00ffffffu;
        slots[assignment] = encoded >> 24;
        assignment_valid[assignment] =
            tokens[assignment] < M && slots[assignment] < T;
      }

      for (uint32_t group = lane; group < kGroups;
           group += kFp4ExpertWave) {
        const uint32_t k0 = group * 32;
        int32_t gate_i8[kDecodedPerGroup];
        int32_t up_i8[kDecodedPerGroup];
#pragma unroll
        for (uint32_t j = 0; j < kDecodedPerGroup; ++j) {
          gate_i8[j] = gate_lds[wave][group * kDecodedPerGroup + j];
          up_i8[j] = up_lds[wave][group * kDecodedPerGroup + j];
        }
        const float gate_scale =
            gfx90a_e8m0_value(gate_scale_lds[wave][group]);
        const float up_scale = gfx90a_e8m0_value(up_scale_lds[wave][group]);
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const uint32_t token = tokens[assignment];
          const size_t xq_group =
              static_cast<size_t>(token) * kGroups + group;
          gate_acc[assignment] += gfx90a_fp4_dot32_i8_prepacked(
              xq + static_cast<size_t>(token) * K + k0, gate_i8,
              x_scale[xq_group] * gate_scale * 0.5f);
          up_acc[assignment] += gfx90a_fp4_dot32_i8_prepacked(
              xq + static_cast<size_t>(token) * K + k0, up_i8,
              x_scale[xq_group] * up_scale * 0.5f);
        }
      }

#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        if (!assignment_valid[assignment]) continue;
        float gate_sum = gate_acc[assignment];
        float up_sum = up_acc[assignment];
#pragma unroll
        for (uint32_t offset = 32; offset > 0; offset >>= 1) {
          gate_sum += __shfl_down(gate_sum, offset, kFp4ExpertWave);
          up_sum += __shfl_down(up_sum, offset, kFp4ExpertWave);
        }
        if (lane == 0) {
          const float gate = fminf(gate_sum, limit);
          const float up = fmaxf(-limit, fminf(up_sum, limit));
          const float activated = gate / (1.0f + expf(-gate));
          const size_t output_assignment =
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          out[output_assignment * I + row] = cast<bf16_t>(activated * up);
        }
      }
    }
  }
}

// Oracle-only hot-expert down.  A 16-lane subgroup owns one (expert, output
// row), decodes the packed K512 row into subgroup-private LDS once, and walks
// all A4 chunks.  The subgroup16 reduction and fixed FP32 slot partial are
// unchanged from production.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_hot_expert_lds_down_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ active_experts,
        const int32_t* __restrict__ block_starts,
        const int32_t* __restrict__ block_counts,
        const int32_t* __restrict__ num_active,
        const float* __restrict__ topk_weights) {
  static_assert(kAssignments == 4 && K == 512,
                "hot-expert down is TP4 A4/K512");
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kSubgroupWidth = 16;
  constexpr uint32_t kSubgroupsPerWave = kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kSubgroupsPerBlock = kNumWaves * kSubgroupsPerWave;
  constexpr uint32_t kDecodedPerGroup = 8;

  __shared__ uint32_t pair_lut[256];
  __shared__ int32_t decoded_lds[kSubgroupsPerBlock][K / 4];
  __shared__ uint8_t scale_lds[kSubgroupsPerBlock][kGroups];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t block_subgroup = wave * kSubgroupsPerWave + subgroup;
  const uint32_t global_subgroup =
      blockIdx.x * kSubgroupsPerBlock + block_subgroup;
  const uint32_t total_subgroups = gridDim.x * kSubgroupsPerBlock;
  const uint32_t active = static_cast<uint32_t>(max(num_active[0], 0));

  for (uint32_t task = global_subgroup; task < active * N;
       task += total_subgroups) {
    const uint32_t active_index = task / N;
    const uint32_t row = task % N;
    const int32_t expert_id = active_experts[active_index];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    const uint32_t block_start =
        static_cast<uint32_t>(max(block_starts[active_index], 0));
    const uint32_t block_count =
        static_cast<uint32_t>(max(block_counts[active_index], 0));

    const uint32_t group = subgroup_lane;
    const size_t weight_base =
        (static_cast<size_t>(expert) * N + row) * (K / 2) + group * 16;
    Gfx90aHotExpertPacked16 packed;
    packed.vector = *reinterpret_cast<const uint4*>(weight + weight_base);
#pragma unroll
    for (uint32_t j = 0; j < kDecodedPerGroup; ++j) {
      decoded_lds[block_subgroup][group * kDecodedPerGroup + j] =
          gfx90a_fp4_pack4_i8_lds(packed.halves[j], pair_lut);
    }
    scale_lds[block_subgroup][group] = weight_scale[
        gfx90a_down_scale_offset<E, N, K>(expert, row, group)];

    for (uint32_t chunk = 0; chunk < block_count; ++chunk) {
      uint32_t tokens[kAssignments];
      uint32_t slots[kAssignments];
      bool assignment_valid[kAssignments];
      float acc[kAssignments] = {};
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[(block_start + chunk) * kAssignments + assignment]);
        tokens[assignment] = encoded & 0x00ffffffu;
        slots[assignment] = encoded >> 24;
        assignment_valid[assignment] =
            tokens[assignment] < M && slots[assignment] < T;
      }

      int32_t weight_i8[kDecodedPerGroup];
#pragma unroll
      for (uint32_t j = 0; j < kDecodedPerGroup; ++j) {
        weight_i8[j] =
            decoded_lds[block_subgroup][group * kDecodedPerGroup + j];
      }
      const float scale = gfx90a_e8m0_value(scale_lds[block_subgroup][group]);
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments;
           ++assignment) {
        if (!assignment_valid[assignment]) continue;
        const size_t input_assignment =
            static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
        const size_t xq_group = input_assignment * kGroups + group;
        acc[assignment] += gfx90a_fp4_dot32_i8_prepacked(
            xq + input_assignment * K + k0, weight_i8,
            x_scale[xq_group] * scale * 0.5f);
      }

#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        if (!assignment_valid[assignment]) continue;
        float sum = acc[assignment];
#pragma unroll
        for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
          sum += __shfl_down(sum, offset, kSubgroupWidth);
        }
        if (subgroup_lane == 0) {
          const size_t output_assignment =
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          partial[output_assignment * N + row] =
              sum * topk_weights[output_assignment];
        }
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves,
          uint32_t kGateBlocks, uint32_t kDownBlocks>
struct Gfx90aFp4HotExpertLdsOracle {
  static void run_gate(const tvm::ffi::TensorView xq,
                       const tvm::ffi::TensorView x_scale,
                       const tvm::ffi::TensorView weight,
                       const tvm::ffi::TensorView weight_scale,
                       const tvm::ffi::TensorView sorted_ids,
                       const tvm::ffi::TensorView active_experts,
                       const tvm::ffi::TensorView block_starts,
                       const tvm::ffi::TensorView block_counts,
                       const tvm::ffi::TensorView num_active,
                       const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    LaunchKernel(kGateBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_hot_expert_lds_gate_kernel<
            E, M, T, I, K, kAssignments, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(active_experts.data_ptr()),
        static_cast<const int32_t*>(block_starts.data_ptr()),
        static_cast<const int32_t*>(block_counts.data_ptr()),
        static_cast<const int32_t*>(num_active.data_ptr()),
        static_cast<float>(limit));
  }

  static void run_down(const tvm::ffi::TensorView xq,
                       const tvm::ffi::TensorView x_scale,
                       const tvm::ffi::TensorView weight,
                       const tvm::ffi::TensorView weight_scale,
                       const tvm::ffi::TensorView sorted_ids,
                       const tvm::ffi::TensorView active_experts,
                       const tvm::ffi::TensorView block_starts,
                       const tvm::ffi::TensorView block_counts,
                       const tvm::ffi::TensorView num_active,
                       const tvm::ffi::TensorView topk_weights,
                       const tvm::ffi::TensorView partial) {
    using namespace host;
    LaunchKernel(kDownBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_hot_expert_lds_down_kernel<
            E, M, T, 4096, I, kAssignments, kNumWaves>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(active_experts.data_ptr()),
        static_cast<const int32_t*>(block_starts.data_ptr()),
        static_cast<const int32_t*>(block_counts.data_ptr()),
        static_cast<const int32_t*>(num_active.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
