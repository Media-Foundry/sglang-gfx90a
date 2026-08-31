#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

union Gfx90aExpertRowPacked16 {
  uint4 vector;
  uint16_t halves[8];
};

// Oracle-only A4/R2 gate/up schedule.  A wave owns one (expert, R2 row tile)
// and visits that expert's consecutive A4 chunks in order.  Each chunk keeps
// the production accumulator and DPP reduction tree; there is no publication
// protocol and no CTA-level ownership of a complete expert.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_row_persistent_gate_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ active_experts,
        const int32_t* __restrict__ block_starts,
        const int32_t* __restrict__ block_counts,
        const int32_t* __restrict__ num_active, float limit) {
  static_assert(kAssignments == 4 && kRows == 2,
                "expert-row oracle preserves production A4/R2");
  __shared__ uint32_t pair_lut[256];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  constexpr uint32_t kTilesPerExpert = I / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t active = static_cast<uint32_t>(max(num_active[0], 0));

  for (uint32_t task = global_wave; task < active * kTilesPerExpert;
       task += total_waves) {
    const uint32_t active_index = task / kTilesPerExpert;
    const uint32_t row0 = (task % kTilesPerExpert) * kRows;
    const int32_t expert_id = active_experts[active_index];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    const uint32_t block_start =
        static_cast<uint32_t>(max(block_starts[active_index], 0));
    const uint32_t block_count =
        static_cast<uint32_t>(max(block_counts[active_index], 0));

    for (uint32_t chunk = 0; chunk < block_count; ++chunk) {
      uint32_t tokens[kAssignments];
      uint32_t slots[kAssignments];
      bool assignment_valid[kAssignments];
      float gate_acc[kAssignments][kRows] = {};
      float up_acc[kAssignments][kRows] = {};
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[(block_start + chunk) * kAssignments + assignment]);
        tokens[assignment] = encoded & 0x00ffffffu;
        slots[assignment] = encoded >> 24;
        assignment_valid[assignment] =
            tokens[assignment] < M && slots[assignment] < T;
      }

      for (uint32_t group = lane; group < K / 32;
           group += kFp4ExpertWave) {
        const uint32_t k0 = group * 32;
        Gfx90aExpertRowPacked16 gate_packed[kRows];
        Gfx90aExpertRowPacked16 up_packed[kRows];
        uint8_t gate_scale_raw[kRows];
        uint8_t up_scale_raw[kRows];
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          const uint32_t gate_row = row0 + r;
          const uint32_t up_row = I + gate_row;
          const size_t gate_base =
              (static_cast<size_t>(expert) * (2 * I) + gate_row) * (K / 2) +
              group * 16;
          const size_t up_base =
              (static_cast<size_t>(expert) * (2 * I) + up_row) * (K / 2) +
              group * 16;
          gate_packed[r].vector =
              *reinterpret_cast<const uint4*>(weight + gate_base);
          up_packed[r].vector =
              *reinterpret_cast<const uint4*>(weight + up_base);
          gate_scale_raw[r] = weight_scale[
              gfx90a_gate_up_scale_offset<E, I, K>(expert, gate_row, group)];
          up_scale_raw[r] = weight_scale[
              gfx90a_gate_up_scale_offset<E, I, K>(expert, up_row, group)];
        }
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          const float gate_scale = gfx90a_e8m0_value(gate_scale_raw[r]);
          const float up_scale = gfx90a_e8m0_value(up_scale_raw[r]);
          int32_t gate_i8[8];
          int32_t up_i8[8];
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            gate_i8[j] = gfx90a_fp4_pack4_i8_lds(
                gate_packed[r].halves[j], pair_lut);
            up_i8[j] =
                gfx90a_fp4_pack4_i8_lds(up_packed[r].halves[j], pair_lut);
          }
#pragma unroll
          for (uint32_t assignment = 0; assignment < kAssignments;
               ++assignment) {
            if (!assignment_valid[assignment]) continue;
            const uint32_t token = tokens[assignment];
            const size_t xq_group =
                static_cast<size_t>(token) * (K / 32) + group;
            gate_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                xq + static_cast<size_t>(token) * K + k0, gate_i8,
                x_scale[xq_group] * gate_scale * 0.5f);
            up_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                xq + static_cast<size_t>(token) * K + k0, up_i8,
                x_scale[xq_group] * up_scale * 0.5f);
          }
        }
      }

#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        if (!assignment_valid[assignment]) continue;
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          gate_acc[assignment][r] +=
              __shfl_down(gate_acc[assignment][r], 32, kFp4ExpertWave);
          up_acc[assignment][r] +=
              __shfl_down(up_acc[assignment][r], 32, kFp4ExpertWave);
          gate_acc[assignment][r] +=
              __shfl_down(gate_acc[assignment][r], 16, kFp4ExpertWave);
          up_acc[assignment][r] +=
              __shfl_down(up_acc[assignment][r], 16, kFp4ExpertWave);
          gate_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x108u>(gate_acc[assignment][r]);
          up_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x108u>(up_acc[assignment][r]);
          gate_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x104u>(gate_acc[assignment][r]);
          up_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x104u>(up_acc[assignment][r]);
          gate_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x102u>(gate_acc[assignment][r]);
          up_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x102u>(up_acc[assignment][r]);
          gate_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x101u>(gate_acc[assignment][r]);
          up_acc[assignment][r] =
              gfx90a_fp4_dpp_add<0x101u>(up_acc[assignment][r]);
          if (lane == 0) {
            const float gate = fminf(gate_acc[assignment][r], limit);
            const float up =
                fmaxf(-limit, fminf(up_acc[assignment][r], limit));
            const float activated = gate / (1.0f + expf(-gate));
            const size_t output_assignment =
                static_cast<size_t>(tokens[assignment]) * T +
                slots[assignment];
            out[output_assignment * I + row0 + r] =
                cast<bf16_t>(activated * up);
          }
        }
      }
    }
  }
}

// Oracle-only logical-scale down schedule.  A 16-lane subgroup owns one
// (expert, R2 output-row tile) and scans consecutive A4 chunks.  Each chunk
// uses the original subgroup16 reduction and writes the original FP32 slot
// partial, so the existing fixed-slot reduction remains bitwise comparable.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_row_persistent_down_kernel(
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
                "M64 expert-row down is TP4 A4/K512");
  constexpr uint32_t kRows = 2;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kSubgroupWidth = 16;
  constexpr uint32_t kSubgroupsPerWave =
      kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kSubgroupsPerBlock =
      kNumWaves * kSubgroupsPerWave;
  constexpr uint32_t kTilesPerExpert = N / kRows;

  __shared__ uint32_t pair_lut[256];
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

  for (uint32_t task = global_subgroup; task < active * kTilesPerExpert;
       task += total_subgroups) {
    const uint32_t active_index = task / kTilesPerExpert;
    const uint32_t row0 = (task % kTilesPerExpert) * kRows;
    const int32_t expert_id = active_experts[active_index];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    const uint32_t block_start =
        static_cast<uint32_t>(max(block_starts[active_index], 0));
    const uint32_t block_count =
        static_cast<uint32_t>(max(block_counts[active_index], 0));

    for (uint32_t chunk = 0; chunk < block_count; ++chunk) {
      uint32_t tokens[kAssignments];
      uint32_t slots[kAssignments];
      bool assignment_valid[kAssignments];
      float acc[kAssignments][kRows] = {};
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[(block_start + chunk) * kAssignments + assignment]);
        tokens[assignment] = encoded & 0x00ffffffu;
        slots[assignment] = encoded >> 24;
        assignment_valid[assignment] =
            tokens[assignment] < M && slots[assignment] < T;
      }

      for (uint32_t group = subgroup_lane; group < kGroups;
           group += kSubgroupWidth) {
        const uint32_t k0 = group * 32;
        Gfx90aExpertRowPacked16 packed[kRows];
        uint8_t scale_raw[kRows];
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          const uint32_t row = row0 + r;
          const size_t weight_base =
              (static_cast<size_t>(expert) * N + row) * (K / 2) +
              group * 16;
          packed[r].vector =
              *reinterpret_cast<const uint4*>(weight + weight_base);
          scale_raw[r] =
              weight_scale[(static_cast<size_t>(expert) * N + row) *
                               kGroups +
                           group];
        }
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          const float scale = gfx90a_e8m0_value(scale_raw[r]);
          int32_t weight_i8[8];
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            weight_i8[j] = gfx90a_fp4_pack4_i8_lds(
                packed[r].halves[j], pair_lut);
          }
#pragma unroll
          for (uint32_t assignment = 0; assignment < kAssignments;
               ++assignment) {
            if (!assignment_valid[assignment]) continue;
            const size_t input_assignment =
                static_cast<size_t>(tokens[assignment]) * T +
                slots[assignment];
            const size_t xq_group = input_assignment * kGroups + group;
            acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                xq + input_assignment * K + k0, weight_i8,
                x_scale[xq_group] * scale * 0.5f);
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
          for (uint32_t offset = kSubgroupWidth / 2; offset > 0;
               offset >>= 1) {
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
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kGateBlocks, uint32_t kDownBlocks>
struct Gfx90aFp4ExpertRowPersistentOracle {
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
        gfx90a_fp4_expert_row_persistent_gate_kernel<
            E, M, T, I, K, kAssignments, kRows, kNumWaves>,
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
    LaunchKernel(kDownBlocks, 4 * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_row_persistent_down_kernel<
            E, M, T, 4096, I, kAssignments, 4>,
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
