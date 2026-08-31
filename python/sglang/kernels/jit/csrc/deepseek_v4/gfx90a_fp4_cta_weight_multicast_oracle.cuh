#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

// Oracle-only M64 A4 weight multicast.  One wave remains the owner of one A4
// chunk, so its accumulators and reduction tree are unchanged.  Four waves in
// a CTA consume consecutive chunks of the same expert while cooperatively
// loading and decoding each packed weight tile only once.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kBlocks>
__global__ void __launch_bounds__(4 * kFp4ExpertWave)
    gfx90a_fp4_cta_multicast_gate_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ descriptor_experts,
        const int32_t* __restrict__ descriptor_starts,
        const int32_t* __restrict__ descriptor_counts,
        const int32_t* __restrict__ num_descriptors, float limit) {
  static_assert(M == 64 && T == 6 && I == 512 && K == 4096,
                "multicast oracle is strict TP4 DSpark M64 gate/up");
  constexpr uint32_t kAssignments = 4;
  constexpr uint32_t kRows = 2;
  constexpr uint32_t kWaves = 4;
  // K1024 phases keep the decoded gate/up tile plus LUT below 8 KiB/CTA.
  // Lane-local accumulation still visits groups lane,lane+64 in the same
  // order as production: phases 0/2 belong to lanes 0..31 and phases 1/3 to
  // lanes 32..63.
  constexpr uint32_t kGroupsPerPhase = 32;
  constexpr uint32_t kPhases = K / 32 / kGroupsPerPhase;
  constexpr uint32_t kLogicalRows = 2 * kRows;  // gate/up for both R2 rows
  constexpr uint32_t kDecodedPerPhase =
      kLogicalRows * kGroupsPerPhase * 8;

  __shared__ uint32_t pair_lut[256];
  __shared__ int32_t decoded[kDecodedPerPhase];
  __shared__ uint8_t scale_raw[kLogicalRows * kGroupsPerPhase];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t descriptor_count =
      static_cast<uint32_t>(max(num_descriptors[0], 0));
  constexpr uint32_t kRowTiles = I / kRows;

  for (uint32_t task = blockIdx.x;
       task < descriptor_count * kRowTiles; task += gridDim.x) {
    const uint32_t descriptor = task / kRowTiles;
    const uint32_t row0 = (task % kRowTiles) * kRows;
    const int32_t expert_id = descriptor_experts[descriptor];
    const uint32_t chunk_start =
        static_cast<uint32_t>(max(descriptor_starts[descriptor], 0));
    const uint32_t chunk_count = static_cast<uint32_t>(
        min(max(descriptor_counts[descriptor], 0), static_cast<int32_t>(kWaves)));
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E) ||
        chunk_count == 0) {
      continue;
    }
    const uint32_t expert = static_cast<uint32_t>(expert_id);

    uint32_t tokens[kAssignments] = {};
    uint32_t slots[kAssignments] = {};
    bool assignment_valid[kAssignments] = {};
    float gate_acc[kAssignments][kRows] = {};
    float up_acc[kAssignments][kRows] = {};
    if (wave < chunk_count) {
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[(chunk_start + wave) * kAssignments + assignment]);
        tokens[assignment] = encoded & 0x00ffffffu;
        slots[assignment] = encoded >> 24;
        assignment_valid[assignment] =
            tokens[assignment] < M && slots[assignment] < T;
      }
    }

#pragma unroll
    for (uint32_t phase = 0; phase < kPhases; ++phase) {
      for (uint32_t item = threadIdx.x; item < kDecodedPerPhase;
           item += blockDim.x) {
        const uint32_t logical_row = item / (kGroupsPerPhase * 8);
        const uint32_t rem = item % (kGroupsPerPhase * 8);
        const uint32_t local_group = rem / 8;
        const uint32_t packed_index = rem % 8;
        const uint32_t r = logical_row % kRows;
        const bool is_up = logical_row >= kRows;
        const uint32_t weight_row = (is_up ? I : 0) + row0 + r;
        const uint32_t group = phase * kGroupsPerPhase + local_group;
        const size_t base =
            (static_cast<size_t>(expert) * (2 * I) + weight_row) * (K / 2) +
            group * 16 + packed_index * 2;
        decoded[item] = gfx90a_fp4_pack4_i8_lds(
            *reinterpret_cast<const uint16_t*>(weight + base), pair_lut);
      }
      for (uint32_t item = threadIdx.x;
           item < kLogicalRows * kGroupsPerPhase; item += blockDim.x) {
        const uint32_t logical_row = item / kGroupsPerPhase;
        const uint32_t local_group = item % kGroupsPerPhase;
        const uint32_t r = logical_row % kRows;
        const bool is_up = logical_row >= kRows;
        const uint32_t weight_row = (is_up ? I : 0) + row0 + r;
        const uint32_t group = phase * kGroupsPerPhase + local_group;
        scale_raw[item] = weight_scale[
            gfx90a_gate_up_scale_offset<E, I, K>(expert, weight_row, group)];
      }
      __syncthreads();

      const bool lane_owns_phase =
          (lane / kGroupsPerPhase) == (phase & 1u);
      if (wave < chunk_count && lane_owns_phase) {
        const uint32_t phase_lane = lane % kGroupsPerPhase;
        const uint32_t group = phase * kGroupsPerPhase + phase_lane;
        const uint32_t k0 = group * 32;
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          int32_t gate_i8[8];
          int32_t up_i8[8];
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            gate_i8[j] =
                decoded[((r * kGroupsPerPhase + phase_lane) * 8) + j];
            up_i8[j] = decoded[
                (((kRows + r) * kGroupsPerPhase + phase_lane) * 8) + j];
          }
          const float gate_scale = gfx90a_e8m0_value(
              scale_raw[r * kGroupsPerPhase + phase_lane]);
          const float up_scale = gfx90a_e8m0_value(
              scale_raw[(kRows + r) * kGroupsPerPhase + phase_lane]);
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
      __syncthreads();
    }

    if (wave < chunk_count) {
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
            const float up = fmaxf(-limit, fminf(up_acc[assignment][r], limit));
            const float activated = gate / (1.0f + expf(-gate));
            const size_t output_assignment =
                static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
            out[output_assignment * I + row0 + r] =
                cast<bf16_t>(activated * up);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kBlocks>
__global__ void __launch_bounds__(4 * kFp4ExpertWave)
    gfx90a_fp4_cta_multicast_down_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ descriptor_experts,
        const int32_t* __restrict__ descriptor_starts,
        const int32_t* __restrict__ descriptor_counts,
        const int32_t* __restrict__ num_descriptors,
        const float* __restrict__ topk_weights) {
  static_assert(M == 64 && T == 6 && N == 4096 && K == 512,
                "multicast oracle is strict TP4 DSpark M64 down");
  constexpr uint32_t kAssignments = 4;
  constexpr uint32_t kWaves = 4;
  constexpr uint32_t kSubgroupWidth = 16;
  constexpr uint32_t kSubgroupsPerWave = 4;
  constexpr uint32_t kRowsPerSubgroup = 2;
  constexpr uint32_t kRowsPerCta =
      kSubgroupsPerWave * kRowsPerSubgroup;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kDecoded = kRowsPerCta * kGroups * 8;

  __shared__ uint32_t pair_lut[256];
  __shared__ int32_t decoded[kDecoded];
  __shared__ uint8_t scale_raw[kRowsPerCta * kGroups];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t descriptor_count =
      static_cast<uint32_t>(max(num_descriptors[0], 0));
  constexpr uint32_t kRowTiles = N / kRowsPerCta;

  for (uint32_t task = blockIdx.x;
       task < descriptor_count * kRowTiles; task += gridDim.x) {
    const uint32_t descriptor = task / kRowTiles;
    const uint32_t row_base = (task % kRowTiles) * kRowsPerCta;
    const int32_t expert_id = descriptor_experts[descriptor];
    const uint32_t chunk_start =
        static_cast<uint32_t>(max(descriptor_starts[descriptor], 0));
    const uint32_t chunk_count = static_cast<uint32_t>(
        min(max(descriptor_counts[descriptor], 0), static_cast<int32_t>(kWaves)));
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E) ||
        chunk_count == 0) {
      continue;
    }
    const uint32_t expert = static_cast<uint32_t>(expert_id);

    uint32_t tokens[kAssignments] = {};
    uint32_t slots[kAssignments] = {};
    bool assignment_valid[kAssignments] = {};
    float acc[kAssignments][kRowsPerSubgroup] = {};
    if (wave < chunk_count) {
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[(chunk_start + wave) * kAssignments + assignment]);
        tokens[assignment] = encoded & 0x00ffffffu;
        slots[assignment] = encoded >> 24;
        assignment_valid[assignment] =
            tokens[assignment] < M && slots[assignment] < T;
      }
    }

    for (uint32_t item = threadIdx.x; item < kDecoded;
         item += blockDim.x) {
      const uint32_t local_row = item / (kGroups * 8);
      const uint32_t rem = item % (kGroups * 8);
      const uint32_t group = rem / 8;
      const uint32_t packed_index = rem % 8;
      const uint32_t row = row_base + local_row;
      const size_t base =
          (static_cast<size_t>(expert) * N + row) * (K / 2) +
          group * 16 + packed_index * 2;
      decoded[item] = gfx90a_fp4_pack4_i8_lds(
          *reinterpret_cast<const uint16_t*>(weight + base), pair_lut);
    }
    for (uint32_t item = threadIdx.x; item < kRowsPerCta * kGroups;
         item += blockDim.x) {
      const uint32_t local_row = item / kGroups;
      const uint32_t group = item % kGroups;
      const uint32_t row = row_base + local_row;
      scale_raw[item] = weight_scale[
          (static_cast<size_t>(expert) * N + row) * kGroups + group];
    }
    __syncthreads();

    if (wave < chunk_count) {
      const uint32_t group = subgroup_lane;
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRowsPerSubgroup; ++r) {
        const uint32_t local_row = subgroup * kRowsPerSubgroup + r;
        int32_t weight_i8[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          weight_i8[j] =
              decoded[((local_row * kGroups + group) * 8) + j];
        }
        const float scale =
            gfx90a_e8m0_value(scale_raw[local_row * kGroups + group]);
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const size_t input_assignment =
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          const size_t xq_group = input_assignment * kGroups + group;
          acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + input_assignment * K + k0, weight_i8,
              x_scale[xq_group] * scale * 0.5f);
        }
      }
    }
    __syncthreads();

    if (wave < chunk_count) {
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        if (!assignment_valid[assignment]) continue;
        const size_t output_assignment =
            static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
        const float routed_weight = topk_weights[output_assignment];
#pragma unroll
        for (uint32_t r = 0; r < kRowsPerSubgroup; ++r) {
#pragma unroll
          for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
            acc[assignment][r] +=
                __shfl_down(acc[assignment][r], offset, kSubgroupWidth);
          }
          if (subgroup_lane == 0) {
            const uint32_t row =
                row_base + subgroup * kRowsPerSubgroup + r;
            partial[output_assignment * N + row] =
                acc[assignment][r] * routed_weight;
          }
        }
      }
    }
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kGateBlocks, uint32_t kDownBlocks>
struct Gfx90aFp4CtaWeightMulticastOracle {
  static void run_gate(const tvm::ffi::TensorView xq,
                       const tvm::ffi::TensorView x_scale,
                       const tvm::ffi::TensorView weight,
                       const tvm::ffi::TensorView weight_scale,
                       const tvm::ffi::TensorView sorted_ids,
                       const tvm::ffi::TensorView descriptor_experts,
                       const tvm::ffi::TensorView descriptor_starts,
                       const tvm::ffi::TensorView descriptor_counts,
                       const tvm::ffi::TensorView num_descriptors,
                       const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    LaunchKernel(kGateBlocks, 4 * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_cta_multicast_gate_kernel<E, M, T, I, K, kGateBlocks>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(descriptor_experts.data_ptr()),
        static_cast<const int32_t*>(descriptor_starts.data_ptr()),
        static_cast<const int32_t*>(descriptor_counts.data_ptr()),
        static_cast<const int32_t*>(num_descriptors.data_ptr()),
        static_cast<float>(limit));
  }

  static void run_down(const tvm::ffi::TensorView xq,
                       const tvm::ffi::TensorView x_scale,
                       const tvm::ffi::TensorView weight,
                       const tvm::ffi::TensorView weight_scale,
                       const tvm::ffi::TensorView sorted_ids,
                       const tvm::ffi::TensorView descriptor_experts,
                       const tvm::ffi::TensorView descriptor_starts,
                       const tvm::ffi::TensorView descriptor_counts,
                       const tvm::ffi::TensorView num_descriptors,
                       const tvm::ffi::TensorView topk_weights,
                       const tvm::ffi::TensorView partial) {
    using namespace host;
    LaunchKernel(kDownBlocks, 4 * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_cta_multicast_down_kernel<E, M, T, 4096, I, kDownBlocks>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(descriptor_experts.data_ptr()),
        static_cast<const int32_t*>(descriptor_starts.data_ptr()),
        static_cast<const int32_t*>(descriptor_counts.data_ptr()),
        static_cast<const int32_t*>(num_descriptors.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
