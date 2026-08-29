#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

// Oracle-only projection-specialized grouped gate/up kernel.  Each adjacent
// wave pair owns one (expert A4 block, output-row tile): the even wave computes
// gate and the odd wave computes up.  One task per pair keeps the cross-wave
// barrier uniform across the full 1024-thread workgroup.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kBlocks,
          uint32_t kPrepacked>
__global__ void __launch_bounds__(16 * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_up_paired_oracle_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  static_assert(kPrepacked == 2, "paired oracle requires the LDS FP4 LUT");
  constexpr uint32_t kPhysicalWaves = 16;
  constexpr uint32_t kTaskPairs = kPhysicalWaves / 2;
  constexpr uint32_t kTilesPerExpertBlock = (I + kRows - 1) / kRows;

  __shared__ uint32_t pair_lut[256];
  __shared__ float gate_exchange[kTaskPairs][kAssignments][kRows];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t physical_wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t pair = physical_wave / 2;
  const bool is_up = (physical_wave & 1u) != 0;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t task = blockIdx.x * kTaskPairs + pair;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;
  const uint32_t total_tasks = valid_blocks * kTilesPerExpertBlock;
  const bool task_valid = task < total_tasks;

  uint32_t tokens[kAssignments] = {};
  uint32_t slots[kAssignments] = {};
  bool assignment_valid[kAssignments] = {};
  uint32_t row0 = 0;
  uint32_t expert = 0;
  if (task_valid) {
    const uint32_t expert_block = task / kTilesPerExpertBlock;
    row0 = (task % kTilesPerExpertBlock) * kRows;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id >= 0 && expert_id < static_cast<int32_t>(E)) {
      expert = static_cast<uint32_t>(expert_id);
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[expert_block * kAssignments + assignment]);
        tokens[assignment] = encoded & 0x00ffffffu;
        slots[assignment] = encoded >> 24;
        assignment_valid[assignment] =
            tokens[assignment] < M && slots[assignment] < T;
      }
    }
  }

  float acc[kAssignments][kRows] = {};
  if (task_valid) {
    for (uint32_t group = lane; group < K / 32;
         group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t local_row = row0 + r;
        if (local_row >= I) continue;
        const uint32_t projection_row = (is_up ? I : 0) + local_row;
        const size_t weight_base =
            (static_cast<size_t>(expert) * (2 * I) + projection_row) *
                (K / 2) +
            group * 16;
        const float weight_group_scale = gfx90a_e8m0_value(
            weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                expert, projection_row, group)]);
        int32_t weight_i8[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          weight_i8[j] = gfx90a_fp4_pack4_i8_lds(
              *reinterpret_cast<const uint16_t*>(weight + weight_base + j * 2),
              pair_lut);
        }
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const uint32_t token = tokens[assignment];
          const size_t xq_group =
              static_cast<size_t>(token) * (K / 32) + group;
          acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + static_cast<size_t>(token) * K + k0,
              weight_i8,
              x_scale[xq_group] * weight_group_scale * 0.5f);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1) {
        acc[assignment][r] +=
            __shfl_down(acc[assignment][r], offset, kFp4ExpertWave);
      }
      if (!is_up && lane == 0) {
        gate_exchange[pair][assignment][r] = acc[assignment][r];
      }
    }
  }
  __syncthreads();

  if (is_up && lane == 0 && task_valid) {
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      if (!assignment_valid[assignment]) continue;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        if (row0 + r >= I) continue;
        const float gate = fminf(gate_exchange[pair][assignment][r], limit);
        const float up = fmaxf(-limit, fminf(acc[assignment][r], limit));
        const float activated = gate / (1.0f + expf(-gate));
        const size_t output_assignment =
            static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
        out[output_assignment * I + row0 + r] =
            cast<bf16_t>(activated * up);
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kBlocks,
          uint32_t kPrepacked = 2>
struct Gfx90aFp4ExpertGateUpPairedOracleKernel {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(kBlocks, 16 * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_gate_up_paired_oracle_kernel<
            E, M, T, I, K, kAssignments, kRows, kBlocks, kPrepacked>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<float>(limit));
  }
};

}  // namespace sglang
