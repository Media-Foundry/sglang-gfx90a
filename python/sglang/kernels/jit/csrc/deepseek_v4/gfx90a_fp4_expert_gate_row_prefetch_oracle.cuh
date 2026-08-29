#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

union Gfx90aGatePackedRow16 {
  uint4 vector;
  uint16_t halves[8];
};

// Oracle-only A4/R2 grouped gate/up.  The task mapping, LUT decode, SDOT,
// accumulation and DPP reduction are production-identical.  The sole schedule
// change is to request gate/up packed rows and scales for both R2 rows before
// decoding or consuming row0.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_row_prefetch_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  static_assert(kRows == 2, "row-prefetch oracle is fixed to R2");
  __shared__ uint32_t pair_lut[256];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  constexpr uint32_t kTilesPerExpertBlock = I / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks = (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = global_wave;
       task < valid_blocks * kTilesPerExpertBlock; task += total_waves) {
    const uint32_t expert_block = task / kTilesPerExpertBlock;
    const uint32_t row0 = (task % kTilesPerExpertBlock) * kRows;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);

    uint32_t tokens[kAssignments];
    uint32_t slots[kAssignments];
    bool assignment_valid[kAssignments];
    float gate_acc[kAssignments][kRows] = {};
    float up_acc[kAssignments][kRows] = {};
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[expert_block * kAssignments + assignment]);
      tokens[assignment] = encoded & 0x00ffffffu;
      slots[assignment] = encoded >> 24;
      assignment_valid[assignment] =
          tokens[assignment] < M && slots[assignment] < T;
    }

    for (uint32_t group = lane; group < K / 32; group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
      Gfx90aGatePackedRow16 gate_packed[kRows];
      Gfx90aGatePackedRow16 up_packed[kRows];
      uint8_t gate_scale_raw[kRows];
      uint8_t up_scale_raw[kRows];

      // All four independent 16-byte row requests and four scale requests are
      // source-level issued before row0's LUT decode and assignment dots.
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t local_row = row0 + r;
        const uint32_t gate_row = local_row;
        const uint32_t up_row = I + local_row;
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
        gate_scale_raw[r] =
            weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                expert, gate_row, group)];
        up_scale_raw[r] =
            weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                expert, up_row, group)];
      }

#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const float gate_scale = gfx90a_e8m0_value(gate_scale_raw[r]);
        const float up_scale = gfx90a_e8m0_value(up_scale_raw[r]);
        int32_t gate_i8[8];
        int32_t up_i8[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          gate_i8[j] =
              gfx90a_fp4_pack4_i8_lds(gate_packed[r].halves[j], pair_lut);
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
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          out[output_assignment * I + row0 + r] =
              cast<bf16_t>(activated * up);
        }
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked = 2>
struct Gfx90aFp4ExpertGateRowPrefetchOracle {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView out, double limit) {
    static_assert(kPrepacked == 2, "row-prefetch requires LDS decode");
    using namespace host;
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_gate_row_prefetch_kernel<
            E, M, T, I, K, kAssignments, kRows, kNumWaves>,
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
