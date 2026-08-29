#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked = 2>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_cta_stage_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  static_assert(kPrepacked == 2 && kAssignments == 4 && kRows == 2 &&
                kNumWaves == 8 && K == 4096);
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kTilesPerExpertBlock = I / kRows;
  static_assert(kTilesPerExpertBlock % kNumWaves == 0);
  constexpr uint32_t kCtaTasksPerExpert =
      kTilesPerExpertBlock / kNumWaves;

  __shared__ uint32_t pair_lut[256];
  __shared__ int8_t staged_x[kAssignments][K];
  __shared__ float staged_scale[kAssignments][kGroups];
  __shared__ uint32_t staged_tokens[kAssignments];
  __shared__ uint32_t staged_slots[kAssignments];
  __shared__ uint32_t staged_valid[kAssignments];

  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kFp4ExpertWave;
  const uint32_t lane = tid % kFp4ExpertWave;
  if (tid < 256) {
    pair_lut[tid] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(tid))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;
  const uint32_t total_cta_tasks = valid_blocks * kCtaTasksPerExpert;

  // CTA-granular loop: every thread executes every barrier. Since 32 CTA
  // tasks cover each expert block, all eight waves share its A4 activation.
  for (uint32_t cta_task = blockIdx.x; cta_task < total_cta_tasks;
       cta_task += gridDim.x) {
    const uint32_t expert_block = cta_task / kCtaTasksPerExpert;
    const uint32_t tile_group = cta_task % kCtaTasksPerExpert;
    const int32_t expert_id = sorted_expert_ids[expert_block];

    if (tid < kAssignments) {
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[expert_block * kAssignments + tid]);
      const uint32_t token = encoded & 0x00ffffffu;
      const uint32_t slot = encoded >> 24;
      staged_tokens[tid] = token;
      staged_slots[tid] = slot;
      staged_valid[tid] = token < M && slot < T;
    }
    __syncthreads();

    for (uint32_t linear = tid; linear < kAssignments * K;
         linear += blockDim.x) {
      const uint32_t assignment = linear / K;
      const uint32_t column = linear % K;
      const uint32_t token = staged_tokens[assignment];
      staged_x[assignment][column] = staged_valid[assignment]
          ? xq[static_cast<size_t>(token) * K + column]
          : 0;
    }
    for (uint32_t linear = tid; linear < kAssignments * kGroups;
         linear += blockDim.x) {
      const uint32_t assignment = linear / kGroups;
      const uint32_t group = linear % kGroups;
      const uint32_t token = staged_tokens[assignment];
      staged_scale[assignment][group] = staged_valid[assignment]
          ? x_scale[static_cast<size_t>(token) * kGroups + group]
          : 0.0f;
    }
    __syncthreads();

    if (expert_id >= 0 && expert_id < static_cast<int32_t>(E)) {
      const uint32_t expert = static_cast<uint32_t>(expert_id);
      const uint32_t row0 =
          (tile_group * kNumWaves + wave) * kRows;
      float gate_acc[kAssignments][kRows] = {};
      float up_acc[kAssignments][kRows] = {};
      for (uint32_t group = lane; group < kGroups;
           group += kFp4ExpertWave) {
        const uint32_t k0 = group * 32;
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
          const float gate_scale = gfx90a_e8m0_value(
              weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                  expert, gate_row, group)]);
          const float up_scale = gfx90a_e8m0_value(
              weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                  expert, up_row, group)]);
          int32_t gate_i8[8];
          int32_t up_i8[8];
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            gate_i8[j] = gfx90a_fp4_pack4_i8_lds(
                *reinterpret_cast<const uint16_t*>(weight + gate_base + j * 2),
                pair_lut);
            up_i8[j] = gfx90a_fp4_pack4_i8_lds(
                *reinterpret_cast<const uint16_t*>(weight + up_base + j * 2),
                pair_lut);
          }
#pragma unroll
          for (uint32_t assignment = 0; assignment < kAssignments;
               ++assignment) {
            if (!staged_valid[assignment]) continue;
            gate_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                staged_x[assignment] + k0, gate_i8,
                staged_scale[assignment][group] * gate_scale * 0.5f);
            up_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                staged_x[assignment] + k0, up_i8,
                staged_scale[assignment][group] * up_scale * 0.5f);
          }
        }
      }

#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        if (!staged_valid[assignment]) continue;
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
                static_cast<size_t>(staged_tokens[assignment]) * T +
                staged_slots[assignment];
            out[output_assignment * I + row0 + r] =
                cast<bf16_t>(activated * up);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked = 2>
struct Gfx90aFp4ExpertGateCtaStageOracle {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_gate_cta_stage_kernel<
            E, M, T, I, K, kAssignments, kRows, kNumWaves, kBlocks,
            kPrepacked>,
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
