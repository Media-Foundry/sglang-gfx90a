#pragma once
#include "gfx90a_fp4_expert_gemv.cuh"
namespace sglang {

__device__ __forceinline__ int32_t gfx90a_khalf_dot32_i32(
    const int8_t* x, const int32_t* packed_weight) {
  int32_t acc = 0;
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    const int32_t xv = *reinterpret_cast<const int32_t*>(x + j * 4);
    acc = __builtin_amdgcn_sdot4(xv, packed_weight[j], acc, false);
  }
  return acc;
}
// loads.  AIter's sorter encodes the token in the low 24 bits and the top-k
// slot in the high 8 bits of sorted_ids.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kPrepacked, bool kUseDpp = false>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_gate_two_wave_khalf_oracle_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  static_assert(K == 4096 && kAssignments == 4 && kRows == 2 &&
                kNumWaves == 8 && kPrepacked == 0 && kUseDpp);
  // Four wave pairs x 64 lanes x sixteen FP32 gate/up partials consumes the
  // full 64-KiB gfx90a workgroup LDS budget. Arithmetic FP4 unpack is required
  // because the production 1-KiB LUT cannot coexist with this exchange.
  __shared__ int32_t half_dot[4][64][16];
  __shared__ float half_scale[4][64][16];
  constexpr uint32_t kTilesPerExpertBlock = (I + kRows - 1) / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t pair = wave / 2;
  const uint32_t half = wave & 1;
  const uint32_t global_pair = blockIdx.x * 4 + pair;
  const uint32_t total_pairs = gridDim.x * 4;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = global_pair;
       task < valid_blocks * kTilesPerExpertBlock;
       task += total_pairs) {
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

    for (uint32_t group = lane + half * 64; group < K / 32; group += 128) {
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t local_row = row0 + r;
        if (local_row >= I) continue;
        const uint32_t gate_row = local_row;
        const uint32_t up_row = I + local_row;
        const size_t gate_base =
            (static_cast<size_t>(expert) * (2 * I) + gate_row) *
                (kPrepacked == 1 ? K : K / 2) +
            group * (kPrepacked == 1 ? 32 : 16);
        const size_t up_base =
            (static_cast<size_t>(expert) * (2 * I) + up_row) *
                (kPrepacked == 1 ? K : K / 2) +
            group * (kPrepacked == 1 ? 32 : 16);
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
          gate_i8[j] = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + gate_base + j * 2));
          up_i8[j] = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + up_base + j * 2));
        }
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const uint32_t token = tokens[assignment];
          const size_t xq_group =
              static_cast<size_t>(token) * (K / 32) + group;
          const int32_t gate_dot = gfx90a_khalf_dot32_i32(
              xq + static_cast<size_t>(token) * K + k0, gate_i8);
          const int32_t up_dot = gfx90a_khalf_dot32_i32(
              xq + static_cast<size_t>(token) * K + k0, up_i8);
          const float gate_combined_scale =
              x_scale[xq_group] * gate_scale * 0.5f;
          const float up_combined_scale =
              x_scale[xq_group] * up_scale * 0.5f;
          const uint32_t field = assignment * 4 + r * 2;
          if (half == 0) {
            gate_acc[assignment][r] +=
                static_cast<float>(gate_dot) * gate_combined_scale;
            up_acc[assignment][r] +=
                static_cast<float>(up_dot) * up_combined_scale;
          } else {
            half_dot[pair][lane][field] = gate_dot;
            half_dot[pair][lane][field + 1] = up_dot;
            half_scale[pair][lane][field] = gate_combined_scale;
            half_scale[pair][lane][field + 1] = up_combined_scale;
          }
        }
      }
    }

    // Exchange the second K-half lane partials without changing the
    // production association: wave0 forms (group[lane] + group[lane+64])
    // before the original wave64 DPP tree.
    __syncthreads();
    if (half == 0) {
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          const uint32_t base = assignment * 4 + r * 2;
          gate_acc[assignment][r] +=
              static_cast<float>(half_dot[pair][lane][base]) *
              half_scale[pair][lane][base];
          up_acc[assignment][r] +=
              static_cast<float>(half_dot[pair][lane][base + 1]) *
              half_scale[pair][lane][base + 1];
        }
      }
    }

#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      if (half != 0 || !assignment_valid[assignment]) continue;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        if constexpr (kUseDpp) {
          // Preserve the established wave64 FP addition tree exactly. DPP
          // row_shl cannot cross a 16-lane row, so offsets 32 and 16 remain
          // shuffle-down operations before the four intra-row DPP steps.
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
        } else {
#pragma unroll
          for (uint32_t offset = 32; offset > 0; offset >>= 1) {
            gate_acc[assignment][r] +=
                __shfl_down(gate_acc[assignment][r], offset, kFp4ExpertWave);
            up_acc[assignment][r] +=
                __shfl_down(up_acc[assignment][r], offset, kFp4ExpertWave);
          }
        }
        if (lane == 0 && row0 + r < I) {
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
    // Do not let second-half waves overwrite the pair's LDS slot for the next
    // task until first-half waves have consumed every partial.
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks>
struct Gfx90aFp4GateTwoWaveKHalfOracle {
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
        gfx90a_fp4_gate_two_wave_khalf_oracle_kernel<
            E, M, T, I, K, kAssignments, kRows, kNumWaves, 0, true>,
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
