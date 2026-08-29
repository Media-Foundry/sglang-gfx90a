#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked = 2>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_gate_expert_owned_oracle_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        uint32_t* __restrict__ counters, uint32_t* __restrict__ ready,
        float limit) {
  static_assert(kPrepacked == 2 && kAssignments == 4 && kRows == 2 &&
                kNumWaves == 8 && K == 4096);
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kTilesPerExpertBlock = I / kRows;
  static_assert(kTilesPerExpertBlock % kNumWaves == 0);
  constexpr uint32_t kCtaTasksPerExpert =
      kTilesPerExpertBlock / kNumWaves;

  __shared__ uint32_t pair_lut[256];
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
  static_assert(kBlocks == 1 || kBlocks == 4 || kBlocks == 8,
                "oracle owner CTA fan-in must be 1/4/8");
  const uint32_t expert_block = blockIdx.x / kBlocks;
  const uint32_t owner_shard = blockIdx.x % kBlocks;
  if (expert_block >= valid_blocks) return;
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

  // One CTA owns one padded A4 expert block. Its W8 waves traverse all 256
  // R2 row tiles in 32 uniform rounds, preserving the original DPP tree.
  for (uint32_t tile_group = owner_shard;
       tile_group < kCtaTasksPerExpert; tile_group += kBlocks) {

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
            const uint32_t token = staged_tokens[assignment];
            gate_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                xq + static_cast<size_t>(token) * K + k0, gate_i8,
                x_scale[static_cast<size_t>(token) * kGroups + group] *
                    gate_scale * 0.5f);
            up_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                xq + static_cast<size_t>(token) * K + k0, up_i8,
                x_scale[static_cast<size_t>(token) * kGroups + group] *
                    up_scale * 0.5f);
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
  }
  __syncthreads();
  if (tid == 0) {
    const uint32_t old = __scoped_atomic_fetch_add(
        counters + expert_block, 1u, __ATOMIC_ACQ_REL,
        __MEMORY_SCOPE_DEVICE);
    if (old % kBlocks == kBlocks - 1) {
      __scoped_atomic_store_n(
          ready + expert_block, old / kBlocks + 1u,
          __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked = 2>
struct Gfx90aFp4GateExpertOwnedOracle {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView counters,
                  const tvm::ffi::TensorView ready,
                  const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto expert_blocks = SymbolicSize{"expert_blocks"};
    TensorMatcher({expert_blocks}).with_dtype<int32_t>().verify(sorted_expert_ids);
    LaunchKernel(static_cast<uint32_t>(expert_blocks.unwrap()) * kBlocks,
                 kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_gate_expert_owned_oracle_kernel<
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
        static_cast<uint32_t*>(counters.data_ptr()),
        static_cast<uint32_t*>(ready.data_ptr()),
        static_cast<float>(limit));
  }
};

}  // namespace sglang
