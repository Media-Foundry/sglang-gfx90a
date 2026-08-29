#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

// Distance-one K-group prefetch oracle for the fixed K4096/wave64 geometry.
// Each lane owns groups lane and lane+64.  The second group's four packed
// gate/up row vectors and scales are made live before the first group is
// decoded and consumed; activation is deliberately not prefetched.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_up_kprefetch_oracle_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  static_assert(K == 4096 && kRows == 2 && kPrepacked == 2,
                "oracle is fixed to K4096/R2/LDS-LUT");
  __shared__ uint32_t pair_lut[256];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  constexpr uint32_t kTilesPerExpertBlock = (I + kRows - 1) / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = global_wave;
       task < valid_blocks * kTilesPerExpertBlock;
       task += total_waves) {
    const uint32_t expert_block = task / kTilesPerExpertBlock;
    const uint32_t row0 = (task % kTilesPerExpertBlock) * kRows;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    uint32_t encoded_ids[kAssignments];
    float gate_acc[kAssignments][kRows] = {};
    float up_acc[kAssignments][kRows] = {};
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      encoded_ids[assignment] = static_cast<uint32_t>(
          sorted_ids[expert_block * kAssignments + assignment]);
    }

    const uint32_t next_group = lane + 64;
    gfx90a_i32x4 next_gate_packed[kRows];
    gfx90a_i32x4 next_up_packed[kRows];
    float next_gate_scale[kRows];
    float next_up_scale[kRows];
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const uint32_t local_row = row0 + r;
      const uint32_t gate_row = local_row;
      const uint32_t up_row = I + local_row;
      if (local_row < I) {
        const size_t gate_base =
            (static_cast<size_t>(expert) * (2 * I) + gate_row) * (K / 2) +
            next_group * 16;
        const size_t up_base =
            (static_cast<size_t>(expert) * (2 * I) + up_row) * (K / 2) +
            next_group * 16;
        next_gate_packed[r] = *reinterpret_cast<const gfx90a_i32x4*>(
            weight + gate_base);
        next_up_packed[r] = *reinterpret_cast<const gfx90a_i32x4*>(
            weight + up_base);
        next_gate_scale[r] = gfx90a_e8m0_value(
            weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                expert, gate_row, next_group)]);
        next_up_scale[r] = gfx90a_e8m0_value(
            weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                expert, up_row, next_group)]);
      }
    }

    // Consume group=lane first, preserving the production FP32 accumulation
    // order. The group=lane+64 packed vectors above remain live meanwhile.
    const uint32_t first_group = lane;
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const uint32_t local_row = row0 + r;
      if (local_row >= I) continue;
      const uint32_t gate_row = local_row;
      const uint32_t up_row = I + local_row;
      const size_t gate_base =
          (static_cast<size_t>(expert) * (2 * I) + gate_row) * (K / 2) +
          first_group * 16;
      const size_t up_base =
          (static_cast<size_t>(expert) * (2 * I) + up_row) * (K / 2) +
          first_group * 16;
      const float gate_scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
              expert, gate_row, first_group)]);
      const float up_scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
              expert, up_row, first_group)]);
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
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t token = encoded_ids[assignment] & 0x00ffffffu;
        const uint32_t slot = encoded_ids[assignment] >> 24;
        if (token >= M || slot >= T) continue;
        const size_t xq_group =
            static_cast<size_t>(token) * (K / 32) + first_group;
        gate_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + first_group * 32,
            gate_i8, x_scale[xq_group] * gate_scale * 0.5f);
        up_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + first_group * 32,
            up_i8, x_scale[xq_group] * up_scale * 0.5f);
      }
    }

    // Decode and consume the already-issued second-group packed vectors.
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r >= I) continue;
      int32_t gate_i8[8];
      int32_t up_i8[8];
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t gate_word =
            static_cast<uint32_t>(next_gate_packed[r][j / 2]);
        const uint32_t up_word =
            static_cast<uint32_t>(next_up_packed[r][j / 2]);
        const uint16_t gate_pair =
            static_cast<uint16_t>(gate_word >> ((j & 1) * 16));
        const uint16_t up_pair =
            static_cast<uint16_t>(up_word >> ((j & 1) * 16));
        gate_i8[j] = gfx90a_fp4_pack4_i8_lds(gate_pair, pair_lut);
        up_i8[j] = gfx90a_fp4_pack4_i8_lds(up_pair, pair_lut);
      }
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t token = encoded_ids[assignment] & 0x00ffffffu;
        const uint32_t slot = encoded_ids[assignment] >> 24;
        if (token >= M || slot >= T) continue;
        const size_t xq_group =
            static_cast<size_t>(token) * (K / 32) + next_group;
        gate_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + next_group * 32,
            gate_i8, x_scale[xq_group] * next_gate_scale[r] * 0.5f);
        up_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + next_group * 32,
            up_i8, x_scale[xq_group] * next_up_scale[r] * 0.5f);
      }
    }

#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      const uint32_t token = encoded_ids[assignment] & 0x00ffffffu;
      const uint32_t slot = encoded_ids[assignment] >> 24;
      if (token >= M || slot >= T) continue;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
        for (uint32_t offset = 32; offset > 0; offset >>= 1) {
          gate_acc[assignment][r] +=
              __shfl_down(gate_acc[assignment][r], offset, kFp4ExpertWave);
          up_acc[assignment][r] +=
              __shfl_down(up_acc[assignment][r], offset, kFp4ExpertWave);
        }
        if (lane == 0 && row0 + r < I) {
          const float gate = fminf(gate_acc[assignment][r], limit);
          const float up = fmaxf(-limit, fminf(up_acc[assignment][r], limit));
          const float activated = gate / (1.0f + expf(-gate));
          out[(static_cast<size_t>(token) * T + slot) * I + row0 + r] =
              cast<bf16_t>(activated * up);
        }
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked = 2>
struct Gfx90aFp4ExpertGateUpKPrefetchOracleKernel {
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
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_gate_up_kprefetch_oracle_kernel<
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
