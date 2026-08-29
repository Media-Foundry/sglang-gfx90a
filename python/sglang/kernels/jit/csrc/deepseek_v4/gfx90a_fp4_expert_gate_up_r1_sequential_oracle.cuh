#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves, uint32_t kBlocks,
          uint32_t kPrepacked>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_up_r1_sequential_oracle_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  static_assert(kPrepacked == 2, "R1 sequential oracle requires LDS LUT");
  __shared__ uint32_t pair_lut[256];
  __shared__ float gate_exchange[kNumWaves][kAssignments];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = global_wave; task < valid_blocks * I;
       task += total_waves) {
    const uint32_t expert_block = task / I;
    const uint32_t row = task % I;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);
    uint32_t encoded_ids[kAssignments];
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      encoded_ids[assignment] = static_cast<uint32_t>(
          sorted_ids[expert_block * kAssignments + assignment]);
    }

    float gate_acc[kAssignments] = {};
    for (uint32_t group = lane; group < K / 32; group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
      const size_t weight_base =
          (static_cast<size_t>(expert) * (2 * I) + row) * (K / 2) +
          group * 16;
      const float weight_group_scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
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
        const uint32_t token = encoded_ids[assignment] & 0x00ffffffu;
        const uint32_t slot = encoded_ids[assignment] >> 24;
        if (token >= M || slot >= T) continue;
        const size_t xq_group =
            static_cast<size_t>(token) * (K / 32) + group;
        gate_acc[assignment] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + k0, weight_i8,
            x_scale[xq_group] * weight_group_scale * 0.5f);
      }
    }
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1) {
        gate_acc[assignment] +=
            __shfl_down(gate_acc[assignment], offset, kFp4ExpertWave);
      }
      if (lane == 0) gate_exchange[wave][assignment] = gate_acc[assignment];
    }

    float up_acc[kAssignments] = {};
    for (uint32_t group = lane; group < K / 32; group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
      const uint32_t up_row = I + row;
      const size_t weight_base =
          (static_cast<size_t>(expert) * (2 * I) + up_row) * (K / 2) +
          group * 16;
      const float weight_group_scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
              expert, up_row, group)]);
      int32_t weight_i8[8];
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        weight_i8[j] = gfx90a_fp4_pack4_i8_lds(
            *reinterpret_cast<const uint16_t*>(weight + weight_base + j * 2),
            pair_lut);
      }
#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        const uint32_t token = encoded_ids[assignment] & 0x00ffffffu;
        const uint32_t slot = encoded_ids[assignment] >> 24;
        if (token >= M || slot >= T) continue;
        const size_t xq_group =
            static_cast<size_t>(token) * (K / 32) + group;
        up_acc[assignment] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + k0, weight_i8,
            x_scale[xq_group] * weight_group_scale * 0.5f);
      }
    }
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1) {
        up_acc[assignment] +=
            __shfl_down(up_acc[assignment], offset, kFp4ExpertWave);
      }
      const uint32_t token = encoded_ids[assignment] & 0x00ffffffu;
      const uint32_t slot = encoded_ids[assignment] >> 24;
      if (lane == 0 && token < M && slot < T) {
        const float gate = fminf(gate_exchange[wave][assignment], limit);
        const float up = fmaxf(-limit, fminf(up_acc[assignment], limit));
        const float activated = gate / (1.0f + expf(-gate));
        out[(static_cast<size_t>(token) * T + slot) * I + row] =
            cast<bf16_t>(activated * up);
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves, uint32_t kBlocks,
          uint32_t kPrepacked = 2>
struct Gfx90aFp4ExpertGateUpR1SequentialOracleKernel {
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
        gfx90a_fp4_expert_gate_up_r1_sequential_oracle_kernel<
            E, M, T, I, K, kAssignments, kNumWaves, kBlocks, kPrepacked>,
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
