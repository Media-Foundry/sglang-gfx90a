#pragma once

// Compiled after gfx90a_fp4_expert_gemv.cuh by a standalone oracle only.

namespace sglang {

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kWaves,
          uint32_t kCtasPerExpert>
__global__ void __launch_bounds__(kWaves * kFp4ExpertWave)
    gfx90a_fp4_gate_wave_quant_oracle_kernel(
        bf16_t* __restrict__ intermediate,
        int8_t* __restrict__ output_q,
        float* __restrict__ output_scale,
        const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        float limit) {
  static_assert(I == 256 && K == 4096,
                "oracle is specialized for TP8 I256/K4096");
  static_assert(kAssignments == 4 && kWaves == 4 && kCtasPerExpert == 2,
                "oracle keeps the audited 2-CTA/4-wave mapping");
  constexpr uint32_t kRowsPerWave = 32;
  constexpr uint32_t kGroupsPerAssignment = I / 32;

  __shared__ uint32_t pair_lut[256];
  __shared__ bf16_t quant_tile[kWaves][kAssignments][kRowsPerWave];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t expert_block = blockIdx.x / kCtasPerExpert;
  const uint32_t cta_shard = blockIdx.x % kCtasPerExpert;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;
  if (expert_block >= valid_blocks) return;
  const int32_t expert_id = sorted_expert_ids[expert_block];
  if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) return;
  const uint32_t expert = static_cast<uint32_t>(expert_id);

  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t i_group = cta_shard * kWaves + wave;
  const uint32_t row_base = i_group * kRowsPerWave;
  uint32_t tokens[kAssignments];
  uint32_t slots[kAssignments];
  bool assignment_valid[kAssignments];
#pragma unroll
  for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
    const uint32_t encoded = static_cast<uint32_t>(
        sorted_ids[expert_block * kAssignments + assignment]);
    tokens[assignment] = encoded & 0x00ffffffu;
    slots[assignment] = encoded >> 24;
    assignment_valid[assignment] =
        tokens[assignment] < M && slots[assignment] < T;
  }

#pragma unroll
  for (uint32_t local_row = 0; local_row < kRowsPerWave; ++local_row) {
    const uint32_t row = row_base + local_row;
    float gate_acc[kAssignments] = {};
    float up_acc[kAssignments] = {};
    for (uint32_t group = lane; group < K / 32;
         group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
      const size_t gate_base =
          (static_cast<size_t>(expert) * (2 * I) + row) * (K / 2) +
          group * 16;
      const size_t up_base =
          (static_cast<size_t>(expert) * (2 * I) + I + row) * (K / 2) +
          group * 16;
      const float gate_scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
              expert, row, group)]);
      const float up_scale = gfx90a_e8m0_value(
          weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
              expert, I + row, group)]);
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
        if (!assignment_valid[assignment]) continue;
        const uint32_t token = tokens[assignment];
        const size_t xq_group =
            static_cast<size_t>(token) * (K / 32) + group;
        gate_acc[assignment] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + k0,
            gate_i8, x_scale[xq_group] * gate_scale * 0.5f);
        up_acc[assignment] += gfx90a_fp4_dot32_i8_prepacked(
            xq + static_cast<size_t>(token) * K + k0,
            up_i8, x_scale[xq_group] * up_scale * 0.5f);
      }
    }

#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      if (!assignment_valid[assignment]) continue;
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1) {
        gate_acc[assignment] +=
            __shfl_down(gate_acc[assignment], offset, kFp4ExpertWave);
        up_acc[assignment] +=
            __shfl_down(up_acc[assignment], offset, kFp4ExpertWave);
      }
      if (lane == 0) {
        const float gate = fminf(gate_acc[assignment], limit);
        const float up = fmaxf(-limit, fminf(up_acc[assignment], limit));
        const float activated = gate / (1.0f + expf(-gate));
        const bf16_t value = cast<bf16_t>(activated * up);
        const size_t output_assignment =
            static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
        intermediate[output_assignment * I + row] = value;
        quant_tile[wave][assignment][local_row] = value;
      }
    }
  }

  // Each wave owns its tile, so a wave barrier is sufficient before lanes
  // 0--15 reload the exact BF16 values and reproduce the existing HIP/Triton
  // group32 quant arithmetic.
  __syncwarp();
#pragma unroll
  for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
    if (!assignment_valid[assignment]) continue;
    if (lane < 16) {
      const float x0 = cast<float>(quant_tile[wave][assignment][lane]);
      const float x1 = cast<float>(quant_tile[wave][assignment][16 + lane]);
      float absmax = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
      for (uint32_t offset = 8; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_xor(absmax, offset, 16));
      }
      const float scale = fmaxf(absmax, 1.0e-10f) / 127.0f;
      const float q0 = fmaxf(-128.0f, fminf(127.0f, x0 / scale));
      const float q1 = fmaxf(-128.0f, fminf(127.0f, x1 / scale));
      const size_t output_assignment =
          static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
      const size_t q_base = output_assignment * I + row_base;
      output_q[q_base + lane] = static_cast<int8_t>(q0);
      output_q[q_base + 16 + lane] = static_cast<int8_t>(q1);
      if (lane == 0) {
        output_scale[output_assignment * kGroupsPerAssignment + i_group] =
            scale;
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kWaves,
          uint32_t kCtasPerExpert>
struct Gfx90aFp4GateWaveQuantOracleKernel {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView intermediate,
                  const tvm::ffi::TensorView output_q,
                  const tvm::ffi::TensorView output_scale,
                  double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto expert_blocks = SymbolicSize{"expert_blocks"};
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({expert_blocks}).with_dtype<int32_t>().with_device(device).verify(sorted_expert_ids);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(intermediate);
    TensorMatcher({M, T, I}).with_dtype<int8_t>().with_device(device).verify(output_q);
    TensorMatcher({M, T, I / 32}).with_dtype<float>().with_device(device).verify(output_scale);
    LaunchKernel(static_cast<uint32_t>(expert_blocks.unwrap()) * kCtasPerExpert,
                 kWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_gate_wave_quant_oracle_kernel<
            E, M, T, I, K, kAssignments, kWaves, kCtasPerExpert>,
        static_cast<bf16_t*>(intermediate.data_ptr()),
        static_cast<int8_t*>(output_q.data_ptr()),
        static_cast<float*>(output_scale.data_ptr()),
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
