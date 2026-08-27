#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kFp4ExpertWave = 64;
using gfx90a_i32x4 = int32_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float gfx90a_fp4_value(uint8_t bits) {
  const uint8_t mag = bits & 7;
  float value;
  switch (mag) {
    case 0: value = 0.0f; break;
    case 1: value = 0.5f; break;
    case 2: value = 1.0f; break;
    case 3: value = 1.5f; break;
    case 4: value = 2.0f; break;
    case 5: value = 3.0f; break;
    case 6: value = 4.0f; break;
    default: value = 6.0f; break;
  }
  return (bits & 8) ? -value : value;
}

__device__ __forceinline__ float gfx90a_e8m0_value(uint8_t exponent) {
  uint32_t bits = static_cast<uint32_t>(exponent) << 23;
  if (exponent == 0) bits = 0x00400000u;
  return __builtin_bit_cast(float, bits);
}

__device__ __forceinline__ float gfx90a_fp4_dot2_fp16(
    float x0, float x1, uint8_t packed_weight, float scale, float acc) {
  const fp16x2_t xh = cast<fp16x2_t>(fp32x2_t{x0, x1});
  const fp16x2_t wh = cast<fp16x2_t>(fp32x2_t{
      gfx90a_fp4_value(packed_weight & 15) * scale,
      gfx90a_fp4_value((packed_weight >> 4) & 15) * scale});
  return amd_mixed_dot(wh, xh, acc, false);
}

__device__ __forceinline__ int8_t gfx90a_fp4_i8_code(uint8_t bits) {
  const uint8_t mag = bits & 7;
  int8_t value;
  switch (mag) {
    case 0: value = 0; break;
    case 1: value = 1; break;
    case 2: value = 2; break;
    case 3: value = 3; break;
    case 4: value = 4; break;
    case 5: value = 6; break;
    case 6: value = 8; break;
    default: value = 12; break;
  }
  return (bits & 8) ? -value : value;
}

__device__ __forceinline__ int32_t gfx90a_fp4_pack4_i8(uint16_t packed) {
  constexpr uint32_t kPositiveLo = 0x03020100u;
  constexpr uint32_t kPositiveHi = 0x0c080604u;
  constexpr uint32_t kNegativeLo = 0xfdfeff00u;
  constexpr uint32_t kNegativeHi = 0xf4f8fafcu;
  const uint32_t value = packed;
  const uint32_t magnitude_selector =
      (value & 0x0007u) | ((value & 0x0070u) << 4) |
      ((value & 0x0700u) << 8) | ((value & 0x7000u) << 12);
  const uint32_t positive = __builtin_amdgcn_perm(
      kPositiveHi, kPositiveLo, magnitude_selector);
  const uint32_t negative = __builtin_amdgcn_perm(
      kNegativeHi, kNegativeLo, magnitude_selector);
  const uint32_t sign_selector =
      ((value & 0x0008u) >> 1) | ((value & 0x0080u) << 3) |
      ((value & 0x0800u) << 7) | ((value & 0x8000u) << 11);
  return static_cast<int32_t>(__builtin_amdgcn_perm(
      negative, positive, sign_selector | 0x03020100u));
}

__device__ __forceinline__ int32_t gfx90a_fp4_pack4_i8_lds(
    uint16_t packed, const uint32_t* pair_lut) {
  const uint32_t lo = pair_lut[packed & 0xffu] & 0xffffu;
  const uint32_t hi = pair_lut[packed >> 8] & 0xffffu;
  return static_cast<int32_t>(lo | (hi << 16));
}

__device__ __forceinline__ float gfx90a_fp4_dot32_i8(
    const int8_t* x, const uint8_t* weight, float combined_scale) {
  int32_t acc = 0;
#pragma unroll
  for (uint32_t j = 0; j < 32; j += 4) {
    const int32_t xv = *reinterpret_cast<const int32_t*>(x + j);
    const uint16_t packed =
        *reinterpret_cast<const uint16_t*>(weight + j / 2);
    acc = __builtin_amdgcn_sdot4(
        xv, gfx90a_fp4_pack4_i8(packed), acc, false);
  }
  return static_cast<float>(acc) * combined_scale;
}

__device__ __forceinline__ float gfx90a_fp4_dot32_i8_prepacked(
    const int8_t* x, const int32_t* packed_weight, float combined_scale) {
  int32_t acc = 0;
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    const int32_t xv = *reinterpret_cast<const int32_t*>(x + j * 4);
    acc = __builtin_amdgcn_sdot4(xv, packed_weight[j], acc, false);
  }
  return static_cast<float>(acc) * combined_scale;
}

template <uint32_t E, uint32_t I, uint32_t K>
__device__ __forceinline__ size_t gfx90a_gate_up_scale_offset(
    uint32_t expert, uint32_t row, uint32_t group) {
  constexpr uint32_t N1 = I / 16;
  constexpr uint32_t K1 = (K / 32) / 8;
  const uint32_t pack2 = row / I;
  const uint32_t local_row = row - pack2 * I;
  const uint32_t n1 = local_row / 16;
  const uint32_t nlane = local_row % 16;
  const uint32_t k1 = group / 8;
  const uint32_t krem = group % 8;
  const uint32_t kpack = krem / 4;
  const uint32_t klane = krem % 4;
  return ((((((static_cast<size_t>(expert) * N1 + n1) * K1 + k1) * 4 +
              klane) * 16 + nlane) * 2 + kpack) * 2 + pack2);
}

template <uint32_t E, uint32_t N, uint32_t K>
__device__ __forceinline__ size_t gfx90a_down_scale_offset(
    uint32_t expert, uint32_t row, uint32_t group) {
  constexpr uint32_t N1 = N / 32;
  constexpr uint32_t K1 = (K / 32) / 8;
  const uint32_t n1 = row / 32;
  const uint32_t nrem = row % 32;
  const uint32_t npack = nrem / 16;
  const uint32_t nlane = nrem % 16;
  const uint32_t k1 = group / 8;
  const uint32_t krem = group % 8;
  const uint32_t kpack = krem / 4;
  const uint32_t klane = krem % 4;
  return ((((((static_cast<size_t>(expert) * N1 + n1) * K1 + k1) * 4 +
              klane) * 16 + nlane) * 2 + kpack) * 2 + npack);
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t GE,
          uint32_t I, uint32_t K,
          uint32_t kRows, uint32_t kNumWaves,
          uint32_t kSlotBegin, uint32_t kSlotEnd>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_up_kernel(
        bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
        const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ expert_mask,
        const int32_t* __restrict__ live_count, float limit) {
  __shared__ int8_t sxq[M == 1 ? K : 1];
  __shared__ float sx_scale[M == 1 ? K / 32 : 1];
  if constexpr (M == 1) {
    if (xq == nullptr) {
      for (uint32_t group = threadIdx.x; group < K / 32;
           group += blockDim.x) {
        const uint32_t k0 = group * 32;
        float local_max = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j) {
          local_max = fmaxf(local_max, fabsf(cast<float>(x[k0 + j])));
        }
        const float scale = fmaxf(local_max / 127.0f, 1.0e-12f);
        const float inv_scale = 1.0f / scale;
        sx_scale[group] = scale;
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j) {
          const float q = nearbyintf(cast<float>(x[k0 + j]) * inv_scale);
          sxq[k0 + j] =
              static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, q)));
        }
      }
      __syncthreads();
    }
  }
  constexpr uint32_t kTilesPerAssignment = (I + kRows - 1) / kRows;
  constexpr uint32_t kOwnedSlots = kSlotEnd - kSlotBegin;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t live =
      live_count == nullptr
          ? M
          : min(static_cast<uint32_t>(max(live_count[0], 0)), M);

  for (uint32_t task = global_wave;
       task < live * kOwnedSlots * kTilesPerAssignment;
       task += total_waves) {
    const uint32_t assignment = task / kTilesPerAssignment;
    const uint32_t token = assignment / kOwnedSlots;
    const uint32_t slot = kSlotBegin + assignment % kOwnedSlots;
    const uint32_t row0 = (task % kTilesPerAssignment) * kRows;
    const int32_t expert_id = expert_ids[static_cast<size_t>(token) * T + slot];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(GE) ||
        (expert_mask != nullptr && expert_mask[expert_id] == 0)) {
      continue;
    }
    const uint32_t global_expert = static_cast<uint32_t>(expert_id);
    const uint32_t expert = global_expert % E;
    float gate_acc[kRows] = {};
    float up_acc[kRows] = {};

    for (uint32_t group = lane; group < K / 32; group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
      float xv[32];
      const bf16_t* x_row = x + static_cast<size_t>(token) * K;
      if constexpr (M != 1) {
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j) {
          xv[j] = cast<float>(x_row[k0 + j]);
        }
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t local_row = row0 + r;
        if (local_row < I) {
          const uint32_t gate_row = local_row;
          const uint32_t up_row = I + local_row;
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
          if (xq != nullptr) {
            const size_t xq_group =
                static_cast<size_t>(token) * (K / 32) + group;
            gate_acc[r] += gfx90a_fp4_dot32_i8(
                xq + static_cast<size_t>(token) * K + k0,
                weight + gate_base,
                x_scale[xq_group] * gate_scale * 0.5f);
            up_acc[r] += gfx90a_fp4_dot32_i8(
                xq + static_cast<size_t>(token) * K + k0,
                weight + up_base,
                x_scale[xq_group] * up_scale * 0.5f);
          } else if constexpr (M == 1) {
            gate_acc[r] += gfx90a_fp4_dot32_i8(
                sxq + k0, weight + gate_base,
                sx_scale[group] * gate_scale * 0.5f);
            up_acc[r] += gfx90a_fp4_dot32_i8(
                sxq + k0, weight + up_base,
                sx_scale[group] * up_scale * 0.5f);
          } else {
#pragma unroll
            for (uint32_t j = 0; j < 32; j += 2) {
            gate_acc[r] = gfx90a_fp4_dot2_fp16(
                xv[j], xv[j + 1], weight[gate_base + j / 2], gate_scale,
                gate_acc[r]);
            up_acc[r] = gfx90a_fp4_dot2_fp16(
                xv[j], xv[j + 1], weight[up_base + j / 2], up_scale,
                up_acc[r]);
            }
          }
        }
      }
    }

#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1) {
        gate_acc[r] += __shfl_down(gate_acc[r], offset, kFp4ExpertWave);
        up_acc[r] += __shfl_down(up_acc[r], offset, kFp4ExpertWave);
      }
      if (lane == 0 && row0 + r < I) {
        const float gate = fminf(gate_acc[r], limit);
        const float up = fmaxf(-limit, fminf(up_acc[r], limit));
        const float activated = gate / (1.0f + expf(-gate));
        out[(static_cast<size_t>(token) * T + slot) * I + row0 + r] =
            cast<bf16_t>(activated * up);
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t GE,
          uint32_t N, uint32_t K,
          uint32_t kRows, uint32_t kNumWaves,
          uint32_t kSlotBegin, uint32_t kSlotEnd>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_down_kernel(
        bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
        const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ expert_mask,
        const float* __restrict__ topk_weights,
        const int32_t* __restrict__ live_count) {
  __shared__ int8_t sxq[M == 1 ? T * K : 1];
  __shared__ float sx_scale[M == 1 ? T * K / 32 : 1];
  if constexpr (M == 1) {
    if (xq == nullptr) {
      for (uint32_t group = kSlotBegin * (K / 32) + threadIdx.x;
           group < kSlotEnd * (K / 32);
           group += blockDim.x) {
        const uint32_t k0 = group * 32;
        float local_max = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j) {
          local_max = fmaxf(local_max, fabsf(cast<float>(x[k0 + j])));
        }
        const float scale = fmaxf(local_max / 127.0f, 1.0e-12f);
        const float inv_scale = 1.0f / scale;
        sx_scale[group] = scale;
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j) {
          const float q = nearbyintf(cast<float>(x[k0 + j]) * inv_scale);
          sxq[k0 + j] =
              static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, q)));
        }
      }
      __syncthreads();
    }
  }
  constexpr uint32_t kTilesPerRow = (N + kRows - 1) / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  constexpr uint32_t kSubgroupWidth = 16;
  constexpr uint32_t kSubgroups = kFp4ExpertWave / kSubgroupWidth;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t live =
      live_count == nullptr
          ? M
          : min(static_cast<uint32_t>(max(live_count[0], 0)), M);

  for (uint32_t task = global_wave; task < live * kTilesPerRow;
       task += total_waves) {
    const uint32_t token = task / kTilesPerRow;
    const uint32_t row0 = (task % kTilesPerRow) * kRows;
    float acc[kRows] = {};

    for (uint32_t slot = kSlotBegin + subgroup; slot < kSlotEnd;
         slot += kSubgroups) {
      const int32_t expert_id =
          expert_ids[static_cast<size_t>(token) * T + slot];
      if (expert_id < 0 || expert_id >= static_cast<int32_t>(GE) ||
          (expert_mask != nullptr && expert_mask[expert_id] == 0)) {
        continue;
      }
      const uint32_t expert = static_cast<uint32_t>(expert_id) % E;
      const float routed_weight =
          topk_weights[static_cast<size_t>(token) * T + slot];
      float slot_acc[kRows] = {};
      for (uint32_t group = subgroup_lane; group < K / 32;
           group += kSubgroupWidth) {
        const uint32_t k0 = group * 32;
        float xv[32];
        const bf16_t* x_row =
            x + (static_cast<size_t>(token) * T + slot) * K;
        if constexpr (M != 1) {
#pragma unroll
          for (uint32_t j = 0; j < 32; ++j) {
            xv[j] = cast<float>(x_row[k0 + j]);
          }
        }
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          const uint32_t row = row0 + r;
          if (row < N) {
            const size_t weight_base =
                (static_cast<size_t>(expert) * N + row) * (K / 2) +
                group * 16;
            const float scale = gfx90a_e8m0_value(
                weight_scale[gfx90a_down_scale_offset<E, N, K>(
                    expert, row, group)]);
            if (xq != nullptr) {
              const size_t xq_group =
                  (static_cast<size_t>(token) * T + slot) * (K / 32) + group;
              slot_acc[r] += gfx90a_fp4_dot32_i8(
                  xq + (static_cast<size_t>(token) * T + slot) * K + k0,
                  weight + weight_base,
                  x_scale[xq_group] * scale * 0.5f);
            } else if constexpr (M == 1) {
              const uint32_t scale_group = slot * (K / 32) + group;
              slot_acc[r] += gfx90a_fp4_dot32_i8(
                  sxq + static_cast<size_t>(slot) * K + k0,
                  weight + weight_base,
                  sx_scale[scale_group] * scale * 0.5f);
            } else {
#pragma unroll
              for (uint32_t j = 0; j < 32; j += 2) {
                slot_acc[r] = gfx90a_fp4_dot2_fp16(
                    xv[j], xv[j + 1], weight[weight_base + j / 2],
                    scale, slot_acc[r]);
              }
            }
          }
        }
      }
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
        for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
          slot_acc[r] += __shfl_down(slot_acc[r], offset, kSubgroupWidth);
        }
        if (subgroup_lane == 0) {
          acc[r] += slot_acc[r] * routed_weight;
        }
      }
    }

#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const float total = __shfl(acc[r], 0, kFp4ExpertWave) +
                          __shfl(acc[r], 16, kFp4ExpertWave) +
                          __shfl(acc[r], 32, kFp4ExpertWave) +
                          __shfl(acc[r], 48, kFp4ExpertWave);
      if (lane == 0 && row0 + r < N) {
        out[static_cast<size_t>(token) * N + row0 + r] = cast<bf16_t>(total);
      }
    }
  }
}

// Prefill routes several tokens to each expert.  Grouping four assignments per
// wave lets all four dot products reuse the same packed FP4 weight and scale
// loads.  AIter's sorter encodes the token in the low 24 bits and the top-k
// slot in the high 8 bits of sorted_ids.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kPrepacked>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_up_grouped_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  __shared__ uint32_t pair_lut[256];
  if constexpr (kPrepacked == 2) {
    if (threadIdx.x < 256) {
      pair_lut[threadIdx.x] = static_cast<uint32_t>(
          gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
    }
    __syncthreads();
  }
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
          if constexpr (kPrepacked == 1) {
            gate_i8[j] = *reinterpret_cast<const int32_t*>(
                weight + gate_base + j * 4);
            up_i8[j] = *reinterpret_cast<const int32_t*>(
                weight + up_base + j * 4);
          } else if constexpr (kPrepacked == 2) {
            gate_i8[j] = gfx90a_fp4_pack4_i8_lds(
                *reinterpret_cast<const uint16_t*>(weight + gate_base + j * 2),
                pair_lut);
            up_i8[j] = gfx90a_fp4_pack4_i8_lds(
                *reinterpret_cast<const uint16_t*>(weight + up_base + j * 2),
                pair_lut);
          } else {
            gate_i8[j] = gfx90a_fp4_pack4_i8(
                *reinterpret_cast<const uint16_t*>(weight + gate_base + j * 2));
            up_i8[j] = gfx90a_fp4_pack4_i8(
                *reinterpret_cast<const uint16_t*>(weight + up_base + j * 2));
          }
        }
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const uint32_t token = tokens[assignment];
          const size_t xq_group =
              static_cast<size_t>(token) * (K / 32) + group;
          gate_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + static_cast<size_t>(token) * K + k0,
              gate_i8,
              x_scale[xq_group] * gate_scale * 0.5f);
          up_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + static_cast<size_t>(token) * K + k0,
              up_i8,
              x_scale[xq_group] * up_scale * 0.5f);
        }
      }
    }

#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      if (!assignment_valid[assignment]) continue;
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
          uint32_t kBlocks, uint32_t kPrepacked = 0>
struct Gfx90aFp4ExpertGateUpGroupedKernel {
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
    if constexpr (kPrepacked == 1) {
      TensorMatcher({E, 2 * I, K}).with_dtype<int8_t>().with_device(device).verify(weight);
    } else {
      TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    }
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_gate_up_grouped_kernel<
            E, M, T, I, K, kAssignments, kRows, kNumWaves, kPrepacked>,
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

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4,
          uint32_t kBroadcastScales = 0, uint32_t kAssignments = 32>
__global__ void __launch_bounds__(kSplit * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_up_mfma32_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  static_assert(kAssignments == 32 || kAssignments == 64);
  constexpr uint32_t kAssignmentHalves = kAssignments / 16;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kTiles = I / 16;
  __shared__ float gate_partial[kSplit][kAssignments * 16];
  __shared__ float up_partial[kSplit][kAssignments * 16];
  const uint32_t lane = threadIdx.x & 63;
  const uint32_t split = threadIdx.x >> 6;
  const uint32_t matrix_index = lane & 15;
  const uint32_t k_lane = (lane >> 4) * 4;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks = (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = blockIdx.x; task < valid_blocks * kTiles;
       task += kBlocks) {
    const uint32_t expert_block = task / kTiles;
    const uint32_t local_row = (task % kTiles) * 16 + matrix_index;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    const uint32_t expert = static_cast<uint32_t>(max(expert_id, 0));
    uint32_t assignment_lane = 0;
    if constexpr (kBroadcastScales != 0) {
      if (lane < kAssignments) {
        assignment_lane = static_cast<uint32_t>(
            sorted_ids[expert_block * kAssignments + lane]);
      }
    }
    uint32_t tokens[kAssignmentHalves][4], slots[kAssignmentHalves][4];
    bool assignment_valid[kAssignmentHalves][4];
#pragma unroll
    for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) {
        const uint32_t assignment = half * 16 + (lane >> 4) * 4 + r;
        const uint32_t encoded = kBroadcastScales != 0
            ? __shfl(assignment_lane, assignment, kFp4ExpertWave)
            : static_cast<uint32_t>(
                  sorted_ids[expert_block * kAssignments + assignment]);
        const uint32_t token = encoded & 0x00ffffffu;
        const uint32_t slot = encoded >> 24;
        assignment_valid[half][r] = token < M && slot < T;
        // AIter pads each expert block with an encoded sentinel.  Keep every
        // subsequently formed address in bounds even for padded MFMA rows;
        // merely masking the load result still leaves room for speculative
        // address generation on gfx90a.
        tokens[half][r] = assignment_valid[half][r] ? token : 0;
        slots[half][r] = assignment_valid[half][r] ? slot : 0;
      }
    }
    uint32_t a_tokens[kAssignmentHalves], a_slots[kAssignmentHalves];
    bool a_valid[kAssignmentHalves];
#pragma unroll
    for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
      const uint32_t assignment = half * 16 + matrix_index;
      const uint32_t encoded = kBroadcastScales != 0
          ? __shfl(assignment_lane, assignment, kFp4ExpertWave)
          : static_cast<uint32_t>(
                sorted_ids[expert_block * kAssignments + assignment]);
      const uint32_t token = encoded & 0x00ffffffu;
      const uint32_t slot = encoded >> 24;
      a_valid[half] = token < M && slot < T;
      a_tokens[half] = a_valid[half] ? token : 0;
      a_slots[half] = a_valid[half] ? slot : 0;
    }
    float gate_acc[kAssignmentHalves][4] = {};
    float up_acc[kAssignmentHalves][4] = {};
    if (expert_id >= 0 && expert_id < static_cast<int32_t>(E)) {
      for (uint32_t group = split; group < kGroups; group += kSplit) {
        const uint32_t k0 = group * 32;
        const size_t gate_base =
            (static_cast<size_t>(expert) * (2 * I) + local_row) * (K / 2) +
            group * 16;
        const size_t up_base =
            (static_cast<size_t>(expert) * (2 * I) + I + local_row) *
                (K / 2) +
            group * 16;
        const int32_t gate_b0 = gfx90a_fp4_pack4_i8(
            *reinterpret_cast<const uint16_t*>(weight + gate_base + k_lane / 2));
        const int32_t gate_b1 = gfx90a_fp4_pack4_i8(
            *reinterpret_cast<const uint16_t*>(weight + gate_base + 8 + k_lane / 2));
        const int32_t up_b0 = gfx90a_fp4_pack4_i8(
            *reinterpret_cast<const uint16_t*>(weight + up_base + k_lane / 2));
        const int32_t up_b1 = gfx90a_fp4_pack4_i8(
            *reinterpret_cast<const uint16_t*>(weight + up_base + 8 + k_lane / 2));
        float gate_scale;
        float up_scale;
        float x_scale_lane = 0.0f;
        if constexpr (kBroadcastScales != 0) {
          float gate_scale_lane = 0.0f;
          float up_scale_lane = 0.0f;
          if (lane < 16) {
            gate_scale_lane = gfx90a_e8m0_value(
                weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                    expert, (task % kTiles) * 16 + lane, group)]);
            up_scale_lane = gfx90a_e8m0_value(
                weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                    expert, I + (task % kTiles) * 16 + lane, group)]);
          }
          gate_scale =
              __shfl(gate_scale_lane, matrix_index, kFp4ExpertWave);
          up_scale = __shfl(up_scale_lane, matrix_index, kFp4ExpertWave);
          if (lane < kAssignments) {
            const uint32_t encoded = assignment_lane;
            const uint32_t token = encoded & 0x00ffffffu;
            const uint32_t slot = encoded >> 24;
            if (token < M && slot < T) {
              x_scale_lane =
                  x_scale[static_cast<size_t>(token) * kGroups + group];
            }
          }
        } else {
          gate_scale = gfx90a_e8m0_value(
              weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                  expert, local_row, group)]);
          up_scale = gfx90a_e8m0_value(
              weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                  expert, I + local_row, group)]);
        }
#pragma unroll
        for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
          const size_t x_base = static_cast<size_t>(a_tokens[half]) * K + k0;
          const int32_t av0 = a_valid[half]
              ? *reinterpret_cast<const int32_t*>(xq + x_base + k_lane)
              : 0;
          const int32_t av1 = a_valid[half]
              ? *reinterpret_cast<const int32_t*>(xq + x_base + 16 + k_lane)
              : 0;
          gfx90a_i32x4 gate_cv{0, 0, 0, 0};
          gfx90a_i32x4 up_cv{0, 0, 0, 0};
          gate_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
              av0, gate_b0, gate_cv, 0, 0, 0);
          gate_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
              av1, gate_b1, gate_cv, 0, 0, 0);
          up_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
              av0, up_b0, up_cv, 0, 0, 0);
          up_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
              av1, up_b1, up_cv, 0, 0, 0);
#pragma unroll
          for (uint32_t r = 0; r < 4; ++r) {
            float xs;
            if constexpr (kBroadcastScales != 0) {
              const uint32_t assignment =
                  half * 16 + (lane >> 4) * 4 + r;
              xs = __shfl(x_scale_lane, assignment, kFp4ExpertWave);
            } else {
              xs = assignment_valid[half][r]
                  ? x_scale[static_cast<size_t>(tokens[half][r]) * kGroups +
                            group]
                  : 0.0f;
            }
            gate_acc[half][r] +=
                static_cast<float>(gate_cv[r]) * xs * gate_scale * 0.5f;
            up_acc[half][r] +=
                static_cast<float>(up_cv[r]) * xs * up_scale * 0.5f;
          }
        }
      }
    }
#pragma unroll
    for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) {
        const uint32_t index =
            lane * (kAssignmentHalves * 4) + half * 4 + r;
        gate_partial[split][index] = gate_acc[half][r];
        up_partial[split][index] = up_acc[half][r];
      }
    }
    __syncthreads();
    if (split == 0) {
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          const uint32_t index =
              lane * (kAssignmentHalves * 4) + half * 4 + r;
          float gate = gate_partial[0][index];
          float up = up_partial[0][index];
#pragma unroll
          for (uint32_t s = 1; s < kSplit; ++s) {
            gate += gate_partial[s][index];
            up += up_partial[s][index];
          }
          if (assignment_valid[half][r] && expert_id >= 0 &&
              expert_id < static_cast<int32_t>(E)) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
            const float activated = gate / (1.0f + expf(-gate));
            const size_t output_assignment =
                static_cast<size_t>(tokens[half][r]) * T + slots[half][r];
            out[output_assignment * I + local_row] =
                cast<bf16_t>(activated * up);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4,
          uint32_t kBroadcastScales = 0, uint32_t kAssignments = 32>
struct Gfx90aFp4ExpertGateUpMfma32Kernel {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(kBlocks, kSplit * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_gate_up_mfma32_kernel<
            E, M, T, I, K, kBlocks, kSplit, kBroadcastScales, kAssignments>,
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

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kPrepacked>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_down_grouped_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        const float* __restrict__ topk_weights) {
  __shared__ uint32_t pair_lut[256];
  if constexpr (kPrepacked == 2) {
    if (threadIdx.x < 256) {
      pair_lut[threadIdx.x] = static_cast<uint32_t>(
          gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
    }
    __syncthreads();
  }
  static_assert(K >= 32 && K % 32 == 0,
                "grouped down requires a positive group-32 K dimension");
  constexpr uint32_t kGroups = K / 32;
  // TP8 shards the down projection to K=256, so only eight group-32 dot
  // products exist.  Keep all lanes productive there instead of carrying an
  // otherwise idle upper half of a 16-lane subgroup through every task and
  // reduction.  Wider K retains the established 16-lane mapping.
  constexpr uint32_t kSubgroupWidth = kGroups < 16 ? kGroups : 16;
  static_assert((kSubgroupWidth & (kSubgroupWidth - 1)) == 0,
                "grouped down subgroup width must be a power of two");
  static_assert(kFp4ExpertWave % kSubgroupWidth == 0,
                "grouped down subgroup width must divide wave64");
  constexpr uint32_t kSubgroupsPerWave = kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kTilesPerExpertBlock = (N + kRows - 1) / kRows;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t block_subgroup = wave * kSubgroupsPerWave + subgroup;
  const uint32_t subgroups_per_block = kNumWaves * kSubgroupsPerWave;
  const uint32_t global_subgroup =
      blockIdx.x * subgroups_per_block + block_subgroup;
  const uint32_t total_subgroups = gridDim.x * subgroups_per_block;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = global_subgroup;
       task < valid_blocks * kTilesPerExpertBlock;
       task += total_subgroups) {
    const uint32_t expert_block = task / kTilesPerExpertBlock;
    const uint32_t row0 = (task % kTilesPerExpertBlock) * kRows;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);

    uint32_t tokens[kAssignments];
    uint32_t slots[kAssignments];
    bool assignment_valid[kAssignments];
    float acc[kAssignments][kRows] = {};
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[expert_block * kAssignments + assignment]);
      tokens[assignment] = encoded & 0x00ffffffu;
      slots[assignment] = encoded >> 24;
      assignment_valid[assignment] =
          tokens[assignment] < M && slots[assignment] < T;
    }

    for (uint32_t group = subgroup_lane; group < K / 32;
         group += kSubgroupWidth) {
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t row = row0 + r;
        if (row >= N) continue;
        const size_t weight_base =
            (static_cast<size_t>(expert) * N + row) *
                (kPrepacked == 1 ? K : K / 2) +
            group * (kPrepacked == 1 ? 32 : 16);
        const float scale = gfx90a_e8m0_value(
            weight_scale[gfx90a_down_scale_offset<E, N, K>(
                expert, row, group)]);
        int32_t weight_i8[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          if constexpr (kPrepacked == 1) {
            weight_i8[j] = *reinterpret_cast<const int32_t*>(
                weight + weight_base + j * 4);
          } else if constexpr (kPrepacked == 2) {
            weight_i8[j] = gfx90a_fp4_pack4_i8_lds(
                *reinterpret_cast<const uint16_t*>(weight + weight_base + j * 2),
                pair_lut);
          } else {
            weight_i8[j] = gfx90a_fp4_pack4_i8(
                *reinterpret_cast<const uint16_t*>(weight + weight_base + j * 2));
          }
        }
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const size_t input_assignment =
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          const size_t xq_group = input_assignment * (K / 32) + group;
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
        for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
          acc[assignment][r] +=
              __shfl_down(acc[assignment][r], offset, kSubgroupWidth);
        }
        if (subgroup_lane == 0 && row0 + r < N) {
          partial[output_assignment * N + row0 + r] =
              acc[assignment][r] * routed_weight;
        }
      }
    }
  }
}

template <uint32_t M, uint32_t T, uint32_t N>
__global__ void gfx90a_fp4_expert_down_reduce_kernel(
    bf16_t* __restrict__ out, const float* __restrict__ partial) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= M * N) return;
  const uint32_t token = index / N;
  const uint32_t row = index - token * N;
  const size_t base = static_cast<size_t>(token) * T * N + row;
  // Match the legacy wave's four subgroup accumulators exactly: subgroup 0
  // owns slots 0/4, subgroup 1 owns 1/5, then lanes 0/16/32/48 are summed.
  static_assert(T == 6, "DSV4 grouped down expects top-k 6");
  const float subgroup0 = partial[base] + partial[base + 4 * N];
  const float subgroup1 = partial[base + N] + partial[base + 5 * N];
  const float acc =
      subgroup0 + subgroup1 + partial[base + 2 * N] + partial[base + 3 * N];
  out[index] = cast<bf16_t>(acc);
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4,
          uint32_t kBroadcastScales = 0, uint32_t kAssignments = 32>
__global__ void __launch_bounds__(kSplit * kFp4ExpertWave)
    gfx90a_fp4_expert_down_mfma32_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        const float* __restrict__ topk_weights) {
  static_assert(kAssignments == 32 || kAssignments == 64);
  constexpr uint32_t kAssignmentHalves = kAssignments / 16;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kTiles = N / 16;
  __shared__ float tile_partial[kSplit][kAssignments * 16];
  const uint32_t lane = threadIdx.x & 63;
  const uint32_t split = threadIdx.x >> 6;
  const uint32_t matrix_index = lane & 15;
  const uint32_t k_lane = (lane >> 4) * 4;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks = (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = blockIdx.x; task < valid_blocks * kTiles;
       task += kBlocks) {
    const uint32_t expert_block = task / kTiles;
    const uint32_t row = (task % kTiles) * 16 + matrix_index;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    const uint32_t expert = static_cast<uint32_t>(max(expert_id, 0));
    uint32_t assignment_lane = 0;
    float assignment_weight_lane = 0.0f;
    if constexpr (kBroadcastScales != 0) {
      if (lane < kAssignments) {
        const uint32_t encoded = static_cast<uint32_t>(
            sorted_ids[expert_block * kAssignments + lane]);
        assignment_lane = encoded;
        const uint32_t token = encoded & 0x00ffffffu;
        const uint32_t slot = encoded >> 24;
        assignment_weight_lane = token < M && slot < T
            ? topk_weights[static_cast<size_t>(token) * T + slot]
            : 0.0f;
      }
    }
    uint32_t tokens[kAssignmentHalves][4], slots[kAssignmentHalves][4];
    bool assignment_valid[kAssignmentHalves][4];
#pragma unroll
    for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) {
        const uint32_t assignment = half * 16 + (lane >> 4) * 4 + r;
        const uint32_t encoded = kBroadcastScales != 0
            ? __shfl(assignment_lane, assignment, kFp4ExpertWave)
            : static_cast<uint32_t>(
                  sorted_ids[expert_block * kAssignments + assignment]);
        const uint32_t token = encoded & 0x00ffffffu;
        const uint32_t slot = encoded >> 24;
        assignment_valid[half][r] = token < M && slot < T;
        tokens[half][r] = assignment_valid[half][r] ? token : 0;
        slots[half][r] = assignment_valid[half][r] ? slot : 0;
      }
    }
    uint32_t a_tokens[kAssignmentHalves], a_slots[kAssignmentHalves];
    bool a_valid[kAssignmentHalves];
#pragma unroll
    for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
      const uint32_t assignment = half * 16 + matrix_index;
      const uint32_t encoded = kBroadcastScales != 0
          ? __shfl(assignment_lane, assignment, kFp4ExpertWave)
          : static_cast<uint32_t>(
                sorted_ids[expert_block * kAssignments + assignment]);
      const uint32_t token = encoded & 0x00ffffffu;
      const uint32_t slot = encoded >> 24;
      a_valid[half] = token < M && slot < T;
      a_tokens[half] = a_valid[half] ? token : 0;
      a_slots[half] = a_valid[half] ? slot : 0;
    }
    float acc[kAssignmentHalves][4] = {};
    if (expert_id >= 0 && expert_id < static_cast<int32_t>(E)) {
      for (uint32_t group = split; group < kGroups; group += kSplit) {
        const uint32_t k0 = group * 32;
        const size_t weight_base =
            (static_cast<size_t>(expert) * N + row) * (K / 2) + group * 16;
        const int32_t bv0 = gfx90a_fp4_pack4_i8(
            *reinterpret_cast<const uint16_t*>(weight + weight_base + k_lane / 2));
        const int32_t bv1 = gfx90a_fp4_pack4_i8(
            *reinterpret_cast<const uint16_t*>(weight + weight_base + 8 + k_lane / 2));
        float weight_s;
        float x_scale_lane = 0.0f;
        if constexpr (kBroadcastScales != 0) {
          float weight_scale_lane = 0.0f;
          if (lane < 16) {
            weight_scale_lane = gfx90a_e8m0_value(
                weight_scale[gfx90a_down_scale_offset<E, N, K>(
                    expert, (task % kTiles) * 16 + lane, group)]);
          }
          weight_s =
              __shfl(weight_scale_lane, matrix_index, kFp4ExpertWave);
          if (lane < kAssignments) {
            const uint32_t encoded = assignment_lane;
            const uint32_t token = encoded & 0x00ffffffu;
            const uint32_t slot = encoded >> 24;
            if (token < M && slot < T) {
              x_scale_lane =
                  x_scale[(static_cast<size_t>(token) * T + slot) * kGroups +
                          group];
            }
          }
        } else {
          weight_s = gfx90a_e8m0_value(
              weight_scale[gfx90a_down_scale_offset<E, N, K>(
                  expert, row, group)]);
        }
#pragma unroll
        for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
          const size_t input_assignment =
              static_cast<size_t>(a_tokens[half]) * T + a_slots[half];
          const size_t x_base = input_assignment * K + k0;
          const int32_t av0 = a_valid[half]
              ? *reinterpret_cast<const int32_t*>(xq + x_base + k_lane)
              : 0;
          const int32_t av1 = a_valid[half]
              ? *reinterpret_cast<const int32_t*>(xq + x_base + 16 + k_lane)
              : 0;
          gfx90a_i32x4 cv{0, 0, 0, 0};
          cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
              av0, bv0, cv, 0, 0, 0);
          cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
              av1, bv1, cv, 0, 0, 0);
#pragma unroll
          for (uint32_t r = 0; r < 4; ++r) {
            float xs;
            if constexpr (kBroadcastScales != 0) {
              const uint32_t assignment =
                  half * 16 + (lane >> 4) * 4 + r;
              xs = __shfl(x_scale_lane, assignment, kFp4ExpertWave);
            } else {
              const size_t scale_index =
                  (static_cast<size_t>(tokens[half][r]) * T +
                   slots[half][r]) *
                      kGroups +
                  group;
              xs = assignment_valid[half][r] ? x_scale[scale_index] : 0.0f;
            }
            acc[half][r] +=
                static_cast<float>(cv[r]) * xs * weight_s * 0.5f;
          }
        }
      }
    }
#pragma unroll
    for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) {
        tile_partial[split]
                    [lane * (kAssignmentHalves * 4) + half * 4 + r] =
                        acc[half][r];
      }
    }
    __syncthreads();
    if (split == 0) {
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          const uint32_t index =
              lane * (kAssignmentHalves * 4) + half * 4 + r;
          float total = tile_partial[0][index];
#pragma unroll
          for (uint32_t s = 1; s < kSplit; ++s) {
            total += tile_partial[s][index];
          }
          if (assignment_valid[half][r] && expert_id >= 0 &&
              expert_id < static_cast<int32_t>(E)) {
            const size_t output_assignment =
                static_cast<size_t>(tokens[half][r]) * T + slots[half][r];
            const uint32_t assignment =
                half * 16 + (lane >> 4) * 4 + r;
            const float routed_weight = kBroadcastScales != 0
                ? __shfl(assignment_weight_lane, assignment,
                         kFp4ExpertWave)
                : topk_weights[output_assignment];
            partial[output_assignment * N + row] =
                total * routed_weight;
          }
        }
      }
    }
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked = 0>
struct Gfx90aFp4ExpertDownGroupedKernel {
  // Expose the producer and fixed-slot reduction separately for standalone
  // occupancy-bucket experiments.  The normal `run` entry point below keeps
  // the production launch sequence unchanged.
  static void run_partial(const tvm::ffi::TensorView xq,
                          const tvm::ffi::TensorView x_scale,
                          const tvm::ffi::TensorView weight,
                          const tvm::ffi::TensorView weight_scale,
                          const tvm::ffi::TensorView sorted_ids,
                          const tvm::ffi::TensorView sorted_expert_ids,
                          const tvm::ffi::TensorView num_valid_ids,
                          const tvm::ffi::TensorView topk_weights,
                          const tvm::ffi::TensorView partial) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    if constexpr (kPrepacked == 1) {
      TensorMatcher({E, N, K}).with_dtype<int8_t>().with_device(device).verify(weight);
    } else {
      TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    }
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partial);
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_down_grouped_kernel<
            E, M, T, N, K, kAssignments, kRows, kNumWaves, kPrepacked>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }

  static void reduce(const tvm::ffi::TensorView partial,
                     const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    constexpr uint32_t kThreads = 256;
    LaunchKernel((M * N + kThreads - 1) / kThreads, kThreads, out.device())(
        gfx90a_fp4_expert_down_reduce_kernel<M, T, N>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(partial.data_ptr()));
  }

  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView topk_weights,
                  const tvm::ffi::TensorView partial,
                  const tvm::ffi::TensorView out) {
    run_partial(xq, x_scale, weight, weight_scale, sorted_ids,
                sorted_expert_ids, num_valid_ids, topk_weights, partial);
    reduce(partial, out);
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4,
          uint32_t kBroadcastScales = 0, uint32_t kAssignments = 32>
struct Gfx90aFp4ExpertDownMfma32Kernel {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView sorted_expert_ids,
                  const tvm::ffi::TensorView num_valid_ids,
                  const tvm::ffi::TensorView topk_weights,
                  const tvm::ffi::TensorView partial,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(kBlocks, kSplit * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_down_mfma32_kernel<
            E, M, T, N, K, kBlocks, kSplit, kBroadcastScales, kAssignments>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
    constexpr uint32_t kThreads = 256;
    LaunchKernel((M * N + kThreads - 1) / kThreads, kThreads, xq.device())(
        gfx90a_fp4_expert_down_reduce_kernel<M, T, N>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(partial.data_ptr()));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t GE,
          uint32_t I, uint32_t K,
          uint32_t kRows, uint32_t kNumWaves, uint32_t kBlocks,
          uint32_t kSlotBegin, uint32_t kSlotEnd>
struct Gfx90aFp4ExpertGateUpKernel {
  static void launch(const tvm::ffi::TensorView x,
                     const tvm::ffi::TensorView weight,
                     const tvm::ffi::TensorView weight_scale,
                     const tvm::ffi::TensorView expert_ids,
                     const tvm::ffi::TensorView out, double limit,
                     const int32_t* expert_mask,
                     const int32_t* live_count,
                     const int8_t* xq = nullptr,
                     const float* x_scale = nullptr) {
    using namespace host;
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, x.device())(
        gfx90a_fp4_expert_gate_up_kernel<
            E, M, T, GE, I, K, kRows, kNumWaves, kSlotBegin, kSlotEnd>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        xq, x_scale,
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()), expert_mask, live_count,
        static_cast<float>(limit));
  }

  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView expert_mask,
                  const tvm::ffi::TensorView live_count,
                  const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(live_count);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, out, limit,
           static_cast<const int32_t*>(expert_mask.data_ptr()),
           static_cast<const int32_t*>(live_count.data_ptr()));
  }

  static void run_prequant(const tvm::ffi::TensorView x,
                           const tvm::ffi::TensorView xq,
                           const tvm::ffi::TensorView x_scale,
                           const tvm::ffi::TensorView weight,
                           const tvm::ffi::TensorView weight_scale,
                           const tvm::ffi::TensorView expert_ids,
                           const tvm::ffi::TensorView expert_mask,
                           const tvm::ffi::TensorView live_count,
                           const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(live_count);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, out, limit,
           static_cast<const int32_t*>(expert_mask.data_ptr()),
           static_cast<const int32_t*>(live_count.data_ptr()),
           static_cast<const int8_t*>(xq.data_ptr()),
           static_cast<const float*>(x_scale.data_ptr()));
  }

  static void run_static(const tvm::ffi::TensorView x,
                         const tvm::ffi::TensorView weight,
                         const tvm::ffi::TensorView weight_scale,
                         const tvm::ffi::TensorView expert_ids,
                         const tvm::ffi::TensorView expert_mask,
                         const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, out, limit,
           static_cast<const int32_t*>(expert_mask.data_ptr()), nullptr);
  }

  static void run_prequant_static(
      const tvm::ffi::TensorView x, const tvm::ffi::TensorView xq,
      const tvm::ffi::TensorView x_scale,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weight_scale,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView expert_mask,
      const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, out, limit,
           static_cast<const int32_t*>(expert_mask.data_ptr()), nullptr,
           static_cast<const int8_t*>(xq.data_ptr()),
           static_cast<const float*>(x_scale.data_ptr()));
  }

  static void run_static_nomask(const tvm::ffi::TensorView x,
                                const tvm::ffi::TensorView weight,
                                const tvm::ffi::TensorView weight_scale,
                                const tvm::ffi::TensorView expert_ids,
                                const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, out, limit, nullptr, nullptr);
  }

  static void run_prequant_static_nomask(
      const tvm::ffi::TensorView x, const tvm::ffi::TensorView xq,
      const tvm::ffi::TensorView x_scale,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weight_scale,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, out, limit, nullptr, nullptr,
           static_cast<const int8_t*>(xq.data_ptr()),
           static_cast<const float*>(x_scale.data_ptr()));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t GE,
          uint32_t N, uint32_t K,
          uint32_t kRows, uint32_t kNumWaves, uint32_t kBlocks,
          uint32_t kSlotBegin, uint32_t kSlotEnd>
struct Gfx90aFp4ExpertDownKernel {
  static void launch(const tvm::ffi::TensorView x,
                     const tvm::ffi::TensorView weight,
                     const tvm::ffi::TensorView weight_scale,
                     const tvm::ffi::TensorView expert_ids,
                     const tvm::ffi::TensorView topk_weights,
                     const tvm::ffi::TensorView out,
                     const int32_t* expert_mask,
                     const int32_t* live_count,
                     const int8_t* xq = nullptr,
                     const float* x_scale = nullptr) {
    using namespace host;
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, x.device())(
        gfx90a_fp4_expert_down_kernel<
            E, M, T, GE, N, K, kRows, kNumWaves, kSlotBegin, kSlotEnd>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        xq, x_scale,
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()), expert_mask,
        static_cast<const float*>(topk_weights.data_ptr()), live_count);
  }

  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView expert_mask,
                  const tvm::ffi::TensorView topk_weights,
                  const tvm::ffi::TensorView live_count,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(live_count);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, topk_weights, out,
           static_cast<const int32_t*>(expert_mask.data_ptr()),
           static_cast<const int32_t*>(live_count.data_ptr()));
  }

  static void run_prequant(const tvm::ffi::TensorView x,
                           const tvm::ffi::TensorView xq,
                           const tvm::ffi::TensorView x_scale,
                           const tvm::ffi::TensorView weight,
                           const tvm::ffi::TensorView weight_scale,
                           const tvm::ffi::TensorView expert_ids,
                           const tvm::ffi::TensorView expert_mask,
                           const tvm::ffi::TensorView topk_weights,
                           const tvm::ffi::TensorView live_count,
                           const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(live_count);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, topk_weights, out,
           static_cast<const int32_t*>(expert_mask.data_ptr()),
           static_cast<const int32_t*>(live_count.data_ptr()),
           static_cast<const int8_t*>(xq.data_ptr()),
           static_cast<const float*>(x_scale.data_ptr()));
  }

  static void run_static(const tvm::ffi::TensorView x,
                         const tvm::ffi::TensorView weight,
                         const tvm::ffi::TensorView weight_scale,
                         const tvm::ffi::TensorView expert_ids,
                         const tvm::ffi::TensorView expert_mask,
                         const tvm::ffi::TensorView topk_weights,
                         const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, topk_weights, out,
           static_cast<const int32_t*>(expert_mask.data_ptr()), nullptr);
  }

  static void run_prequant_static(
      const tvm::ffi::TensorView x, const tvm::ffi::TensorView xq,
      const tvm::ffi::TensorView x_scale,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weight_scale,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView expert_mask,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({GE}).with_dtype<int32_t>().with_device(device).verify(expert_mask);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, topk_weights, out,
           static_cast<const int32_t*>(expert_mask.data_ptr()), nullptr,
           static_cast<const int8_t*>(xq.data_ptr()),
           static_cast<const float*>(x_scale.data_ptr()));
  }

  static void run_static_nomask(const tvm::ffi::TensorView x,
                                const tvm::ffi::TensorView weight,
                                const tvm::ffi::TensorView weight_scale,
                                const tvm::ffi::TensorView expert_ids,
                                const tvm::ffi::TensorView topk_weights,
                                const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, topk_weights, out, nullptr,
           nullptr);
  }

  static void run_prequant_static_nomask(
      const tvm::ffi::TensorView x, const tvm::ffi::TensorView xq,
      const tvm::ffi::TensorView x_scale,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weight_scale,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    launch(x, weight, weight_scale, expert_ids, topk_weights, out, nullptr,
           nullptr, static_cast<const int8_t*>(xq.data_ptr()),
           static_cast<const float*>(x_scale.data_ptr()));
  }
};

}  // namespace sglang
