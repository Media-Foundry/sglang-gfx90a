#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kFp4ExpertWave = 64;

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
          uint32_t kRows, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_up_kernel(
        bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ expert_mask,
        const int32_t* __restrict__ live_count, float limit) {
  constexpr uint32_t kTilesPerAssignment = (I + kRows - 1) / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t live = min(static_cast<uint32_t>(max(live_count[0], 0)), M);

  for (uint32_t task = global_wave;
       task < live * T * kTilesPerAssignment;
       task += total_waves) {
    const uint32_t assignment = task / kTilesPerAssignment;
    const uint32_t token = assignment / T;
    const uint32_t slot = assignment % T;
    const uint32_t row0 = (task % kTilesPerAssignment) * kRows;
    const int32_t expert_id = expert_ids[static_cast<size_t>(token) * T + slot];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(GE) ||
        expert_mask[expert_id] == 0) {
      continue;
    }
    const uint32_t global_expert = static_cast<uint32_t>(expert_id);
    const uint32_t expert = global_expert % E;
    float gate_acc[kRows] = {};
    float up_acc[kRows] = {};

    for (uint32_t group = lane; group < K / 32; group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
      float xv[32];
#pragma unroll
      for (uint32_t j = 0; j < 32; ++j) {
        xv[j] = cast<float>(x[static_cast<size_t>(token) * K + k0 + j]);
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
#pragma unroll
          for (uint32_t j = 0; j < 32; ++j) {
            const uint8_t gw = weight[gate_base + j / 2];
            const uint8_t uw = weight[up_base + j / 2];
            const uint32_t shift = (j & 1) * 4;
            gate_acc[r] = fmaf(xv[j],
                               gfx90a_fp4_value((gw >> shift) & 15) * gate_scale,
                               gate_acc[r]);
            up_acc[r] = fmaf(xv[j],
                             gfx90a_fp4_value((uw >> shift) & 15) * up_scale,
                             up_acc[r]);
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
          uint32_t kRows, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_down_kernel(
        bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ expert_mask,
        const float* __restrict__ topk_weights,
        const int32_t* __restrict__ live_count) {
  constexpr uint32_t kTilesPerRow = (N + kRows - 1) / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t global_wave = blockIdx.x * kNumWaves + wave;
  const uint32_t total_waves = gridDim.x * kNumWaves;
  const uint32_t live = min(static_cast<uint32_t>(max(live_count[0], 0)), M);

  for (uint32_t task = global_wave; task < live * kTilesPerRow;
       task += total_waves) {
    const uint32_t token = task / kTilesPerRow;
    const uint32_t row0 = (task % kTilesPerRow) * kRows;
    float acc[kRows] = {};

    for (uint32_t slot = 0; slot < T; ++slot) {
      const int32_t expert_id =
          expert_ids[static_cast<size_t>(token) * T + slot];
      if (expert_id < 0 || expert_id >= static_cast<int32_t>(GE) ||
          expert_mask[expert_id] == 0) {
        continue;
      }
      const uint32_t expert = static_cast<uint32_t>(expert_id) % E;
      const float routed_weight =
          topk_weights[static_cast<size_t>(token) * T + slot];
      for (uint32_t group = lane; group < K / 32;
           group += kFp4ExpertWave) {
        const uint32_t k0 = group * 32;
        float xv[32];
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j) {
          xv[j] = cast<float>(
              x[(static_cast<size_t>(token) * T + slot) * K + k0 + j]);
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
#pragma unroll
            for (uint32_t j = 0; j < 32; ++j) {
              const uint8_t packed = weight[weight_base + j / 2];
              const uint32_t shift = (j & 1) * 4;
              acc[r] = fmaf(
                  xv[j],
                  gfx90a_fp4_value((packed >> shift) & 15) * scale *
                      routed_weight,
                  acc[r]);
            }
          }
        }
      }
    }

#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1) {
        acc[r] += __shfl_down(acc[r], offset, kFp4ExpertWave);
      }
      if (lane == 0 && row0 + r < N) {
        out[static_cast<size_t>(token) * N + row0 + r] =
            cast<bf16_t>(acc[r]);
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t GE,
          uint32_t I, uint32_t K,
          uint32_t kRows, uint32_t kNumWaves>
struct Gfx90aFp4ExpertGateUpKernel {
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
    constexpr uint32_t kBlocks = 256;
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, device.unwrap())(
        gfx90a_fp4_expert_gate_up_kernel<E, M, T, GE, I, K, kRows, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<const int32_t*>(expert_mask.data_ptr()),
        static_cast<const int32_t*>(live_count.data_ptr()),
        static_cast<float>(limit));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t GE,
          uint32_t N, uint32_t K,
          uint32_t kRows, uint32_t kNumWaves>
struct Gfx90aFp4ExpertDownKernel {
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
    constexpr uint32_t kBlocks = 256;
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, device.unwrap())(
        gfx90a_fp4_expert_down_kernel<E, M, T, GE, N, K, kRows, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<const int32_t*>(expert_mask.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<const int32_t*>(live_count.data_ptr()));
  }
};

}  // namespace sglang
