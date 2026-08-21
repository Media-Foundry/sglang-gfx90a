#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

using namespace device;

// Decode-only CDNA2 shared-expert gate/up projection. The checkpoint stores
// [gate; up] as adjacent rows. A wave computes both dot products while x is
// staged once, then writes the bounded SwiGLU result directly.
constexpr uint32_t kGfx90aGatedVec = 16 / sizeof(bf16_t);
constexpr uint32_t kGfx90aGatedWave = 64;

__device__ __forceinline__ float gfx90a_gated_dot8(const float4 wv,
                                                    const float4 xv) {
  const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
    const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
    acc = fmaf(w0, x0, acc);
    acc = fmaf(w1, x1, acc);
  }
  return acc;
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aGatedWave)
    gfx90a_bf16_gated_gemv_kernel(bf16_t* __restrict__ out,
                                  const bf16_t* __restrict__ x,
                                  const bf16_t* __restrict__ weight,
                                  float limit) {
  static_assert(N % 2 == 0, "gate/up row count must be even");
  constexpr uint32_t kOut = N / 2;
  __shared__ bf16_t sx[K];

  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kGfx90aGatedVec; k < K;
       k += kNumWaves * kGfx90aGatedWave * kGfx90aGatedVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  }
  __syncthreads();

  const uint32_t wave = tid / kGfx90aGatedWave;
  const uint32_t lane = tid % kGfx90aGatedWave;
  const uint32_t row0 = (blockIdx.x * kNumWaves + wave) * kRows;
  if (row0 >= kOut) return;

  float gate[kRows];
  float up[kRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
    gate[r] = 0.0f;
    up[r] = 0.0f;
  }

  constexpr uint32_t kStep =
      kGfx90aGatedWave * kGfx90aGatedVec * kUnroll;
  for (uint32_t k = lane * kGfx90aGatedVec * kUnroll; k < K; k += kStep) {
    float4 xv[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[u] = *reinterpret_cast<const float4*>(sx + k +
                                               u * kGfx90aGatedVec);
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < kOut) {
        const bf16_t* wg = weight + static_cast<size_t>(row0 + r) * K + k;
        const bf16_t* wu =
            weight + static_cast<size_t>(kOut + row0 + r) * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          gate[r] += gfx90a_gated_dot8(
              *reinterpret_cast<const float4*>(wg +
                                                u * kGfx90aGatedVec),
              xv[u]);
          up[r] += gfx90a_gated_dot8(
              *reinterpret_cast<const float4*>(wu +
                                                u * kGfx90aGatedVec),
              xv[u]);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      gate[r] += __shfl_down(gate[r], offset, kGfx90aGatedWave);
      up[r] += __shfl_down(up[r], offset, kGfx90aGatedWave);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < kOut) {
        const float g = fminf(gate[r], limit);
        const float u = fmaxf(fminf(up[r], limit), -limit);
        const float silu = g / (1.0f + expf(-g));
        out[row0 + r] = cast<bf16_t>(silu * u);
      }
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
struct Gfx90aBf16GatedGemvKernel {
  static_assert(K % (kGfx90aGatedWave * kGfx90aGatedVec * kUnroll) == 0,
                "K must cover complete wave64 vector strides");

  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out, float limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({1, N / 2})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);

    constexpr uint32_t kRowsPerBlock = kRows * kNumWaves;
    constexpr uint32_t kBlocks =
        (N / 2 + kRowsPerBlock - 1) / kRowsPerBlock;
    LaunchKernel(kBlocks, kNumWaves * kGfx90aGatedWave, device.unwrap())(
        gfx90a_bf16_gated_gemv_kernel<N, K, kRows, kUnroll, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()), limit);
  }
};

}  // namespace sglang
