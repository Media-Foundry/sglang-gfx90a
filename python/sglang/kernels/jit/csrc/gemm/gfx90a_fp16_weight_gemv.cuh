#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kGfx90aFp16GemvVec = 8;
constexpr uint32_t kGfx90aFp16GemvWave = 64;

// CDNA2 v_dot2_f32_f16 performs two FP16 multiplies with FP32 accumulation.
// Activations stay BF16 at the public boundary and are converted in registers;
// only the cached projection weight uses FP16 storage.
__device__ __forceinline__ float gfx90a_fp16_dot8(const float4 wv,
                                                  const float4 xv) {
  const fp16x2_t* w2 = reinterpret_cast<const fp16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const fp16x2_t xh = cast<fp16x2_t>(cast<fp32x2_t>(x2[i]));
    acc = amd_mixed_dot(w2[i], xh, acc, false);
  }
  return acc;
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aFp16GemvWave)
    gfx90a_fp16_weight_gemv_kernel(bf16_t* __restrict__ out,
                                   const bf16_t* __restrict__ x,
                                   const fp16_t* __restrict__ weight) {
  __shared__ bf16_t sx[K];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kGfx90aFp16GemvVec; k < K;
       k += kNumWaves * kGfx90aFp16GemvWave * kGfx90aFp16GemvVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  }
  __syncthreads();

  const uint32_t wave = tid / kGfx90aFp16GemvWave;
  const uint32_t lane = tid % kGfx90aFp16GemvWave;
  const uint32_t row0 = (blockIdx.x * kNumWaves + wave) * kRows;
  if (row0 >= N) return;

  float acc[kRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) acc[r] = 0.0f;

  constexpr uint32_t kStep =
      kGfx90aFp16GemvWave * kGfx90aFp16GemvVec * kUnroll;
  for (uint32_t k = lane * kGfx90aFp16GemvVec * kUnroll; k < K; k += kStep) {
    float4 xv[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[u] = *reinterpret_cast<const float4*>(sx + k +
                                               u * kGfx90aFp16GemvVec);
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        const fp16_t* wr = weight + static_cast<size_t>(row0 + r) * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          acc[r] += gfx90a_fp16_dot8(
              *reinterpret_cast<const float4*>(wr +
                                                u * kGfx90aFp16GemvVec),
              xv[u]);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset, kGfx90aFp16GemvWave);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) out[row0 + r] = cast<bf16_t>(acc[r]);
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
struct Gfx90aFp16WeightGemvKernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<fp16_t>().with_device(device).verify(weight);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    constexpr uint32_t kRowsPerBlock = kRows * kNumWaves;
    constexpr uint32_t kBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    LaunchKernel(kBlocks, kNumWaves * kGfx90aFp16GemvWave, device.unwrap())(
        gfx90a_fp16_weight_gemv_kernel<N, K, kRows, kUnroll, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const fp16_t*>(weight.data_ptr()));
  }
};

}  // namespace sglang
