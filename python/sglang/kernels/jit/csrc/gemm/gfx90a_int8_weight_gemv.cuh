#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kGfx90aInt8GemvWave = 64;
constexpr uint32_t kGfx90aInt8GemvVec = 16;

// Single-token CDNA2 GEMV with a per-row symmetric INT8 weight cache. The
// public activation remains BF16. Each workgroup quantizes that small vector
// into LDS and then uses V_DOT4_I32_I8 for the large weight scan.
template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aInt8GemvWave)
    gfx90a_int8_weight_gemv_kernel(
        bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
        const int8_t* __restrict__ weight,
        const float* __restrict__ weight_scale) {
  __shared__ int8_t sx[K];
  __shared__ float wave_max[kNumWaves];
  __shared__ float inv_scale;
  __shared__ float x_scale;

  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kGfx90aInt8GemvWave;
  const uint32_t lane = tid % kGfx90aInt8GemvWave;

  float local_max = 0.0f;
  for (uint32_t k = tid; k < K;
       k += kNumWaves * kGfx90aInt8GemvWave) {
    local_max = fmaxf(local_max, fabsf(cast<float>(x[k])));
  }
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    local_max =
        fmaxf(local_max,
              __shfl_down(local_max, offset, kGfx90aInt8GemvWave));
  }
  if (lane == 0) wave_max[wave] = local_max;
  __syncthreads();

  if (wave == 0) {
    float block_max = lane < kNumWaves ? wave_max[lane] : 0.0f;
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      block_max =
          fmaxf(block_max,
                __shfl_down(block_max, offset, kGfx90aInt8GemvWave));
    }
    if (lane == 0) {
      const float scale = fmaxf(block_max / 127.0f, 1.0e-12f);
      x_scale = scale;
      inv_scale = 1.0f / scale;
    }
  }
  __syncthreads();

  for (uint32_t k = tid; k < K;
       k += kNumWaves * kGfx90aInt8GemvWave) {
    const float q = nearbyintf(cast<float>(x[k]) * inv_scale);
    sx[k] = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, q)));
  }
  __syncthreads();

  const uint32_t row0 = (blockIdx.x * kNumWaves + wave) * kRows;
  if (row0 >= N) return;

  int32_t acc[kRows] = {};
  constexpr uint32_t kStep =
      kGfx90aInt8GemvWave * kGfx90aInt8GemvVec * kUnroll;
  for (uint32_t k = lane * kGfx90aInt8GemvVec * kUnroll; k < K;
       k += kStep) {
    int32_t xv[kUnroll][4];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const float4 packed = *reinterpret_cast<const float4*>(
          sx + k + u * kGfx90aInt8GemvVec);
      *reinterpret_cast<float4*>(xv[u]) = packed;
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        const int8_t* wr = weight + static_cast<size_t>(row0 + r) * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          const float4 packed = *reinterpret_cast<const float4*>(
              wr + u * kGfx90aInt8GemvVec);
          int32_t wv[4];
          *reinterpret_cast<float4*>(wv) = packed;
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) {
            acc[r] =
                __builtin_amdgcn_sdot4(xv[u][i], wv[i], acc[r], false);
          }
        }
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset, kGfx90aInt8GemvWave);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        out[row0 + r] = cast<bf16_t>(static_cast<float>(acc[r]) * x_scale *
                                     weight_scale[row0 + r]);
      }
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
struct Gfx90aInt8WeightGemvKernel {
  static_assert(K % (kGfx90aInt8GemvWave * kGfx90aInt8GemvVec * kUnroll) ==
                    0,
                "K must cover complete wave64 vector strides");

  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<int8_t>().with_device(device).verify(weight);
    TensorMatcher({N})
        .with_dtype<float>()
        .with_device(device)
        .verify(weight_scale);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    constexpr uint32_t kRowsPerBlock = kRows * kNumWaves;
    constexpr uint32_t kBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    LaunchKernel(kBlocks, kNumWaves * kGfx90aInt8GemvWave, device.unwrap())(
        gfx90a_int8_weight_gemv_kernel<N, K, kRows, kUnroll, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const int8_t*>(weight.data_ptr()),
        static_cast<const float*>(weight_scale.data_ptr()));
  }
};

}  // namespace sglang
