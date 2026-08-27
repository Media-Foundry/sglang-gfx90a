#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kGfx90aInt8M32Wave = 64;
constexpr uint32_t kGfx90aInt8M32Vec = 16;

// This is deliberately a two-kernel research prototype.  Keeping the rowwise
// activation quantizer separate makes its cost visible and avoids the M*K LDS
// footprint that a direct extension of the M=1 GEMV would require.
template <uint32_t M, uint32_t K, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aInt8M32Wave)
    gfx90a_int8_m32_quant_kernel(int8_t* __restrict__ qx,
                                float* __restrict__ x_scale,
                                const bf16_t* __restrict__ x) {
  __shared__ float wave_max[kNumWaves];
  __shared__ float row_inv_scale;

  const uint32_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kGfx90aInt8M32Wave;
  const uint32_t lane = tid % kGfx90aInt8M32Wave;
  if (row >= M) return;

  const bf16_t* xr = x + static_cast<size_t>(row) * K;
  int8_t* qr = qx + static_cast<size_t>(row) * K;
  float local_max = 0.0f;
  for (uint32_t k = tid; k < K;
       k += kNumWaves * kGfx90aInt8M32Wave) {
    local_max = fmaxf(local_max, fabsf(cast<float>(xr[k])));
  }
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    local_max =
        fmaxf(local_max,
              __shfl_down(local_max, offset, kGfx90aInt8M32Wave));
  }
  if (lane == 0) wave_max[wave] = local_max;
  __syncthreads();

  if (wave == 0) {
    float block_max = lane < kNumWaves ? wave_max[lane] : 0.0f;
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      block_max =
          fmaxf(block_max,
                __shfl_down(block_max, offset, kGfx90aInt8M32Wave));
    }
    if (lane == 0) {
      const float scale = fmaxf(block_max / 127.0f, 1.0e-12f);
      x_scale[row] = scale;
      row_inv_scale = 1.0f / scale;
    }
  }
  __syncthreads();

  for (uint32_t k = tid; k < K;
       k += kNumWaves * kGfx90aInt8M32Wave) {
    const float q = nearbyintf(cast<float>(xr[k]) * row_inv_scale);
    qr[k] = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, q)));
  }
}

// Each wave owns kOutRows output channels and an A-tile of token rows.  The
// packed INT8 weight is loaded once, then reused for all kATile activations.
// This is the key distinction from launching M independent GEMVs.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t kATile,
          uint32_t kOutRows, uint32_t kUnroll, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aInt8M32Wave)
    gfx90a_int8_weight_gemm_m32_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ qx,
        const float* __restrict__ x_scale,
        const int8_t* __restrict__ weight,
        const float* __restrict__ weight_scale) {
  const uint32_t wave = threadIdx.x / kGfx90aInt8M32Wave;
  const uint32_t lane = threadIdx.x % kGfx90aInt8M32Wave;
  const uint32_t m0 = blockIdx.y * kATile;
  const uint32_t n0 = (blockIdx.x * kNumWaves + wave) * kOutRows;
  if (m0 >= M || n0 >= N) return;

  int32_t acc[kATile][kOutRows] = {};
  constexpr uint32_t kStep =
      kGfx90aInt8M32Wave * kGfx90aInt8M32Vec * kUnroll;
  for (uint32_t k = lane * kGfx90aInt8M32Vec * kUnroll; k < K;
       k += kStep) {
    int32_t xv[kATile][kUnroll][4];
#pragma unroll
    for (uint32_t a = 0; a < kATile; ++a) {
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const float4 packed = *reinterpret_cast<const float4*>(
            qx + static_cast<size_t>(m0 + a) * K + k +
            u * kGfx90aInt8M32Vec);
        *reinterpret_cast<float4*>(xv[a][u]) = packed;
      }
    }
#pragma unroll
    for (uint32_t r = 0; r < kOutRows; ++r) {
      if (n0 + r < N) {
        const int8_t* wr =
            weight + static_cast<size_t>(n0 + r) * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          const float4 packed = *reinterpret_cast<const float4*>(
              wr + u * kGfx90aInt8M32Vec);
          int32_t wv[4];
          *reinterpret_cast<float4*>(wv) = packed;
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (uint32_t a = 0; a < kATile; ++a) {
              acc[a][r] = __builtin_amdgcn_sdot4(
                  xv[a][u][i], wv[i], acc[a][r], false);
            }
          }
        }
      }
    }
  }

#pragma unroll
  for (uint32_t a = 0; a < kATile; ++a) {
#pragma unroll
    for (uint32_t r = 0; r < kOutRows; ++r) {
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1) {
        acc[a][r] +=
            __shfl_down(acc[a][r], offset, kGfx90aInt8M32Wave);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t a = 0; a < kATile; ++a) {
#pragma unroll
      for (uint32_t r = 0; r < kOutRows; ++r) {
        if (m0 + a < M && n0 + r < N) {
          out[static_cast<size_t>(m0 + a) * N + n0 + r] =
              cast<bf16_t>(static_cast<float>(acc[a][r]) * x_scale[m0 + a] *
                           weight_scale[n0 + r]);
        }
      }
    }
  }
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t kATile,
          uint32_t kOutRows, uint32_t kUnroll, uint32_t kNumWaves>
struct Gfx90aInt8WeightGemmM32Kernel {
  static_assert(M % kATile == 0, "M must cover complete activation tiles");
  static_assert(K % (kGfx90aInt8M32Wave * kGfx90aInt8M32Vec * kUnroll) ==
                    0,
                "K must cover complete wave64 vector strides");

  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView qx,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<int8_t>().with_device(device).verify(weight);
    TensorMatcher({N})
        .with_dtype<float>()
        .with_device(device)
        .verify(weight_scale);
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(qx);
    TensorMatcher({M})
        .with_dtype<float>()
        .with_device(device)
        .verify(x_scale);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    LaunchKernel(M, kNumWaves * kGfx90aInt8M32Wave, device.unwrap())(
        gfx90a_int8_m32_quant_kernel<M, K, kNumWaves>,
        static_cast<int8_t*>(qx.data_ptr()),
        static_cast<float*>(x_scale.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()));

    constexpr uint32_t kRowsPerBlock = kOutRows * kNumWaves;
    constexpr uint32_t kBlocksX = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    constexpr uint32_t kBlocksY = (M + kATile - 1) / kATile;
    LaunchKernel(dim3(kBlocksX, kBlocksY),
                 kNumWaves * kGfx90aInt8M32Wave, device.unwrap())(
        gfx90a_int8_weight_gemm_m32_kernel<M, N, K, kATile, kOutRows,
                                           kUnroll, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const int8_t*>(qx.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const int8_t*>(weight.data_ptr()),
        static_cast<const float*>(weight_scale.data_ptr()));
  }
};

}  // namespace sglang
