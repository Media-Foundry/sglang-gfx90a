#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/warp.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

namespace sglang {

using namespace device;

constexpr uint32_t kQwenHcWave = 64;
constexpr uint32_t kQwenHcVec = 8;

__device__ __forceinline__ float qwen_hc_dot8(const float4 wv, const float4 xv) {
  const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
    const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
    acc = fmaf(w0, x0, acc);
    acc = fmaf(w1, x1, acc);
  }
  return acc;
}

// Qwen4 HC down projection: [1,10240] x [320,10240]^T -> FP32 [1,320].
// Four wave64s per block, two output rows per wave.  Unlike the persistent
// Triton path this needs neither atomics nor a software grid barrier.
template <uint32_t kRows>
__global__ __launch_bounds__(256) void qwen_hc_down_kernel(
    const bf16_t* __restrict__ x, const bf16_t* __restrict__ weight,
    float* __restrict__ out) {
  constexpr uint32_t K = 10240;
  constexpr uint32_t N = 320;
  __shared__ bf16_t sx[K];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kQwenHcVec; k < K;
       k += blockDim.x * kQwenHcVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  }
  __syncthreads();
  const uint32_t wave = tid / kQwenHcWave;
  const uint32_t lane = tid % kQwenHcWave;
  const uint32_t row0 = (blockIdx.x * 4 + wave) * kRows;
  if (row0 >= N) return;
  float acc[kRows] = {};
  for (uint32_t k = lane * kQwenHcVec; k < K;
       k += kQwenHcWave * kQwenHcVec) {
    const float4 xv = *reinterpret_cast<const float4*>(sx + k);
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const float4 wv = *reinterpret_cast<const float4*>(
          weight + static_cast<size_t>(row0 + r) * K + k);
      acc[r] += qwen_hc_dot8(wv, xv);
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
    for (uint32_t offset = 32; offset; offset >>= 1)
      acc[r] += __shfl_down(acc[r], offset, 64);
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r)
      if (row0 + r < N) out[row0 + r] = acc[r];
  }
}

// Experimental producer fusion: the first eight down-projection CTAs also
// replay the exact 8-way/32-thread HC inject-weight dot decomposition.  The
// gate depends only on x, so its partials can be carried across the block and
// consumed after the TP all-reduce without a separate gate launch.
template <uint32_t kRows>
__global__ __launch_bounds__(256) void qwen_hc_down_gate_kernel(
    const bf16_t* __restrict__ x, const bf16_t* __restrict__ weight,
    const bf16_t* __restrict__ inject_weight, float* __restrict__ out,
    float* __restrict__ gate_partials) {
  constexpr uint32_t K = 10240;
  constexpr uint32_t N = 320;
  constexpr uint32_t HC = 4;
  __shared__ bf16_t sx[K];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kQwenHcVec; k < K;
       k += blockDim.x * kQwenHcVec)
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  __syncthreads();

  const uint32_t wave = tid / kQwenHcWave;
  const uint32_t lane = tid % kQwenHcWave;
  const uint32_t row0 = (blockIdx.x * 4 + wave) * kRows;
  float acc[kRows] = {};
  for (uint32_t k = lane * kQwenHcVec; k < K;
       k += kQwenHcWave * kQwenHcVec) {
    const float4 xv = *reinterpret_cast<const float4*>(sx + k);
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r)
      if (row0 + r < N)
        acc[r] += qwen_hc_dot8(*reinterpret_cast<const float4*>(
            weight + static_cast<size_t>(row0 + r) * K + k), xv);
  }
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
    for (uint32_t offset = 32; offset; offset >>= 1)
      acc[r] += __shfl_down(acc[r], offset, 64);
  if (lane == 0)
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r)
      if (row0 + r < N) out[row0 + r] = acc[r];

  if (blockIdx.x < 8 && tid < HC * 32) {
    const uint32_t c = tid / 32;
    const uint32_t sublane = tid % 32;
    const uint32_t ref_tid = blockIdx.x * 32 + sublane;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 5; ++j) {
      const uint32_t vec = ref_tid + j * 256;
      const bf16x2_t* nv = reinterpret_cast<const bf16x2_t*>(sx + vec * 8);
      const bf16x2_t* wv = reinterpret_cast<const bf16x2_t*>(
          inject_weight + static_cast<size_t>(c) * K + vec * 8);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(nv[i]);
        const auto [wx, wy] = cast<fp32x2_t>(wv[i]);
        sum += nx * wx + ny * wy;
      }
    }
    sum = warp::reduce_sum<32>(sum);
    if (sublane == 0) gate_partials[blockIdx.x * HC + c] = sum;
  }
}

// HC up projection plus sigmoid gate and four-stream weighted mean.
// A wave computes two hidden columns and keeps all four gates in registers.
template <uint32_t kRows>
__global__ __launch_bounds__(256) void qwen_hc_up_mix_kernel(
    const bf16_t* __restrict__ x, const float* __restrict__ down,
    const bf16_t* __restrict__ weight, bf16_t* __restrict__ out) {
  constexpr uint32_t HC = 4;
  constexpr uint32_t HS = 2560;
  constexpr uint32_t R = 320;
  __shared__ bf16_t st[R];
  const uint32_t tid = threadIdx.x;
  for (uint32_t r = tid; r < R; r += blockDim.x) {
    const float a = down[r] * 0.25f;
    st[r] = cast<bf16_t>(a / (1.0f + expf(-a)));
  }
  __syncthreads();
  const uint32_t wave = tid / kQwenHcWave;
  const uint32_t lane = tid % kQwenHcWave;
  const uint32_t j0 = (blockIdx.x * 4 + wave) * kRows;
  if (j0 >= HS) return;
  float acc[kRows][HC] = {};
  for (uint32_t r = lane; r < R; r += kQwenHcWave) {
    const float tv = cast<float>(st[r]);
#pragma unroll
    for (uint32_t jr = 0; jr < kRows; ++jr) {
      const uint32_t j = j0 + jr;
      if (j < HS) {
#pragma unroll
        for (uint32_t g = 0; g < HC; ++g) {
          acc[jr][g] = fmaf(
              cast<float>(weight[(static_cast<size_t>(g) * HS + j) * R + r]),
              tv, acc[jr][g]);
        }
      }
    }
  }
#pragma unroll
  for (uint32_t jr = 0; jr < kRows; ++jr)
#pragma unroll
    for (uint32_t g = 0; g < HC; ++g)
#pragma unroll
      for (uint32_t offset = 32; offset; offset >>= 1)
        acc[jr][g] += __shfl_down(acc[jr][g], offset, 64);
  if (lane == 0) {
#pragma unroll
    for (uint32_t jr = 0; jr < kRows; ++jr) {
      const uint32_t j = j0 + jr;
      if (j < HS) {
        float value = 0.0f;
#pragma unroll
        for (uint32_t g = 0; g < HC; ++g) {
          const float gate = 1.0f / (1.0f + expf(-acc[jr][g]));
          value += gate * cast<float>(x[g * HS + j]);
        }
        out[j] = cast<bf16_t>(value * 0.25f);
      }
    }
  }
}

template <uint32_t kDownRows, uint32_t kUpRows>
struct Gfx90aQwenHcMix {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView w_down,
                  const tvm::ffi::TensorView w_up,
                  const tvm::ffi::TensorView workspace,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({1, 10240}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({320, 10240}).with_dtype<bf16_t>().with_device(device).verify(w_down);
    TensorMatcher({10240, 320}).with_dtype<bf16_t>().with_device(device).verify(w_up);
    TensorMatcher({1, 320}).with_dtype<float>().with_device(device).verify(workspace);
    TensorMatcher({1, 2560}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel((320 + 4 * kDownRows - 1) / (4 * kDownRows), 256, x.device())(
        qwen_hc_down_kernel<kDownRows>, static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w_down.data_ptr()),
        static_cast<float*>(workspace.data_ptr()));
    LaunchKernel((2560 + 4 * kUpRows - 1) / (4 * kUpRows), 256, x.device())(
        qwen_hc_up_mix_kernel<kUpRows>, static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const float*>(workspace.data_ptr()),
        static_cast<const bf16_t*>(w_up.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
  }

  static void run_with_gate(const tvm::ffi::TensorView x,
                            const tvm::ffi::TensorView w_down,
                            const tvm::ffi::TensorView w_up,
                            const tvm::ffi::TensorView inject_weight,
                            const tvm::ffi::TensorView workspace,
                            const tvm::ffi::TensorView gate_partials,
                            const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({1, 10240}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({320, 10240}).with_dtype<bf16_t>().with_device(device).verify(w_down);
    TensorMatcher({10240, 320}).with_dtype<bf16_t>().with_device(device).verify(w_up);
    TensorMatcher({4, 10240}).with_dtype<bf16_t>().with_device(device).verify(inject_weight);
    TensorMatcher({1, 320}).with_dtype<float>().with_device(device).verify(workspace);
    TensorMatcher({1, 8, 4}).with_dtype<float>().with_device(device).verify(gate_partials);
    TensorMatcher({1, 2560}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel((320 + 4 * kDownRows - 1) / (4 * kDownRows), 256, x.device())(
        qwen_hc_down_gate_kernel<kDownRows>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w_down.data_ptr()),
        static_cast<const bf16_t*>(inject_weight.data_ptr()),
        static_cast<float*>(workspace.data_ptr()),
        static_cast<float*>(gate_partials.data_ptr()));
    LaunchKernel((2560 + 4 * kUpRows - 1) / (4 * kUpRows), 256, x.device())(
        qwen_hc_up_mix_kernel<kUpRows>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const float*>(workspace.data_ptr()),
        static_cast<const bf16_t*>(w_up.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
  }
};

}  // namespace sglang
