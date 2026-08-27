#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kAttnQK = 1024;
constexpr uint32_t kAttnQN = 8192;
constexpr uint32_t kAttnQWave = 64;
constexpr uint32_t kAttnQWaves = 4;
constexpr uint32_t kAttnQRows = 2;
constexpr uint32_t kAttnQVec = 8;

__device__ __forceinline__ float attn_q_dot8(const float4 wv, const float4 xv) {
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

__global__ void __launch_bounds__(kAttnQWaves * kAttnQWave)
    gfx90a_attn_q_dual_gemv_kernel(
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ w0,
        const bf16_t* __restrict__ w1,
        bf16_t* __restrict__ out0,
        bf16_t* __restrict__ out1) {
  __shared__ bf16_t sx[kAttnQK];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kAttnQVec; k < kAttnQK;
       k += blockDim.x * kAttnQVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  }
  __syncthreads();

  const uint32_t wave = tid / kAttnQWave;
  const uint32_t lane = tid % kAttnQWave;
  constexpr uint32_t kRowsPerBlock = kAttnQRows * kAttnQWaves;
  const uint32_t global_row0 = blockIdx.x * kRowsPerBlock + wave * kAttnQRows;
  const bool second = global_row0 >= kAttnQN;
  const uint32_t local_row0 = second ? global_row0 - kAttnQN : global_row0;
  const bf16_t* weight = second ? w1 : w0;
  bf16_t* output = second ? out1 : out0;
  if (local_row0 >= kAttnQN) return;

  float acc[kAttnQRows] = {};
  constexpr uint32_t kUnroll = 1;
  constexpr uint32_t kStep = kAttnQWave * kAttnQVec * kUnroll;
  for (uint32_t k = lane * kAttnQVec * kUnroll; k < kAttnQK; k += kStep) {
    const float4 xv = *reinterpret_cast<const float4*>(sx + k);
#pragma unroll
    for (uint32_t r = 0; r < kAttnQRows; ++r) {
      acc[r] += attn_q_dot8(
          *reinterpret_cast<const float4*>(
              weight + static_cast<size_t>(local_row0 + r) * kAttnQK + k),
          xv);
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < kAttnQRows; ++r) {
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset, kAttnQWave);
    }
    if (lane == 0) output[local_row0 + r] = cast<bf16_t>(acc[r]);
  }
}

struct Gfx90aAttnQDualGemvKernel {
  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w0,
      const tvm::ffi::TensorView w1,
      const tvm::ffi::TensorView out0,
      const tvm::ffi::TensorView out1) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kAttnQK}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({kAttnQN, kAttnQK}).with_dtype<bf16_t>().with_device(device).verify(w0);
    TensorMatcher({kAttnQN, kAttnQK}).with_dtype<bf16_t>().with_device(device).verify(w1);
    TensorMatcher({1, kAttnQN}).with_dtype<bf16_t>().with_device(device).verify(out0);
    TensorMatcher({1, kAttnQN}).with_dtype<bf16_t>().with_device(device).verify(out1);
    constexpr uint32_t kBlocks =
        (2 * kAttnQN) / (kAttnQRows * kAttnQWaves);
    LaunchKernel(kBlocks, kAttnQWaves * kAttnQWave, device.unwrap())(
        gfx90a_attn_q_dual_gemv_kernel,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w0.data_ptr()),
        static_cast<const bf16_t*>(w1.data_ptr()),
        static_cast<bf16_t*>(out0.data_ptr()),
        static_cast<bf16_t*>(out1.data_ptr()));
  }
};

}  // namespace sglang
