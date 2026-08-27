#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kAttnPrepK = 4096;
constexpr uint32_t kAttnPrepN0 = 1536;
constexpr uint32_t kAttnPrepN1 = 2048;
constexpr uint32_t kAttnPrepN2 = 512;
constexpr uint32_t kAttnPrepN3 = 64;
constexpr uint32_t kAttnPrepWave = 64;
constexpr uint32_t kAttnPrepWaves = 8;
constexpr uint32_t kAttnPrepVec = 8;

__device__ __forceinline__ float attn_prep_dot8(
    const float4 wv, const float4 xv) {
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

__global__ void __launch_bounds__(kAttnPrepWaves * kAttnPrepWave)
    gfx90a_attn_prepare_gemv_kernel(
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ w0,
        const bf16_t* __restrict__ w1,
        const bf16_t* __restrict__ w2,
        const bf16_t* __restrict__ w3,
        bf16_t* __restrict__ out0,
        float* __restrict__ out1,
        float* __restrict__ out2,
        bf16_t* __restrict__ out3) {
  __shared__ bf16_t sx[kAttnPrepK];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kAttnPrepVec; k < kAttnPrepK;
       k += blockDim.x * kAttnPrepVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  }
  __syncthreads();

  const uint32_t lane = tid % kAttnPrepWave;
  const uint32_t global_row = blockIdx.x * kAttnPrepWaves + tid / kAttnPrepWave;
  constexpr uint32_t kTotalN =
      kAttnPrepN0 + kAttnPrepN1 + kAttnPrepN2 + kAttnPrepN3;
  if (global_row >= kTotalN) return;

  const bf16_t* weight;
  uint32_t local_row;
  if (global_row < kAttnPrepN0) {
    weight = w0;
    local_row = global_row;
  } else if (global_row < kAttnPrepN0 + kAttnPrepN1) {
    weight = w1;
    local_row = global_row - kAttnPrepN0;
  } else if (global_row < kAttnPrepN0 + kAttnPrepN1 + kAttnPrepN2) {
    weight = w2;
    local_row = global_row - kAttnPrepN0 - kAttnPrepN1;
  } else {
    weight = w3;
    local_row = global_row - kAttnPrepN0 - kAttnPrepN1 - kAttnPrepN2;
  }

  float acc = 0.0f;
  constexpr uint32_t kUnroll = 2;
  constexpr uint32_t kStep = kAttnPrepWave * kAttnPrepVec * kUnroll;
  for (uint32_t k = lane * kAttnPrepVec * kUnroll; k < kAttnPrepK;
       k += kStep) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t ku = k + u * kAttnPrepVec;
      acc += attn_prep_dot8(
          *reinterpret_cast<const float4*>(
              weight + static_cast<size_t>(local_row) * kAttnPrepK + ku),
          *reinterpret_cast<const float4*>(sx + ku));
    }
  }
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset, kAttnPrepWave);
  }
  if (lane != 0) return;

  const bf16_t rounded = cast<bf16_t>(acc);
  if (global_row < kAttnPrepN0) {
    out0[local_row] = rounded;
  } else if (global_row < kAttnPrepN0 + kAttnPrepN1) {
    out1[local_row] = cast<float>(rounded);
  } else if (global_row < kAttnPrepN0 + kAttnPrepN1 + kAttnPrepN2) {
    out2[local_row] = cast<float>(rounded);
  } else {
    out3[local_row] = rounded;
  }
}

struct Gfx90aAttnPrepareGemvKernel {
  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w0,
      const tvm::ffi::TensorView w1,
      const tvm::ffi::TensorView w2,
      const tvm::ffi::TensorView w3,
      const tvm::ffi::TensorView out0,
      const tvm::ffi::TensorView out1,
      const tvm::ffi::TensorView out2,
      const tvm::ffi::TensorView out3) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kAttnPrepK}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({kAttnPrepN0, kAttnPrepK}).with_dtype<bf16_t>().with_device(device).verify(w0);
    TensorMatcher({kAttnPrepN1, kAttnPrepK}).with_dtype<bf16_t>().with_device(device).verify(w1);
    TensorMatcher({kAttnPrepN2, kAttnPrepK}).with_dtype<bf16_t>().with_device(device).verify(w2);
    TensorMatcher({kAttnPrepN3, kAttnPrepK}).with_dtype<bf16_t>().with_device(device).verify(w3);
    TensorMatcher({1, kAttnPrepN0}).with_dtype<bf16_t>().with_device(device).verify(out0);
    TensorMatcher({1, kAttnPrepN1}).with_dtype<float>().with_device(device).verify(out1);
    TensorMatcher({1, kAttnPrepN2}).with_dtype<float>().with_device(device).verify(out2);
    TensorMatcher({1, kAttnPrepN3}).with_dtype<bf16_t>().with_device(device).verify(out3);
    constexpr uint32_t kTotalN =
        kAttnPrepN0 + kAttnPrepN1 + kAttnPrepN2 + kAttnPrepN3;
    constexpr uint32_t kBlocks =
        (kTotalN + kAttnPrepWaves - 1) / kAttnPrepWaves;
    LaunchKernel(kBlocks, kAttnPrepWaves * kAttnPrepWave, device.unwrap())(
        gfx90a_attn_prepare_gemv_kernel,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w0.data_ptr()),
        static_cast<const bf16_t*>(w1.data_ptr()),
        static_cast<const bf16_t*>(w2.data_ptr()),
        static_cast<const bf16_t*>(w3.data_ptr()),
        static_cast<bf16_t*>(out0.data_ptr()),
        static_cast<float*>(out1.data_ptr()),
        static_cast<float*>(out2.data_ptr()),
        static_cast<bf16_t*>(out3.data_ptr()));
  }
};

}  // namespace sglang
