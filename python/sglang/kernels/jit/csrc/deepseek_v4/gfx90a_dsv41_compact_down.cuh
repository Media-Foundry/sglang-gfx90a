// Copyright 2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/tensor.h>
#include <tvm/ffi/container/tensor.h>
#include <cmath>

namespace sglang {

// A16W4 v1 weight and scale shuffles have DIFFERENT K strides for K=384:
// weights hold 3 x 128 values; scales hold 2 x 8 groups (512 values).
// Decode their logical addresses independently. Do not feed K=384 into
// CKTile's 256-wide stage-2 pipeline or round the scale stride down to 12.
__device__ inline float dsv41_fp4(uint32_t code) {
  const uint32_t a = code & 7;
  const float v = a < 4 ? 0.5f * a : (a < 6 ? float(a - 2) : float(2 * a - 8));
  return code & 8 ? -v : v;
}

__global__ void dsv41_compact_down_kernel(
    const bf16_t* intermediate, const uint8_t* weight, const uint8_t* scale,
    const int32_t* ids, const float* routes, float* partial,
    uint32_t assignments, uint32_t experts, uint32_t logical_k) {
  constexpr uint32_t N = 5120, K = 384;
  const uint32_t lane = threadIdx.x % 64;
  const uint32_t n = blockIdx.x * 4 + threadIdx.x / 64;
  const uint32_t a = blockIdx.y;
  if (n >= N || a >= assignments) return;
  const int32_t e = ids[a];
  float sum = 0;
  if (e >= 0 && uint32_t(e) < experts) {
    const float rw = routes[a];
    for (uint32_t packed_k = lane; packed_k < logical_k / 2; packed_k += 64) {
      const uint32_t k = packed_k * 2;
      const uint64_t wi = uint64_t(e) * N * (K / 2) +
          ((((n / 16) * (K / 128) + k / 128) * 4 + (k % 128) / 32) * 16 + n % 16) * 16 + (k % 32) / 2;
      const uint32_t g = k / 32;
      const uint64_t si = (((((uint64_t(e) * (N / 32) + n / 32) * 2 + g / 8) * 4 + g % 4) * 16 + n % 16) * 2 + (g % 8) / 4) * 2 + (n % 32) / 16;
      const uint8_t codes = weight[wi];
      const float s = ldexpf(1.0f, int(scale[si]) - 127);
      // Official W4A16 math applies the route before W2, rounded to BF16.
      const float x0 = float(bf16_t(float(intermediate[uint64_t(a) * K + k]) * rw));
      const float x1 = float(bf16_t(float(intermediate[uint64_t(a) * K + k + 1]) * rw));
      sum = fmaf(x0, dsv41_fp4(codes & 15) * s, sum);
      sum = fmaf(x1, dsv41_fp4(codes >> 4) * s, sum);
    }
  }
  for (uint32_t offset = 32; offset; offset >>= 1)
    sum += __shfl_down(sum, offset, 64);
  if (lane == 0) partial[uint64_t(a) * N + n] = sum;
}

__global__ void dsv41_compact_down_reduce(const float* partial, bf16_t* out,
                                         uint32_t rows) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 5120) return;
  const uint32_t m = i / 5120, n = i % 5120;
  float sum = 0;
  // Fixed original top-k-slot order; never atomic append/accumulate.
  for (uint32_t slot = 0; slot < 6; ++slot)
    sum += float(bf16_t(partial[(uint64_t(m) * 6 + slot) * 5120 + n]));
  out[i] = bf16_t(sum);
}

struct Gfx90aDsv41CompactDown {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w,
                  const tvm::ffi::TensorView s, const tvm::ffi::TensorView ids,
                  const tvm::ffi::TensorView routes, const tvm::ffi::TensorView partial,
                  const tvm::ffi::TensorView out, int64_t logical_k) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    const int64_t m = x.size(0), e = w.size(0);
    RuntimeCheck(logical_k > 0 && logical_k <= 384 && logical_k % 32 == 0,
                 "V4.1 compact down requires K aligned to 32 and <=384");
    TensorMatcher({m, 6, 384}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({e, 5120, 192}).with_dtype<uint8_t>().with_device(device).verify(w);
    TensorMatcher({e * 5120, 16}).with_dtype<uint8_t>().with_device(device).verify(s);
    TensorMatcher({m, 6}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({m, 6}).with_dtype<float>().with_device(device).verify(routes);
    TensorMatcher({m, 6, 5120}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({m, 5120}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(dim3(1280, m * 6), 256, device.unwrap())(
        dsv41_compact_down_kernel, static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()), static_cast<const uint8_t*>(s.data_ptr()),
        static_cast<const int32_t*>(ids.data_ptr()), static_cast<const float*>(routes.data_ptr()),
        static_cast<float*>(partial.data_ptr()), uint32_t(m * 6), uint32_t(e), uint32_t(logical_k));
    LaunchKernel((m * 5120 + 255) / 256, 256, device.unwrap())(
        dsv41_compact_down_reduce, static_cast<const float*>(partial.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()), uint32_t(m));
  }
};
}  // namespace sglang
