#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using gfx90a_i32x4_oracle = int32_t __attribute__((ext_vector_type(4)));
using gfx90a_i32x16_oracle = int32_t __attribute__((ext_vector_type(16)));

__global__ void gfx90a_mfma_i8_4x4_probe_kernel(
    int32_t* __restrict__ out, const int32_t* __restrict__ a,
    const int32_t* __restrict__ b, uint32_t cbsz, uint32_t abid,
    uint32_t blgp) {
  const uint32_t lane = threadIdx.x;
  gfx90a_i32x4_oracle acc = {0, 0, 0, 0};
  // Control operands must be compile-time immediates.  The switch deliberately
  // exposes the default and a small audit set without touching production.
  if (cbsz == 0 && abid == 0 && blgp == 0) {
    acc = __builtin_amdgcn_mfma_i32_4x4x4i8(a[lane], b[lane], acc, 0, 0, 0);
  } else if (cbsz == 1 && abid == 0 && blgp == 0) {
    acc = __builtin_amdgcn_mfma_i32_4x4x4i8(a[lane], b[lane], acc, 1, 0, 0);
  } else if (cbsz == 2 && abid == 0 && blgp == 0) {
    acc = __builtin_amdgcn_mfma_i32_4x4x4i8(a[lane], b[lane], acc, 2, 0, 0);
  } else if (cbsz == 0 && abid == 0 && blgp == 1) {
    acc = __builtin_amdgcn_mfma_i32_4x4x4i8(a[lane], b[lane], acc, 0, 0, 1);
  }
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) out[lane * 4 + i] = acc[i];
}

struct Gfx90aMfmaI8_4x4ProbeKernel {
  static void run(const tvm::ffi::TensorView a,
                  const tvm::ffi::TensorView b,
                  const tvm::ffi::TensorView out,
                  int64_t cbsz, int64_t abid, int64_t blgp) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({64}).with_dtype<int32_t>().with_device(device).verify(a);
    TensorMatcher({64}).with_dtype<int32_t>().with_device(device).verify(b);
    TensorMatcher({64, 4}).with_dtype<int32_t>().with_device(device).verify(out);
    LaunchKernel(1, 64, a.device())(
        gfx90a_mfma_i8_4x4_probe_kernel,
        static_cast<int32_t*>(out.data_ptr()),
        static_cast<const int32_t*>(a.data_ptr()),
        static_cast<const int32_t*>(b.data_ptr()),
        static_cast<uint32_t>(cbsz), static_cast<uint32_t>(abid),
        static_cast<uint32_t>(blgp));
  }
};

__global__ void gfx90a_mfma_i8_a4n4k32_oracle_kernel(
    int32_t* __restrict__ mfma_out, int32_t* __restrict__ sdot_out,
    float* __restrict__ mfma_scaled, float* __restrict__ sdot_scaled,
    const int8_t* __restrict__ x, const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale) {
  const uint32_t lane = threadIdx.x;
  const uint32_t local = lane & 3u;
  gfx90a_i32x4_oracle acc = {0, 0, 0, 0};
#pragma unroll
  for (uint32_t k0 = 0; k0 < 32; k0 += 4) {
    int32_t a = 0;
    int32_t b = 0;
    if (lane < 4) {
      a = *reinterpret_cast<const int32_t*>(x + local * 32 + k0);
      b = *reinterpret_cast<const int32_t*>(weight + local * 32 + k0);
    }
    acc = __builtin_amdgcn_mfma_i32_4x4x4i8(a, b, acc, 0, 0, 0);
  }
  if (lane < 4) {
#pragma unroll
    for (uint32_t assignment = 0; assignment < 4; ++assignment) {
      const uint32_t index = assignment * 4 + local;
      mfma_out[index] = acc[assignment];
      mfma_scaled[index] = static_cast<float>(acc[assignment]) *
                           x_scale[assignment] * weight_scale[local];
    }
  }

  if (lane < 16) {
    const uint32_t assignment = lane >> 2;
    const uint32_t row = lane & 3u;
    int32_t ref = 0;
#pragma unroll
    for (uint32_t k0 = 0; k0 < 32; k0 += 4) {
      const int32_t a = *reinterpret_cast<const int32_t*>(
          x + assignment * 32 + k0);
      const int32_t b = *reinterpret_cast<const int32_t*>(
          weight + row * 32 + k0);
      ref = __builtin_amdgcn_sdot4(a, b, ref, false);
    }
    sdot_out[lane] = ref;
    sdot_scaled[lane] = static_cast<float>(ref) *
                        x_scale[assignment] * weight_scale[row];
  }
}

struct Gfx90aMfmaI8A4N4K32OracleKernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView mfma_out,
                  const tvm::ffi::TensorView sdot_out,
                  const tvm::ffi::TensorView mfma_scaled,
                  const tvm::ffi::TensorView sdot_scaled) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({4, 32}).with_dtype<int8_t>().with_device(device).verify(x);
    TensorMatcher({4, 32}).with_dtype<int8_t>().with_device(device).verify(weight);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(weight_scale);
    TensorMatcher({4, 4}).with_dtype<int32_t>().with_device(device).verify(mfma_out);
    TensorMatcher({4, 4}).with_dtype<int32_t>().with_device(device).verify(sdot_out);
    TensorMatcher({4, 4}).with_dtype<float>().with_device(device).verify(mfma_scaled);
    TensorMatcher({4, 4}).with_dtype<float>().with_device(device).verify(sdot_scaled);
    LaunchKernel(1, 64, x.device())(
        gfx90a_mfma_i8_a4n4k32_oracle_kernel,
        static_cast<int32_t*>(mfma_out.data_ptr()),
        static_cast<int32_t*>(sdot_out.data_ptr()),
        static_cast<float*>(mfma_scaled.data_ptr()),
        static_cast<float*>(sdot_scaled.data_ptr()),
        static_cast<const int8_t*>(x.data_ptr()),
        static_cast<const int8_t*>(weight.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const float*>(weight_scale.data_ptr()));
  }
};

template <bool kRunMfma, bool kRunSdot>
__global__ void gfx90a_mfma_i8_m32n32k32_oracle_kernel(
    int32_t* __restrict__ mfma_out, int32_t* __restrict__ sdot_out,
    const int8_t* __restrict__ x, const int8_t* __restrict__ weight) {
  const uint32_t lane = threadIdx.x;
  const uint32_t matrix_lane = lane & 31u;
  if constexpr (kRunMfma) {
    gfx90a_i32x16_oracle acc = {};
#pragma unroll
    for (uint32_t k_group = 0; k_group < 4; ++k_group) {
      const uint32_t k0 = k_group * 8 + (lane >> 5) * 4;
      const int32_t a = *reinterpret_cast<const int32_t*>(
          x + matrix_lane * 32 + k0);
      const int32_t b = *reinterpret_cast<const int32_t*>(
          weight + matrix_lane * 32 + k0);
      acc = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, acc, 0, 0, 0);
    }
#pragma unroll
    for (uint32_t v = 0; v < 16; ++v) {
      const uint32_t row = (lane >> 5) * 4 + (v & 3u) + 8 * (v >> 2);
      mfma_out[row * 32 + matrix_lane] = acc[v];
    }
  }

  if constexpr (kRunSdot) {
    for (uint32_t index = lane; index < 32 * 32; index += 64) {
      const uint32_t row = index / 32;
      const uint32_t col = index % 32;
      int32_t dot = 0;
#pragma unroll
      for (uint32_t k = 0; k < 32; k += 4) {
        dot = __builtin_amdgcn_sdot4(
            *reinterpret_cast<const int32_t*>(x + row * 32 + k),
            *reinterpret_cast<const int32_t*>(weight + col * 32 + k),
            dot, false);
      }
      sdot_out[index] = dot;
    }
  }
}

template <bool kRunMfma, bool kRunSdot>
struct Gfx90aMfmaI8M32N32K32OracleKernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView mfma_out,
                  const tvm::ffi::TensorView sdot_out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({32, 32}).with_dtype<int8_t>().with_device(device).verify(x);
    TensorMatcher({32, 32}).with_dtype<int8_t>().with_device(device).verify(weight);
    TensorMatcher({32, 32}).with_dtype<int32_t>().with_device(device).verify(mfma_out);
    TensorMatcher({32, 32}).with_dtype<int32_t>().with_device(device).verify(sdot_out);
    LaunchKernel(1, 64, x.device())(
        gfx90a_mfma_i8_m32n32k32_oracle_kernel<kRunMfma, kRunSdot>,
        static_cast<int32_t*>(mfma_out.data_ptr()),
        static_cast<int32_t*>(sdot_out.data_ptr()),
        static_cast<const int8_t*>(x.data_ptr()),
        static_cast<const int8_t*>(weight.data_ptr()));
  }
};

}  // namespace sglang
