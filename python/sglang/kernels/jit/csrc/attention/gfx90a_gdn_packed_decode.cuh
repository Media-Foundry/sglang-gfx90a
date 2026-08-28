#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

constexpr uint32_t kGdnH = 4;
constexpr uint32_t kGdnHV = 12;
constexpr uint32_t kGdnD = 128;

__device__ __forceinline__ float gdn_wave_sum(float x) {
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1)
    x += __shfl_down(x, offset, 64);
  return x;
}

template <uint32_t kRows, bool kABf16, bool kDtBf16>
__global__ __launch_bounds__(64, 2) void gfx90a_gdn_packed_decode_kernel(
    const __hip_bfloat16* __restrict__ mixed,
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ b,
    const void* __restrict__ A_log,
    const void* __restrict__ dt_bias,
    float* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    __hip_bfloat16* __restrict__ out) {
  const uint32_t tile = blockIdx.x;
  const uint32_t hv = blockIdx.y;
  const uint32_t lane = threadIdx.x;
  const int32_t slot = state_indices[0];
  if (slot < 0) {
    const uint32_t row = tile * kRows + lane;
    if (row < kGdnD) out[hv * kGdnD + row] = __float2bfloat16(0.0f);
    return;
  }

  const uint32_t h = hv / (kGdnHV / kGdnH);
  const float q0_raw = __bfloat162float(mixed[h * kGdnD + lane]);
  const float q1_raw = __bfloat162float(mixed[h * kGdnD + lane + 64]);
  const uint32_t kbase = kGdnH * kGdnD + h * kGdnD;
  const float k0_raw = __bfloat162float(mixed[kbase + lane]);
  const float k1_raw = __bfloat162float(mixed[kbase + lane + 64]);
  float qn = gdn_wave_sum(q0_raw * q0_raw + q1_raw * q1_raw);
  float kn = gdn_wave_sum(k0_raw * k0_raw + k1_raw * k1_raw);
  qn = __shfl(qn, 0, 64);
  kn = __shfl(kn, 0, 64);
  constexpr float scale = 0.08838834764831845f;  // rsqrt(128)
  const float qscale = rsqrtf(qn + 1.0e-6f) * scale;
  const float kscale = rsqrtf(kn + 1.0e-6f);
  const float q0 = q0_raw * qscale;
  const float q1 = q1_raw * qscale;
  const float k0 = k0_raw * kscale;
  const float k1 = k1_raw * kscale;

  const float a_log_value = kABf16
      ? __bfloat162float(static_cast<const __hip_bfloat16*>(A_log)[hv])
      : static_cast<const float*>(A_log)[hv];
  const float dt_bias_value = kDtBf16
      ? __bfloat162float(static_cast<const __hip_bfloat16*>(dt_bias)[hv])
      : static_cast<const float*>(dt_bias)[hv];
  const float x = __bfloat162float(a[hv]) + dt_bias_value;
  const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
  const float alpha = expf(-expf(a_log_value) * softplus);
  // Match the Triton path's BF16 round-trip of sigmoid(beta).
  const __hip_bfloat16 beta_bf16 = __float2bfloat16(1.0f / (1.0f + expf(-__bfloat162float(b[hv]))));
  const float beta = __bfloat162float(beta_bf16);

  const uint32_t row0 = tile * kRows;
  const size_t state_head =
      (static_cast<size_t>(slot) * kGdnHV + hv) * kGdnD * kGdnD;
  float s0[kRows];
  float s1[kRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
    const size_t base = state_head + (row0 + r) * kGdnD;
    s0[r] = state[base + lane] * alpha;
    s1[r] = state[base + lane + 64] * alpha;
  }

  const uint32_t vbase = 2 * kGdnH * kGdnD + hv * kGdnD;
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
    float hk = gdn_wave_sum(s0[r] * k0 + s1[r] * k1);
    hk = __shfl(hk, 0, 64);
    const float delta = (__bfloat162float(mixed[vbase + row0 + r]) - hk) * beta;
    s0[r] += delta * k0;
    s1[r] += delta * k1;
    float y = gdn_wave_sum(s0[r] * q0 + s1[r] * q1);
    if (lane == 0) out[hv * kGdnD + row0 + r] = __float2bfloat16(y);
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
    const size_t base = state_head + (row0 + r) * kGdnD;
    state[base + lane] = s0[r];
    state[base + lane + 64] = s1[r];
  }
}

template <uint32_t kRows, bool kABf16, bool kDtBf16>
struct Gfx90aGdnPackedDecode {
  static void run(const tvm::ffi::TensorView mixed,
                  const tvm::ffi::TensorView a,
                  const tvm::ffi::TensorView b,
                  const tvm::ffi::TensorView A_log,
                  const tvm::ffi::TensorView dt_bias,
                  const tvm::ffi::TensorView state,
                  const tvm::ffi::TensorView state_indices,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, 2560}).with_dtype<__hip_bfloat16>().with_device(device).verify(mixed);
    TensorMatcher({1, 12}).with_dtype<__hip_bfloat16>().with_device(device).verify(a);
    TensorMatcher({1, 12}).with_dtype<__hip_bfloat16>().with_device(device).verify(b);
    if constexpr (kABf16)
      TensorMatcher({12}).with_dtype<__hip_bfloat16>().with_device(device).verify(A_log);
    else
      TensorMatcher({12}).with_dtype<float>().with_device(device).verify(A_log);
    if constexpr (kDtBf16)
      TensorMatcher({12}).with_dtype<__hip_bfloat16>().with_device(device).verify(dt_bias);
    else
      TensorMatcher({12}).with_dtype<float>().with_device(device).verify(dt_bias);
    TensorMatcher({-1, 12, 128, 128}).with_dtype<float>().with_device(device).verify(state);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(state_indices);
    TensorMatcher({1, 1, 12, 128}).with_dtype<__hip_bfloat16>().with_device(device).verify(out);
    static_assert(kGdnD % kRows == 0);
    LaunchKernel(dim3(kGdnD / kRows, 12), 64, device.unwrap())(
        gfx90a_gdn_packed_decode_kernel<kRows, kABf16, kDtBf16>,
        static_cast<const __hip_bfloat16*>(mixed.data_ptr()),
        static_cast<const __hip_bfloat16*>(a.data_ptr()),
        static_cast<const __hip_bfloat16*>(b.data_ptr()),
        A_log.data_ptr(), dt_bias.data_ptr(),
        static_cast<float*>(state.data_ptr()),
        static_cast<const int32_t*>(state_indices.data_ptr()),
        static_cast<__hip_bfloat16*>(out.data_ptr()));
  }
};

}  // namespace sglang
