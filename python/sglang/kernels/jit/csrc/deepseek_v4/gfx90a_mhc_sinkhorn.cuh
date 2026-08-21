#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

constexpr uint32_t kMhcWaveSize = 64;
constexpr uint32_t kMhcSize = 4;
constexpr uint32_t kMhcMixSize = 24;
#ifndef SGLANG_MHC_SINKHORN_ITERS
#define SGLANG_MHC_SINKHORN_ITERS 20
#endif
constexpr uint32_t kMhcSinkhornIters = SGLANG_MHC_SINKHORN_ITERS;

__device__ __forceinline__ float mhc_row_sum(float value) {
  value += __shfl_xor(value, 1, kMhcWaveSize);
  value += __shfl_xor(value, 2, kMhcWaveSize);
  return value;
}

__device__ __forceinline__ float mhc_col_sum(float value) {
  value += __shfl_xor(value, 4, kMhcWaveSize);
  value += __shfl_xor(value, 8, kMhcWaveSize);
  return value;
}

__device__ __forceinline__ float mhc_row_max(float value) {
  value = fmaxf(value, __shfl_xor(value, 1, kMhcWaveSize));
  value = fmaxf(value, __shfl_xor(value, 2, kMhcWaveSize));
  return value;
}

__global__ void __launch_bounds__(kMhcWaveSize)
    gfx90a_mhc_sinkhorn_kernel(const float* __restrict__ mixes,
                               const float* __restrict__ hc_scale,
                               const float* __restrict__ hc_base,
                               float* __restrict__ pre,
                               float* __restrict__ post,
                               float* __restrict__ comb,
                               float eps) {
  const uint32_t lane = threadIdx.x;
  const uint32_t token = blockIdx.x;
  const float* mix = mixes + static_cast<size_t>(token) * kMhcMixSize;

  if (lane < kMhcSize) {
    const float pre_logit = fmaf(mix[lane], hc_scale[0], hc_base[lane]);
    const float post_logit =
        fmaf(mix[kMhcSize + lane], hc_scale[1], hc_base[kMhcSize + lane]);
    pre[static_cast<size_t>(token) * kMhcSize + lane] =
        1.0f / (1.0f + expf(-pre_logit)) + eps;
    post[static_cast<size_t>(token) * kMhcSize + lane] =
        2.0f / (1.0f + expf(-post_logit));
  }

  float value = 0.0f;
  if (lane < kMhcSize * kMhcSize) {
    value = fmaf(mix[2 * kMhcSize + lane], hc_scale[2],
                 hc_base[2 * kMhcSize + lane]);
  }
  value = expf(value - mhc_row_max(value));
  value = value / mhc_row_sum(value) + eps;
  value = value / (mhc_col_sum(value) + eps);

#pragma unroll
  for (uint32_t iter = 1; iter < kMhcSinkhornIters; ++iter) {
    value = value / (mhc_row_sum(value) + eps);
    value = value / (mhc_col_sum(value) + eps);
  }

  if (lane < kMhcSize * kMhcSize) {
    comb[static_cast<size_t>(token) * kMhcSize * kMhcSize + lane] = value;
  }
}

struct Gfx90aMhcSinkhornKernel {
  static void run(const tvm::ffi::TensorView mixes,
                  const tvm::ffi::TensorView hc_scale,
                  const tvm::ffi::TensorView hc_base,
                  const tvm::ffi::TensorView pre,
                  const tvm::ffi::TensorView post,
                  const tvm::ffi::TensorView comb,
                  float eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T, 1, kMhcMixSize}).with_dtype<float>().with_device(device).verify(mixes);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(hc_scale);
    TensorMatcher({kMhcMixSize}).with_dtype<float>().with_device(device).verify(hc_base);
    TensorMatcher({T, 1, kMhcSize}).with_dtype<float>().with_device(device).verify(pre);
    TensorMatcher({T, 1, kMhcSize}).with_dtype<float>().with_device(device).verify(post);
    TensorMatcher({T, 1, kMhcSize, kMhcSize}).with_dtype<float>().with_device(device).verify(comb);

    LaunchKernel(static_cast<uint32_t>(T.unwrap()), kMhcWaveSize,
                 device.unwrap())(
        gfx90a_mhc_sinkhorn_kernel,
        static_cast<const float*>(mixes.data_ptr()),
        static_cast<const float*>(hc_scale.data_ptr()),
        static_cast<const float*>(hc_base.data_ptr()),
        static_cast<float*>(pre.data_ptr()),
        static_cast<float*>(post.data_ptr()),
        static_cast<float*>(comb.data_ptr()), eps);
  }
};

}  // namespace sglang
