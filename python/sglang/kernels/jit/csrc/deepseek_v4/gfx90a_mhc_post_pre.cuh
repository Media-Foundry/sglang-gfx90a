#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

#ifndef SGLANG_MHC_SINKHORN_ITERS
#define SGLANG_MHC_SINKHORN_ITERS 20
#endif

using namespace device;

constexpr uint32_t kMhcFusedHc = 4;
constexpr uint32_t kMhcFusedHidden = 4096;
constexpr uint32_t kMhcFusedK = kMhcFusedHc * kMhcFusedHidden;
constexpr uint32_t kMhcFusedMix = 24;
constexpr uint32_t kMhcFusedWave = 64;
constexpr uint32_t kMhcFusedWaves = 8;
constexpr uint32_t kMhcFusedThreads = kMhcFusedWave * kMhcFusedWaves;
constexpr uint32_t kMhcFinishWaves = 8;
constexpr uint32_t kMhcFinishThreads = kMhcFusedWave * kMhcFinishWaves;

__device__ __forceinline__ float mhc_fused_wave_sum(float value) {
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, kMhcFusedWave);
  }
  return value;
}

__device__ __forceinline__ float mhc_fused_row_sum(float value) {
  value += __shfl_xor(value, 1, kMhcFusedWave);
  value += __shfl_xor(value, 2, kMhcFusedWave);
  return value;
}

__device__ __forceinline__ float mhc_fused_col_sum(float value) {
  value += __shfl_xor(value, 4, kMhcFusedWave);
  value += __shfl_xor(value, 8, kMhcFusedWave);
  return value;
}

__device__ __forceinline__ float mhc_fused_row_max(float value) {
  value = fmaxf(value, __shfl_xor(value, 1, kMhcFusedWave));
  value = fmaxf(value, __shfl_xor(value, 2, kMhcFusedWave));
  return value;
}

__global__ void __launch_bounds__(kMhcFusedThreads, 1)
    gfx90a_mhc_post_pre_kernel(
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ residual,
        const float* __restrict__ previous_post,
        const float* __restrict__ previous_comb,
        const float* __restrict__ fn,
        const float* __restrict__ hc_scale,
        const float* __restrict__ hc_base,
        const bf16_t* __restrict__ norm_weight,
        bf16_t* __restrict__ residual_out,
        float* __restrict__ post_out,
        float* __restrict__ comb_out,
        bf16_t* __restrict__ layer_input_out,
        float rms_eps,
        float sinkhorn_eps,
        float post_multiplier,
        float norm_eps) {
  __shared__ bf16_t current[kMhcFusedK];
  __shared__ float mixes[kMhcFusedMix];
  __shared__ float pre[kMhcFusedHc];
  __shared__ float post[kMhcFusedHc];
  __shared__ float comb[kMhcFusedHc * kMhcFusedHc];
  __shared__ float wave_partial[kMhcFusedWaves];
  __shared__ float pre_rms;
  __shared__ float output_rms;

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kMhcFusedWave;
  const uint32_t lane = tid % kMhcFusedWave;
  const size_t residual_base = static_cast<size_t>(token) * kMhcFusedK;
  const size_t x_base = static_cast<size_t>(token) * kMhcFusedHidden;

  // Previous post + current x -> next 4-way residual. Store BF16 before the
  // pre-mix, matching the unfused boundary's rounding exactly.
  float local_sq = 0.0f;
  for (uint32_t idx = tid; idx < kMhcFusedK; idx += kMhcFusedThreads) {
    const uint32_t out_hc = idx / kMhcFusedHidden;
    const uint32_t h = idx % kMhcFusedHidden;
    float value = previous_post[static_cast<size_t>(token) * kMhcFusedHc + out_hc] *
                  cast<float>(x[x_base + h]);
#pragma unroll
    for (uint32_t in_hc = 0; in_hc < kMhcFusedHc; ++in_hc) {
      value = fmaf(
          previous_comb[static_cast<size_t>(token) * 16 + in_hc * 4 + out_hc],
          cast<float>(residual[residual_base + in_hc * kMhcFusedHidden + h]),
          value);
    }
    const bf16_t rounded = cast<bf16_t>(value);
    current[idx] = rounded;
    residual_out[residual_base + idx] = rounded;
    const float rounded_f = cast<float>(rounded);
    local_sq = fmaf(rounded_f, rounded_f, local_sq);
  }
  local_sq = mhc_fused_wave_sum(local_sq);
  if (lane == 0) wave_partial[wave] = local_sq;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kMhcFusedWaves; ++i) total += wave_partial[i];
    pre_rms = rsqrtf(total / static_cast<float>(kMhcFusedK) + rms_eps);
  }
  __syncthreads();

  // Eight waves own three of the 24 mix rows each. The 32-KiB current
  // residual is shared once; FP32 checkpoint weights are streamed once.
  const uint32_t row0 = wave * 3;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
  for (uint32_t k = lane; k < kMhcFusedK; k += kMhcFusedWave) {
    const float v = cast<float>(current[k]);
    acc0 = fmaf(fn[static_cast<size_t>(row0 + 0) * kMhcFusedK + k], v, acc0);
    acc1 = fmaf(fn[static_cast<size_t>(row0 + 1) * kMhcFusedK + k], v, acc1);
    acc2 = fmaf(fn[static_cast<size_t>(row0 + 2) * kMhcFusedK + k], v, acc2);
  }
  acc0 = mhc_fused_wave_sum(acc0);
  acc1 = mhc_fused_wave_sum(acc1);
  acc2 = mhc_fused_wave_sum(acc2);
  if (lane == 0) {
    mixes[row0 + 0] = acc0 * pre_rms;
    mixes[row0 + 1] = acc1 * pre_rms;
    mixes[row0 + 2] = acc2 * pre_rms;
  }
  __syncthreads();

  // Wave zero performs the complete 4x4 Sinkhorn in registers.
  if (wave == 0) {
    if (lane < kMhcFusedHc) {
      const float pre_logit =
          fmaf(mixes[lane], hc_scale[0], hc_base[lane]);
      const float post_logit = fmaf(mixes[4 + lane], hc_scale[1], hc_base[4 + lane]);
      pre[lane] = 1.0f / (1.0f + expf(-pre_logit)) + sinkhorn_eps;
      post[lane] = post_multiplier / (1.0f + expf(-post_logit));
    }
    float value = 0.0f;
    if (lane < 16) {
      value = fmaf(mixes[8 + lane], hc_scale[2], hc_base[8 + lane]);
    }
    value = expf(value - mhc_fused_row_max(value));
    value = value / mhc_fused_row_sum(value) + sinkhorn_eps;
    value = value / (mhc_fused_col_sum(value) + sinkhorn_eps);
#pragma unroll
    for (uint32_t iter = 1; iter < SGLANG_MHC_SINKHORN_ITERS; ++iter) {
      value = value / (mhc_fused_row_sum(value) + sinkhorn_eps);
      value = value / (mhc_fused_col_sum(value) + sinkhorn_eps);
    }
    if (lane < 16) comb[lane] = value;
  }
  __syncthreads();

  if (tid < 4) {
    post_out[static_cast<size_t>(token) * 4 + tid] = post[tid];
  }
  if (tid < 16) {
    comb_out[static_cast<size_t>(token) * 16 + tid] = comb[tid];
  }

  // Weighted sum and the following RMSNorm. Each thread keeps its eight
  // hidden positions in registers across the block reduction.
  float y_values[kMhcFusedHidden / kMhcFusedThreads];
  float y_sq = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kMhcFusedHidden / kMhcFusedThreads; ++i) {
    const uint32_t h = tid + i * kMhcFusedThreads;
    float y = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      y = fmaf(pre[j], cast<float>(current[j * kMhcFusedHidden + h]), y);
    }
    // The unfused weighted-sum materializes BF16 before RMSNorm. Preserve
    // that rounding boundary so routing/logits remain reference-equivalent.
    const float rounded_y = cast<float>(cast<bf16_t>(y));
    y_values[i] = rounded_y;
    y_sq = fmaf(rounded_y, rounded_y, y_sq);
  }
  y_sq = mhc_fused_wave_sum(y_sq);
  if (lane == 0) wave_partial[wave] = y_sq;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kMhcFusedWaves; ++i) total += wave_partial[i];
    output_rms = rsqrtf(total / static_cast<float>(kMhcFusedHidden) + norm_eps);
  }
  __syncthreads();
#pragma unroll
  for (uint32_t i = 0; i < kMhcFusedHidden / kMhcFusedThreads; ++i) {
    const uint32_t h = tid + i * kMhcFusedThreads;
    layer_input_out[x_base + h] = cast<bf16_t>(
        y_values[i] * output_rms * cast<float>(norm_weight[h]));
  }
}

// Finish the MHC boundary after the bandwidth-heavy 24x16384 pre-mix has run
// as a multi-CTA Triton kernel.  Keeping that GEMV distributed over the CUs is
// essential for M=1 on CDNA2; this CTA fuses only the launch-bound tail:
// sigmoid/Sinkhorn, weighted residual sum, and the following RMSNorm.
__global__ void __launch_bounds__(kMhcFinishThreads, 2)
    gfx90a_mhc_finish_kernel(
        const bf16_t* __restrict__ residual,
        const float* __restrict__ mixes_in,
        const float* __restrict__ hc_scale,
        const float* __restrict__ hc_base,
        const bf16_t* __restrict__ norm_weight,
        float* __restrict__ post_out,
        float* __restrict__ comb_out,
        bf16_t* __restrict__ layer_input_out,
        float sinkhorn_eps,
        float post_multiplier,
        float norm_eps) {
  __shared__ float pre[kMhcFusedHc];
  __shared__ float post[kMhcFusedHc];
  __shared__ float comb[kMhcFusedHc * kMhcFusedHc];
  __shared__ float wave_partial[kMhcFinishWaves];
  __shared__ float output_rms;

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kMhcFusedWave;
  const uint32_t lane = tid % kMhcFusedWave;
  const size_t residual_base = static_cast<size_t>(token) * kMhcFusedK;
  const size_t x_base = static_cast<size_t>(token) * kMhcFusedHidden;
  const float* mixes = mixes_in + static_cast<size_t>(token) * kMhcFusedMix;

  if (wave == 0) {
    if (lane < kMhcFusedHc) {
      const float pre_logit = fmaf(mixes[lane], hc_scale[0], hc_base[lane]);
      const float post_logit =
          fmaf(mixes[4 + lane], hc_scale[1], hc_base[4 + lane]);
      pre[lane] = 1.0f / (1.0f + expf(-pre_logit)) + sinkhorn_eps;
      post[lane] = post_multiplier / (1.0f + expf(-post_logit));
    }
    float value = 0.0f;
    if (lane < 16) {
      value = fmaf(mixes[8 + lane], hc_scale[2], hc_base[8 + lane]);
    }
    value = expf(value - mhc_fused_row_max(value));
    value = value / mhc_fused_row_sum(value) + sinkhorn_eps;
    value = value / (mhc_fused_col_sum(value) + sinkhorn_eps);
#pragma unroll
    for (uint32_t iter = 1; iter < SGLANG_MHC_SINKHORN_ITERS; ++iter) {
      value = value / (mhc_fused_row_sum(value) + sinkhorn_eps);
      value = value / (mhc_fused_col_sum(value) + sinkhorn_eps);
    }
    if (lane < 16) comb[lane] = value;
  }
  __syncthreads();

  if (tid < 4) post_out[static_cast<size_t>(token) * 4 + tid] = post[tid];
  if (tid < 16) comb_out[static_cast<size_t>(token) * 16 + tid] = comb[tid];

  float y_values[kMhcFusedHidden / kMhcFinishThreads];
  float y_sq = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kMhcFusedHidden / kMhcFinishThreads; ++i) {
    const uint32_t h = tid + i * kMhcFinishThreads;
    float y = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      y = fmaf(pre[j], cast<float>(residual[residual_base + j * kMhcFusedHidden + h]), y);
    }
    const float rounded_y = cast<float>(cast<bf16_t>(y));
    y_values[i] = rounded_y;
    y_sq = fmaf(rounded_y, rounded_y, y_sq);
  }
  y_sq = mhc_fused_wave_sum(y_sq);
  if (lane == 0) wave_partial[wave] = y_sq;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kMhcFinishWaves; ++i) total += wave_partial[i];
    output_rms = rsqrtf(total / static_cast<float>(kMhcFusedHidden) + norm_eps);
  }
  __syncthreads();
#pragma unroll
  for (uint32_t i = 0; i < kMhcFusedHidden / kMhcFinishThreads; ++i) {
    const uint32_t h = tid + i * kMhcFinishThreads;
    layer_input_out[x_base + h] = cast<bf16_t>(
        y_values[i] * output_rms * cast<float>(norm_weight[h]));
  }
}

struct Gfx90aMhcPostPreKernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView previous_post,
                  const tvm::ffi::TensorView previous_comb,
                  const tvm::ffi::TensorView fn,
                  const tvm::ffi::TensorView hc_scale,
                  const tvm::ffi::TensorView hc_base,
                  const tvm::ffi::TensorView norm_weight,
                  const tvm::ffi::TensorView residual_out,
                  const tvm::ffi::TensorView post_out,
                  const tvm::ffi::TensorView comb_out,
                  const tvm::ffi::TensorView layer_input_out,
                  float rms_eps,
                  float sinkhorn_eps,
                  float post_multiplier,
                  float norm_eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T, 4096}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({T, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({T, 4}).with_dtype<float>().with_device(device).verify(previous_post);
    TensorMatcher({T, 4, 4}).with_dtype<float>().with_device(device).verify(previous_comb);
    TensorMatcher({24, 16384}).with_dtype<float>().with_device(device).verify(fn);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(hc_scale);
    TensorMatcher({24}).with_dtype<float>().with_device(device).verify(hc_base);
    TensorMatcher({4096}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    TensorMatcher({T, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(residual_out);
    TensorMatcher({T, 4}).with_dtype<float>().with_device(device).verify(post_out);
    TensorMatcher({T, 4, 4}).with_dtype<float>().with_device(device).verify(comb_out);
    TensorMatcher({T, 4096}).with_dtype<bf16_t>().with_device(device).verify(layer_input_out);
    LaunchKernel(static_cast<uint32_t>(T.unwrap()), kMhcFusedThreads,
                 device.unwrap())(
        gfx90a_mhc_post_pre_kernel,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(previous_post.data_ptr()),
        static_cast<const float*>(previous_comb.data_ptr()),
        static_cast<const float*>(fn.data_ptr()),
        static_cast<const float*>(hc_scale.data_ptr()),
        static_cast<const float*>(hc_base.data_ptr()),
        static_cast<const bf16_t*>(norm_weight.data_ptr()),
        static_cast<bf16_t*>(residual_out.data_ptr()),
        static_cast<float*>(post_out.data_ptr()),
        static_cast<float*>(comb_out.data_ptr()),
        static_cast<bf16_t*>(layer_input_out.data_ptr()),
        rms_eps, sinkhorn_eps, post_multiplier, norm_eps);
  }
};

struct Gfx90aMhcFinishKernel {
  static void run(const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView mixes,
                  const tvm::ffi::TensorView hc_scale,
                  const tvm::ffi::TensorView hc_base,
                  const tvm::ffi::TensorView norm_weight,
                  const tvm::ffi::TensorView post_out,
                  const tvm::ffi::TensorView comb_out,
                  const tvm::ffi::TensorView layer_input_out,
                  float sinkhorn_eps,
                  float post_multiplier,
                  float norm_eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({T, 24}).with_dtype<float>().with_device(device).verify(mixes);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(hc_scale);
    TensorMatcher({24}).with_dtype<float>().with_device(device).verify(hc_base);
    TensorMatcher({4096}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    TensorMatcher({T, 4}).with_dtype<float>().with_device(device).verify(post_out);
    TensorMatcher({T, 4, 4}).with_dtype<float>().with_device(device).verify(comb_out);
    TensorMatcher({T, 4096}).with_dtype<bf16_t>().with_device(device).verify(layer_input_out);
    LaunchKernel(static_cast<uint32_t>(T.unwrap()), kMhcFinishThreads,
                 device.unwrap())(
        gfx90a_mhc_finish_kernel,
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(mixes.data_ptr()),
        static_cast<const float*>(hc_scale.data_ptr()),
        static_cast<const float*>(hc_base.data_ptr()),
        static_cast<const bf16_t*>(norm_weight.data_ptr()),
        static_cast<float*>(post_out.data_ptr()),
        static_cast<float*>(comb_out.data_ptr()),
        static_cast<bf16_t*>(layer_input_out.data_ptr()),
        sinkhorn_eps, post_multiplier, norm_eps);
  }
};

}  // namespace sglang
