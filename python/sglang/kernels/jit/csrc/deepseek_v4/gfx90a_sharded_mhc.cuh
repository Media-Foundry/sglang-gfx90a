#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

#ifndef SGLANG_SHARDED_MHC_SINKHORN_ITERS
#define SGLANG_SHARDED_MHC_SINKHORN_ITERS 20
#endif

using namespace device;

// TP8-specific MHC boundary.  Hidden and residual state remain sharded over
// the hidden dimension; only the 24 pre-mix dots plus one residual sum of
// squares are reduced between stage 1 and stage 2.
constexpr uint32_t kShardedMhcHc = 4;
constexpr uint32_t kShardedMhcHidden = 512;
constexpr uint32_t kShardedMhcFullHidden = 4096;
constexpr uint32_t kShardedMhcK = kShardedMhcHc * kShardedMhcHidden;
constexpr uint32_t kShardedMhcFullK = kShardedMhcHc * kShardedMhcFullHidden;
constexpr uint32_t kShardedMhcMix = 24;
constexpr uint32_t kShardedMhcStats = kShardedMhcMix + 1;
constexpr uint32_t kShardedMhcWave = 64;
constexpr uint32_t kShardedMhcWaves = 8;
constexpr uint32_t kShardedMhcThreads = kShardedMhcWave * kShardedMhcWaves;

__device__ __forceinline__ float sharded_mhc_wave_sum(float value) {
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, kShardedMhcWave);
  }
  return value;
}

__device__ __forceinline__ float sharded_mhc_row_sum(float value) {
  value += __shfl_xor(value, 1, kShardedMhcWave);
  value += __shfl_xor(value, 2, kShardedMhcWave);
  return value;
}

__device__ __forceinline__ float sharded_mhc_col_sum(float value) {
  value += __shfl_xor(value, 4, kShardedMhcWave);
  value += __shfl_xor(value, 8, kShardedMhcWave);
  return value;
}

__device__ __forceinline__ float sharded_mhc_row_max(float value) {
  value = fmaxf(value, __shfl_xor(value, 1, kShardedMhcWave));
  value = fmaxf(value, __shfl_xor(value, 2, kShardedMhcWave));
  return value;
}

// Materialize the same BF16 hc_post boundary as the replicated reference,
// then produce rank-local contributions to [24 dot products, residual ss].
__global__ void __launch_bounds__(kShardedMhcThreads, 1)
    gfx90a_sharded_mhc_stage1_kernel(
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ residual,
        const float* __restrict__ previous_post,
        const float* __restrict__ previous_comb,
        const half* __restrict__ fn_shard,
        bf16_t* __restrict__ residual_out,
        float* __restrict__ stats_out) {
  __shared__ bf16_t current[kShardedMhcK];
  __shared__ float wave_partial[kShardedMhcWaves];

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kShardedMhcWave;
  const uint32_t lane = tid % kShardedMhcWave;
  const size_t residual_base = static_cast<size_t>(token) * kShardedMhcK;
  const size_t x_base = static_cast<size_t>(token) * kShardedMhcHidden;
  const size_t stats_base = static_cast<size_t>(token) * kShardedMhcStats;

  float local_sq = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kShardedMhcK / kShardedMhcThreads; ++i) {
    const uint32_t idx = tid + i * kShardedMhcThreads;
    const uint32_t out_hc = idx / kShardedMhcHidden;
    const uint32_t h = idx % kShardedMhcHidden;
    float value = previous_post[static_cast<size_t>(token) * kShardedMhcHc + out_hc] *
                  cast<float>(x[x_base + h]);
#pragma unroll
    for (uint32_t in_hc = 0; in_hc < kShardedMhcHc; ++in_hc) {
      value = fmaf(
          previous_comb[static_cast<size_t>(token) * 16 + in_hc * 4 + out_hc],
          cast<float>(residual[residual_base + in_hc * kShardedMhcHidden + h]),
          value);
    }
    const bf16_t rounded = cast<bf16_t>(value);
    current[idx] = rounded;
    residual_out[residual_base + idx] = rounded;
    const float rounded_f = cast<float>(rounded);
    local_sq = fmaf(rounded_f, rounded_f, local_sq);
  }
  local_sq = sharded_mhc_wave_sum(local_sq);
  if (lane == 0) wave_partial[wave] = local_sq;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kShardedMhcWaves; ++i) total += wave_partial[i];
    stats_out[stats_base + kShardedMhcMix] = total;
  }
  __syncthreads();

  // Each wave owns three rows of the local 24x2048 FP16 fn shard.
  const uint32_t row0 = wave * 3;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
  for (uint32_t k = lane; k < kShardedMhcK; k += kShardedMhcWave) {
    const float value = cast<float>(current[k]);
    acc0 = fmaf(cast<float>(fn_shard[static_cast<size_t>(row0 + 0) * kShardedMhcK + k]),
                value, acc0);
    acc1 = fmaf(cast<float>(fn_shard[static_cast<size_t>(row0 + 1) * kShardedMhcK + k]),
                value, acc1);
    acc2 = fmaf(cast<float>(fn_shard[static_cast<size_t>(row0 + 2) * kShardedMhcK + k]),
                value, acc2);
  }
  acc0 = sharded_mhc_wave_sum(acc0);
  acc1 = sharded_mhc_wave_sum(acc1);
  acc2 = sharded_mhc_wave_sum(acc2);
  if (lane == 0) {
    stats_out[stats_base + row0 + 0] = acc0;
    stats_out[stats_base + row0 + 1] = acc1;
    stats_out[stats_base + row0 + 2] = acc2;
  }
}

// Consume all-reduced stats, run the replicated scalar MHC math, and retain
// only the local hidden shard of the weighted residual sum.  y is rounded to
// BF16 before its local sum of squares, matching the unfused boundary.
__global__ void __launch_bounds__(kShardedMhcThreads, 2)
    gfx90a_sharded_mhc_stage2_kernel(
        const bf16_t* __restrict__ residual,
        const float* __restrict__ global_stats,
        const float* __restrict__ hc_scale,
        const float* __restrict__ hc_base,
        bf16_t* __restrict__ y_rounded_out,
        float* __restrict__ post_out,
        float* __restrict__ comb_out,
        float* __restrict__ y_sumsq_out,
        float rms_eps,
        float sinkhorn_eps,
        float post_multiplier) {
  __shared__ float pre[kShardedMhcHc];
  __shared__ float post[kShardedMhcHc];
  __shared__ float comb[kShardedMhcHc * kShardedMhcHc];
  __shared__ float mixes[kShardedMhcMix];
  __shared__ float wave_partial[kShardedMhcWaves];

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kShardedMhcWave;
  const uint32_t lane = tid % kShardedMhcWave;
  const size_t residual_base = static_cast<size_t>(token) * kShardedMhcK;
  const size_t stats_base = static_cast<size_t>(token) * kShardedMhcStats;
  const float inv_rms = rsqrtf(
      global_stats[stats_base + kShardedMhcMix] /
          static_cast<float>(kShardedMhcFullK) +
      rms_eps);

  if (tid < kShardedMhcMix) {
    mixes[tid] = global_stats[stats_base + tid] * inv_rms;
  }
  __syncthreads();

  if (wave == 0) {
    if (lane < kShardedMhcHc) {
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
    value = expf(value - sharded_mhc_row_max(value));
    value = value / sharded_mhc_row_sum(value) + sinkhorn_eps;
    value = value / (sharded_mhc_col_sum(value) + sinkhorn_eps);
#pragma unroll
    for (uint32_t iter = 1; iter < SGLANG_SHARDED_MHC_SINKHORN_ITERS; ++iter) {
      value = value / (sharded_mhc_row_sum(value) + sinkhorn_eps);
      value = value / (sharded_mhc_col_sum(value) + sinkhorn_eps);
    }
    if (lane < 16) comb[lane] = value;
  }
  __syncthreads();

  if (tid < 4) post_out[static_cast<size_t>(token) * 4 + tid] = post[tid];
  if (tid < 16) comb_out[static_cast<size_t>(token) * 16 + tid] = comb[tid];

  const uint32_t h = tid;
  float y = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kShardedMhcHc; ++j) {
    y = fmaf(pre[j],
             cast<float>(residual[residual_base + j * kShardedMhcHidden + h]), y);
  }
  const bf16_t rounded = cast<bf16_t>(y);
  y_rounded_out[static_cast<size_t>(token) * kShardedMhcHidden + h] = rounded;
  const float rounded_f = cast<float>(rounded);
  float y_sq = rounded_f * rounded_f;
  y_sq = sharded_mhc_wave_sum(y_sq);
  if (lane == 0) wave_partial[wave] = y_sq;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kShardedMhcWaves; ++i) total += wave_partial[i];
    y_sumsq_out[token] = total;
  }
}

// Normalize the local hidden shard after the caller all-reduces y_sumsq.
__global__ void __launch_bounds__(kShardedMhcThreads, 2)
    gfx90a_sharded_mhc_stage3_kernel(
        const bf16_t* __restrict__ y_rounded,
        const float* __restrict__ global_y_sumsq,
        const bf16_t* __restrict__ norm_weight_shard,
        bf16_t* __restrict__ layer_input_out,
        float norm_eps) {
  const uint32_t token = blockIdx.x;
  const uint32_t h = threadIdx.x;
  const size_t base = static_cast<size_t>(token) * kShardedMhcHidden;
  const float inv_rms = rsqrtf(
      global_y_sumsq[token] / static_cast<float>(kShardedMhcFullHidden) + norm_eps);
  layer_input_out[base + h] = cast<bf16_t>(
      cast<float>(y_rounded[base + h]) * inv_rms *
      cast<float>(norm_weight_shard[h]));
}

struct Gfx90aShardedMhcStage1Kernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView previous_post,
                  const tvm::ffi::TensorView previous_comb,
                  const tvm::ffi::TensorView fn_shard,
                  const tvm::ffi::TensorView residual_out,
                  const tvm::ffi::TensorView stats_out) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T, 512}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({T, 4, 512}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({T, 4}).with_dtype<float>().with_device(device).verify(previous_post);
    TensorMatcher({T, 4, 4}).with_dtype<float>().with_device(device).verify(previous_comb);
    TensorMatcher({24, 2048}).with_dtype<half>().with_device(device).verify(fn_shard);
    TensorMatcher({T, 4, 512}).with_dtype<bf16_t>().with_device(device).verify(residual_out);
    TensorMatcher({T, 25}).with_dtype<float>().with_device(device).verify(stats_out);
    LaunchKernel(static_cast<uint32_t>(T.unwrap()), kShardedMhcThreads,
                 device.unwrap())(
        gfx90a_sharded_mhc_stage1_kernel,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(previous_post.data_ptr()),
        static_cast<const float*>(previous_comb.data_ptr()),
        static_cast<const half*>(fn_shard.data_ptr()),
        static_cast<bf16_t*>(residual_out.data_ptr()),
        static_cast<float*>(stats_out.data_ptr()));
  }
};

struct Gfx90aShardedMhcStage2Kernel {
  static void run(const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView global_stats,
                  const tvm::ffi::TensorView hc_scale,
                  const tvm::ffi::TensorView hc_base,
                  const tvm::ffi::TensorView y_rounded_out,
                  const tvm::ffi::TensorView post_out,
                  const tvm::ffi::TensorView comb_out,
                  const tvm::ffi::TensorView y_sumsq_out,
                  float rms_eps,
                  float sinkhorn_eps,
                  float post_multiplier) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T, 4, 512}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({T, 25}).with_dtype<float>().with_device(device).verify(global_stats);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(hc_scale);
    TensorMatcher({24}).with_dtype<float>().with_device(device).verify(hc_base);
    TensorMatcher({T, 512}).with_dtype<bf16_t>().with_device(device).verify(y_rounded_out);
    TensorMatcher({T, 4}).with_dtype<float>().with_device(device).verify(post_out);
    TensorMatcher({T, 4, 4}).with_dtype<float>().with_device(device).verify(comb_out);
    TensorMatcher({T}).with_dtype<float>().with_device(device).verify(y_sumsq_out);
    LaunchKernel(static_cast<uint32_t>(T.unwrap()), kShardedMhcThreads,
                 device.unwrap())(
        gfx90a_sharded_mhc_stage2_kernel,
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(global_stats.data_ptr()),
        static_cast<const float*>(hc_scale.data_ptr()),
        static_cast<const float*>(hc_base.data_ptr()),
        static_cast<bf16_t*>(y_rounded_out.data_ptr()),
        static_cast<float*>(post_out.data_ptr()),
        static_cast<float*>(comb_out.data_ptr()),
        static_cast<float*>(y_sumsq_out.data_ptr()),
        rms_eps, sinkhorn_eps, post_multiplier);
  }
};

struct Gfx90aShardedMhcStage3Kernel {
  static void run(const tvm::ffi::TensorView y_rounded,
                  const tvm::ffi::TensorView global_y_sumsq,
                  const tvm::ffi::TensorView norm_weight_shard,
                  const tvm::ffi::TensorView layer_input_out,
                  float norm_eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T, 512}).with_dtype<bf16_t>().with_device(device).verify(y_rounded);
    TensorMatcher({T}).with_dtype<float>().with_device(device).verify(global_y_sumsq);
    TensorMatcher({512}).with_dtype<bf16_t>().with_device(device).verify(norm_weight_shard);
    TensorMatcher({T, 512}).with_dtype<bf16_t>().with_device(device).verify(layer_input_out);
    LaunchKernel(static_cast<uint32_t>(T.unwrap()), kShardedMhcThreads,
                 device.unwrap())(
        gfx90a_sharded_mhc_stage3_kernel,
        static_cast<const bf16_t*>(y_rounded.data_ptr()),
        static_cast<const float*>(global_y_sumsq.data_ptr()),
        static_cast<const bf16_t*>(norm_weight_shard.data_ptr()),
        static_cast<bf16_t*>(layer_input_out.data_ptr()), norm_eps);
  }
};

}  // namespace sglang
