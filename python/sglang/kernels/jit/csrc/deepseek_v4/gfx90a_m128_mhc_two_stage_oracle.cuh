#pragma once

#include "gfx90a_mhc_post_pre.cuh"

namespace sglang {

// Standalone-only M128 boundary experiment.  Stage 0 tiles tokens by 16 and
// hidden K by 1024.  Wave 0 materializes the BF16 post-combined residual and
// its RMS partial; two waves then use MFMA 16x16x16 FP16 to emit split-K dot
// partials for the 24 MHC mix rows.  Stage 1 reduces those fixed split slots,
// performs Sinkhorn, the weighted residual, and the following RMSNorm.
constexpr uint32_t kMhc2M = 128;
constexpr uint32_t kMhc2TokenTile = 16;
constexpr uint32_t kMhc2Splits = 16;
constexpr uint32_t kMhc2SplitK = kMhcFusedK / kMhc2Splits;
constexpr uint32_t kMhc2ProducerWaves = 2;
constexpr uint32_t kMhc2ProducerThreads = kMhc2ProducerWaves * kMhcFusedWave;

using mhc2_half4 = _Float16 __attribute__((ext_vector_type(4)));
using mhc2_float4 = float __attribute__((ext_vector_type(4)));

__global__ void __launch_bounds__(kMhc2ProducerThreads, 2)
    gfx90a_m128_mhc_producer_kernel(
        const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ residual,
        const float* __restrict__ previous_post,
        const float* __restrict__ previous_comb,
        const half* __restrict__ fn,
        bf16_t* __restrict__ residual_out,
        float* __restrict__ rms_partial,
        float* __restrict__ dot_partial) {
  const uint32_t token_tile = blockIdx.x / kMhc2Splits;
  const uint32_t split = blockIdx.x % kMhc2Splits;
  const uint32_t wave = threadIdx.x / kMhcFusedWave;
  const uint32_t lane = threadIdx.x % kMhcFusedWave;
  const uint32_t token_lane = lane & 15u;
  const uint32_t k_lane = lane >> 4;
  const uint32_t token = token_tile * kMhc2TokenTile + token_lane;
  const uint32_t split_k0 = split * kMhc2SplitK;

  // One wave owns all 16 tokens' post-combine slice.  Four lanes cooperate on
  // each token so the XOR16/XOR32 tree produces one exact split RMS partial.
  float sq = 0.0f;
  if (wave == 0) {
    for (uint32_t kk = 0; kk < kMhc2SplitK; kk += 16) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t k = split_k0 + kk + k_lane * 4 + j;
        const uint32_t out_hc = k / kMhcFusedHidden;
        const uint32_t h = k % kMhcFusedHidden;
        const size_t residual_base = static_cast<size_t>(token) * kMhcFusedK;
        float value =
            previous_post[static_cast<size_t>(token) * 4 + out_hc] *
            cast<float>(x[static_cast<size_t>(token) * kMhcFusedHidden + h]);
#pragma unroll
        for (uint32_t in_hc = 0; in_hc < 4; ++in_hc) {
          value = fmaf(
              previous_comb[static_cast<size_t>(token) * 16 + in_hc * 4 +
                            out_hc],
              cast<float>(residual[residual_base +
                                   in_hc * kMhcFusedHidden + h]),
              value);
        }
        const bf16_t rounded = cast<bf16_t>(value);
        residual_out[residual_base + k] = rounded;
        const float rf = cast<float>(rounded);
        sq = fmaf(rf, rf, sq);
      }
    }
    sq += __shfl_xor(sq, 16, kMhcFusedWave);
    sq += __shfl_xor(sq, 32, kMhcFusedWave);
    if (lane < 16) {
      rms_partial[(static_cast<size_t>(split) * kMhc2M) + token] = sq;
    }
  }

  // Publish stage-0 BF16 values to the second wave before either wave reads
  // the completed slice for MFMA.  The communication is local to this CTA.
  __threadfence_block();
  __syncthreads();

  const uint32_t n0 = wave * 16;
  mhc2_float4 c = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t kk = 0; kk < kMhc2SplitK; kk += 16) {
    mhc2_half4 a;
    mhc2_half4 b;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t k = split_k0 + kk + k_lane * 4 + j;
      a[j] = static_cast<_Float16>(cast<float>(
          residual_out[static_cast<size_t>(token) * kMhcFusedK + k]));
      const uint32_t n = n0 + token_lane;
      b[j] = n < kMhcFusedMix
                 ? static_cast<_Float16>(
                       fn[static_cast<size_t>(n) * kMhcFusedK + k])
                 : static_cast<_Float16>(0.0f);
    }
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  }

  // MFMA C layout: lane%16 owns N, lane/16 selects the first M row, and the
  // four accumulators advance M by four.
  const uint32_t n = n0 + token_lane;
  const uint32_t m0 = k_lane;
  if (n < kMhcFusedMix) {
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t out_token = token_tile * 16 + m0 + i * 4;
      dot_partial[(static_cast<size_t>(split) * kMhc2M + out_token) *
                      kMhcFusedMix +
                  n] = c[i];
    }
  }
}

__global__ void __launch_bounds__(kMhcFinishThreads, 2)
    gfx90a_m128_mhc_consumer_kernel(
        const bf16_t* __restrict__ residual,
        const float* __restrict__ rms_partial,
        const float* __restrict__ dot_partial,
        const float* __restrict__ hc_scale,
        const float* __restrict__ hc_base,
        const bf16_t* __restrict__ norm_weight,
        float* __restrict__ mixes_out,
        float* __restrict__ post_out,
        float* __restrict__ comb_out,
        bf16_t* __restrict__ layer_input_out,
        float rms_eps,
        float sinkhorn_eps,
        float post_multiplier,
        float norm_eps) {
  __shared__ float mixes[kMhcFusedMix];
  __shared__ float pre[kMhcFusedHc];
  __shared__ float post[kMhcFusedHc];
  __shared__ float comb[kMhcFusedHc * kMhcFusedHc];
  __shared__ float wave_partial[kMhcFinishWaves];
  __shared__ float pre_rms;
  __shared__ float output_rms;

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kMhcFusedWave;
  const uint32_t lane = tid % kMhcFusedWave;
  const size_t residual_base = static_cast<size_t>(token) * kMhcFusedK;
  const size_t x_base = static_cast<size_t>(token) * kMhcFusedHidden;

  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t split = 0; split < kMhc2Splits; ++split) {
      total += rms_partial[static_cast<size_t>(split) * kMhc2M + token];
    }
    pre_rms = rsqrtf(total / static_cast<float>(kMhcFusedK) + rms_eps);
  }
  if (tid < kMhcFusedMix) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t split = 0; split < kMhc2Splits; ++split) {
      total += dot_partial[(static_cast<size_t>(split) * kMhc2M + token) *
                               kMhcFusedMix +
                           tid];
    }
    mixes[tid] = total;
  }
  __syncthreads();
  if (tid < kMhcFusedMix) {
    mixes[tid] *= pre_rms;
    mixes_out[static_cast<size_t>(token) * kMhcFusedMix + tid] = mixes[tid];
  }
  __syncthreads();

  if (wave == 0) {
    if (lane < kMhcFusedHc) {
      const float pre_logit =
          fmaf(mixes[lane], hc_scale[0], hc_base[lane]);
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
      y = fmaf(pre[j],
               cast<float>(residual[residual_base + j * kMhcFusedHidden + h]),
               y);
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
    output_rms =
        rsqrtf(total / static_cast<float>(kMhcFusedHidden) + norm_eps);
  }
  __syncthreads();
#pragma unroll
  for (uint32_t i = 0; i < kMhcFusedHidden / kMhcFinishThreads; ++i) {
    const uint32_t h = tid + i * kMhcFinishThreads;
    layer_input_out[x_base + h] = cast<bf16_t>(
        y_values[i] * output_rms * cast<float>(norm_weight[h]));
  }
}

struct Gfx90aM128MhcTwoStageOracle {
  static void producer(const tvm::ffi::TensorView x,
                       const tvm::ffi::TensorView residual,
                       const tvm::ffi::TensorView previous_post,
                       const tvm::ffi::TensorView previous_comb,
                       const tvm::ffi::TensorView fn,
                       const tvm::ffi::TensorView residual_out,
                       const tvm::ffi::TensorView rms_partial,
                       const tvm::ffi::TensorView dot_partial) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({128, 4096}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({128, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({128, 4}).with_dtype<float>().with_device(device).verify(previous_post);
    TensorMatcher({128, 4, 4}).with_dtype<float>().with_device(device).verify(previous_comb);
    TensorMatcher({24, 16384}).with_dtype<half>().with_device(device).verify(fn);
    TensorMatcher({128, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(residual_out);
    TensorMatcher({16, 128}).with_dtype<float>().with_device(device).verify(rms_partial);
    TensorMatcher({16, 128, 24}).with_dtype<float>().with_device(device).verify(dot_partial);
    LaunchKernel((kMhc2M / kMhc2TokenTile) * kMhc2Splits,
                 kMhc2ProducerThreads, device.unwrap())(
        gfx90a_m128_mhc_producer_kernel,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(previous_post.data_ptr()),
        static_cast<const float*>(previous_comb.data_ptr()),
        static_cast<const half*>(fn.data_ptr()),
        static_cast<bf16_t*>(residual_out.data_ptr()),
        static_cast<float*>(rms_partial.data_ptr()),
        static_cast<float*>(dot_partial.data_ptr()));
  }

  static void consumer(const tvm::ffi::TensorView residual,
                       const tvm::ffi::TensorView rms_partial,
                       const tvm::ffi::TensorView dot_partial,
                       const tvm::ffi::TensorView hc_scale,
                       const tvm::ffi::TensorView hc_base,
                       const tvm::ffi::TensorView norm_weight,
                       const tvm::ffi::TensorView mixes_out,
                       const tvm::ffi::TensorView post_out,
                       const tvm::ffi::TensorView comb_out,
                       const tvm::ffi::TensorView layer_input_out,
                       float rms_eps, float sinkhorn_eps,
                       float post_multiplier, float norm_eps) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({128, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({16, 128}).with_dtype<float>().with_device(device).verify(rms_partial);
    TensorMatcher({16, 128, 24}).with_dtype<float>().with_device(device).verify(dot_partial);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(hc_scale);
    TensorMatcher({24}).with_dtype<float>().with_device(device).verify(hc_base);
    TensorMatcher({4096}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    TensorMatcher({128, 24}).with_dtype<float>().with_device(device).verify(mixes_out);
    TensorMatcher({128, 4}).with_dtype<float>().with_device(device).verify(post_out);
    TensorMatcher({128, 4, 4}).with_dtype<float>().with_device(device).verify(comb_out);
    TensorMatcher({128, 4096}).with_dtype<bf16_t>().with_device(device).verify(layer_input_out);
    LaunchKernel(kMhc2M, kMhcFinishThreads, device.unwrap())(
        gfx90a_m128_mhc_consumer_kernel,
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(rms_partial.data_ptr()),
        static_cast<const float*>(dot_partial.data_ptr()),
        static_cast<const float*>(hc_scale.data_ptr()),
        static_cast<const float*>(hc_base.data_ptr()),
        static_cast<const bf16_t*>(norm_weight.data_ptr()),
        static_cast<float*>(mixes_out.data_ptr()),
        static_cast<float*>(post_out.data_ptr()),
        static_cast<float*>(comb_out.data_ptr()),
        static_cast<bf16_t*>(layer_input_out.data_ptr()),
        rms_eps, sinkhorn_eps, post_multiplier, norm_eps);
  }
};

}  // namespace sglang
