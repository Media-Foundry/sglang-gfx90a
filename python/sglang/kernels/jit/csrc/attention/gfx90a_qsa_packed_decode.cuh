#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace sglang {

__device__ __forceinline__ float qsa_wave_sum(float value) {
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1)
    value += __shfl_down(value, offset, 64);
  return value;
}

__device__ __forceinline__ float qsa_wave_max(float value) {
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_down(value, offset, 64));
  return value;
}

template <uint32_t H, uint32_t TOPK>
__global__ __launch_bounds__(256) void gfx90a_qsa_packed_decode_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ packed_k,
    const __hip_bfloat16* __restrict__ packed_v,
    const int32_t* __restrict__ cu_seqlens_k,
    __hip_bfloat16* __restrict__ out,
    float scale) {
  constexpr uint32_t D = 256;
  const uint32_t head = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 63;
  const uint32_t wave = tid >> 6;
  const int32_t count_raw = cu_seqlens_k[1] - cu_seqlens_k[0];
  const uint32_t count =
      count_raw <= 0 ? 0u : min(static_cast<uint32_t>(count_raw), TOPK);

  __shared__ float scores[TOPK];
  __shared__ float wave_partials[4];
  __shared__ float block_value;

  const __hip_bfloat16* q_head = q + head * D;
  for (uint32_t pos = wave; pos < count; pos += 4) {
    float dot = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t d = lane * 4 + i;
      dot += __bfloat162float(q_head[d]) *
             __bfloat162float(packed_k[pos * D + d]);
    }
    dot = qsa_wave_sum(dot);
    if (lane == 0) scores[pos] = dot * scale;
  }
  __syncthreads();

  if (count == 0) {
    out[head * D + tid] = __float2bfloat16(0.0f);
    return;
  }

  float local_max = -FLT_MAX;
  for (uint32_t pos = tid; pos < count; pos += 256)
    local_max = fmaxf(local_max, scores[pos]);
  local_max = qsa_wave_max(local_max);
  if (lane == 0) wave_partials[wave] = local_max;
  __syncthreads();
  if (tid == 0) {
    float value = wave_partials[0];
#pragma unroll
    for (uint32_t i = 1; i < 4; ++i) value = fmaxf(value, wave_partials[i]);
    block_value = value;
  }
  __syncthreads();
  const float max_score = block_value;

  float local_sum = 0.0f;
  for (uint32_t pos = tid; pos < count; pos += 256)
    local_sum += expf(scores[pos] - max_score);
  local_sum = qsa_wave_sum(local_sum);
  if (lane == 0) wave_partials[wave] = local_sum;
  __syncthreads();
  if (tid == 0) {
    block_value = wave_partials[0] + wave_partials[1] + wave_partials[2] +
                  wave_partials[3];
  }
  __syncthreads();
  const float inv_sum = 1.0f / block_value;

  float value = 0.0f;
  for (uint32_t pos = 0; pos < count; ++pos) {
    const float probability = expf(scores[pos] - max_score) * inv_sum;
    value += probability * __bfloat162float(packed_v[pos * D + tid]);
  }
  out[head * D + tid] = __float2bfloat16(value);
}

template <uint32_t B, uint32_t H, uint32_t TOPK, uint32_t SPLITS>
__global__ __launch_bounds__(256) void gfx90a_qsa_packed_split_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ packed_k,
    const __hip_bfloat16* __restrict__ packed_v,
    const int32_t* __restrict__ cu_seqlens_k,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float scale) {
  constexpr uint32_t D = 256;
  constexpr uint32_t MAX_CHUNK = (TOPK + SPLITS - 1) / SPLITS;
  const uint32_t head = blockIdx.x;
  const uint32_t split = blockIdx.y;
  const uint32_t batch = blockIdx.z;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 63;
  const uint32_t wave = tid >> 6;
  const int32_t packed_begin = cu_seqlens_k[batch];
  const int32_t count_raw = cu_seqlens_k[batch + 1] - packed_begin;
  const uint32_t count =
      count_raw <= 0 ? 0u : min(static_cast<uint32_t>(count_raw), TOPK);
  const uint32_t chunk = (count + SPLITS - 1) / SPLITS;
  const uint32_t begin = min(split * chunk, count);
  const uint32_t end = min(begin + chunk, count);
  const uint32_t local_count = end - begin;

  __shared__ float scores[MAX_CHUNK];
  __shared__ float wave_partials[4];
  __shared__ float block_value;

  const __hip_bfloat16* q_head = q + (batch * H + head) * D;
  for (uint32_t local = wave; local < local_count; local += 4) {
    const uint32_t pos = begin + local;
    float dot = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t d = lane * 4 + i;
      dot += __bfloat162float(q_head[d]) *
             __bfloat162float(packed_k[(packed_begin + pos) * D + d]);
    }
    dot = qsa_wave_sum(dot);
    if (lane == 0) scores[local] = dot * scale;
  }
  __syncthreads();

  const uint64_t partial_index =
      (static_cast<uint64_t>(batch) * H + head) * SPLITS + split;
  float local_max = -FLT_MAX;
  for (uint32_t local = tid; local < local_count; local += 256)
    local_max = fmaxf(local_max, scores[local]);
  local_max = qsa_wave_max(local_max);
  if (lane == 0) wave_partials[wave] = local_max;
  __syncthreads();
  if (tid == 0) {
    float value = wave_partials[0];
#pragma unroll
    for (uint32_t i = 1; i < 4; ++i) value = fmaxf(value, wave_partials[i]);
    block_value = value;
    partial_max[partial_index] = value;
  }
  __syncthreads();
  const float max_score = block_value;

  float local_sum = 0.0f;
  for (uint32_t local = tid; local < local_count; local += 256)
    local_sum += expf(scores[local] - max_score);
  local_sum = qsa_wave_sum(local_sum);
  if (lane == 0) wave_partials[wave] = local_sum;
  __syncthreads();
  if (tid == 0) {
    block_value = wave_partials[0] + wave_partials[1] + wave_partials[2] +
                  wave_partials[3];
    partial_sum[partial_index] = block_value;
  }
  __syncthreads();

  float value = 0.0f;
  for (uint32_t local = 0; local < local_count; ++local) {
    const float probability = expf(scores[local] - max_score);
    value += probability *
             __bfloat162float(
                 packed_v[(packed_begin + begin + local) * D + tid]);
  }
  partial_out[(partial_index * D) + tid] = value;
}

template <uint32_t B, uint32_t H, uint32_t SPLITS>
__global__ __launch_bounds__(256) void gfx90a_qsa_packed_reduce_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __hip_bfloat16* __restrict__ out) {
  constexpr uint32_t D = 256;
  const uint32_t head = blockIdx.x;
  const uint32_t batch = blockIdx.z;
  const uint32_t d = threadIdx.x;
  __shared__ float global_max;
  __shared__ float denominator;
  if (d == 0) {
    float m = -FLT_MAX;
#pragma unroll
    for (uint32_t split = 0; split < SPLITS; ++split)
      m = fmaxf(m, partial_max[(batch * H + head) * SPLITS + split]);
    float l = 0.0f;
#pragma unroll
    for (uint32_t split = 0; split < SPLITS; ++split) {
      const uint32_t index = (batch * H + head) * SPLITS + split;
      l += partial_sum[index] * expf(partial_max[index] - m);
    }
    global_max = m;
    denominator = l;
  }
  __syncthreads();
  float value = 0.0f;
#pragma unroll
  for (uint32_t split = 0; split < SPLITS; ++split) {
    const uint32_t index = (batch * H + head) * SPLITS + split;
    value += partial_out[index * D + d] *
             expf(partial_max[index] - global_max);
  }
  out[(batch * H + head) * D + d] =
      __float2bfloat16(denominator > 0.0f ? value / denominator : 0.0f);
}

template <uint32_t B, uint32_t H, uint32_t TOPK>
struct Gfx90aQsaPackedDecode {
  static void run(const tvm::ffi::TensorView q,
                  const tvm::ffi::TensorView packed_k,
                  const tvm::ffi::TensorView packed_v,
                  const tvm::ffi::TensorView cu_seqlens_k,
                  const tvm::ffi::TensorView partial_out,
                  const tvm::ffi::TensorView partial_max,
                  const tvm::ffi::TensorView partial_sum,
                  const tvm::ffi::TensorView out,
                  double scale) {
    constexpr uint32_t SPLITS = 8;
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, H, 256}).with_dtype<__hip_bfloat16>().with_device(device).verify(q);
    TensorMatcher({B * TOPK, 1, 256}).with_dtype<__hip_bfloat16>().with_device(device).verify(packed_k);
    TensorMatcher({B * TOPK, 1, 256}).with_dtype<__hip_bfloat16>().with_device(device).verify(packed_v);
    TensorMatcher({B + 1}).with_dtype<int32_t>().with_device(device).verify(cu_seqlens_k);
    TensorMatcher({B, H, SPLITS, 256}).with_dtype<float>().with_device(device).verify(partial_out);
    TensorMatcher({B, H, SPLITS}).with_dtype<float>().with_device(device).verify(partial_max);
    TensorMatcher({B, H, SPLITS}).with_dtype<float>().with_device(device).verify(partial_sum);
    TensorMatcher({B, H, 256}).with_dtype<__hip_bfloat16>().with_device(device).verify(out);
    LaunchKernel(dim3(H, SPLITS, B), 256, q.device())(
        gfx90a_qsa_packed_split_kernel<B, H, TOPK, SPLITS>,
        static_cast<const __hip_bfloat16*>(q.data_ptr()),
        static_cast<const __hip_bfloat16*>(packed_k.data_ptr()),
        static_cast<const __hip_bfloat16*>(packed_v.data_ptr()),
        static_cast<const int32_t*>(cu_seqlens_k.data_ptr()),
        static_cast<float*>(partial_out.data_ptr()),
        static_cast<float*>(partial_max.data_ptr()),
        static_cast<float*>(partial_sum.data_ptr()),
        static_cast<float>(scale));
    LaunchKernel(dim3(H, 1, B), 256, q.device())(
        gfx90a_qsa_packed_reduce_kernel<B, H, SPLITS>,
        static_cast<const float*>(partial_out.data_ptr()),
        static_cast<const float*>(partial_max.data_ptr()),
        static_cast<const float*>(partial_sum.data_ptr()),
        static_cast<__hip_bfloat16*>(out.data_ptr()));
  }
};

}  // namespace sglang
