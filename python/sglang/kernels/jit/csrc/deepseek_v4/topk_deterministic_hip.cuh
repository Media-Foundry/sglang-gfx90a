#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace sglang {

// Total order: score descending, then logical position ascending. Normalize
// signed zero; rank NaNs below finite scores. Never perturb the score itself.
__device__ inline uint64_t deterministic_topk_key(float score, uint32_t index) {
  uint32_t bits = __float_as_uint(score == 0.0f ? 0.0f : score);
  uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  if ((bits & 0x7fffffffu) > 0x7f800000u) ordered = 0;
  return (uint64_t(ordered) << 32) | (0xffffffffu - index);
}

__global__ void deterministic_topk_hip(
    const float* scores, const int32_t* lengths, const int32_t* pages,
    int32_t* output, int32_t* raw, int64_t score_stride, int64_t page_stride,
    int32_t page_size, int32_t k) {
  const int row = blockIdx.x, lane = threadIdx.x;
  const int length = lengths[row] > 0 ? lengths[row] : 0;
  scores += row * score_stride;
  pages += row * page_stride;
  output += row * k;
  if (raw) raw += row * k;
  for (int i = lane; i < k; i += 256) {
    output[i] = -1;
    if (raw) raw[i] = -1;
  }
  if (length <= k) {
    for (int i = lane; i < length; i += 256) {
      const int index = length - 1 - i;
      output[i] = pages[index / page_size] * page_size + index % page_size;
      if (raw) raw[i] = index;
    }
    return;
  }
  __shared__ int histogram[256];
  __shared__ uint64_t prefix, mask;
  __shared__ int remaining, done, count[4], offset;
  if (lane == 0) {
    prefix = mask = 0;
    remaining = k;
    done = offset = 0;
  }
  __syncthreads();
  // Histogram atomics only count keys. Selection and publication have no
  // atomic-append ordering dependency and need no bounded candidate queue.
  for (int shift = 56; shift >= 0; shift -= 8) {
    histogram[lane] = 0;
    __syncthreads();
    for (int index = lane; index < length; index += 256) {
      const uint64_t key = deterministic_topk_key(scores[index], index);
      if ((key & mask) == prefix) atomicAdd(histogram + ((key >> shift) & 255), 1);
    }
    __syncthreads();
    if (lane == 0) {
      for (int bucket = 255; bucket >= 0; --bucket) {
        if (remaining > histogram[bucket]) remaining -= histogram[bucket];
        else {
          prefix |= uint64_t(bucket) << shift;
          mask |= uint64_t(255) << shift;
          done = remaining == histogram[bucket];
          break;
        }
      }
    }
    __syncthreads();
    if (done) break;
  }
  // Canonical attention accumulation order: descending LOGICAL index.
  // Every lane executes ballot, including lanes outside the final tile.
  const uint64_t threshold = prefix;
  for (int base = length - 1; base >= 0; base -= 256) {
    const int index = base - lane;
    const bool selected = index >= 0 && deterministic_topk_key(scores[index], index) >= threshold;
    const uint64_t ballot = __ballot(selected);
    const int wave = lane / 64, wave_lane = lane % 64;
    const int before = __popcll(ballot & ((uint64_t(1) << wave_lane) - 1));
    if (wave_lane == 0) count[wave] = __popcll(ballot);
    __syncthreads();
    int position = offset + before;
    for (int w = 0; w < wave; ++w) position += count[w];
    if (selected) {
      output[position] = pages[index / page_size] * page_size + index % page_size;
      if (raw) raw[position] = index;
    }
    __syncthreads();
    if (lane == 0) offset += count[0] + count[1] + count[2] + count[3];
    __syncthreads();
  }
}

// All 256 threads participate; the caller must synchronize before reusing
// wave_totals after this scan. Integer sums keep selection exact.
__device__ inline int topk_scan256(int value, int* wave_totals) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  for (int delta = 1; delta < 64; delta <<= 1) {
    const int other = __shfl_up(value, delta, 64);
    if (lane >= delta) value += other;
  }
  if (lane == 63) wave_totals[wave] = value;
  __syncthreads();
  for (int w = 0; w < wave; ++w) value += wave_totals[w];
  return value;
}

__global__ void deterministic_topk_hip_fast(
    const float* scores, const int32_t* lengths, const int32_t* pages,
    int32_t* output, int32_t* raw, int64_t score_stride, int64_t page_stride,
    int32_t page_size, int32_t k) {
  const int row = blockIdx.x, lane = threadIdx.x;
  const int length = lengths[row] > 0 ? lengths[row] : 0;
  scores += row * score_stride;
  pages += row * page_stride;
  output += row * k;
  if (raw) raw += row * k;
  for (int i = lane; i < k; i += 256) {
    output[i] = -1;
    if (raw) raw[i] = -1;
  }
  if (length <= k) {
    for (int i = lane; i < length; i += 256) {
      const int index = length - 1 - i;
      output[i] = pages[index / page_size] * page_size + index % page_size;
      if (raw) raw[i] = index;
    }
    return;
  }
  __shared__ int histogram[4 * 256];
  __shared__ int wave_totals[4], remaining, done;
  __shared__ uint64_t prefix, mask;
  if (lane == 0) {
    prefix = mask = 0;
    remaining = k;
    done = 0;
  }
  __syncthreads();
  for (int shift = 56; shift >= 0; shift -= 8) {
    for (int i = lane; i < 4 * 256; i += 256) histogram[i] = 0;
    __syncthreads();
    // Privatize counting by wave; atomics never choose winning IDs.
    for (int index = lane; index < length; index += 256) {
      const uint64_t key = deterministic_topk_key(scores[index], index);
      if ((key & mask) == prefix)
        atomicAdd(histogram + (lane >> 6) * 256 + ((key >> shift) & 255), 1);
    }
    __syncthreads();
    const int bucket = 255 - lane;
    const int n = histogram[bucket] + histogram[256 + bucket] +
                  histogram[512 + bucket] + histogram[768 + bucket];
    const int target = remaining;
    const int cumulative = topk_scan256(n, wave_totals);
    // Exactly one bucket spans the requested rank. No serial 256-bin walk.
    if (cumulative >= target && cumulative - n < target) {
      prefix |= uint64_t(bucket) << shift;
      mask |= uint64_t(255) << shift;
      remaining = target - (cumulative - n);
      done = remaining == n;
    }
    __syncthreads();
    if (done) break;
  }
  // Each thread owns a contiguous descending-ID interval. A single block
  // scan assigns output offsets; no per-256-position publication barriers.
  const uint64_t threshold = prefix;
  const int chunk = (length + 255) / 256;
  const int begin = length - 1 - lane * chunk;
  const int end = begin - chunk > -1 ? begin - chunk : -1;
  int selected = 0;
  for (int index = begin; index > end; --index)
    selected += deterministic_topk_key(scores[index], index) >= threshold;
  int position = topk_scan256(selected, wave_totals) - selected;
  for (int index = begin; index > end; --index) {
    if (deterministic_topk_key(scores[index], index) >= threshold) {
      output[position] = pages[index / page_size] * page_size + index % page_size;
      if (raw) raw[position] = index;
      ++position;
    }
  }
}

template <bool Fast = false>
struct DeterministicTopKHip {
  static void transform(
      tvm::ffi::TensorView scores, tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView pages, tvm::ffi::TensorView output, int32_t page_size,
      tvm::ffi::Optional<tvm::ffi::TensorView> raw) {
    using namespace host;
    auto B = SymbolicSize{"batch"}, S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_stride"}, K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device).verify(pages);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(output);
    int32_t* raw_ptr = nullptr;
    if (raw.has_value()) {
      TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(raw.value());
      raw_ptr = static_cast<int32_t*>(raw.value().data_ptr());
    }
    RuntimeCheck(page_size > 0, "page_size must be positive");
    RuntimeCheck(K.unwrap() > 0 && K.unwrap() <= 1024, "topk must be in (0, 1024]");
    constexpr auto kernel = Fast ? deterministic_topk_hip_fast : deterministic_topk_hip;
    LaunchKernel(B.unwrap(), 256, device.unwrap())(
        kernel, static_cast<float*>(scores.data_ptr()),
        static_cast<int32_t*>(lengths.data_ptr()), static_cast<int32_t*>(pages.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()), raw_ptr, S.unwrap(), P.unwrap(),
        page_size, static_cast<int32_t>(K.unwrap()));
  }
};
}  // namespace sglang
