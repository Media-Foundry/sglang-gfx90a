#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

template <uint32_t E, uint32_t M, uint32_t T, uint32_t A>
__global__ void mq4g128_sorter_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_assignments,
    int32_t* __restrict__ sorted_experts) {
  static_assert((E & (E - 1)) == 0, "expert count must be a power of two");
  constexpr uint32_t kTotal = M * T;
  __shared__ int32_t counts[E];
  __shared__ int32_t scan[E];
  __shared__ int32_t cursors[E];
  __shared__ int32_t next_group;
  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid; i < E; i += blockDim.x) {
    counts[i] = 0;
    scan[i] = 0;
  }
  for (uint32_t i = tid; i < kTotal * A; i += blockDim.x)
    sorted_assignments[i] = -1;
  for (uint32_t i = tid; i < kTotal; i += blockDim.x)
    sorted_experts[i] = -1;
  __syncthreads();

  for (uint32_t assignment = tid; assignment < kTotal; assignment += blockDim.x) {
    const int32_t expert = expert_ids[assignment];
    if (expert >= 0 && expert < static_cast<int32_t>(E))
      atomicAdd(counts + expert, 1);
  }
  __syncthreads();
  for (uint32_t i = tid; i < E; i += blockDim.x)
    scan[i] = (counts[i] + A - 1) / A;
  __syncthreads();

  // Work-efficient exclusive Blelloch scan over per-expert A4 group counts.
  for (uint32_t offset = 1; offset < E; offset <<= 1) {
    const uint32_t index = (tid + 1) * offset * 2 - 1;
    if (index < E) scan[index] += scan[index - offset];
    __syncthreads();
  }
  if (tid == 0) scan[E - 1] = 0;
  __syncthreads();
  for (uint32_t offset = E / 2; offset >= 1; offset >>= 1) {
    const uint32_t index = (tid + 1) * offset * 2 - 1;
    if (index < E) {
      const int32_t left = scan[index - offset];
      scan[index - offset] = scan[index];
      scan[index] += left;
    }
    __syncthreads();
  }
  for (uint32_t expert = tid; expert < E; expert += blockDim.x) {
    cursors[expert] = scan[expert] * A;
    const uint32_t groups = (counts[expert] + A - 1) / A;
    for (uint32_t group = 0; group < groups; ++group)
      sorted_experts[scan[expert] + group] = expert;
  }
  __syncthreads();
  if (tid == 0)
    next_group = scan[E - 1] + (counts[E - 1] + A - 1) / A;
  __syncthreads();
  for (uint32_t assignment = tid; assignment < kTotal; assignment += blockDim.x) {
    const int32_t expert = expert_ids[assignment];
    if (expert >= 0 && expert < static_cast<int32_t>(E)) {
      const uint32_t write = atomicAdd(cursors + expert, 1);
      sorted_assignments[write] = assignment;
    } else {
      // Preserve invalid/remote assignments as explicit zero-fill work.  The
      // grouped projection returns a dense [M,T,N] tensor, just like indexed;
      // leaving these slots uninitialized corrupts the subsequent EP sum.
      const uint32_t group = atomicAdd(&next_group, 1);
      sorted_assignments[group * A] = assignment;
    }
  }
}

template <uint32_t K>
__device__ __forceinline__ float mq4g128_dot_row(
    const uint8_t* __restrict__ row, const float* __restrict__ x,
    uint32_t lane) {
  static_assert(K % 128 == 0);
  float acc = 0.0f;
#pragma unroll
  for (uint32_t group = 0; group < K / 128; ++group) {
    const uint8_t* block = row + group * 72;
    const float scale = *reinterpret_cast<const float*>(block);
    const float zero = *reinterpret_cast<const float*>(block + 4);
    const uint16_t packed =
        *reinterpret_cast<const uint16_t*>(block + 8 + lane * 2);
    const uint32_t base = group * 128 + lane * 4;
    acc += (scale * static_cast<float>(packed & 15) + zero) * x[base];
    acc += (scale * static_cast<float>((packed >> 4) & 15) + zero) * x[base + 1];
    acc += (scale * static_cast<float>((packed >> 8) & 15) + zero) * x[base + 2];
    acc += (scale * static_cast<float>((packed >> 12) & 15) + zero) * x[base + 3];
  }
#pragma unroll
  for (uint32_t offset = 16; offset != 0; offset >>= 1)
    acc += __shfl_down(acc, offset, 32);
  return acc;
}

template <uint32_t K, uint32_t A>
__device__ __forceinline__ void mq4g128_dot_row_grouped(
    const uint8_t* __restrict__ row,
    const float* const __restrict__ inputs[A], uint32_t lane,
    float (&acc)[A]) {
  static_assert(K % 128 == 0);
#pragma unroll
  for (uint32_t a = 0; a < A; ++a) acc[a] = 0.0f;
#pragma unroll
  for (uint32_t group = 0; group < K / 128; ++group) {
    const uint8_t* block = row + group * 72;
    const float scale = *reinterpret_cast<const float*>(block);
    const float zero = *reinterpret_cast<const float*>(block + 4);
    const uint16_t packed =
        *reinterpret_cast<const uint16_t*>(block + 8 + lane * 2);
    const float w0 = scale * static_cast<float>(packed & 15) + zero;
    const float w1 = scale * static_cast<float>((packed >> 4) & 15) + zero;
    const float w2 = scale * static_cast<float>((packed >> 8) & 15) + zero;
    const float w3 = scale * static_cast<float>((packed >> 12) & 15) + zero;
    const uint32_t base = group * 128 + lane * 4;
#pragma unroll
    for (uint32_t a = 0; a < A; ++a) {
      const float* x = inputs[a];
      if (x != nullptr)
        acc[a] += w0 * x[base] + w1 * x[base + 1] +
                  w2 * x[base + 2] + w3 * x[base + 3];
    }
  }
#pragma unroll
  for (uint32_t offset = 16; offset != 0; offset >>= 1)
#pragma unroll
    for (uint32_t a = 0; a < A; ++a)
      acc[a] += __shfl_down(acc[a], offset, 32);
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K>
__global__ __launch_bounds__(64) void mq4g128_indexed_kernel(
    const float* __restrict__ x, const uint8_t* __restrict__ weight,
    const int32_t* __restrict__ expert_ids, float* __restrict__ out) {
  const uint32_t subgroup = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = blockIdx.x * 2 + subgroup;
  const uint32_t slot = blockIdx.y;
  const uint32_t token = blockIdx.z;
  if (row >= N) return;
  const int32_t expert = expert_ids[token * T + slot];
  if (expert < 0 || expert >= static_cast<int32_t>(E)) {
    if (lane == 0) out[(token * T + slot) * N + row] = 0.0f;
    return;
  }
  constexpr uint64_t kRowBytes = (K / 128) * 72;
  const uint8_t* wrow = weight +
      (static_cast<uint64_t>(expert) * N + row) * kRowBytes;
  const float value = mq4g128_dot_row<K>(wrow, x + token * K, lane);
  if (lane == 0) out[(token * T + slot) * N + row] = value;
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t A>
__global__ __launch_bounds__(64) void mq4g128_grouped_kernel(
    const float* __restrict__ x, const uint8_t* __restrict__ weight,
    const int32_t* __restrict__ sorted_assignments,
    const int32_t* __restrict__ sorted_experts,
    float* __restrict__ out) {
  const uint32_t subgroup = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = blockIdx.x * 2 + subgroup;
  const uint32_t group = blockIdx.y;
  if (row >= N) return;
  const int32_t expert = sorted_experts[group];
  if (expert < 0 || expert >= static_cast<int32_t>(E)) {
    const int32_t assignment = sorted_assignments[group * A];
    if (lane == 0 && assignment >= 0)
      out[static_cast<uint64_t>(assignment) * N + row] = 0.0f;
    return;
  }
  const float* inputs[A];
  int32_t assignments[A];
#pragma unroll
  for (uint32_t a = 0; a < A; ++a) {
    const int32_t assignment = sorted_assignments[group * A + a];
    assignments[a] = assignment;
    inputs[a] = assignment >= 0 ? x + (assignment / T) * K : nullptr;
  }
  constexpr uint64_t kRowBytes = (K / 128) * 72;
  const uint8_t* wrow = weight +
      (static_cast<uint64_t>(expert) * N + row) * kRowBytes;
  float values[A];
  mq4g128_dot_row_grouped<K, A>(wrow, inputs, lane, values);
  if (lane == 0) {
#pragma unroll
    for (uint32_t a = 0; a < A; ++a)
      if (assignments[a] >= 0)
        out[static_cast<uint64_t>(assignments[a]) * N + row] = values[a];
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K>
struct Gfx90aMq4g128Indexed {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<float>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 128, 72}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel(dim3((N + 1) / 2, T, M), dim3(64), x.device())(
        mq4g128_indexed_kernel<E, M, T, N, K>,
        static_cast<const float*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<float*>(out.data_ptr()));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t A>
struct Gfx90aMq4g128Sorter {
  static void run(const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView sorted_assignments,
                  const tvm::ffi::TensorView sorted_experts) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M * T * A}).with_dtype<int32_t>().with_device(device).verify(sorted_assignments);
    TensorMatcher({M * T}).with_dtype<int32_t>().with_device(device).verify(sorted_experts);
    LaunchKernel(1, 256, expert_ids.device())(
        mq4g128_sorter_kernel<E, M, T, A>,
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<int32_t*>(sorted_assignments.data_ptr()),
        static_cast<int32_t*>(sorted_experts.data_ptr()));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t A, uint32_t G>
struct Gfx90aMq4g128Grouped {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView sorted_assignments,
                  const tvm::ffi::TensorView sorted_experts,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<float>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 128, 72}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({G * A}).with_dtype<int32_t>().with_device(device).verify(sorted_assignments);
    TensorMatcher({G}).with_dtype<int32_t>().with_device(device).verify(sorted_experts);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel(dim3((N + 1) / 2, G), dim3(64), x.device())(
        mq4g128_grouped_kernel<E, M, T, N, K, A>,
        static_cast<const float*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const int32_t*>(sorted_assignments.data_ptr()),
        static_cast<const int32_t*>(sorted_experts.data_ptr()),
        static_cast<float*>(out.data_ptr()));
  }
};

}  // namespace sglang
