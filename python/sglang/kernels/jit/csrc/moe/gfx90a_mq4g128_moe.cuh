#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

template <uint32_t M, uint32_t T, uint32_t E>
__global__ void mq4g128_remap_topk_kernel(
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ local_expert_mapping,
    int32_t* __restrict__ local_ids) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * T) return;
  const int32_t expert = expert_ids[i];
  local_ids[i] =
      expert >= 0 && expert < static_cast<int32_t>(E)
          ? local_expert_mapping[expert]
          : -1;
}

template <uint32_t M, uint32_t T, uint32_t E>
struct Gfx90aMq4g128RemapTopk {
  static void run(const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView local_expert_mapping,
                  const tvm::ffi::TensorView local_ids) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(local_expert_mapping);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(local_ids);
    LaunchKernel((M * T + 63) / 64, 64, expert_ids.device())(
        mq4g128_remap_topk_kernel<M, T, E>,
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<const int32_t*>(local_expert_mapping.data_ptr()),
        static_cast<int32_t*>(local_ids.data_ptr()));
  }
};

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

template <uint32_t K, bool Symmetric = false>
__device__ __forceinline__ float mq4g128_dot_row(
    const uint8_t* __restrict__ row, const float* __restrict__ x,
    uint32_t lane) {
  static_assert(K % 128 == 0);
  float acc = 0.0f;
#pragma unroll
  for (uint32_t group = 0; group < K / 128; ++group) {
    constexpr uint32_t kGroupBytes = Symmetric ? 68 : 72;
    constexpr uint32_t kCodeOffset = Symmetric ? 4 : 8;
    const uint8_t* block = row + group * kGroupBytes;
    const float scale = *reinterpret_cast<const float*>(block);
    const float zero = Symmetric
        ? 0.0f
        : *reinterpret_cast<const float*>(block + 4);
    const uint16_t packed =
        *reinterpret_cast<const uint16_t*>(block + kCodeOffset + lane * 2);
    const uint32_t base = group * 128 + lane * 4;
    if constexpr (Symmetric) {
      acc += scale * static_cast<float>(static_cast<int32_t>(packed & 15) - 8) * x[base];
      acc += scale * static_cast<float>(static_cast<int32_t>((packed >> 4) & 15) - 8) * x[base + 1];
      acc += scale * static_cast<float>(static_cast<int32_t>((packed >> 8) & 15) - 8) * x[base + 2];
      acc += scale * static_cast<float>(static_cast<int32_t>((packed >> 12) & 15) - 8) * x[base + 3];
    } else {
      acc += (scale * static_cast<float>(packed & 15) + zero) * x[base];
      acc += (scale * static_cast<float>((packed >> 4) & 15) + zero) * x[base + 1];
      acc += (scale * static_cast<float>((packed >> 8) & 15) + zero) * x[base + 2];
      acc += (scale * static_cast<float>((packed >> 12) & 15) + zero) * x[base + 3];
    }
  }
#pragma unroll
  for (uint32_t offset = 16; offset != 0; offset >>= 1)
    acc += __shfl_down(acc, offset, 32);
  return acc;
}

template <uint32_t K, uint32_t A, bool Symmetric = false>
__device__ __forceinline__ void mq4g128_dot_row_grouped(
    const uint8_t* __restrict__ row,
    const float* const __restrict__ inputs[A], uint32_t lane,
    float (&acc)[A]) {
  static_assert(K % 128 == 0);
#pragma unroll
  for (uint32_t a = 0; a < A; ++a) acc[a] = 0.0f;
#pragma unroll
  for (uint32_t group = 0; group < K / 128; ++group) {
    constexpr uint32_t kGroupBytes = Symmetric ? 68 : 72;
    constexpr uint32_t kCodeOffset = Symmetric ? 4 : 8;
    const uint8_t* block = row + group * kGroupBytes;
    const float scale = *reinterpret_cast<const float*>(block);
    const float zero = Symmetric ? 0.0f : *reinterpret_cast<const float*>(block + 4);
    const uint16_t packed =
        *reinterpret_cast<const uint16_t*>(block + kCodeOffset + lane * 2);
    const int32_t q_offset = Symmetric ? 8 : 0;
    const float w0 = scale * static_cast<float>(static_cast<int32_t>(packed & 15) - q_offset) + zero;
    const float w1 = scale * static_cast<float>(static_cast<int32_t>((packed >> 4) & 15) - q_offset) + zero;
    const float w2 = scale * static_cast<float>(static_cast<int32_t>((packed >> 8) & 15) - q_offset) + zero;
    const float w3 = scale * static_cast<float>(static_cast<int32_t>((packed >> 12) & 15) - q_offset) + zero;
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

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          bool Symmetric = false>
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
  constexpr uint64_t kRowBytes = (K / 128) * (Symmetric ? 68 : 72);
  const uint8_t* wrow = weight +
      (static_cast<uint64_t>(expert) * N + row) * kRowBytes;
  const float value = mq4g128_dot_row<K, Symmetric>(wrow, x + token * K, lane);
  if (lane == 0) out[(token * T + slot) * N + row] = value;
}

// M32 expert-owned metadata: valid local assignments are packed into
// one contiguous segment per expert. offsets[E] is the total valid count;
// remote/invalid assignments remain -1 in the unused tail.
template <uint32_t E, uint32_t M, uint32_t T>
__global__ void mq4g128_expert_owned_sorter_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ assignments) {
  static_assert((E & (E - 1)) == 0, "expert count must be a power of two");
  constexpr uint32_t kTotal = M * T;
  __shared__ int32_t counts[E];
  __shared__ int32_t scan[E];
  __shared__ int32_t cursors[E];
  const uint32_t tid = threadIdx.x;
  for (uint32_t expert = tid; expert < E; expert += blockDim.x) {
    counts[expert] = 0;
    scan[expert] = 0;
  }
  for (uint32_t assignment = tid; assignment < kTotal;
       assignment += blockDim.x)
    assignments[assignment] = -1;
  __syncthreads();

  for (uint32_t assignment = tid; assignment < kTotal;
       assignment += blockDim.x) {
    const int32_t expert = expert_ids[assignment];
    if (expert >= 0 && expert < static_cast<int32_t>(E))
      atomicAdd(counts + expert, 1);
  }
  __syncthreads();
  for (uint32_t expert = tid; expert < E; expert += blockDim.x)
    scan[expert] = counts[expert];
  __syncthreads();

  for (uint32_t stride = 1; stride < E; stride <<= 1) {
    const uint32_t index = (tid + 1) * stride * 2 - 1;
    if (index < E) scan[index] += scan[index - stride];
    __syncthreads();
  }
  if (tid == 0) scan[E - 1] = 0;
  __syncthreads();
  for (uint32_t stride = E / 2; stride >= 1; stride >>= 1) {
    const uint32_t index = (tid + 1) * stride * 2 - 1;
    if (index < E) {
      const int32_t left = scan[index - stride];
      scan[index - stride] = scan[index];
      scan[index] += left;
    }
    __syncthreads();
  }
  for (uint32_t expert = tid; expert < E; expert += blockDim.x) {
    offsets[expert] = scan[expert];
    cursors[expert] = scan[expert];
  }
  if (tid == 0) offsets[E] = scan[E - 1] + counts[E - 1];
  __syncthreads();

  for (uint32_t assignment = tid; assignment < kTotal;
       assignment += blockDim.x) {
    const int32_t expert = expert_ids[assignment];
    if (expert >= 0 && expert < static_cast<int32_t>(E)) {
      const uint32_t write = atomicAdd(cursors + expert, 1);
      assignments[write] = assignment;
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t W, bool Symmetric = false>
__global__ __launch_bounds__(32 * W) void mq4g128_expert_owned_kernel(
    const float* __restrict__ x, const uint8_t* __restrict__ weight,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ assignments, float* __restrict__ out) {
  const uint32_t subgroup = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  static_assert(W == 2 || W == 4 || W == 8);
  const uint32_t row = blockIdx.x * W + subgroup;
  const uint32_t expert = blockIdx.y;
  if (row >= N) return;
  const int32_t begin = offsets[expert];
  const int32_t end = offsets[expert + 1];
  if (begin == end) return;
  constexpr uint64_t kRowBytes = (K / 128) * (Symmetric ? 68 : 72);
  const uint8_t* wrow =
      weight + (static_cast<uint64_t>(expert) * N + row) * kRowBytes;
  for (int32_t index = begin; index < end; ++index) {
    const int32_t assignment = assignments[index];
    const float value = mq4g128_dot_row<K, Symmetric>(
        wrow, x + static_cast<size_t>(assignment / T) * K, lane);
    if (lane == 0)
      out[static_cast<uint64_t>(assignment) * N + row] = value;
  }
}

// BS1 EP path: collapse the static top-k grid dimension into each row CTA.
// With EP4 most of the ten routed IDs are remote on any one rank.  The normal
// `(N/2,T)` grid still schedules a CTA for every remote slot; this form keeps
// enough row parallelism for gfx90a while paying CTA setup only once per pair
// of output rows.  The dot and reduction order for every valid assignment is
// identical to mq4g128_indexed_kernel.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          bool Symmetric = false>
__global__ __launch_bounds__(64) void mq4g128_indexed_persistent_slots_kernel(
    const float* __restrict__ x, const uint8_t* __restrict__ weight,
    const int32_t* __restrict__ expert_ids, float* __restrict__ out) {
  const uint32_t subgroup = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = blockIdx.x * 2 + subgroup;
  if (row >= N) return;
  constexpr uint64_t kRowBytes = (K / 128) * (Symmetric ? 68 : 72);
#pragma unroll
  for (uint32_t assignment = 0; assignment < M * T; ++assignment) {
    const int32_t expert = expert_ids[assignment];
    float value = 0.0f;
    if (expert >= 0 && expert < static_cast<int32_t>(E)) {
      const uint8_t* wrow = weight +
          (static_cast<uint64_t>(expert) * N + row) * kRowBytes;
      value = mq4g128_dot_row<K, Symmetric>(
          wrow, x + static_cast<size_t>(assignment / T) * K, lane);
    }
    if (lane == 0)
      out[static_cast<uint64_t>(assignment) * N + row] = value;
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t A, bool Symmetric = false>
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
  constexpr uint64_t kRowBytes = (K / 128) * (Symmetric ? 68 : 72);
  const uint8_t* wrow = weight +
      (static_cast<uint64_t>(expert) * N + row) * kRowBytes;
  float values[A];
  mq4g128_dot_row_grouped<K, A, Symmetric>(wrow, inputs, lane, values);
  if (lane == 0) {
#pragma unroll
    for (uint32_t a = 0; a < A; ++a)
      if (assignments[a] >= 0)
        out[static_cast<uint64_t>(assignments[a]) * N + row] = values[a];
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          bool Symmetric = false>
struct Gfx90aMq4g128Indexed {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<float>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 128, Symmetric ? 68 : 72}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel(dim3((N + 1) / 2, T, M), dim3(64), x.device())(
        mq4g128_indexed_kernel<E, M, T, N, K, Symmetric>,
        static_cast<const float*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<float*>(out.data_ptr()));
  }
};

template <uint32_t E, uint32_t M, uint32_t T>
struct Gfx90aMq4g128ExpertOwnedSorter {
  static void run(const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView offsets,
                  const tvm::ffi::TensorView assignments) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({E + 1}).with_dtype<int32_t>().with_device(device).verify(offsets);
    TensorMatcher({M * T}).with_dtype<int32_t>().with_device(device).verify(assignments);
    LaunchKernel(1, 256, expert_ids.device())(
        mq4g128_expert_owned_sorter_kernel<E, M, T>,
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<int32_t*>(offsets.data_ptr()),
        static_cast<int32_t*>(assignments.data_ptr()));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t W, bool Symmetric = false>
struct Gfx90aMq4g128ExpertOwned {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView offsets,
                  const tvm::ffi::TensorView assignments,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<float>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 128, Symmetric ? 68 : 72}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E + 1}).with_dtype<int32_t>().with_device(device).verify(offsets);
    TensorMatcher({M * T}).with_dtype<int32_t>().with_device(device).verify(assignments);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel(dim3((N + W - 1) / W, E), 32 * W, x.device())(
        mq4g128_expert_owned_kernel<E, M, T, N, K, W, Symmetric>,
        static_cast<const float*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const int32_t*>(offsets.data_ptr()),
        static_cast<const int32_t*>(assignments.data_ptr()),
        static_cast<float*>(out.data_ptr()));
  }
};

template <uint32_t K>
__device__ __forceinline__ float mq4g128_symmetric_sdot_row(
    const uint8_t* __restrict__ row, const int8_t* __restrict__ x,
    const float* __restrict__ x_scale, uint32_t lane) {
  static_assert(K % 128 == 0);
  float acc = 0.0f;
#pragma unroll
  for (uint32_t group = 0; group < K / 128; ++group) {
    const uint8_t* block = row + group * 68;
    const float weight_scale = *reinterpret_cast<const float*>(block);
    const uint16_t packed =
        *reinterpret_cast<const uint16_t*>(block + 4 + lane * 2);
    // Expand four packed nibbles into byte lanes with a single gfx90a
    // V_PERM_B32. Adding 0x78 cannot carry across byte lanes because each
    // nibble is at most 15; flipping the sign bit then maps [0, 15] exactly
    // onto the signed INT4 codebook [-8, 7].
    const uint32_t even = packed & 0x0f0fu;
    const uint32_t odd = (packed >> 4) & 0x0f0fu;
    const uint32_t qbytes =
        __builtin_amdgcn_perm(odd, even, 0x05010400u);
    const uint32_t weight_i8 =
        (qbytes + 0x78787878u) ^ 0x80808080u;
    const uint32_t base = group * 128 + lane * 4;
    const int32_t input_i8 = *reinterpret_cast<const int32_t*>(x + base);
    const int32_t dot = __builtin_amdgcn_sdot4(
        static_cast<int32_t>(weight_i8), input_i8, 0, false);
    acc += static_cast<float>(dot) * (weight_scale * x_scale[group]);
  }
#pragma unroll
  for (uint32_t offset = 16; offset != 0; offset >>= 1)
    acc += __shfl_down(acc, offset, 32);
  return acc;
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t W>
__global__ __launch_bounds__(32 * W) void mq4g128_expert_owned_sdot_kernel(
    const int8_t* __restrict__ x, const float* __restrict__ x_scale,
    const uint8_t* __restrict__ weight, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ assignments, float* __restrict__ out) {
  const uint32_t subgroup = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = blockIdx.x * W + subgroup;
  const uint32_t expert = blockIdx.y;
  if (row >= N) return;
  const int32_t begin = offsets[expert];
  const int32_t end = offsets[expert + 1];
  if (begin == end) return;
  constexpr uint64_t kRowBytes = (K / 128) * 68;
  const uint8_t* wrow =
      weight + (static_cast<uint64_t>(expert) * N + row) * kRowBytes;
  for (int32_t index = begin; index < end; ++index) {
    const int32_t assignment = assignments[index];
    const uint32_t token = assignment / T;
    const float value = mq4g128_symmetric_sdot_row<K>(
        wrow, x + static_cast<size_t>(token) * K,
        x_scale + static_cast<size_t>(token) * (K / 128), lane);
    if (lane == 0)
      out[static_cast<uint64_t>(assignment) * N + row] = value;
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t W>
struct Gfx90aMq4g128ExpertOwnedSdot {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView offsets,
                  const tvm::ffi::TensorView assignments,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(x);
    TensorMatcher({M, K / 128}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 128, 68}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E + 1}).with_dtype<int32_t>().with_device(device).verify(offsets);
    TensorMatcher({M * T}).with_dtype<int32_t>().with_device(device).verify(assignments);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel(dim3((N + W - 1) / W, E), 32 * W, x.device())(
        mq4g128_expert_owned_sdot_kernel<E, M, T, N, K, W>,
        static_cast<const int8_t*>(x.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const int32_t*>(offsets.data_ptr()),
        static_cast<const int32_t*>(assignments.data_ptr()),
        static_cast<float*>(out.data_ptr()));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          bool Symmetric = false>
struct Gfx90aMq4g128PersistentSlots {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<float>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 128, Symmetric ? 68 : 72}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel((N + 1) / 2, 64, x.device())(
        mq4g128_indexed_persistent_slots_kernel<E, M, T, N, K, Symmetric>,
        static_cast<const float*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<float*>(out.data_ptr()));
  }
};

template <uint32_t T, uint32_t N>
__global__ void mq4g128_weighted_reduce_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ router_weights,
    bf16_t* __restrict__ out) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N) return;
  float values[T];
#pragma unroll
  for (uint32_t slot = 0; slot < T; ++slot) {
    volatile float weighted = partials[static_cast<uint64_t>(slot) * N + row] *
                              router_weights[slot];
    values[slot] = weighted;
  }
  static_assert(T == 10, "Qwen routed reduction requires top-k 10");
  // Match ATen Reduce.cuh's vt0=4 thread_reduce_impl exactly. Four independent
  // accumulators consume slots with stride four, then combine in lane order.
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  acc0 += values[0]; acc1 += values[1]; acc2 += values[2]; acc3 += values[3];
  acc0 += values[4]; acc1 += values[5]; acc2 += values[6]; acc3 += values[7];
  acc0 += values[8]; acc1 += values[9];
  float sum = acc0 + acc1;
  sum += acc2;
  sum += acc3;
  out[row] = cast<bf16_t>(sum);
}

template <uint32_t T, uint32_t N>
struct Gfx90aMq4g128WeightedReduce {
  static void run(const tvm::ffi::TensorView partials,
                  const tvm::ffi::TensorView router_weights,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({1, T, N}).with_dtype<float>().with_device(device).verify(partials);
    TensorMatcher({1, T}).with_dtype<float>().with_device(device).verify(router_weights);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel((N + 255) / 256, 256, partials.device())(
        mq4g128_weighted_reduce_kernel<T, N>,
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const float*>(router_weights.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
  }
};

template <uint32_t M, uint32_t T, uint32_t N>
__global__ void mq4g128_masked_weighted_reduce_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ router_weights,
    const int32_t* __restrict__ expert_ids, bf16_t* __restrict__ out) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t token = blockIdx.y;
  if (row >= N || token >= M) return;
  float values[T];
#pragma unroll
  for (uint32_t slot = 0; slot < T; ++slot) {
    const uint64_t assignment = static_cast<uint64_t>(token) * T + slot;
    volatile float weighted = expert_ids[assignment] >= 0
        ? partials[assignment * N + row] * router_weights[assignment]
        : 0.0f;
    values[slot] = weighted;
  }
  static_assert(T == 10, "Qwen routed reduction requires top-k 10");
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  acc0 += values[0]; acc1 += values[1]; acc2 += values[2]; acc3 += values[3];
  acc0 += values[4]; acc1 += values[5]; acc2 += values[6]; acc3 += values[7];
  acc0 += values[8]; acc1 += values[9];
  float sum = acc0 + acc1;
  sum += acc2;
  sum += acc3;
  out[static_cast<uint64_t>(token) * N + row] = cast<bf16_t>(sum);
}

template <uint32_t M, uint32_t T, uint32_t N>
struct Gfx90aMq4g128MaskedWeightedReduce {
  static void run(const tvm::ffi::TensorView partials,
                  const tvm::ffi::TensorView router_weights,
                  const tvm::ffi::TensorView expert_ids,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partials);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(router_weights);
    TensorMatcher({M, T}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(dim3((N + 255) / 256, M), 256, partials.device())(
        mq4g128_masked_weighted_reduce_kernel<M, T, N>,
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const float*>(router_weights.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
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
          uint32_t A, uint32_t G, bool Symmetric = false>
struct Gfx90aMq4g128Grouped {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView sorted_assignments,
                  const tvm::ffi::TensorView sorted_experts,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<float>().with_device(device).verify(x);
    TensorMatcher({E, N, K / 128, Symmetric ? 68 : 72}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({G * A}).with_dtype<int32_t>().with_device(device).verify(sorted_assignments);
    TensorMatcher({G}).with_dtype<int32_t>().with_device(device).verify(sorted_experts);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel(dim3((N + 1) / 2, G), dim3(64), x.device())(
        mq4g128_grouped_kernel<E, M, T, N, K, A, Symmetric>,
        static_cast<const float*>(x.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const int32_t*>(sorted_assignments.data_ptr()),
        static_cast<const int32_t*>(sorted_experts.data_ptr()),
        static_cast<float*>(out.data_ptr()));
  }
};

}  // namespace sglang
