#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

template <uint32_t E, uint32_t kAssignments, uint32_t P>
__global__ void gfx90a_bf16_batched_build_runs_kernel(
    const int32_t* __restrict__ sorted_experts,
    const int32_t* __restrict__ num_valid_ids,
    int32_t* __restrict__ block_starts,
    int32_t* __restrict__ block_counts,
    int32_t* __restrict__ overflow) {
  const uint32_t expert = threadIdx.x;
  if (expert >= E) return;
  const uint32_t blocks =
      static_cast<uint32_t>(max(num_valid_ids[0], 0)) / kAssignments;
  uint32_t first = blocks;
  uint32_t count = 0;
  for (uint32_t block = 0; block < blocks; ++block) {
    if (sorted_experts[block] == static_cast<int32_t>(expert)) {
      if (count == 0) first = block;
      ++count;
    }
  }
  block_starts[expert] = static_cast<int32_t>(first);
  block_counts[expert] = static_cast<int32_t>(count);
  if (count * kAssignments > P) atomicExch(overflow, 1);
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t H,
          uint32_t P, uint32_t kAssignments>
__global__ void gfx90a_bf16_batched_pack_kernel(
    bf16_t* __restrict__ expert_x, int32_t* __restrict__ route_rows,
    const bf16_t* __restrict__ x, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ block_starts,
    const int32_t* __restrict__ block_counts) {
  constexpr uint32_t kVec = 4;
  constexpr uint32_t kHVec = H / kVec;
  const uint32_t tasks = E * P * kHVec;
  for (uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
       task < tasks; task += blockDim.x * gridDim.x) {
    const uint32_t h_vec = task % kHVec;
    const uint32_t erow = task / kHVec;
    const uint32_t row = erow % P;
    const uint32_t expert = erow / P;
    const uint32_t count = static_cast<uint32_t>(max(block_counts[expert], 0));
    const bool in_range = row < count * kAssignments;
    uint32_t token = M;
    uint32_t slot = T;
    if (in_range) {
      const uint32_t block = static_cast<uint32_t>(block_starts[expert]) +
                             row / kAssignments;
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[block * kAssignments + row % kAssignments]);
      token = encoded & 0x00ffffffu;
      slot = encoded >> 24;
    }
    auto* dst = reinterpret_cast<uint64_t*>(expert_x) +
                static_cast<size_t>(erow) * kHVec + h_vec;
    if (token < M && slot < T) {
      *dst = reinterpret_cast<const uint64_t*>(x)[
          static_cast<size_t>(token) * kHVec + h_vec];
      if (h_vec == 0) route_rows[static_cast<size_t>(token) * T + slot] = row;
    } else {
      *dst = 0;
    }
  }
}

template <uint32_t E, uint32_t P, uint32_t I>
__global__ void gfx90a_bf16_batched_swiglu_kernel(
    bf16_t* __restrict__ out, const bf16_t* __restrict__ gate_up,
    float limit) {
  const uint32_t elements = E * P * I;
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    const uint32_t row = index / I;
    const uint32_t col = index - row * I;
    float gate = cast<float>(gate_up[static_cast<size_t>(row) * (2 * I) + col]);
    float up = cast<float>(gate_up[static_cast<size_t>(row) * (2 * I) + I + col]);
    gate = fminf(gate, limit);
    up = fmaxf(-limit, fminf(up, limit));
    out[index] = cast<bf16_t>((gate / (1.0f + expf(-gate))) * up);
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t H, uint32_t P>
__global__ void gfx90a_bf16_batched_reduce_kernel(
    bf16_t* __restrict__ out, const bf16_t* __restrict__ expert_out,
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ route_rows,
    const float* __restrict__ topk_weights) {
  const uint32_t elements = M * H;
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    const uint32_t token = index / H;
    const uint32_t col = index - token * H;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t slot = 0; slot < T; ++slot) {
      const size_t route = static_cast<size_t>(token) * T + slot;
      const int32_t expert = topk_ids[route];
      const int32_t row = route_rows[route];
      if (expert >= 0 && expert < static_cast<int32_t>(E) &&
          row >= 0 && row < static_cast<int32_t>(P)) {
        const size_t src = (static_cast<size_t>(expert) * P + row) * H + col;
        acc += cast<float>(expert_out[src]) * topk_weights[route];
      }
    }
    out[index] = cast<bf16_t>(acc);
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t H,
          uint32_t I, uint32_t P, uint32_t kAssignments, uint32_t kBlocks>
struct Gfx90aBf16BatchedMoeOracle {
  static void build_runs(const tvm::ffi::TensorView sorted_experts,
                         const tvm::ffi::TensorView num_valid_ids,
                         const tvm::ffi::TensorView block_starts,
                         const tvm::ffi::TensorView block_counts,
                         const tvm::ffi::TensorView overflow) {
    using namespace host;
    LaunchKernel(1, E, sorted_experts.device())(
        gfx90a_bf16_batched_build_runs_kernel<E, kAssignments, P>,
        static_cast<const int32_t*>(sorted_experts.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<int32_t*>(block_starts.data_ptr()),
        static_cast<int32_t*>(block_counts.data_ptr()),
        static_cast<int32_t*>(overflow.data_ptr()));
  }

  static void pack(const tvm::ffi::TensorView x,
                   const tvm::ffi::TensorView sorted_ids,
                   const tvm::ffi::TensorView block_starts,
                   const tvm::ffi::TensorView block_counts,
                   const tvm::ffi::TensorView expert_x,
                   const tvm::ffi::TensorView route_rows) {
    using namespace host;
    LaunchKernel(kBlocks, 256, x.device())(
        gfx90a_bf16_batched_pack_kernel<E, M, T, H, P, kAssignments>,
        static_cast<bf16_t*>(expert_x.data_ptr()),
        static_cast<int32_t*>(route_rows.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(block_starts.data_ptr()),
        static_cast<const int32_t*>(block_counts.data_ptr()));
  }

  static void swiglu(const tvm::ffi::TensorView gate_up,
                     const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    LaunchKernel(kBlocks, 256, gate_up.device())(
        gfx90a_bf16_batched_swiglu_kernel<E, P, I>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(gate_up.data_ptr()),
        static_cast<float>(limit));
  }

  static void reduce(const tvm::ffi::TensorView expert_out,
                     const tvm::ffi::TensorView topk_ids,
                     const tvm::ffi::TensorView route_rows,
                     const tvm::ffi::TensorView topk_weights,
                     const tvm::ffi::TensorView out) {
    using namespace host;
    LaunchKernel(kBlocks, 256, expert_out.device())(
        gfx90a_bf16_batched_reduce_kernel<E, M, T, H, P>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(expert_out.data_ptr()),
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<const int32_t*>(route_rows.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
