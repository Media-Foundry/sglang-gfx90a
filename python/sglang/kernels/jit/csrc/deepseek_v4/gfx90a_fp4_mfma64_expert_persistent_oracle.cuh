#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

template <uint32_t E, uint32_t kAssignments>
__global__ void gfx90a_fp4_mfma64_build_expert_runs_kernel(
    const int32_t* __restrict__ sorted_experts,
    const int32_t* __restrict__ num_valid_ids,
    int32_t* __restrict__ active_experts,
    int32_t* __restrict__ block_starts,
    int32_t* __restrict__ block_counts) {
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
  active_experts[expert] = static_cast<int32_t>(expert);
  block_starts[expert] = static_cast<int32_t>(first);
  block_counts[expert] = static_cast<int32_t>(count);
}

// Oracle-only MFMA64 gate/up schedule for large prefill.  One CTA owns an
// (expert, I16 tile), then walks that expert's consecutive A64 assignment
// blocks.  The per-block MFMA and split reduction order are identical to the
// production kernel; only the task mapping changes so the second block can
// reuse the first block's weight footprint through the gfx90a caches.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4,
          uint32_t kAssignments = 64>
__global__ void __launch_bounds__(kSplit * kFp4ExpertWave)
    gfx90a_fp4_mfma64_expert_persistent_gate_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ active_experts,
        const int32_t* __restrict__ block_starts,
        const int32_t* __restrict__ block_counts,
        const int32_t* __restrict__ num_active, float limit) {
  static_assert(kAssignments == 64);
  constexpr uint32_t kAssignmentHalves = 4;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kTiles = I / 16;
  __shared__ float gate_partial[kSplit][kAssignments * 16];
  __shared__ float up_partial[kSplit][kAssignments * 16];
  const uint32_t lane = threadIdx.x & 63;
  const uint32_t split = threadIdx.x >> 6;
  const uint32_t matrix_index = lane & 15;
  const uint32_t k_lane = (lane >> 4) * 4;
  const uint32_t active = static_cast<uint32_t>(max(num_active[0], 0));

  for (uint32_t task = blockIdx.x; task < active * kTiles; task += kBlocks) {
    const uint32_t active_index = task / kTiles;
    const uint32_t tile = task % kTiles;
    const uint32_t local_row = tile * 16 + matrix_index;
    const int32_t expert_id = active_experts[active_index];
    const uint32_t expert = static_cast<uint32_t>(max(expert_id, 0));
    const uint32_t block_start =
        static_cast<uint32_t>(max(block_starts[active_index], 0));
    const uint32_t block_count =
        static_cast<uint32_t>(max(block_counts[active_index], 0));

    for (uint32_t chunk = 0; chunk < block_count; ++chunk) {
      const uint32_t expert_block = block_start + chunk;
      uint32_t assignment_lane = 0;
      if (lane < kAssignments) {
        assignment_lane = static_cast<uint32_t>(
            sorted_ids[expert_block * kAssignments + lane]);
      }
      uint32_t tokens[kAssignmentHalves][4], slots[kAssignmentHalves][4];
      bool assignment_valid[kAssignmentHalves][4];
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          const uint32_t assignment = half * 16 + (lane >> 4) * 4 + r;
          const uint32_t encoded =
              __shfl(assignment_lane, assignment, kFp4ExpertWave);
          const uint32_t token = encoded & 0x00ffffffu;
          const uint32_t slot = encoded >> 24;
          assignment_valid[half][r] = token < M && slot < T;
          tokens[half][r] = assignment_valid[half][r] ? token : 0;
          slots[half][r] = assignment_valid[half][r] ? slot : 0;
        }
      }
      uint32_t a_tokens[kAssignmentHalves];
      bool a_valid[kAssignmentHalves];
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
        const uint32_t encoded = __shfl(
            assignment_lane, half * 16 + matrix_index, kFp4ExpertWave);
        const uint32_t token = encoded & 0x00ffffffu;
        const uint32_t slot = encoded >> 24;
        a_valid[half] = token < M && slot < T;
        a_tokens[half] = a_valid[half] ? token : 0;
      }
      float gate_acc[kAssignmentHalves][4] = {};
      float up_acc[kAssignmentHalves][4] = {};
      if (expert_id >= 0 && expert_id < static_cast<int32_t>(E)) {
        for (uint32_t group = split; group < kGroups; group += kSplit) {
          const uint32_t k0 = group * 32;
          const size_t gate_base =
              (static_cast<size_t>(expert) * (2 * I) + local_row) *
                  (K / 2) +
              group * 16;
          const size_t up_base =
              (static_cast<size_t>(expert) * (2 * I) + I + local_row) *
                  (K / 2) +
              group * 16;
          const int32_t gate_b0 = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + gate_base +
                                                  k_lane / 2));
          const int32_t gate_b1 = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + gate_base + 8 +
                                                  k_lane / 2));
          const int32_t up_b0 = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + up_base +
                                                  k_lane / 2));
          const int32_t up_b1 = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + up_base + 8 +
                                                  k_lane / 2));
          float gate_scale_lane = 0.0f;
          float up_scale_lane = 0.0f;
          if (lane < 16) {
            gate_scale_lane = gfx90a_e8m0_value(
                weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                    expert, tile * 16 + lane, group)]);
            up_scale_lane = gfx90a_e8m0_value(
                weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                    expert, I + tile * 16 + lane, group)]);
          }
          const float gate_scale =
              __shfl(gate_scale_lane, matrix_index, kFp4ExpertWave);
          const float up_scale =
              __shfl(up_scale_lane, matrix_index, kFp4ExpertWave);
          float x_scale_lane = 0.0f;
          if (lane < kAssignments) {
            const uint32_t token = assignment_lane & 0x00ffffffu;
            const uint32_t slot = assignment_lane >> 24;
            if (token < M && slot < T) {
              x_scale_lane =
                  x_scale[static_cast<size_t>(token) * kGroups + group];
            }
          }
#pragma unroll
          for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
            const size_t x_base = static_cast<size_t>(a_tokens[half]) * K + k0;
            const int32_t av0 = a_valid[half]
                ? *reinterpret_cast<const int32_t*>(xq + x_base + k_lane)
                : 0;
            const int32_t av1 = a_valid[half]
                ? *reinterpret_cast<const int32_t*>(xq + x_base + 16 + k_lane)
                : 0;
            gfx90a_i32x4 gate_cv{0, 0, 0, 0};
            gfx90a_i32x4 up_cv{0, 0, 0, 0};
            gate_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
                av0, gate_b0, gate_cv, 0, 0, 0);
            gate_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
                av1, gate_b1, gate_cv, 0, 0, 0);
            up_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
                av0, up_b0, up_cv, 0, 0, 0);
            up_cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
                av1, up_b1, up_cv, 0, 0, 0);
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
              const uint32_t assignment =
                  half * 16 + (lane >> 4) * 4 + r;
              const float xs = __shfl(
                  x_scale_lane, assignment, kFp4ExpertWave);
              gate_acc[half][r] += static_cast<float>(gate_cv[r]) * xs *
                                   gate_scale * 0.5f;
              up_acc[half][r] +=
                  static_cast<float>(up_cv[r]) * xs * up_scale * 0.5f;
            }
          }
        }
      }
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          const uint32_t index =
              lane * (kAssignmentHalves * 4) + half * 4 + r;
          gate_partial[split][index] = gate_acc[half][r];
          up_partial[split][index] = up_acc[half][r];
        }
      }
      __syncthreads();
      if (split == 0) {
#pragma unroll
        for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
          for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t index =
                lane * (kAssignmentHalves * 4) + half * 4 + r;
            float gate = gate_partial[0][index];
            float up = up_partial[0][index];
#pragma unroll
            for (uint32_t s = 1; s < kSplit; ++s) {
              gate += gate_partial[s][index];
              up += up_partial[s][index];
            }
            if (assignment_valid[half][r] && expert_id >= 0 &&
                expert_id < static_cast<int32_t>(E)) {
              gate = fminf(gate, limit);
              up = fmaxf(-limit, fminf(up, limit));
              const float activated = gate / (1.0f + expf(-gate));
              const size_t output_assignment =
                  static_cast<size_t>(tokens[half][r]) * T +
                  slots[half][r];
              out[output_assignment * I + local_row] =
                  cast<bf16_t>(activated * up);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4,
          uint32_t kAssignments = 64>
struct Gfx90aFp4Mfma64ExpertPersistentGateOracle {
  static void build_runs(const tvm::ffi::TensorView sorted_experts,
                         const tvm::ffi::TensorView num_valid_ids,
                         const tvm::ffi::TensorView active_experts,
                         const tvm::ffi::TensorView block_starts,
                         const tvm::ffi::TensorView block_counts) {
    using namespace host;
    LaunchKernel(1, E, sorted_experts.device())(
        gfx90a_fp4_mfma64_build_expert_runs_kernel<E, kAssignments>,
        static_cast<const int32_t*>(sorted_experts.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<int32_t*>(active_experts.data_ptr()),
        static_cast<int32_t*>(block_starts.data_ptr()),
        static_cast<int32_t*>(block_counts.data_ptr()));
  }

  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView sorted_ids,
                  const tvm::ffi::TensorView active_experts,
                  const tvm::ffi::TensorView block_starts,
                  const tvm::ffi::TensorView block_counts,
                  const tvm::ffi::TensorView num_active,
                  const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    LaunchKernel(kBlocks, kSplit * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_mfma64_expert_persistent_gate_kernel<
            E, M, T, I, K, kBlocks, kSplit, kAssignments>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(active_experts.data_ptr()),
        static_cast<const int32_t*>(block_starts.data_ptr()),
        static_cast<const int32_t*>(block_counts.data_ptr()),
        static_cast<const int32_t*>(num_active.data_ptr()),
        static_cast<float>(limit));
  }
};

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 2,
          uint32_t kAssignments = 64>
__global__ void __launch_bounds__(kSplit * kFp4ExpertWave)
    gfx90a_fp4_mfma64_expert_persistent_down_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ active_experts,
        const int32_t* __restrict__ block_starts,
        const int32_t* __restrict__ block_counts,
        const int32_t* __restrict__ num_active,
        const float* __restrict__ topk_weights) {
  static_assert(kAssignments == 64);
  constexpr uint32_t kAssignmentHalves = 4;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kTiles = N / 16;
  __shared__ float tile_partial[kSplit][kAssignments * 16];
  const uint32_t lane = threadIdx.x & 63;
  const uint32_t split = threadIdx.x >> 6;
  const uint32_t matrix_index = lane & 15;
  const uint32_t k_lane = (lane >> 4) * 4;
  const uint32_t active = static_cast<uint32_t>(max(num_active[0], 0));

  for (uint32_t task = blockIdx.x; task < active * kTiles; task += kBlocks) {
    const uint32_t active_index = task / kTiles;
    const uint32_t tile = task % kTiles;
    const uint32_t row = tile * 16 + matrix_index;
    const int32_t expert_id = active_experts[active_index];
    const uint32_t expert = static_cast<uint32_t>(max(expert_id, 0));
    const uint32_t block_start =
        static_cast<uint32_t>(max(block_starts[active_index], 0));
    const uint32_t block_count =
        static_cast<uint32_t>(max(block_counts[active_index], 0));

    for (uint32_t chunk = 0; chunk < block_count; ++chunk) {
      const uint32_t expert_block = block_start + chunk;
      uint32_t assignment_lane = 0;
      float assignment_weight_lane = 0.0f;
      if (lane < kAssignments) {
        assignment_lane = static_cast<uint32_t>(
            sorted_ids[expert_block * kAssignments + lane]);
        const uint32_t token = assignment_lane & 0x00ffffffu;
        const uint32_t slot = assignment_lane >> 24;
        if (token < M && slot < T) {
          assignment_weight_lane =
              topk_weights[static_cast<size_t>(token) * T + slot];
        }
      }
      uint32_t tokens[kAssignmentHalves][4], slots[kAssignmentHalves][4];
      bool assignment_valid[kAssignmentHalves][4];
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          const uint32_t assignment = half * 16 + (lane >> 4) * 4 + r;
          const uint32_t encoded =
              __shfl(assignment_lane, assignment, kFp4ExpertWave);
          const uint32_t token = encoded & 0x00ffffffu;
          const uint32_t slot = encoded >> 24;
          assignment_valid[half][r] = token < M && slot < T;
          tokens[half][r] = assignment_valid[half][r] ? token : 0;
          slots[half][r] = assignment_valid[half][r] ? slot : 0;
        }
      }
      uint32_t a_tokens[kAssignmentHalves], a_slots[kAssignmentHalves];
      bool a_valid[kAssignmentHalves];
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
        const uint32_t encoded = __shfl(
            assignment_lane, half * 16 + matrix_index, kFp4ExpertWave);
        const uint32_t token = encoded & 0x00ffffffu;
        const uint32_t slot = encoded >> 24;
        a_valid[half] = token < M && slot < T;
        a_tokens[half] = a_valid[half] ? token : 0;
        a_slots[half] = a_valid[half] ? slot : 0;
      }
      float acc[kAssignmentHalves][4] = {};
      if (expert_id >= 0 && expert_id < static_cast<int32_t>(E)) {
        for (uint32_t group = split; group < kGroups; group += kSplit) {
          const uint32_t k0 = group * 32;
          const size_t weight_base =
              (static_cast<size_t>(expert) * N + row) * (K / 2) +
              group * 16;
          const int32_t bv0 = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + weight_base +
                                                  k_lane / 2));
          const int32_t bv1 = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + weight_base + 8 +
                                                  k_lane / 2));
          float weight_scale_lane = 0.0f;
          if (lane < 16) {
            weight_scale_lane = gfx90a_e8m0_value(
                weight_scale[gfx90a_down_scale_offset<E, N, K>(
                    expert, tile * 16 + lane, group)]);
          }
          const float weight_s =
              __shfl(weight_scale_lane, matrix_index, kFp4ExpertWave);
          float x_scale_lane = 0.0f;
          if (lane < kAssignments) {
            const uint32_t token = assignment_lane & 0x00ffffffu;
            const uint32_t slot = assignment_lane >> 24;
            if (token < M && slot < T) {
              x_scale_lane =
                  x_scale[(static_cast<size_t>(token) * T + slot) * kGroups +
                          group];
            }
          }
#pragma unroll
          for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
            const size_t input_assignment =
                static_cast<size_t>(a_tokens[half]) * T + a_slots[half];
            const size_t x_base = input_assignment * K + k0;
            const int32_t av0 = a_valid[half]
                ? *reinterpret_cast<const int32_t*>(xq + x_base + k_lane)
                : 0;
            const int32_t av1 = a_valid[half]
                ? *reinterpret_cast<const int32_t*>(xq + x_base + 16 + k_lane)
                : 0;
            gfx90a_i32x4 cv{0, 0, 0, 0};
            cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
                av0, bv0, cv, 0, 0, 0);
            cv = __builtin_amdgcn_mfma_i32_16x16x16i8(
                av1, bv1, cv, 0, 0, 0);
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
              const uint32_t assignment =
                  half * 16 + (lane >> 4) * 4 + r;
              const float xs = __shfl(
                  x_scale_lane, assignment, kFp4ExpertWave);
              acc[half][r] +=
                  static_cast<float>(cv[r]) * xs * weight_s * 0.5f;
            }
          }
        }
      }
#pragma unroll
      for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          const uint32_t index =
              lane * (kAssignmentHalves * 4) + half * 4 + r;
          tile_partial[split][index] = acc[half][r];
        }
      }
      __syncthreads();
      if (split == 0) {
#pragma unroll
        for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
          for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t index =
                lane * (kAssignmentHalves * 4) + half * 4 + r;
            float total = tile_partial[0][index];
#pragma unroll
            for (uint32_t s = 1; s < kSplit; ++s) {
              total += tile_partial[s][index];
            }
            if (assignment_valid[half][r] && expert_id >= 0 &&
                expert_id < static_cast<int32_t>(E)) {
              const size_t output_assignment =
                  static_cast<size_t>(tokens[half][r]) * T +
                  slots[half][r];
              const uint32_t assignment =
                  half * 16 + (lane >> 4) * 4 + r;
              const float routed_weight = __shfl(
                  assignment_weight_lane, assignment, kFp4ExpertWave);
              partial[output_assignment * N + row] =
                  total * routed_weight;
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 2,
          uint32_t kAssignments = 64>
struct Gfx90aFp4Mfma64ExpertPersistentDownOracle {
  static void run_partial(const tvm::ffi::TensorView xq,
                          const tvm::ffi::TensorView x_scale,
                          const tvm::ffi::TensorView weight,
                          const tvm::ffi::TensorView weight_scale,
                          const tvm::ffi::TensorView sorted_ids,
                          const tvm::ffi::TensorView active_experts,
                          const tvm::ffi::TensorView block_starts,
                          const tvm::ffi::TensorView block_counts,
                          const tvm::ffi::TensorView num_active,
                          const tvm::ffi::TensorView topk_weights,
                          const tvm::ffi::TensorView partial) {
    using namespace host;
    LaunchKernel(kBlocks, kSplit * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_mfma64_expert_persistent_down_kernel<
            E, M, T, N, K, kBlocks, kSplit, kAssignments>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(active_experts.data_ptr()),
        static_cast<const int32_t*>(block_starts.data_ptr()),
        static_cast<const int32_t*>(block_counts.data_ptr()),
        static_cast<const int32_t*>(num_active.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }

  static void reduce(const tvm::ffi::TensorView partial,
                     const tvm::ffi::TensorView out) {
    using namespace host;
    constexpr uint32_t kThreads = 256;
    LaunchKernel((M * N + kThreads - 1) / kThreads, kThreads, out.device())(
        gfx90a_fp4_expert_down_reduce_kernel<M, T, N>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(partial.data_ptr()));
  }
};

}  // namespace sglang
