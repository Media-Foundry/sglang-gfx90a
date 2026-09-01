#pragma once

// Standalone-only oracle for the large-prefill DSV4 routed gate/up path.
//
// Production's MFMA64 specialization means 64 assignments per expert block;
// its output tile is still only 16 intermediate columns.  A group-32 INT8
// quantization group therefore spans two independently scheduled tasks.  This
// oracle changes only that ownership: one CTA computes the adjacent I16 tiles
// sequentially, reusing the production split-K partial storage, then performs
// the existing group-32 quantization locally.  It is deliberately not wired to
// a production selector.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include "gfx90a_fp4_expert_gemv.cuh"

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

using namespace device;

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4, bool kWriteIntermediate = false>
__global__ void __launch_bounds__(kSplit * kFp4ExpertWave)
    gfx90a_fp4_mfma64_gate_quant_oracle_kernel(
        bf16_t* __restrict__ intermediate, int8_t* __restrict__ output_q,
        float* __restrict__ output_scale, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids, float limit) {
  constexpr uint32_t kAssignments = 64;
  constexpr uint32_t kAssignmentHalves = kAssignments / 16;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kTiles32 = I / 32;
  static_assert(I % 32 == 0);
  static_assert(kSplit == 4,
                "oracle preserves the production MFMA64 split-4 order");

  // Same split storage as one production I16 task.  The two I16 tiles execute
  // sequentially, so this storage is not doubled.  Activated BF16 values are
  // retained only long enough to form the local group-32 quantization group.
  __shared__ float gate_partial[kSplit][kAssignments * 16];
  __shared__ float up_partial[kSplit][kAssignments * 16];
  __shared__ bf16_t activated_bf16[kAssignments][32];

  const uint32_t lane = threadIdx.x & 63;
  const uint32_t split = threadIdx.x >> 6;
  const uint32_t matrix_index = lane & 15;
  const uint32_t assignment_quad = lane >> 4;
  const uint32_t k_lane = assignment_quad * 4;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks = (valid + kAssignments - 1) / kAssignments;

  for (uint32_t task = blockIdx.x; task < valid_blocks * kTiles32;
       task += kBlocks) {
    const uint32_t expert_block = task / kTiles32;
    const uint32_t tile32 = task % kTiles32;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    const uint32_t expert = static_cast<uint32_t>(max(expert_id, 0));

    // Each lane in a wave loads one encoded assignment and broadcasts it to
    // the MFMA output lanes, exactly like production kBroadcastScales=1.
    const uint32_t assignment_lane = lane < kAssignments
        ? static_cast<uint32_t>(
              sorted_ids[expert_block * kAssignments + lane])
        : static_cast<uint32_t>(M);

    uint32_t tokens[kAssignmentHalves][4];
    uint32_t slots[kAssignmentHalves][4];
    bool assignment_valid[kAssignmentHalves][4];
#pragma unroll
    for (uint32_t half = 0; half < kAssignmentHalves; ++half) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) {
        const uint32_t assignment = half * 16 + assignment_quad * 4 + r;
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
      const uint32_t assignment = half * 16 + matrix_index;
      const uint32_t encoded =
          __shfl(assignment_lane, assignment, kFp4ExpertWave);
      const uint32_t token = encoded & 0x00ffffffu;
      const uint32_t slot = encoded >> 24;
      a_valid[half] = token < M && slot < T;
      a_tokens[half] = a_valid[half] ? token : 0;
    }

    // Sequentially evaluate the two production-identical I16 tiles.  The
    // explicit CTA barrier at the end of each half protects reuse of the
    // split partial arrays; it is not a cross-CTA publication protocol.
#pragma unroll
    for (uint32_t tile_half = 0; tile_half < 2; ++tile_half) {
      const uint32_t local_row =
          tile32 * 32 + tile_half * 16 + matrix_index;
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
                    expert, tile32 * 32 + tile_half * 16 + lane, group)]);
            up_scale_lane = gfx90a_e8m0_value(
                weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                    expert, I + tile32 * 32 + tile_half * 16 + lane,
                    group)]);
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
            const size_t x_base =
                static_cast<size_t>(a_tokens[half]) * K + k0;
            const int32_t av0 = a_valid[half]
                ? *reinterpret_cast<const int32_t*>(xq + x_base + k_lane)
                : 0;
            const int32_t av1 = a_valid[half]
                ? *reinterpret_cast<const int32_t*>(xq + x_base + 16 +
                                                     k_lane)
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
                  half * 16 + assignment_quad * 4 + r;
              const float xs = __shfl(x_scale_lane, assignment,
                                      kFp4ExpertWave);
              gate_acc[half][r] +=
                  static_cast<float>(gate_cv[r]) * xs * gate_scale * 0.5f;
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
            const uint32_t assignment =
                half * 16 + assignment_quad * 4 + r;
            if (assignment_valid[half][r] && expert_id >= 0 &&
                expert_id < static_cast<int32_t>(E)) {
              gate = fminf(gate, limit);
              up = fmaxf(-limit, fminf(up, limit));
              const float activated = gate / (1.0f + expf(-gate));
              // This BF16 boundary is part of the model: the standalone
              // quantizer reloads BF16, not the pre-rounded FP32 product.
              activated_bf16[assignment][tile_half * 16 + matrix_index] =
                  cast<bf16_t>(activated * up);
            }
          }
        }
      }
      __syncthreads();
    }

    // Sixteen 16-lane subgroups process sixteen assignments at a time.  Four
    // uniform rounds cover all 64 assignments without any inter-CTA state.
    const uint32_t subgroup = threadIdx.x / 16;
    const uint32_t subgroup_lane = threadIdx.x & 15;
#pragma unroll
    for (uint32_t round = 0; round < 4; ++round) {
      const uint32_t assignment = round * 16 + subgroup;
      const uint32_t encoded =
          static_cast<uint32_t>(
              sorted_ids[expert_block * kAssignments + assignment]);
      const uint32_t token = encoded & 0x00ffffffu;
      const uint32_t slot = encoded >> 24;
      const bool is_valid = token < M && slot < T && expert_id >= 0 &&
          expert_id < static_cast<int32_t>(E);
      const float x0 = is_valid
          ? cast<float>(activated_bf16[assignment][subgroup_lane])
          : 0.0f;
      const float x1 = is_valid
          ? cast<float>(activated_bf16[assignment][16 + subgroup_lane])
          : 0.0f;
      float absmax = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
      for (uint32_t offset = 8; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_xor(absmax, offset, 16));
      }
      const float scale = fmaxf(absmax, 1.0e-10f) / 127.0f;
      const float q0 = fmaxf(-128.0f, fminf(127.0f, x0 / scale));
      const float q1 = fmaxf(-128.0f, fminf(127.0f, x1 / scale));
      if (is_valid) {
        const size_t output_assignment =
            static_cast<size_t>(token) * T + slot;
        const uint32_t row0 = tile32 * 32 + subgroup_lane;
        output_q[output_assignment * I + row0] =
            static_cast<int8_t>(q0);
        output_q[output_assignment * I + row0 + 16] =
            static_cast<int8_t>(q1);
        if constexpr (kWriteIntermediate) {
          intermediate[output_assignment * I + row0] =
              activated_bf16[assignment][subgroup_lane];
          intermediate[output_assignment * I + row0 + 16] =
              activated_bf16[assignment][16 + subgroup_lane];
        }
        if (subgroup_lane == 0) {
          output_scale[output_assignment * (I / 32) + tile32] = scale;
        }
      }
    }
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kBlocks, uint32_t kSplit = 4>
struct Gfx90aFp4Mfma64GateQuantOracle {
  static void verify(const tvm::ffi::TensorView& xq,
                     const tvm::ffi::TensorView& x_scale,
                     const tvm::ffi::TensorView& weight,
                     const tvm::ffi::TensorView& weight_scale,
                     const tvm::ffi::TensorView& sorted_ids,
                     const tvm::ffi::TensorView& sorted_expert_ids,
                     const tvm::ffi::TensorView& num_valid_ids,
                     const tvm::ffi::TensorView& output_q,
                     const tvm::ffi::TensorView& output_scale) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32})
        .with_dtype<float>()
        .with_device(device)
        .verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({E, 2 * I, K / 32})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(weight_scale);
    auto sorted_count = SymbolicSize{"sorted_count"};
    auto expert_block_count = SymbolicSize{"expert_block_count"};
    TensorMatcher({sorted_count})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(sorted_ids);
    TensorMatcher({expert_block_count})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(sorted_expert_ids);
    TensorMatcher({2})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(num_valid_ids);
    TensorMatcher({M, T, I})
        .with_dtype<int8_t>()
        .with_device(device)
        .verify(output_q);
    TensorMatcher({M, T, I / 32})
        .with_dtype<float>()
        .with_device(device)
        .verify(output_scale);
  }

  static void run_quant(const tvm::ffi::TensorView xq,
                        const tvm::ffi::TensorView x_scale,
                        const tvm::ffi::TensorView weight,
                        const tvm::ffi::TensorView weight_scale,
                        const tvm::ffi::TensorView sorted_ids,
                        const tvm::ffi::TensorView sorted_expert_ids,
                        const tvm::ffi::TensorView num_valid_ids,
                        const tvm::ffi::TensorView output_q,
                        const tvm::ffi::TensorView output_scale,
                        double limit) {
    verify(xq, x_scale, weight, weight_scale, sorted_ids,
           sorted_expert_ids, num_valid_ids, output_q, output_scale);
    host::LaunchKernel(kBlocks, kSplit * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_mfma64_gate_quant_oracle_kernel<
            E, M, T, I, K, kBlocks, kSplit, false>,
        static_cast<bf16_t*>(nullptr),
        static_cast<int8_t*>(output_q.data_ptr()),
        static_cast<float*>(output_scale.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<float>(limit));
  }

  static void run_debug(const tvm::ffi::TensorView xq,
                        const tvm::ffi::TensorView x_scale,
                        const tvm::ffi::TensorView weight,
                        const tvm::ffi::TensorView weight_scale,
                        const tvm::ffi::TensorView sorted_ids,
                        const tvm::ffi::TensorView sorted_expert_ids,
                        const tvm::ffi::TensorView num_valid_ids,
                        const tvm::ffi::TensorView intermediate,
                        const tvm::ffi::TensorView output_q,
                        const tvm::ffi::TensorView output_scale,
                        double limit) {
    using namespace host;
    verify(xq, x_scale, weight, weight_scale, sorted_ids,
           sorted_expert_ids, num_valid_ids, output_q, output_scale);
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, I})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(intermediate);
    host::LaunchKernel(kBlocks, kSplit * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_mfma64_gate_quant_oracle_kernel<
            E, M, T, I, K, kBlocks, kSplit, true>,
        static_cast<bf16_t*>(intermediate.data_ptr()),
        static_cast<int8_t*>(output_q.data_ptr()),
        static_cast<float*>(output_scale.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
        static_cast<float>(limit));
  }
};

}  // namespace sglang
