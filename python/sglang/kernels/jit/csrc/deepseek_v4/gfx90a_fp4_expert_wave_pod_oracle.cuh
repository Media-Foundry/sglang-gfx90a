#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

// Standalone-only experiment.  Four waves in a CTA consume up to four A4
// blocks belonging to the same expert.  Every wave keeps the production A4
// accumulator and reduction tree.  There is deliberately no LDS payload,
// cross-wave communication, barrier, counter, or dynamic task queue: the
// experiment asks whether simultaneous ordinary global loads naturally hit in
// gfx90a's per-CU caches.
constexpr uint32_t kGfx90aWavePodWaves = 4;

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows>
__global__ void __launch_bounds__(kGfx90aWavePodWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_gate_wave_pod_oracle_kernel(
        bf16_t* __restrict__ out, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ pod_blocks,
        const int32_t* __restrict__ num_pods, float limit) {
  static_assert(kAssignments == 4, "wave-pod oracle is intentionally A4-only");
  static_assert(kRows == 2, "wave-pod gate preserves the TP4 R2 shape");
  constexpr uint32_t kTiles = (I + kRows - 1) / kRows;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t pods = static_cast<uint32_t>(max(num_pods[0], 0));

  // A CTA, rather than a wave, owns (pod,row-tile).  wave selects one A4
  // block from that expert's pod.  All waves therefore issue the same weight
  // rows at approximately the same time while remaining mathematically
  // independent.
  for (uint32_t task = blockIdx.x; task < pods * kTiles;
       task += gridDim.x) {
    const uint32_t pod = task / kTiles;
    const uint32_t row0 = (task % kTiles) * kRows;
    const int32_t expert_block =
        pod_blocks[static_cast<size_t>(pod) * kGfx90aWavePodWaves + wave];
    if (expert_block < 0) continue;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);

    uint32_t tokens[kAssignments];
    uint32_t slots[kAssignments];
    bool assignment_valid[kAssignments];
    float gate_acc[kAssignments][kRows] = {};
    float up_acc[kAssignments][kRows] = {};
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[static_cast<size_t>(expert_block) * kAssignments +
                     assignment]);
      tokens[assignment] = encoded & 0x00ffffffu;
      slots[assignment] = encoded >> 24;
      assignment_valid[assignment] =
          tokens[assignment] < M && slots[assignment] < T;
    }

    for (uint32_t group = lane; group < K / 32; group += kFp4ExpertWave) {
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t local_row = row0 + r;
        if (local_row >= I) continue;
        const uint32_t gate_row = local_row;
        const uint32_t up_row = I + local_row;
        const size_t gate_base =
            (static_cast<size_t>(expert) * (2 * I) + gate_row) * (K / 2) +
            group * 16;
        const size_t up_base =
            (static_cast<size_t>(expert) * (2 * I) + up_row) * (K / 2) +
            group * 16;
        const float gate_scale = gfx90a_e8m0_value(
            weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                expert, gate_row, group)]);
        const float up_scale = gfx90a_e8m0_value(
            weight_scale[gfx90a_gate_up_scale_offset<E, I, K>(
                expert, up_row, group)]);
        int32_t gate_i8[8];
        int32_t up_i8[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          gate_i8[j] = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + gate_base + j * 2));
          up_i8[j] = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + up_base + j * 2));
        }
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const uint32_t token = tokens[assignment];
          const size_t xq_group =
              static_cast<size_t>(token) * (K / 32) + group;
          gate_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + static_cast<size_t>(token) * K + k0, gate_i8,
              x_scale[xq_group] * gate_scale * 0.5f);
          up_acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + static_cast<size_t>(token) * K + k0, up_i8,
              x_scale[xq_group] * up_scale * 0.5f);
        }
      }
    }

#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      if (!assignment_valid[assignment]) continue;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        float gate_sum = gate_acc[assignment][r];
        float up_sum = up_acc[assignment][r];
#pragma unroll
        for (uint32_t offset = 32; offset > 0; offset >>= 1) {
          gate_sum += __shfl_down(gate_sum, offset, kFp4ExpertWave);
          up_sum += __shfl_down(up_sum, offset, kFp4ExpertWave);
        }
        if (lane == 0 && row0 + r < I) {
          const float gate = fminf(gate_sum, limit);
          const float up = fmaxf(-limit, fminf(up_sum, limit));
          const float activated = gate / (1.0f + expf(-gate));
          const size_t output_assignment =
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          out[output_assignment * I + row0 + r] =
              cast<bf16_t>(activated * up);
        }
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kRows>
__global__ void __launch_bounds__(kGfx90aWavePodWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_down_wave_pod_oracle_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ pod_blocks,
        const int32_t* __restrict__ num_pods,
        const float* __restrict__ topk_weights) {
  static_assert(kAssignments == 4, "wave-pod oracle is intentionally A4-only");
  static_assert(kRows == 2, "wave-pod down preserves the TP4 R2 shape");
  static_assert(K % 32 == 0 && K / 32 == 16,
                "strict TP4 local-I512 uses sixteen group-32 lanes");
  constexpr uint32_t kSubgroupWidth = 16;
  constexpr uint32_t kSubgroupsPerWave = 4;
  constexpr uint32_t kTiles = (N + kRows - 1) / kRows;
  constexpr uint32_t kTileGroups =
      (kTiles + kSubgroupsPerWave - 1) / kSubgroupsPerWave;
  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t pods = static_cast<uint32_t>(max(num_pods[0], 0));

  for (uint32_t task = blockIdx.x; task < pods * kTileGroups;
       task += gridDim.x) {
    const uint32_t pod = task / kTileGroups;
    const uint32_t tile = (task % kTileGroups) * kSubgroupsPerWave + subgroup;
    if (tile >= kTiles) continue;
    const uint32_t row0 = tile * kRows;
    const int32_t expert_block =
        pod_blocks[static_cast<size_t>(pod) * kGfx90aWavePodWaves + wave];
    if (expert_block < 0) continue;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    if (expert_id < 0 || expert_id >= static_cast<int32_t>(E)) continue;
    const uint32_t expert = static_cast<uint32_t>(expert_id);

    uint32_t tokens[kAssignments];
    uint32_t slots[kAssignments];
    bool assignment_valid[kAssignments];
    float acc[kAssignments][kRows] = {};
#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[static_cast<size_t>(expert_block) * kAssignments +
                     assignment]);
      tokens[assignment] = encoded & 0x00ffffffu;
      slots[assignment] = encoded >> 24;
      assignment_valid[assignment] =
          tokens[assignment] < M && slots[assignment] < T;
    }

    for (uint32_t group = subgroup_lane; group < K / 32;
         group += kSubgroupWidth) {
      const uint32_t k0 = group * 32;
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t row = row0 + r;
        if (row >= N) continue;
        const size_t weight_base =
            (static_cast<size_t>(expert) * N + row) * (K / 2) + group * 16;
        const float scale = gfx90a_e8m0_value(
            weight_scale[gfx90a_down_scale_offset<E, N, K>(
                expert, row, group)]);
        int32_t weight_i8[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          weight_i8[j] = gfx90a_fp4_pack4_i8(
              *reinterpret_cast<const uint16_t*>(weight + weight_base + j * 2));
        }
#pragma unroll
        for (uint32_t assignment = 0; assignment < kAssignments;
             ++assignment) {
          if (!assignment_valid[assignment]) continue;
          const size_t input_assignment =
              static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
          const size_t xq_group = input_assignment * (K / 32) + group;
          acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
              xq + input_assignment * K + k0, weight_i8,
              x_scale[xq_group] * scale * 0.5f);
        }
      }
    }

#pragma unroll
    for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
      if (!assignment_valid[assignment]) continue;
      const size_t output_assignment =
          static_cast<size_t>(tokens[assignment]) * T + slots[assignment];
      const float routed_weight = topk_weights[output_assignment];
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
        for (uint32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
          acc[assignment][r] +=
              __shfl_down(acc[assignment][r], offset, kSubgroupWidth);
        }
        if (subgroup_lane == 0 && row0 + r < N) {
          partial[output_assignment * N + row0 + r] =
              acc[assignment][r] * routed_weight;
        }
      }
    }
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t I, uint32_t K,
          uint32_t kAssignments, uint32_t kRows, uint32_t kGateBlocks,
          uint32_t kDownBlocks>
struct Gfx90aFp4ExpertWavePodOracle {
  static constexpr uint32_t kMaxPods = E * ((M + 15) / 16);

  static void run_gate(const tvm::ffi::TensorView xq,
                       const tvm::ffi::TensorView x_scale,
                       const tvm::ffi::TensorView weight,
                       const tvm::ffi::TensorView weight_scale,
                       const tvm::ffi::TensorView sorted_ids,
                       const tvm::ffi::TensorView sorted_expert_ids,
                       const tvm::ffi::TensorView pod_blocks,
                       const tvm::ffi::TensorView num_pods,
                       const tvm::ffi::TensorView out, double limit) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 2 * I, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 2 * I, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({kMaxPods, kGfx90aWavePodWaves}).with_dtype<int32_t>().with_device(device).verify(pod_blocks);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_pods);
    TensorMatcher({M, T, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(kGateBlocks, kGfx90aWavePodWaves * kFp4ExpertWave,
                 xq.device())(
        gfx90a_fp4_expert_gate_wave_pod_oracle_kernel<
            E, M, T, I, K, kAssignments, kRows>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(pod_blocks.data_ptr()),
        static_cast<const int32_t*>(num_pods.data_ptr()),
        static_cast<float>(limit));
  }

  static void run_down(const tvm::ffi::TensorView xq,
                       const tvm::ffi::TensorView x_scale,
                       const tvm::ffi::TensorView weight,
                       const tvm::ffi::TensorView weight_scale,
                       const tvm::ffi::TensorView sorted_ids,
                       const tvm::ffi::TensorView sorted_expert_ids,
                       const tvm::ffi::TensorView pod_blocks,
                       const tvm::ffi::TensorView num_pods,
                       const tvm::ffi::TensorView topk_weights,
                       const tvm::ffi::TensorView partial) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, I}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, I / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, 4096, I / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, 4096, I / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({kMaxPods, kGfx90aWavePodWaves}).with_dtype<int32_t>().with_device(device).verify(pod_blocks);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_pods);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, T, 4096}).with_dtype<float>().with_device(device).verify(partial);
    LaunchKernel(kDownBlocks, kGfx90aWavePodWaves * kFp4ExpertWave,
                 xq.device())(
        gfx90a_fp4_expert_down_wave_pod_oracle_kernel<
            E, M, T, 4096, I, kAssignments, kRows>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(pod_blocks.data_ptr()),
        static_cast<const int32_t*>(num_pods.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
