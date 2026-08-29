#pragma once

#include "gfx90a_fp4_expert_down_row_prefetch_oracle.cuh"

namespace sglang {

// Oracle-only TP4 A4/R2/W8 grouped down kernel.  A CTA's 32 subgroup tasks
// address consecutive output rows of one expert/A4 block.  Stage that A4
// activation tile once in LDS, while preserving the row-prefetch weight path,
// subgroup16 dot order, partial layout and reduction order.
template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves,
          uint32_t kBlocks, uint32_t kPrepacked>
__global__ void __launch_bounds__(kNumWaves * kFp4ExpertWave)
    gfx90a_fp4_expert_down_cta_xq_staging_kernel(
        float* __restrict__ partial, const int8_t* __restrict__ xq,
        const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ sorted_expert_ids,
        const int32_t* __restrict__ num_valid_ids,
        const float* __restrict__ topk_weights) {
  static_assert(kPrepacked == 2);
  static_assert(kAssignments == 4);
  constexpr uint32_t kRows = 2;
  constexpr uint32_t kGroups = K / 32;
  constexpr uint32_t kSubgroupWidth = 16;
  constexpr uint32_t kSubgroupsPerWave = kFp4ExpertWave / kSubgroupWidth;
  constexpr uint32_t kSubgroupsPerBlock = kNumWaves * kSubgroupsPerWave;
  constexpr uint32_t kTilesPerExpertBlock = (N + kRows - 1) / kRows;
  static_assert(K % 32 == 0 && kGroups >= 16);
  static_assert(kTilesPerExpertBlock % kSubgroupsPerBlock == 0,
                "CTA tasks must not cross an expert block");
  static_assert((kBlocks * kSubgroupsPerBlock) % kTilesPerExpertBlock == 0,
                "grid stride must preserve expert-block alignment");

  __shared__ uint32_t pair_lut[256];
  __shared__ uint32_t staged_tokens[kAssignments];
  __shared__ uint32_t staged_slots[kAssignments];
  __shared__ uint32_t staged_valid[kAssignments];
  __shared__ int8_t staged_xq[kAssignments * K];
  __shared__ float staged_scale[kAssignments * kGroups];

  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x % kFp4ExpertWave;
  const uint32_t subgroup = lane / kSubgroupWidth;
  const uint32_t subgroup_lane = lane % kSubgroupWidth;
  const uint32_t wave = threadIdx.x / kFp4ExpertWave;
  const uint32_t block_subgroup = wave * kSubgroupsPerWave + subgroup;
  const uint32_t valid = max(num_valid_ids[0], 0);
  const uint32_t valid_blocks =
      (valid + kAssignments - 1) / kAssignments;
  const uint32_t total_tasks = valid_blocks * kTilesPerExpertBlock;

  for (uint32_t cta_base = blockIdx.x * kSubgroupsPerBlock;
       cta_base < total_tasks;
       cta_base += gridDim.x * kSubgroupsPerBlock) {
    const uint32_t expert_block = cta_base / kTilesPerExpertBlock;
    if (threadIdx.x < kAssignments) {
      const uint32_t encoded = static_cast<uint32_t>(
          sorted_ids[expert_block * kAssignments + threadIdx.x]);
      const uint32_t token = encoded & 0x00ffffffu;
      const uint32_t slot = encoded >> 24;
      staged_tokens[threadIdx.x] = token;
      staged_slots[threadIdx.x] = slot;
      staged_valid[threadIdx.x] = token < M && slot < T;
    }
    __syncthreads();

    for (uint32_t linear = threadIdx.x; linear < kAssignments * K;
         linear += blockDim.x) {
      const uint32_t assignment = linear / K;
      const uint32_t k = linear - assignment * K;
      if (staged_valid[assignment]) {
        const size_t input_assignment =
            static_cast<size_t>(staged_tokens[assignment]) * T +
            staged_slots[assignment];
        staged_xq[linear] = xq[input_assignment * K + k];
      } else {
        staged_xq[linear] = 0;
      }
    }
    for (uint32_t linear = threadIdx.x;
         linear < kAssignments * kGroups; linear += blockDim.x) {
      const uint32_t assignment = linear / kGroups;
      const uint32_t group = linear - assignment * kGroups;
      if (staged_valid[assignment]) {
        const size_t input_assignment =
            static_cast<size_t>(staged_tokens[assignment]) * T +
            staged_slots[assignment];
        staged_scale[linear] = x_scale[input_assignment * kGroups + group];
      } else {
        staged_scale[linear] = 0.0f;
      }
    }
    __syncthreads();

    const uint32_t task = cta_base + block_subgroup;
    const uint32_t row0 = (task % kTilesPerExpertBlock) * kRows;
    const int32_t expert_id = sorted_expert_ids[expert_block];
    const bool task_active =
        task < total_tasks && expert_id >= 0 &&
        expert_id < static_cast<int32_t>(E);

    if (task_active) {
      const uint32_t expert = static_cast<uint32_t>(expert_id);
      float acc[kAssignments][kRows] = {};
      for (uint32_t group = subgroup_lane; group < kGroups;
           group += kSubgroupWidth) {
        const uint32_t k0 = group * 32;
        Gfx90aFp4PackedRow16 packed[kRows];
        uint8_t scale_raw[kRows];
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          const uint32_t row = row0 + r;
          if (row < N) {
            const size_t weight_base =
                (static_cast<size_t>(expert) * N + row) * (K / 2) + group * 16;
            packed[r].vector =
                *reinterpret_cast<const uint4*>(weight + weight_base);
            scale_raw[r] = weight_scale[
                gfx90a_down_scale_offset<E, N, K>(expert, row, group)];
          } else {
            packed[r].vector = make_uint4(0, 0, 0, 0);
            scale_raw[r] = 0;
          }
        }
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
          if (row0 + r >= N) continue;
          const float scale = gfx90a_e8m0_value(scale_raw[r]);
          int32_t weight_i8[8];
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            weight_i8[j] =
                gfx90a_fp4_pack4_i8_lds(packed[r].halves[j], pair_lut);
          }
#pragma unroll
          for (uint32_t assignment = 0; assignment < kAssignments;
               ++assignment) {
            if (!staged_valid[assignment]) continue;
            acc[assignment][r] += gfx90a_fp4_dot32_i8_prepacked(
                staged_xq + assignment * K + k0, weight_i8,
                staged_scale[assignment * kGroups + group] * scale * 0.5f);
          }
        }
      }

#pragma unroll
      for (uint32_t assignment = 0; assignment < kAssignments; ++assignment) {
        if (!staged_valid[assignment]) continue;
        const size_t output_assignment =
            static_cast<size_t>(staged_tokens[assignment]) * T +
            staged_slots[assignment];
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
    __syncthreads();
  }
}

template <uint32_t E, uint32_t M, uint32_t T, uint32_t N, uint32_t K,
          uint32_t kAssignments, uint32_t kNumWaves, uint32_t kBlocks,
          uint32_t kPrepacked = 2>
struct Gfx90aFp4ExpertDownCtaXqStagingOracle {
  static void run_partial(const tvm::ffi::TensorView xq,
                          const tvm::ffi::TensorView x_scale,
                          const tvm::ffi::TensorView weight,
                          const tvm::ffi::TensorView weight_scale,
                          const tvm::ffi::TensorView sorted_ids,
                          const tvm::ffi::TensorView sorted_expert_ids,
                          const tvm::ffi::TensorView num_valid_ids,
                          const tvm::ffi::TensorView topk_weights,
                          const tvm::ffi::TensorView partial) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, T, K}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({M, T, K / 32}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(num_valid_ids);
    TensorMatcher({M, T}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({M, T, N}).with_dtype<float>().with_device(device).verify(partial);
    LaunchKernel(kBlocks, kNumWaves * kFp4ExpertWave, xq.device())(
        gfx90a_fp4_expert_down_cta_xq_staging_kernel<
            E, M, T, N, K, kAssignments, kNumWaves, kBlocks, kPrepacked>,
        static_cast<float*>(partial.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()),
        static_cast<const int32_t*>(sorted_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_expert_ids.data_ptr()),
        static_cast<const int32_t*>(num_valid_ids.data_ptr()),
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
