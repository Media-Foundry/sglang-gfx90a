#pragma once

#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <stdexcept>

#include "dsv4_unified_sparse_decode_ck.cuh"

// Oracle-only gamma-one target-verify kernel.  Nothing in the production
// selector includes this header.  It intentionally accepts only M64, whose
// row order is [anchor_0, draft_0, anchor_1, draft_1, ...].
namespace ck_tile::dsv4 {

template <int Tile>
__device__ bool pair_slots_prefix_compatible(
    const UnifiedSparseDecodeArgs& args, const int begin0, const int begin1,
    const int valid0, const int valid1, int slots[2][Tile],
    int* compatible_flag) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) *compatible_flag = 1;
  if (tid < Tile) {
    slots[0][tid] = tid < valid0 ? args.kv_indices[begin0 + tid] : -1;
    slots[1][tid] = tid < valid1 ? args.kv_indices[begin1 + tid] : -1;
  }
  __syncthreads();
  const int common = min(valid0, valid1);
  if (tid < common && slots[0][tid] != slots[1][tid]) {
    atomicExch(compatible_flag, 0);
  }
  __syncthreads();
  return *compatible_flag != 0;
}

// Eight waves form two four-wave groups.  A group retains the exact production
// split2 mapping for one row: the same wave owns the same K128 QK fragment and
// D128 PV fragment, and the same four-wave FP32 sum is evaluated in wave order.
// Both groups consume one LDS KV tile when their physical slot vectors match.
// Different/tail tiles use independent LDS banks, preserving each row's tile
// and online-softmax order.
__global__ __launch_bounds__(kWaveSize * 8) void
unified_sparse_decode_d512_mfma_pair_split_core_kernel(
    UnifiedSparseDecodeArgs args, UnifiedSparseDecodeWorkspace workspace,
    int splits) {
  using WarpGemm = WarpGemmMfmaBf16Bf16F32M16N16K16;
  using AVec =
      ext_vector_t<bf16_t, WarpGemm::AWarpTensor::get_thread_buffer_size()>;
  using BVec =
      ext_vector_t<bf16_t, WarpGemm::BWarpTensor::get_thread_buffer_size()>;
  using CVec =
      ext_vector_t<float, WarpGemm::CWarpTensor::get_thread_buffer_size()>;

  constexpr int kTile = 16;
  constexpr int kWavesPerRow = 4;
  constexpr int kRows = 2;
  constexpr int kCtaWaves = kRows * kWavesPerRow;
  constexpr int kCtaThreads = kCtaWaves * kWaveSize;
  constexpr int kOutputTilesPerWave =
      (kHeadDim / kTile) / kWavesPerRow;
  constexpr int kValuesPerLane = 4;
  constexpr int kKvLdsStride = kHeadDim + 4;
  constexpr int kQFragmentsPerWave =
      (kHeadDim / kWavesPerRow) / kTile;

  __shared__ int kv_slots[kRows][kTile];
  __shared__ int compatible_slots;
  __shared__ bf16_t kv_tile[kRows][kTile * kKvLdsStride];
  __shared__ float
      score_partial[kRows][kWavesPerRow][kLocalHeads * kTile];
  __shared__ bf16_t probabilities[kRows][kLocalHeads * kTile];
  __shared__ float alpha_shared[kRows][kLocalHeads];

  const int pair = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kWaveSize;
  const int row = wave / kWavesPerRow;
  const int row_wave = wave % kWavesPerRow;
  const int row_tid = tid % (kWavesPerRow * kWaveSize);
  const int lane = tid % kWaveSize;
  const int matrix_lane = lane % kTile;
  const int k_group = lane / kTile;
  const int token = pair * kRows + row;

  int begin[kRows];
  int end[kRows];
  int first_tile[kRows];
  int last_tile[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int token_r = pair * kRows + r;
    begin[r] = args.kv_indptr[token_r];
    end[r] = args.kv_indptr[token_r + 1];
    const int total_tiles = (end[r] - begin[r] + kTile - 1) / kTile;
    const int tiles_per_split = (total_tiles + splits - 1) / splits;
    first_tile[r] = split * tiles_per_split;
    last_tile[r] = min(total_tiles, first_tile[r] + tiles_per_split);
  }
  const int row_steps = last_tile[row] - first_tile[row];
  const int pair_steps =
      max(last_tile[0] - first_tile[0], last_tile[1] - first_tile[1]);

  // Identical to production PreloadQ=true, with row_wave replacing wave.
  AVec q_fragments[kQFragmentsPerWave];
#pragma unroll
  for (int fragment = 0; fragment < kQFragmentsPerWave; ++fragment) {
    const int k_begin =
        row_wave * (kHeadDim / kWavesPerRow) + fragment * kTile;
#pragma unroll
    for (int k = 0; k < kValuesPerLane; ++k) {
      const int d = k_begin + k_group * kValuesPerLane + k;
      const long q_offset =
          (static_cast<long>(token) * args.heads + matrix_lane) * kHeadDim + d;
      q_fragments[fragment][k] = bit_cast<bf16_t>(args.q[q_offset]);
    }
  }

  const int softmax_head = row_tid / kTile;
  const int softmax_key = row_tid % kTile;
  float softmax_max = -__builtin_inff();
  float softmax_norm = 0.0f;
  float accumulator[kOutputTilesPerWave][kValuesPerLane];
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
#pragma unroll
    for (int n = 0; n < kOutputTilesPerWave; ++n) {
      accumulator[n][i] = 0.0f;
    }
  }

  for (int step = 0; step < pair_steps; ++step) {
    int tile_begin[kRows];
    int valid_keys[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const bool active = step < last_tile[r] - first_tile[r];
      tile_begin[r] = begin[r] + (first_tile[r] + step) * kTile;
      valid_keys[r] = active ? min(kTile, end[r] - tile_begin[r]) : 0;
    }

    // The shorter row may be a strict prefix of the longer row.  Loading the
    // longer tile once remains exact: the shorter row's valid_keys mask gives
    // the extra key probability zero before PV.
    const bool share_tile = pair_slots_prefix_compatible<kTile>(
        args, tile_begin[0], tile_begin[1], valid_keys[0], valid_keys[1],
        kv_slots, &compatible_slots);
    const int shared_source = valid_keys[1] > valid_keys[0] ? 1 : 0;

    // All 512 threads cooperatively load the anchor bank.  The draft bank is
    // loaded only when it cannot alias that exact physical-slot vector.
    constexpr int kTileElements = kTile * kHeadDim;
    for (int linear = tid; linear < kTileElements; linear += kCtaThreads) {
      const int key = linear / kHeadDim;
      const int d = linear % kHeadDim;
      const int slot0 = kv_slots[share_tile ? shared_source : 0][key];
      kv_tile[0][key * kKvLdsStride + d] =
          slot0 >= 0 && slot0 < args.pool_slots
              ? bit_cast<bf16_t>(
                    args.unified_kv[static_cast<long>(slot0) * kHeadDim + d])
              : type_convert<bf16_t>(0.0f);
      if (!share_tile) {
        const int slot1 = kv_slots[1][key];
        kv_tile[1][key * kKvLdsStride + d] =
            slot1 >= 0 && slot1 < args.pool_slots
                ? bit_cast<bf16_t>(
                      args.unified_kv[static_cast<long>(slot1) * kHeadDim + d])
                : type_convert<bf16_t>(0.0f);
      }
    }
    __syncthreads();

    const int kv_bank = row == 1 && !share_tile ? 1 : 0;
    typename WarpGemm::CWarpTensor score_tensor;
    score_tensor.get_thread_buffer().template set_as<CVec>(number<0>{},
                                                            CVec{0.0f});

    const int qk_begin = row_wave * (kHeadDim / kWavesPerRow);
    const int qk_end = qk_begin + kHeadDim / kWavesPerRow;
    for (int k_begin = qk_begin; k_begin < qk_end; k_begin += kTile) {
      AVec q_vec{};
      BVec k_vec{};
#pragma unroll
      for (int k = 0; k < kValuesPerLane; ++k) {
        const int d = k_begin + k_group * kValuesPerLane + k;
        q_vec[k] = q_fragments[(k_begin - qk_begin) / kTile][k];
        k_vec[k] = matrix_lane < valid_keys[row]
                       ? kv_tile[kv_bank][matrix_lane * kKvLdsStride + d]
                       : type_convert<bf16_t>(0.0f);
      }
      typename WarpGemm::AWarpTensor q_tensor;
      typename WarpGemm::BWarpTensor k_tensor;
      q_tensor.get_thread_buffer().template set_as<AVec>(number<0>{}, q_vec);
      k_tensor.get_thread_buffer().template set_as<BVec>(number<0>{}, k_vec);
      WarpGemm{}(score_tensor, q_tensor, k_tensor);
    }

    const CVec score_vec =
        score_tensor.get_thread_buffer().template get_as<CVec>()[number<0>{}];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const int head = k_group * kValuesPerLane + i;
      score_partial[row][row_wave][head * kTile + matrix_lane] = score_vec[i];
    }
    __syncthreads();

    const bool row_active = step < row_steps;
    float alpha = 1.0f;
    if (row_active) {
      float score = 0.0f;
#pragma unroll
      for (int qk_wave = 0; qk_wave < kWavesPerRow; ++qk_wave) {
        score += score_partial[row][qk_wave]
                              [softmax_head * kTile + softmax_key];
      }
      score = softmax_key < valid_keys[row]
                  ? score * args.softmax_scale
                  : -__builtin_inff();

      float block_max = score;
#pragma unroll
      for (int offset = kTile / 2; offset > 0; offset >>= 1) {
        block_max = fmaxf(block_max,
                          __shfl_down(block_max, offset, kTile));
      }
      block_max = __shfl(block_max, 0, kTile);

      const float next_max = fmaxf(softmax_max, block_max);
      alpha = expf(softmax_max - next_max);
      const float weight = softmax_key < valid_keys[row]
                               ? expf(score - next_max)
                               : 0.0f;
      float block_sum = weight;
#pragma unroll
      for (int offset = kTile / 2; offset > 0; offset >>= 1) {
        block_sum += __shfl_down(block_sum, offset, kTile);
      }
      block_sum = __shfl(block_sum, 0, kTile);
      softmax_norm = softmax_norm * alpha + block_sum;
      softmax_max = next_max;
      probabilities[row][softmax_head * kTile + softmax_key] =
          type_convert<bf16_t>(weight);
      if (softmax_key == 0) alpha_shared[row][softmax_head] = alpha;
    } else {
      probabilities[row][softmax_head * kTile + softmax_key] =
          type_convert<bf16_t>(0.0f);
      if (softmax_key == 0) alpha_shared[row][softmax_head] = 1.0f;
    }
    __syncthreads();

    typename WarpGemm::CWarpTensor value_tensors[kOutputTilesPerWave];
#pragma unroll
    for (int n = 0; n < kOutputTilesPerWave; ++n) {
      value_tensors[n].get_thread_buffer().template set_as<CVec>(number<0>{},
                                                                 CVec{0.0f});
    }
    AVec p_vec{};
#pragma unroll
    for (int k = 0; k < kValuesPerLane; ++k) {
      p_vec[k] = probabilities[row]
                              [matrix_lane * kTile +
                               k_group * kValuesPerLane + k];
    }
    typename WarpGemm::AWarpTensor p_tensor;
    p_tensor.get_thread_buffer().template set_as<AVec>(number<0>{}, p_vec);

#pragma unroll
    for (int n = 0; n < kOutputTilesPerWave; ++n) {
      BVec v_vec{};
      const int output_tile = row_wave * kOutputTilesPerWave + n;
#pragma unroll
      for (int k = 0; k < kValuesPerLane; ++k) {
        const int key = k_group * kValuesPerLane + k;
        const int d = output_tile * kTile + matrix_lane;
        v_vec[k] = key < valid_keys[row]
                       ? kv_tile[kv_bank][key * kKvLdsStride + d]
                       : type_convert<bf16_t>(0.0f);
      }
      typename WarpGemm::BWarpTensor v_tensor;
      v_tensor.get_thread_buffer().template set_as<BVec>(number<0>{}, v_vec);
      WarpGemm{}(value_tensors[n], p_tensor, v_tensor);
    }

#pragma unroll
    for (int n = 0; n < kOutputTilesPerWave; ++n) {
      const CVec value_vec = value_tensors[n]
                                 .get_thread_buffer()
                                 .template get_as<CVec>()[number<0>{}];
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        const int head = k_group * kValuesPerLane + i;
        accumulator[n][i] =
            accumulator[n][i] * alpha_shared[row][head] + value_vec[i];
      }
    }
    __syncthreads();
  }

  const std::size_t partial_row =
      (static_cast<std::size_t>(token) * splits + split) * args.heads;
  if (softmax_key == 0) {
    workspace.max_partial[partial_row + softmax_head] = softmax_max;
    workspace.norm_partial[partial_row + softmax_head] = softmax_norm;
  }
#pragma unroll
  for (int n = 0; n < kOutputTilesPerWave; ++n) {
    const int output_tile = row_wave * kOutputTilesPerWave + n;
    const int d = output_tile * kTile + matrix_lane;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const int head = k_group * kValuesPerLane + i;
      workspace.output_partial[(partial_row + head) * kHeadDim + d] =
          accumulator[n][i];
    }
  }
}

inline hipError_t launch_unified_sparse_decode_d512_pair_split2_qreg(
    const UnifiedSparseDecodeArgs& args, void* workspace_ptr,
    hipStream_t stream = nullptr) {
  if (!is_supported(args) || args.tokens != 64 || workspace_ptr == nullptr) {
    return hipErrorInvalidValue;
  }
  constexpr int kSplits = 2;
  const std::size_t rows =
      static_cast<std::size_t>(args.tokens) * kSplits * args.heads;
  auto* base = static_cast<float*>(workspace_ptr);
  const UnifiedSparseDecodeWorkspace workspace{base, base + rows,
                                                base + rows * 2};
  const dim3 core_grid(args.tokens / 2, kSplits, 1);
  const dim3 core_block(kWaveSize * 8, 1, 1);
  hipLaunchKernelGGL(unified_sparse_decode_d512_mfma_pair_split_core_kernel,
                     core_grid, core_block, 0, stream, args, workspace,
                     kSplits);
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) return status;

  const dim3 reduce_grid(args.tokens, args.heads, 1);
  const dim3 reduce_block(kThreads, 1, 1);
  hipLaunchKernelGGL(unified_sparse_decode_d512_mfma_split_reduce_kernel,
                     reduce_grid, reduce_block, 0, stream, args, workspace,
                     kSplits);
  return hipGetLastError();
}

}  // namespace ck_tile::dsv4

namespace sglang {

struct Gfx90aDsv4UnifiedSparsePairOracle {
  static void run(const tvm::ffi::TensorView q,
                  const tvm::ffi::TensorView unified_kv,
                  const tvm::ffi::TensorView kv_indices,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView attn_sink,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView workspace,
                  double softmax_scale,
                  int64_t compress_ratio) {
    if (q.ndim() != 3 || q.size(0) != 64 || q.size(1) != 16 ||
        q.size(2) != 512 || unified_kv.ndim() != 2 ||
        unified_kv.size(1) != 512 || output.ndim() != 3 ||
        output.size(0) != 64 || output.size(1) != 16 ||
        output.size(2) != 512 || kv_indptr.size(0) != 65 ||
        compress_ratio != 128) {
      throw std::runtime_error(
          "gfx90a DSV4 pair-query oracle requires M64/C128/H16/D512");
    }
    ck_tile::dsv4::UnifiedSparseDecodeArgs args{
        static_cast<const ck::bhalf_t*>(q.data_ptr()),
        static_cast<const ck::bhalf_t*>(unified_kv.data_ptr()),
        static_cast<const int32_t*>(kv_indices.data_ptr()),
        static_cast<const int32_t*>(kv_indptr.data_ptr()),
        static_cast<const float*>(attn_sink.data_ptr()),
        static_cast<ck::bhalf_t*>(output.data_ptr()),
        64,
        16,
        static_cast<int32_t>(unified_kv.size(0)),
        static_cast<float>(softmax_scale)};
    const auto stream = sglang::host::LaunchKernel::resolve_device(q.device());
    const hipError_t status =
        ck_tile::dsv4::launch_unified_sparse_decode_d512_pair_split2_qreg(
            args, workspace.data_ptr(), stream);
    if (status != hipSuccess) {
      throw std::runtime_error("gfx90a DSV4 pair-query oracle launch failed");
    }
  }
};

}  // namespace sglang
