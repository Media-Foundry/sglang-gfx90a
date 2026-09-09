// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
// Isolated numerical oracle: production core remains unchanged.
// BF16 probability hi+lo products preserve BF16 KV range without FP16 casts.
// Derived from the production split core; extra PV MFMA may cost occupancy/time.
#include "gfx90a_dsv4_sparse_h8_oracle.cuh"
namespace ck_tile::dsv4 {
template <bool StageQ, bool PreloadQ, bool PipelineKV, int Heads = 16,
          bool PairH8 = false>
__global__ __launch_bounds__(kWaveSize * 4) void
unified_sparse_decode_d512_mfma_refine_probability_kernel(UnifiedSparseDecodeArgs args,
                                                   UnifiedSparseDecodeWorkspace workspace,
                                                   int splits)
{
    using WarpGemm = WarpGemmMfmaBf16Bf16F32M16N16K16;
    using AVec = ext_vector_t<bf16_t, WarpGemm::AWarpTensor::get_thread_buffer_size()>;
    using BVec = ext_vector_t<bf16_t, WarpGemm::BWarpTensor::get_thread_buffer_size()>;
    using CVec = ext_vector_t<float, WarpGemm::CWarpTensor::get_thread_buffer_size()>;

    constexpr int kTile = 16;
    constexpr int kWaves = 4;
    constexpr int kOutputTilesPerWave = (kHeadDim / kTile) / kWaves;
    constexpr int kValuesPerLane = 4;
    constexpr int kKvLdsStride = kHeadDim + 4;
    constexpr int kQLdsStride = kHeadDim + 4;
    constexpr int kQFragmentsPerWave = (kHeadDim / kWaves) / kTile;
    static_assert(!(StageQ && PreloadQ));

    __shared__ bf16_t q_tile[StageQ ? kLocalHeads * kQLdsStride : 1];
    __shared__ int kv_slots[kTile];
    __shared__ bf16_t kv_tile[kTile * kKvLdsStride];
    __shared__ float score_partial[kWaves][kLocalHeads * kTile];
    __shared__ bf16_t probabilities[kLocalHeads * kTile];
    __shared__ bf16_t probabilities_lo[kLocalHeads * kTile];
    __shared__ float alpha_shared[kLocalHeads];

    static_assert(!PairH8 || Heads == 8);
    const int work_item = static_cast<int>(blockIdx.x);
    const int token = PairH8 ? work_item * 2 : work_item;
    const int split = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int matrix_lane = lane % kTile;
    const int k_group = lane / kTile;
    // PairH8 consumes the longer second row's ordered gather list.  The
    // wrapper validates that row0 is an exact prefix of row1.  The two halves
    // of the MFMA M16 tile retain independent lengths/softmax states.
    const int gather_token = PairH8 ? token + 1 : token;
    const int begin = args.kv_indptr[gather_token];
    const int end = args.kv_indptr[gather_token + 1];
    const int total_tiles = (end - begin + kTile - 1) / kTile;
    const int tiles_per_split = (total_tiles + splits - 1) / splits;
    const int first_tile = split * tiles_per_split;
    const int last_tile = min(total_tiles, first_tile + tiles_per_split);

    if constexpr(StageQ)
    {
        for(int linear = tid; linear < kLocalHeads * kHeadDim; linear += kWaveSize * kWaves)
        {
            const int head = linear / kHeadDim;
            const int d = linear % kHeadDim;
            const long q_offset =
                (static_cast<long>(token) * args.heads + head) * kHeadDim + d;
            q_tile[head * kQLdsStride + d] = head < Heads
                ? bit_cast<bf16_t>(args.q[q_offset]) : type_convert<bf16_t>(0.0f);
        }
        __syncthreads();
    }

    AVec q_fragments[PreloadQ ? kQFragmentsPerWave : 1];
    if constexpr(PreloadQ)
    {
#pragma unroll
        for(int fragment = 0; fragment < kQFragmentsPerWave; ++fragment)
        {
            const int k_begin = wave * (kHeadDim / kWaves) + fragment * kTile;
#pragma unroll
            for(int k = 0; k < kValuesPerLane; ++k)
            {
                const int d = k_begin + k_group * kValuesPerLane + k;
                const int q_token = PairH8 ? token + matrix_lane / Heads : token;
                const int q_head = PairH8 ? matrix_lane % Heads : matrix_lane;
                const long q_offset =
                    (static_cast<long>(q_token) * args.heads + q_head) * kHeadDim + d;
                q_fragments[fragment][k] = (PairH8 || matrix_lane < Heads)
                    ? bit_cast<bf16_t>(args.q[q_offset]) : type_convert<bf16_t>(0.0f);
            }
        }
    }

    constexpr int kKvElementsPerThread = (kTile * kHeadDim) / (kWaveSize * kWaves);
    bf16_t next_kv_fragment[PipelineKV ? kKvElementsPerThread : 1];
    if constexpr(PipelineKV)
    {
        if(first_tile < last_tile)
        {
            const int tile_begin = begin + first_tile * kTile;
            const int valid_keys = min(kTile, end - tile_begin);
            if(tid < kTile)
                kv_slots[tid] = tid < valid_keys ? args.kv_indices[tile_begin + tid] : -1;
            __syncthreads();
#pragma unroll
            for(int i = 0; i < kKvElementsPerThread; ++i)
            {
                const int linear = tid + i * kWaveSize * kWaves;
                const int key = linear / kHeadDim;
                const int d = linear % kHeadDim;
                const int slot = kv_slots[key];
                kv_tile[key * kKvLdsStride + d] =
                    slot >= 0 && slot < args.pool_slots
                        ? bit_cast<bf16_t>(
                              args.unified_kv[static_cast<long>(slot) * kHeadDim + d])
                        : type_convert<bf16_t>(0.0f);
            }
            __syncthreads();
        }
    }

    const int softmax_head = tid / kTile;
    const int softmax_key = tid % kTile;
    float softmax_max = -__builtin_inff();
    float softmax_norm = 0.0f;
    float accumulator[kOutputTilesPerWave][kValuesPerLane];
#pragma unroll
    for(int i = 0; i < kValuesPerLane; ++i)
    {
#pragma unroll
        for(int n = 0; n < kOutputTilesPerWave; ++n) accumulator[n][i] = 0.0f;
    }

    for(int tile_index = first_tile; tile_index < last_tile; ++tile_index)
    {
        const int tile_begin = begin + tile_index * kTile;
        const int valid_keys = min(kTile, end - tile_begin);
        if constexpr(!PipelineKV)
        {
            if(tid < kTile)
                kv_slots[tid] = tid < valid_keys ? args.kv_indices[tile_begin + tid] : -1;
            __syncthreads();

            for(int linear = tid; linear < kTile * kHeadDim; linear += kWaveSize * kWaves)
            {
                const int key = linear / kHeadDim;
                const int d = linear % kHeadDim;
                const int slot = kv_slots[key];
                kv_tile[key * kKvLdsStride + d] =
                    slot >= 0 && slot < args.pool_slots
                        ? bit_cast<bf16_t>(
                              args.unified_kv[static_cast<long>(slot) * kHeadDim + d])
                        : type_convert<bf16_t>(0.0f);
            }
            __syncthreads();
        }

        const bool has_next_tile = tile_index + 1 < last_tile;
        if constexpr(PipelineKV)
        {
            if(has_next_tile)
            {
                const int next_begin = tile_begin + kTile;
                const int next_valid = min(kTile, end - next_begin);
                if(tid < kTile)
                    kv_slots[tid] = tid < next_valid ? args.kv_indices[next_begin + tid] : -1;
                __syncthreads();
#pragma unroll
                for(int i = 0; i < kKvElementsPerThread; ++i)
                {
                    const int linear = tid + i * kWaveSize * kWaves;
                    const int key = linear / kHeadDim;
                    const int d = linear % kHeadDim;
                    const int slot = kv_slots[key];
                    next_kv_fragment[i] =
                        slot >= 0 && slot < args.pool_slots
                            ? bit_cast<bf16_t>(
                                  args.unified_kv[static_cast<long>(slot) * kHeadDim + d])
                            : type_convert<bf16_t>(0.0f);
                }
            }
        }

        typename WarpGemm::CWarpTensor score_tensor;
        score_tensor.get_thread_buffer().template set_as<CVec>(number<0>{}, CVec{0.0f});

        const int qk_begin = wave * (kHeadDim / kWaves);
        const int qk_end = qk_begin + kHeadDim / kWaves;
        for(int k_begin = qk_begin; k_begin < qk_end; k_begin += kTile)
        {
            AVec q_vec{};
            BVec k_vec{};
#pragma unroll
            for(int k = 0; k < kValuesPerLane; ++k)
            {
                const int d = k_begin + k_group * kValuesPerLane + k;
                if constexpr(StageQ)
                    q_vec[k] = q_tile[matrix_lane * kQLdsStride + d];
                else if constexpr(PreloadQ)
                    q_vec[k] = q_fragments[(k_begin - qk_begin) / kTile][k];
                else
                {
                    const long q_offset =
                        (static_cast<long>(token) * args.heads + matrix_lane) * kHeadDim + d;
                    q_vec[k] = matrix_lane < Heads
                        ? bit_cast<bf16_t>(args.q[q_offset]) : type_convert<bf16_t>(0.0f);
                }
                k_vec[k] = kv_tile[matrix_lane * kKvLdsStride + d];
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
        for(int i = 0; i < kValuesPerLane; ++i)
        {
            const int head = k_group * kValuesPerLane + i;
            score_partial[wave][head * kTile + matrix_lane] = score_vec[i];
        }
        __syncthreads();

        float score = 0.0f;
#pragma unroll
        for(int qk_wave = 0; qk_wave < kWaves; ++qk_wave)
            score += score_partial[qk_wave][softmax_head * kTile + softmax_key];
        const int score_token = PairH8 ? token + softmax_head / Heads : token;
        const int score_length = PairH8
            ? args.kv_indptr[score_token + 1] - args.kv_indptr[score_token]
            : end - begin;
        const int score_valid_keys =
            max(0, min(kTile, score_length - tile_index * kTile));
        score = softmax_key < score_valid_keys ? score * args.softmax_scale : -__builtin_inff();

        float block_max = score;
#pragma unroll
        for(int offset = kTile / 2; offset > 0; offset >>= 1)
            block_max = fmaxf(block_max, __shfl_down(block_max, offset, kTile));
        block_max = __shfl(block_max, 0, kTile);

        const bool score_row_active = score_valid_keys > 0;
        const float next_max = score_row_active ? fmaxf(softmax_max, block_max)
                                                : softmax_max;
        const float alpha = score_row_active ? expf(softmax_max - next_max) : 1.0f;
        const float weight = score_row_active && softmax_key < score_valid_keys
            ? expf(score - next_max) : 0.0f;
        float block_sum = weight;
#pragma unroll
        for(int offset = kTile / 2; offset > 0; offset >>= 1)
            block_sum += __shfl_down(block_sum, offset, kTile);
        block_sum = __shfl(block_sum, 0, kTile);

        if(score_row_active)
        {
            softmax_norm = softmax_norm * alpha + block_sum;
            softmax_max = next_max;
        }
        const bf16_t p_hi = type_convert<bf16_t>(weight);
        probabilities[softmax_head * kTile + softmax_key] = p_hi;
        probabilities_lo[softmax_head * kTile + softmax_key] =
            type_convert<bf16_t>(weight - type_convert<float>(p_hi));
        if(softmax_key == 0) alpha_shared[softmax_head] = alpha;
        __syncthreads();

        typename WarpGemm::CWarpTensor value_tensors[kOutputTilesPerWave];
#pragma unroll
        for(int n = 0; n < kOutputTilesPerWave; ++n)
            value_tensors[n].get_thread_buffer().template set_as<CVec>(number<0>{},
                                                                      CVec{0.0f});

        AVec p_vec{};
        AVec p_lo_vec{};
#pragma unroll
        for(int k = 0; k < kValuesPerLane; ++k)
        {
            const int p_index = matrix_lane * kTile + k_group * kValuesPerLane + k;
            p_vec[k] = probabilities[p_index];
            p_lo_vec[k] = probabilities_lo[p_index];
        }
        typename WarpGemm::AWarpTensor p_tensor;
        p_tensor.get_thread_buffer().template set_as<AVec>(number<0>{}, p_vec);
        typename WarpGemm::AWarpTensor p_lo_tensor;
        p_lo_tensor.get_thread_buffer().template set_as<AVec>(number<0>{}, p_lo_vec);

#pragma unroll
        for(int n = 0; n < kOutputTilesPerWave; ++n)
        {
            BVec v_vec{};
            const int output_tile = wave * kOutputTilesPerWave + n;
#pragma unroll
            for(int k = 0; k < kValuesPerLane; ++k)
            {
                const int key = k_group * kValuesPerLane + k;
                const int d = output_tile * kTile + matrix_lane;
                v_vec[k] = kv_tile[key * kKvLdsStride + d];
            }
            typename WarpGemm::BWarpTensor v_tensor;
            v_tensor.get_thread_buffer().template set_as<BVec>(number<0>{}, v_vec);
            WarpGemm{}(value_tensors[n], p_tensor, v_tensor);
            WarpGemm{}(value_tensors[n], p_lo_tensor, v_tensor);
        }

#pragma unroll
        for(int n = 0; n < kOutputTilesPerWave; ++n)
        {
            const CVec value_vec =
                value_tensors[n].get_thread_buffer().template get_as<CVec>()[number<0>{}];
#pragma unroll
            for(int i = 0; i < kValuesPerLane; ++i)
            {
                const int head = k_group * kValuesPerLane + i;
                accumulator[n][i] = accumulator[n][i] * alpha_shared[head] + value_vec[i];
            }
        }
        __syncthreads();

        if constexpr(PipelineKV)
        {
            if(has_next_tile)
            {
#pragma unroll
                for(int i = 0; i < kKvElementsPerThread; ++i)
                {
                    const int linear = tid + i * kWaveSize * kWaves;
                    const int key = linear / kHeadDim;
                    const int d = linear % kHeadDim;
                    kv_tile[key * kKvLdsStride + d] = next_kv_fragment[i];
                }
                __syncthreads();
            }
        }
    }

    const int output_token = PairH8 ? token + softmax_head / Heads : token;
    const int output_head = PairH8 ? softmax_head % Heads : softmax_head;
    const std::size_t stat_partial_row =
        (static_cast<std::size_t>(output_token) * splits + split) * args.heads;
    if(softmax_key == 0 && (PairH8 || softmax_head < Heads))
    {
        workspace.max_partial[stat_partial_row + output_head] = softmax_max;
        workspace.norm_partial[stat_partial_row + output_head] = softmax_norm;
    }

#pragma unroll
    for(int n = 0; n < kOutputTilesPerWave; ++n)
    {
        const int output_tile = wave * kOutputTilesPerWave + n;
        const int d = output_tile * kTile + matrix_lane;
#pragma unroll
        for(int i = 0; i < kValuesPerLane; ++i)
        {
            const int combined_head = k_group * kValuesPerLane + i;
            const int value_token = PairH8 ? token + combined_head / Heads : token;
            const int value_head = PairH8 ? combined_head % Heads : combined_head;
            if(PairH8 || combined_head < Heads)
            {
                const std::size_t value_partial_row =
                    (static_cast<std::size_t>(value_token) * splits + split) * args.heads;
                workspace.output_partial[(value_partial_row + value_head) * kHeadDim + d] = accumulator[n][i];
            }
        }
    }
}


} // namespace ck_tile::dsv4

namespace sglang {
struct Gfx90aDsv4SparseH8RefinedProbabilityOracle {
  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView kv,
                  tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,
                  tvm::ffi::TensorView sink, tvm::ffi::TensorView out,
                  tvm::ffi::TensorView scratch, double scale) {
    const int tokens = q.size(0);
    if(q.ndim()!=3 || tokens<=0 || tokens>128 || q.size(1)!=8 || q.size(2)!=512 ||
       kv.ndim()!=2 || kv.size(1)!=512 || out.ndim()!=3 ||
       out.size(0)!=tokens || out.size(1)!=8 || out.size(2)!=512 ||
       indptr.size(0)!=tokens+1 || sink.size(0)!=8 ||
       scratch.numel() < static_cast<int64_t>(tokens)*2*8*514*4)
      throw std::runtime_error("H8 refined oracle shape/workspace mismatch");
    ck_tile::dsv4::UnifiedSparseDecodeArgs args{
      static_cast<const ck::bhalf_t*>(q.data_ptr()),
      static_cast<const ck::bhalf_t*>(kv.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<const int32_t*>(indptr.data_ptr()),
      static_cast<const float*>(sink.data_ptr()),
      static_cast<ck::bhalf_t*>(out.data_ptr()), tokens, 8,
      static_cast<int32_t>(kv.size(0)), static_cast<float>(scale)};
    auto stream = sglang::host::LaunchKernel::resolve_device(q.device());
    const auto workspace = ck_tile::dsv4::partition_mfma_split_workspace(scratch.data_ptr(), args, 2);
    hipLaunchKernelGGL(
      (ck_tile::dsv4::unified_sparse_decode_d512_mfma_refine_probability_kernel<false,true,true,8>),
      dim3(tokens,2,1), dim3(256,1,1), 0, stream, args, workspace, 2);
    auto status = hipGetLastError();
    if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
    hipLaunchKernelGGL(ck_tile::dsv4::unified_sparse_decode_d512_mfma_split_reduce_kernel,
      dim3(tokens,8,1), dim3(256,1,1), 0, stream, args, workspace, 2);
    status = hipGetLastError();
    if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  }
};

struct Gfx90aDsv4SparseH8PairRefinedProbabilityOracle {
  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView kv,
                  tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,
                  tvm::ffi::TensorView sink, tvm::ffi::TensorView out,
                  tvm::ffi::TensorView scratch, double scale) {
    const int tokens = q.size(0);
    if(q.ndim()!=3 || tokens!=128 || q.size(1)!=8 || q.size(2)!=512 ||
       kv.ndim()!=2 || kv.size(1)!=512 || out.ndim()!=3 ||
       out.size(0)!=tokens || out.size(1)!=8 || out.size(2)!=512 ||
       indptr.size(0)!=tokens+1 || sink.size(0)!=8 ||
       scratch.numel() < static_cast<int64_t>(tokens)*2*8*514*4)
      throw std::runtime_error("H8 pair refined oracle requires M128/H8/D512");
    ck_tile::dsv4::UnifiedSparseDecodeArgs args{
      static_cast<const ck::bhalf_t*>(q.data_ptr()),
      static_cast<const ck::bhalf_t*>(kv.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<const int32_t*>(indptr.data_ptr()),
      static_cast<const float*>(sink.data_ptr()),
      static_cast<ck::bhalf_t*>(out.data_ptr()), tokens, 8,
      static_cast<int32_t>(kv.size(0)), static_cast<float>(scale)};
    auto stream = sglang::host::LaunchKernel::resolve_device(q.device());
    const auto workspace = ck_tile::dsv4::partition_mfma_split_workspace(scratch.data_ptr(), args, 2);
    hipLaunchKernelGGL(
      (ck_tile::dsv4::unified_sparse_decode_d512_mfma_refine_probability_kernel<false,true,true,8,true>),
      dim3(tokens/2,2,1), dim3(256,1,1), 0, stream, args, workspace, 2);
    auto status = hipGetLastError();
    if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
    hipLaunchKernelGGL(ck_tile::dsv4::unified_sparse_decode_d512_mfma_split_reduce_kernel,
      dim3(tokens,8,1), dim3(256,1,1), 0, stream, args, workspace, 2);
    status = hipGetLastError();
    if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  }
};

struct Gfx90aDsv4SparseH8PairOracle {
  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView kv,
                  tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,
                  tvm::ffi::TensorView sink, tvm::ffi::TensorView out,
                  tvm::ffi::TensorView scratch, double scale) {
    const int tokens = q.size(0);
    if(q.ndim()!=3 || tokens!=128 || q.size(1)!=8 || q.size(2)!=512 ||
       kv.ndim()!=2 || kv.size(1)!=512 || out.ndim()!=3 ||
       out.size(0)!=tokens || out.size(1)!=8 || out.size(2)!=512 ||
       indptr.size(0)!=tokens+1 || sink.size(0)!=8 ||
       scratch.numel() < static_cast<int64_t>(tokens)*2*8*514*4)
      throw std::runtime_error("H8 pair oracle requires M128/H8/D512");
    ck_tile::dsv4::UnifiedSparseDecodeArgs args{
      static_cast<const ck::bhalf_t*>(q.data_ptr()),
      static_cast<const ck::bhalf_t*>(kv.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<const int32_t*>(indptr.data_ptr()),
      static_cast<const float*>(sink.data_ptr()),
      static_cast<ck::bhalf_t*>(out.data_ptr()), tokens, 8,
      static_cast<int32_t>(kv.size(0)), static_cast<float>(scale)};
    auto stream = sglang::host::LaunchKernel::resolve_device(q.device());
    const auto workspace = ck_tile::dsv4::partition_mfma_split_workspace(scratch.data_ptr(), args, 2);
    hipLaunchKernelGGL(
      (ck_tile::dsv4::unified_sparse_decode_d512_mfma_split_core_kernel<false,true,true,8,true>),
      dim3(tokens/2,2,1), dim3(256,1,1), 0, stream, args, workspace, 2);
    auto status = hipGetLastError();
    if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
    hipLaunchKernelGGL(ck_tile::dsv4::unified_sparse_decode_d512_mfma_split_reduce_kernel,
      dim3(tokens,8,1), dim3(256,1,1), 0, stream, args, workspace, 2);
    status = hipGetLastError();
    if(status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  }
};
}
