// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile::dsv4 {

inline constexpr int kHeadDim = 512;
inline constexpr int kLocalHeads = 16;
inline constexpr int kMaxDecodeM = 128;
inline constexpr int kThreads = 256;
inline constexpr int kWaveSize = 64;
inline constexpr int kDefaultHeadGroup = 4;
inline constexpr int kDefaultKvTile = 8;

struct UnifiedSparseDecodeArgs
{
    const ck::bhalf_t* q;          // [tokens, heads, 512]
    const ck::bhalf_t* unified_kv; // [pool_slots, 512], page size one
    const int32_t* kv_indices;     // packed physical slots
    const int32_t* kv_indptr;      // [tokens + 1]
    const float* attn_sink;        // [heads], natural-log domain
    ck::bhalf_t* output;           // [tokens, heads, 512]
    int32_t tokens;
    int32_t heads;
    int32_t pool_slots;
    float softmax_scale;
};

inline bool is_supported(const UnifiedSparseDecodeArgs& args)
{
    return args.q != nullptr && args.unified_kv != nullptr && args.kv_indices != nullptr &&
           args.kv_indptr != nullptr && args.attn_sink != nullptr && args.output != nullptr &&
           args.tokens > 0 && args.tokens <= kMaxDecodeM &&
           args.heads == kLocalHeads && args.pool_slots > 0;
}

// Semantically complete direct-ragged baseline. One workgroup owns one
// (token, head); each lane owns two D elements. It never gathers KV into a
// temporary tensor and never materializes a score matrix. The online update
// is in natural-log space; attn_sink is folded as a virtual key with zero V.
__global__ void unified_sparse_decode_d512_reference_kernel(UnifiedSparseDecodeArgs args)
{
    const int token = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    const int lane  = static_cast<int>(threadIdx.x);
    const int d0    = lane;
    const int d1    = lane + kThreads;

    __shared__ float reduction[kThreads];
    __shared__ float score_shared;

    const long q_base = (static_cast<long>(token) * args.heads + head) * kHeadDim;
    const float q0    = ck::type_convert<float>(args.q[q_base + d0]);
    const float q1    = ck::type_convert<float>(args.q[q_base + d1]);

    float max_value = -__builtin_inff();
    float normalizer = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int begin = args.kv_indptr[token];
    const int end   = args.kv_indptr[token + 1];

    for(int i = begin; i < end; ++i)
    {
        const int slot = args.kv_indices[i];
        if(slot < 0 || slot >= args.pool_slots)
        {
            continue;
        }

        const long kv_base = static_cast<long>(slot) * kHeadDim;
        const float v0     = ck::type_convert<float>(args.unified_kv[kv_base + d0]);
        const float v1     = ck::type_convert<float>(args.unified_kv[kv_base + d1]);
        reduction[lane]    = q0 * v0 + q1 * v1;
        __syncthreads();

        for(int offset = kThreads / 2; offset > 0; offset >>= 1)
        {
            if(lane < offset)
            {
                reduction[lane] += reduction[lane + offset];
            }
            __syncthreads();
        }

        if(lane == 0)
        {
            score_shared = reduction[0] * args.softmax_scale;
        }
        __syncthreads();

        const float score = score_shared;
        const float next_max = fmaxf(max_value, score);
        const float old_scale = expf(max_value - next_max);
        const float weight = expf(score - next_max);
        normalizer = normalizer * old_scale + weight;
        acc0       = acc0 * old_scale + weight * v0;
        acc1       = acc1 * old_scale + weight * v1;
        max_value  = next_max;
        __syncthreads();
    }

    const float sink = args.attn_sink[head];
    const float final_max = fmaxf(max_value, sink);
    const float kv_scale  = expf(max_value - final_max);
    const float sink_weight = expf(sink - final_max);
    const float denominator = normalizer * kv_scale + sink_weight;
    const float inv_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;

    args.output[q_base + d0] = ck::type_convert<ck::bhalf_t>(acc0 * kv_scale * inv_denominator);
    args.output[q_base + d1] = ck::type_convert<ck::bhalf_t>(acc1 * kv_scale * inv_denominator);
}

// Four wave64s own four heads and reuse an eight-row KV tile from LDS. A wave
// reduces each D512 dot product with shuffles, so the hot loop has no
// workgroup-wide reduction and only two barriers per eight KV rows.
template <int HeadGroup, int KvTile>
__global__ __launch_bounds__(HeadGroup * kWaveSize) void
unified_sparse_decode_d512_wave_tiled_kernel(UnifiedSparseDecodeArgs args)
{
    __shared__ ck::bhalf_t kv_tile[KvTile * kHeadDim];

    const int token      = static_cast<int>(blockIdx.x);
    const int head_group = static_cast<int>(blockIdx.y) * HeadGroup;
    const int tid        = static_cast<int>(threadIdx.x);
    const int wave       = tid / kWaveSize;
    const int lane       = tid % kWaveSize;
    const int head       = head_group + wave;

    const long q_base = (static_cast<long>(token) * args.heads + head) * kHeadDim;
    float q_fragment[kHeadDim / kWaveSize];
    float accumulator[kHeadDim / kWaveSize];

#pragma unroll
    for(int i = 0; i < kHeadDim / kWaveSize; ++i)
    {
        const int d       = lane + i * kWaveSize;
        q_fragment[i]     = ck::type_convert<float>(args.q[q_base + d]);
        accumulator[i]    = 0.0f;
    }

    float max_value = -__builtin_inff();
    float normalizer = 0.0f;
    const int begin = args.kv_indptr[token];
    const int end   = args.kv_indptr[token + 1];

    for(int tile_begin = begin; tile_begin < end; tile_begin += KvTile)
    {
        const int valid_rows = min(KvTile, end - tile_begin);
        const int tile_elements = valid_rows * kHeadDim;

        for(int linear = tid; linear < tile_elements; linear += HeadGroup * kWaveSize)
        {
            const int row  = linear / kHeadDim;
            const int d    = linear % kHeadDim;
            const int slot = args.kv_indices[tile_begin + row];
            kv_tile[linear] = slot >= 0 && slot < args.pool_slots
                                  ? args.unified_kv[static_cast<long>(slot) * kHeadDim + d]
                                  : ck::type_convert<ck::bhalf_t>(0.0f);
        }
        __syncthreads();

        for(int row = 0; row < valid_rows; ++row)
        {
            float dot = 0.0f;
#pragma unroll
            for(int i = 0; i < kHeadDim / kWaveSize; ++i)
            {
                const int d = lane + i * kWaveSize;
                dot += q_fragment[i] *
                       ck::type_convert<float>(kv_tile[row * kHeadDim + d]);
            }

#pragma unroll
            for(int offset = kWaveSize / 2; offset > 0; offset >>= 1)
            {
                dot += __shfl_down(dot, offset);
            }

            const float score = __shfl(dot, 0) * args.softmax_scale;
            const float next_max = fmaxf(max_value, score);
            const float old_scale = expf(max_value - next_max);
            const float weight = expf(score - next_max);
            normalizer = normalizer * old_scale + weight;

#pragma unroll
            for(int i = 0; i < kHeadDim / kWaveSize; ++i)
            {
                const int d = lane + i * kWaveSize;
                const float value = ck::type_convert<float>(kv_tile[row * kHeadDim + d]);
                accumulator[i] = accumulator[i] * old_scale + weight * value;
            }
            max_value = next_max;
        }
        __syncthreads();
    }

    const float sink = args.attn_sink[head];
    const float final_max = fmaxf(max_value, sink);
    const float kv_scale  = expf(max_value - final_max);
    const float sink_weight = expf(sink - final_max);
    const float denominator = normalizer * kv_scale + sink_weight;
    const float output_scale = denominator > 0.0f ? kv_scale / denominator : 0.0f;

#pragma unroll
    for(int i = 0; i < kHeadDim / kWaveSize; ++i)
    {
        const int d = lane + i * kWaveSize;
        args.output[q_base + d] =
            ck::type_convert<ck::bhalf_t>(accumulator[i] * output_scale);
    }
}

// One wave computes four heads. M4N64K16 maps lane%4 to a head and the 64
// lanes to KV/output columns, so both QK and P*V use MFMA without a score
// matrix in global memory.
__global__ __launch_bounds__(kWaveSize) void
unified_sparse_decode_d512_mfma_kernel(UnifiedSparseDecodeArgs args)
{
    using WarpGemm = WarpGemmMfmaBf16Bf16F32M4N64K16;
    using AVec = ext_vector_t<bf16_t, WarpGemm::AWarpTensor::get_thread_buffer_size()>;
    using BVec = ext_vector_t<bf16_t, WarpGemm::BWarpTensor::get_thread_buffer_size()>;
    using CVec = ext_vector_t<float, WarpGemm::CWarpTensor::get_thread_buffer_size()>;

    constexpr int kHeadsPerWave = 4;
    constexpr int kKeysPerTile = 64;
    constexpr int kOutputTiles = kHeadDim / kKeysPerTile;
    constexpr int kMmaK = 16;

    __shared__ bf16_t probabilities[kHeadsPerWave * kKeysPerTile];

    const int token = static_cast<int>(blockIdx.x);
    const int head_base = static_cast<int>(blockIdx.y) * kHeadsPerWave;
    const int lane = static_cast<int>(threadIdx.x);
    const int lane_head = lane % kHeadsPerWave;
    const int begin = args.kv_indptr[token];
    const int end = args.kv_indptr[token + 1];

    float max_value[kHeadsPerWave];
    float normalizer[kHeadsPerWave];
    float accumulator[kOutputTiles][kHeadsPerWave];
#pragma unroll
    for(int h = 0; h < kHeadsPerWave; ++h)
    {
        max_value[h] = -__builtin_inff();
        normalizer[h] = 0.0f;
#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n) accumulator[n][h] = 0.0f;
    }

    for(int tile_begin = begin; tile_begin < end; tile_begin += kKeysPerTile)
    {
        const int valid_keys = min(kKeysPerTile, end - tile_begin);
        typename WarpGemm::CWarpTensor score_tensor;
        score_tensor.get_thread_buffer().template set_as<CVec>(number<0>{}, CVec{0.0f});

#pragma unroll
        for(int k_begin = 0; k_begin < kHeadDim; k_begin += kMmaK)
        {
            AVec q_vec{};
            BVec k_vec{};
            const int key = lane;
            const int slot = key < valid_keys ? args.kv_indices[tile_begin + key] : -1;
#pragma unroll
            for(int k = 0; k < kMmaK; ++k)
            {
                const long q_offset =
                    (static_cast<long>(token) * args.heads + head_base + lane_head) * kHeadDim +
                    k_begin + k;
                q_vec[k] = bit_cast<bf16_t>(args.q[q_offset]);
                k_vec[k] = slot >= 0 && slot < args.pool_slots
                               ? bit_cast<bf16_t>(args.unified_kv[static_cast<long>(slot) *
                                                                      kHeadDim +
                                                                  k_begin + k])
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
        float alpha[kHeadsPerWave];
#pragma unroll
        for(int h = 0; h < kHeadsPerWave; ++h)
        {
            const float score =
                lane < valid_keys ? score_vec[h] * args.softmax_scale : -__builtin_inff();
            float block_max = score;
#pragma unroll
            for(int offset = kWaveSize / 2; offset > 0; offset >>= 1)
                block_max = fmaxf(block_max, __shfl_down(block_max, offset));
            block_max = __shfl(block_max, 0);

            const float next_max = fmaxf(max_value[h], block_max);
            alpha[h] = expf(max_value[h] - next_max);
            const float weight = lane < valid_keys ? expf(score - next_max) : 0.0f;
            float block_sum = weight;
#pragma unroll
            for(int offset = kWaveSize / 2; offset > 0; offset >>= 1)
                block_sum += __shfl_down(block_sum, offset);
            block_sum = __shfl(block_sum, 0);

            normalizer[h] = normalizer[h] * alpha[h] + block_sum;
            max_value[h] = next_max;
            probabilities[h * kKeysPerTile + lane] = type_convert<bf16_t>(weight);
        }
        __syncthreads();

        typename WarpGemm::CWarpTensor value_tensors[kOutputTiles];
#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n)
            value_tensors[n].get_thread_buffer().template set_as<CVec>(number<0>{},
                                                                      CVec{0.0f});

#pragma unroll
        for(int key_begin = 0; key_begin < kKeysPerTile; key_begin += kMmaK)
        {
            AVec p_vec{};
#pragma unroll
            for(int k = 0; k < kMmaK; ++k)
                p_vec[k] = probabilities[lane_head * kKeysPerTile + key_begin + k];

            typename WarpGemm::AWarpTensor p_tensor;
            p_tensor.get_thread_buffer().template set_as<AVec>(number<0>{}, p_vec);

#pragma unroll
            for(int n = 0; n < kOutputTiles; ++n)
            {
                BVec v_vec{};
#pragma unroll
                for(int k = 0; k < kMmaK; ++k)
                {
                    const int key = key_begin + k;
                    const int slot =
                        key < valid_keys ? args.kv_indices[tile_begin + key] : -1;
                    const int d = n * kKeysPerTile + lane;
                    v_vec[k] = slot >= 0 && slot < args.pool_slots
                                   ? bit_cast<bf16_t>(
                                         args.unified_kv[static_cast<long>(slot) * kHeadDim + d])
                                   : type_convert<bf16_t>(0.0f);
                }
                typename WarpGemm::BWarpTensor v_tensor;
                v_tensor.get_thread_buffer().template set_as<BVec>(number<0>{}, v_vec);
                WarpGemm{}(value_tensors[n], p_tensor, v_tensor);
            }
        }

#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n)
        {
            const CVec value_vec =
                value_tensors[n].get_thread_buffer().template get_as<CVec>()[number<0>{}];
#pragma unroll
            for(int h = 0; h < kHeadsPerWave; ++h)
                accumulator[n][h] = accumulator[n][h] * alpha[h] + value_vec[h];
        }
        __syncthreads();
    }

#pragma unroll
    for(int h = 0; h < kHeadsPerWave; ++h)
    {
        const float sink = args.attn_sink[head_base + h];
        const float final_max = fmaxf(max_value[h], sink);
        const float kv_scale = expf(max_value[h] - final_max);
        const float sink_weight = expf(sink - final_max);
        const float denominator = normalizer[h] * kv_scale + sink_weight;
        const float output_scale = denominator > 0.0f ? kv_scale / denominator : 0.0f;
#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n)
        {
            const int d = n * kKeysPerTile + lane;
            const long output_offset =
                (static_cast<long>(token) * args.heads + head_base + h) * kHeadDim + d;
            args.output[output_offset] =
                ck::type_convert<ck::bhalf_t>(accumulator[n][h] * output_scale);
        }
    }
}

// A native 16x16x16 MFMA wave owns all 16 heads for one token. Each key tile
// has 16 rows; the C register layout is transposed through a small LDS tile so
// it becomes the A layout required by P*V.
__global__ __launch_bounds__(kWaveSize) void
unified_sparse_decode_d512_mfma_m16_kernel(UnifiedSparseDecodeArgs args)
{
    using WarpGemm = WarpGemmMfmaBf16Bf16F32M16N16K16;
    using AVec = ext_vector_t<bf16_t, WarpGemm::AWarpTensor::get_thread_buffer_size()>;
    using BVec = ext_vector_t<bf16_t, WarpGemm::BWarpTensor::get_thread_buffer_size()>;
    using CVec = ext_vector_t<float, WarpGemm::CWarpTensor::get_thread_buffer_size()>;

    constexpr int kTile = 16;
    constexpr int kOutputTiles = kHeadDim / kTile;
    constexpr int kValuesPerLane = 4;

    __shared__ bf16_t probabilities[kLocalHeads * kTile];

    const int token = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int matrix_lane = lane % kTile;
    const int k_group = lane / kTile;
    const int begin = args.kv_indptr[token];
    const int end = args.kv_indptr[token + 1];

    float max_value[kValuesPerLane];
    float normalizer[kValuesPerLane];
    float accumulator[kOutputTiles][kValuesPerLane];
#pragma unroll
    for(int i = 0; i < kValuesPerLane; ++i)
    {
        max_value[i] = -__builtin_inff();
        normalizer[i] = 0.0f;
#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n) accumulator[n][i] = 0.0f;
    }

    for(int tile_begin = begin; tile_begin < end; tile_begin += kTile)
    {
        const int valid_keys = min(kTile, end - tile_begin);
        typename WarpGemm::CWarpTensor score_tensor;
        score_tensor.get_thread_buffer().template set_as<CVec>(number<0>{}, CVec{0.0f});

#pragma unroll
        for(int k_begin = 0; k_begin < kHeadDim; k_begin += kTile)
        {
            AVec q_vec{};
            BVec k_vec{};
            const int key = matrix_lane;
            const int slot = key < valid_keys ? args.kv_indices[tile_begin + key] : -1;
#pragma unroll
            for(int k = 0; k < kValuesPerLane; ++k)
            {
                const int d = k_begin + k_group * kValuesPerLane + k;
                const long q_offset =
                    (static_cast<long>(token) * args.heads + matrix_lane) * kHeadDim + d;
                q_vec[k] = bit_cast<bf16_t>(args.q[q_offset]);
                k_vec[k] = slot >= 0 && slot < args.pool_slots
                               ? bit_cast<bf16_t>(
                                     args.unified_kv[static_cast<long>(slot) * kHeadDim + d])
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
        float alpha[kValuesPerLane];
#pragma unroll
        for(int i = 0; i < kValuesPerLane; ++i)
        {
            const int head = k_group * kValuesPerLane + i;
            const float score =
                matrix_lane < valid_keys ? score_vec[i] * args.softmax_scale : -__builtin_inff();
            float block_max = score;
#pragma unroll
            for(int offset = kTile / 2; offset > 0; offset >>= 1)
                block_max = fmaxf(block_max, __shfl_down(block_max, offset, kTile));
            block_max = __shfl(block_max, 0, kTile);

            const float next_max = fmaxf(max_value[i], block_max);
            alpha[i] = expf(max_value[i] - next_max);
            const float weight = matrix_lane < valid_keys ? expf(score - next_max) : 0.0f;
            float block_sum = weight;
#pragma unroll
            for(int offset = kTile / 2; offset > 0; offset >>= 1)
                block_sum += __shfl_down(block_sum, offset, kTile);
            block_sum = __shfl(block_sum, 0, kTile);

            normalizer[i] = normalizer[i] * alpha[i] + block_sum;
            max_value[i] = next_max;
            probabilities[head * kTile + matrix_lane] = type_convert<bf16_t>(weight);
        }
        __syncthreads();

        typename WarpGemm::CWarpTensor value_tensors[kOutputTiles];
#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n)
            value_tensors[n].get_thread_buffer().template set_as<CVec>(number<0>{},
                                                                      CVec{0.0f});

        AVec p_vec{};
#pragma unroll
        for(int k = 0; k < kValuesPerLane; ++k)
            p_vec[k] = probabilities[matrix_lane * kTile + k_group * kValuesPerLane + k];
        typename WarpGemm::AWarpTensor p_tensor;
        p_tensor.get_thread_buffer().template set_as<AVec>(number<0>{}, p_vec);

#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n)
        {
            BVec v_vec{};
#pragma unroll
            for(int k = 0; k < kValuesPerLane; ++k)
            {
                const int key = k_group * kValuesPerLane + k;
                const int slot = key < valid_keys ? args.kv_indices[tile_begin + key] : -1;
                const int d = n * kTile + matrix_lane;
                v_vec[k] = slot >= 0 && slot < args.pool_slots
                               ? bit_cast<bf16_t>(
                                     args.unified_kv[static_cast<long>(slot) * kHeadDim + d])
                               : type_convert<bf16_t>(0.0f);
            }
            typename WarpGemm::BWarpTensor v_tensor;
            v_tensor.get_thread_buffer().template set_as<BVec>(number<0>{}, v_vec);
            WarpGemm{}(value_tensors[n], p_tensor, v_tensor);
        }

#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n)
        {
            const CVec value_vec =
                value_tensors[n].get_thread_buffer().template get_as<CVec>()[number<0>{}];
#pragma unroll
            for(int i = 0; i < kValuesPerLane; ++i)
                accumulator[n][i] = accumulator[n][i] * alpha[i] + value_vec[i];
        }
        __syncthreads();
    }

#pragma unroll
    for(int i = 0; i < kValuesPerLane; ++i)
    {
        const int head = k_group * kValuesPerLane + i;
        const float sink = args.attn_sink[head];
        const float final_max = fmaxf(max_value[i], sink);
        const float kv_scale = expf(max_value[i] - final_max);
        const float sink_weight = expf(sink - final_max);
        const float denominator = normalizer[i] * kv_scale + sink_weight;
        const float output_scale = denominator > 0.0f ? kv_scale / denominator : 0.0f;
#pragma unroll
        for(int n = 0; n < kOutputTiles; ++n)
        {
            const int d = n * kTile + matrix_lane;
            const long output_offset =
                (static_cast<long>(token) * args.heads + head) * kHeadDim + d;
            args.output[output_offset] =
                ck::type_convert<ck::bhalf_t>(accumulator[n][i] * output_scale);
        }
    }
}

struct UnifiedSparseDecodeWorkspace
{
    float* max_partial;    // [tokens, splits, heads]
    float* norm_partial;   // [tokens, splits, heads]
    float* output_partial; // [tokens, splits, heads, 512]
};

inline constexpr int kMfmaSplits = 2;

inline std::size_t get_mfma_split_workspace_size(const UnifiedSparseDecodeArgs& args,
                                                  int splits = kMfmaSplits)
{
    const std::size_t rows = static_cast<std::size_t>(args.tokens) * splits * args.heads;
    return (rows * 2 + rows * kHeadDim) * sizeof(float);
}

inline UnifiedSparseDecodeWorkspace
partition_mfma_split_workspace(void* workspace,
                               const UnifiedSparseDecodeArgs& args,
                               int splits = kMfmaSplits)
{
    const std::size_t rows = static_cast<std::size_t>(args.tokens) * splits * args.heads;
    auto* base = static_cast<float*>(workspace);
    return {base, base + rows, base + rows * 2};
}

// Four waves share a token/split CTA. Each wave owns D128 of the output, which
// cuts accumulator pressure by 4x. QK is deliberately recomputed per wave in
// this first split implementation so all four gfx90a SIMDs stay occupied.
template <bool StageQ, bool PreloadQ, bool PipelineKV>
__global__ __launch_bounds__(kWaveSize * 4) void
unified_sparse_decode_d512_mfma_split_core_kernel(UnifiedSparseDecodeArgs args,
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
    __shared__ float alpha_shared[kLocalHeads];

    const int token = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int matrix_lane = lane % kTile;
    const int k_group = lane / kTile;
    const int begin = args.kv_indptr[token];
    const int end = args.kv_indptr[token + 1];
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
            q_tile[head * kQLdsStride + d] = bit_cast<bf16_t>(args.q[q_offset]);
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
                const long q_offset =
                    (static_cast<long>(token) * args.heads + matrix_lane) * kHeadDim + d;
                q_fragments[fragment][k] = bit_cast<bf16_t>(args.q[q_offset]);
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
                    q_vec[k] = bit_cast<bf16_t>(args.q[q_offset]);
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
        score = softmax_key < valid_keys ? score * args.softmax_scale : -__builtin_inff();

        float block_max = score;
#pragma unroll
        for(int offset = kTile / 2; offset > 0; offset >>= 1)
            block_max = fmaxf(block_max, __shfl_down(block_max, offset, kTile));
        block_max = __shfl(block_max, 0, kTile);

        const float next_max = fmaxf(softmax_max, block_max);
        const float alpha = expf(softmax_max - next_max);
        const float weight = softmax_key < valid_keys ? expf(score - next_max) : 0.0f;
        float block_sum = weight;
#pragma unroll
        for(int offset = kTile / 2; offset > 0; offset >>= 1)
            block_sum += __shfl_down(block_sum, offset, kTile);
        block_sum = __shfl(block_sum, 0, kTile);

        softmax_norm = softmax_norm * alpha + block_sum;
        softmax_max = next_max;
        probabilities[softmax_head * kTile + softmax_key] = type_convert<bf16_t>(weight);
        if(softmax_key == 0) alpha_shared[softmax_head] = alpha;
        __syncthreads();

        typename WarpGemm::CWarpTensor value_tensors[kOutputTilesPerWave];
#pragma unroll
        for(int n = 0; n < kOutputTilesPerWave; ++n)
            value_tensors[n].get_thread_buffer().template set_as<CVec>(number<0>{},
                                                                      CVec{0.0f});

        AVec p_vec{};
#pragma unroll
        for(int k = 0; k < kValuesPerLane; ++k)
            p_vec[k] = probabilities[matrix_lane * kTile + k_group * kValuesPerLane + k];
        typename WarpGemm::AWarpTensor p_tensor;
        p_tensor.get_thread_buffer().template set_as<AVec>(number<0>{}, p_vec);

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

    const std::size_t partial_row =
        (static_cast<std::size_t>(token) * splits + split) * args.heads;
    if(softmax_key == 0)
    {
        workspace.max_partial[partial_row + softmax_head] = softmax_max;
        workspace.norm_partial[partial_row + softmax_head] = softmax_norm;
    }

#pragma unroll
    for(int n = 0; n < kOutputTilesPerWave; ++n)
    {
        const int output_tile = wave * kOutputTilesPerWave + n;
        const int d = output_tile * kTile + matrix_lane;
#pragma unroll
        for(int i = 0; i < kValuesPerLane; ++i)
        {
            const int head = k_group * kValuesPerLane + i;
            workspace.output_partial[(partial_row + head) * kHeadDim + d] = accumulator[n][i];
        }
    }
}

__global__ __launch_bounds__(kThreads) void unified_sparse_decode_d512_mfma_split_reduce_kernel(
    UnifiedSparseDecodeArgs args, UnifiedSparseDecodeWorkspace workspace, int splits)
{
    const int token = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t row0 =
        static_cast<std::size_t>(token) * splits * args.heads + head;

    float final_max = args.attn_sink[head];
#pragma unroll
    for(int split = 0; split < splits; ++split)
        final_max = fmaxf(final_max, workspace.max_partial[row0 + split * args.heads]);

    float denominator = expf(args.attn_sink[head] - final_max);
#pragma unroll
    for(int split = 0; split < splits; ++split)
    {
        const std::size_t row = row0 + split * args.heads;
        denominator += workspace.norm_partial[row] *
                       expf(workspace.max_partial[row] - final_max);
    }
    const float inv_denominator = denominator > 0.0f ? 1.0f / denominator : 0.0f;

#pragma unroll
    for(int i = 0; i < kHeadDim / kThreads; ++i)
    {
        const int d = lane + i * kThreads;
        float value = 0.0f;
#pragma unroll
        for(int split = 0; split < splits; ++split)
        {
            const std::size_t row = row0 + split * args.heads;
            value += workspace.output_partial[row * kHeadDim + d] *
                     expf(workspace.max_partial[row] - final_max);
        }
        const long output_offset =
            (static_cast<long>(token) * args.heads + head) * kHeadDim + d;
        args.output[output_offset] =
            ck::type_convert<ck::bhalf_t>(value * inv_denominator);
    }
}

inline hipError_t launch_unified_sparse_decode_d512_reference(
    const UnifiedSparseDecodeArgs& args, hipStream_t stream = nullptr)
{
    if(!is_supported(args))
    {
        return hipErrorInvalidValue;
    }

    const dim3 grid(args.tokens, args.heads, 1);
    const dim3 block(kThreads, 1, 1);
    hipLaunchKernelGGL(
        unified_sparse_decode_d512_reference_kernel, grid, block, 0, stream, args);
    return hipGetLastError();
}

template <int HeadGroup, int KvTile>
inline hipError_t launch_unified_sparse_decode_d512_wave_tiled(
    const UnifiedSparseDecodeArgs& args, hipStream_t stream = nullptr)
{
    static_assert(HeadGroup > 0 && HeadGroup <= kLocalHeads);
    static_assert(KvTile > 0);
    static_assert(HeadGroup * kWaveSize <= 1024);

    if(!is_supported(args) || args.heads % HeadGroup != 0)
    {
        return hipErrorInvalidValue;
    }

    const dim3 grid(args.tokens, args.heads / HeadGroup, 1);
    const dim3 block(HeadGroup * kWaveSize, 1, 1);
    hipLaunchKernelGGL((unified_sparse_decode_d512_wave_tiled_kernel<HeadGroup, KvTile>),
                       grid,
                       block,
                       0,
                       stream,
                       args);
    return hipGetLastError();
}

inline hipError_t launch_unified_sparse_decode_d512_h4_k8(const UnifiedSparseDecodeArgs& args,
                                                           hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_wave_tiled<4, 8>(args, stream);
}

inline hipError_t launch_unified_sparse_decode_d512_h8_k8(const UnifiedSparseDecodeArgs& args,
                                                           hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_wave_tiled<8, 8>(args, stream);
}

inline hipError_t launch_unified_sparse_decode_d512_h8_k16(const UnifiedSparseDecodeArgs& args,
                                                            hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_wave_tiled<8, 16>(args, stream);
}

inline hipError_t launch_unified_sparse_decode_d512_mfma(const UnifiedSparseDecodeArgs& args,
                                                          hipStream_t stream = nullptr)
{
    if(!is_supported(args) || args.heads % 4 != 0)
    {
        return hipErrorInvalidValue;
    }

    const dim3 grid(args.tokens, args.heads / 4, 1);
    const dim3 block(kWaveSize, 1, 1);
    hipLaunchKernelGGL(unified_sparse_decode_d512_mfma_kernel, grid, block, 0, stream, args);
    return hipGetLastError();
}

inline hipError_t launch_unified_sparse_decode_d512_mfma_m16(
    const UnifiedSparseDecodeArgs& args, hipStream_t stream = nullptr)
{
    if(!is_supported(args))
    {
        return hipErrorInvalidValue;
    }

    const dim3 grid(args.tokens, 1, 1);
    const dim3 block(kWaveSize, 1, 1);
    hipLaunchKernelGGL(unified_sparse_decode_d512_mfma_m16_kernel, grid, block, 0, stream, args);
    return hipGetLastError();
}

template <bool StageQ, bool PreloadQ, bool PipelineKV>
inline hipError_t launch_unified_sparse_decode_d512_mfma_split2_impl(
    const UnifiedSparseDecodeArgs& args,
    void* workspace_ptr,
    int splits,
    hipStream_t stream = nullptr)
{
    if(!is_supported(args) || workspace_ptr == nullptr)
    {
        return hipErrorInvalidValue;
    }
    if(splits <= 0) return hipErrorInvalidValue;
    const std::size_t rows = static_cast<std::size_t>(args.tokens) * splits * args.heads;
    auto* base = static_cast<float*>(workspace_ptr);
    const UnifiedSparseDecodeWorkspace workspace{base, base + rows, base + rows * 2};
    const dim3 core_grid(args.tokens, splits, 1);
    const dim3 core_block(kWaveSize * 4, 1, 1);
    hipLaunchKernelGGL(
        (unified_sparse_decode_d512_mfma_split_core_kernel<StageQ, PreloadQ, PipelineKV>),
                       core_grid,
                       core_block,
                       0,
                       stream,
                       args,
                       workspace,
                       splits);
    hipError_t status = hipGetLastError();
    if(status != hipSuccess) return status;

    const dim3 reduce_grid(args.tokens, args.heads, 1);
    const dim3 reduce_block(kThreads, 1, 1);
    hipLaunchKernelGGL(unified_sparse_decode_d512_mfma_split_reduce_kernel,
                       reduce_grid,
                       reduce_block,
                       0,
                       stream,
                       args,
                       workspace,
                       splits);
    return hipGetLastError();
}

inline hipError_t launch_unified_sparse_decode_d512_mfma_split2(
    const UnifiedSparseDecodeArgs& args, void* workspace_ptr, hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_mfma_split2_impl<false, false, false>(
        args, workspace_ptr, 2, stream);
}

inline hipError_t launch_unified_sparse_decode_d512_mfma_split2_qlds(
    const UnifiedSparseDecodeArgs& args, void* workspace_ptr, hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_mfma_split2_impl<true, false, false>(args,
                                                                                  workspace_ptr,
                                                                                  2,
                                                                                  stream);
}

inline hipError_t launch_unified_sparse_decode_d512_mfma_split2_qreg(
    const UnifiedSparseDecodeArgs& args, void* workspace_ptr, hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_mfma_split2_impl<false, true, false>(args,
                                                                                  workspace_ptr,
                                                                                  2,
                                                                                  stream);
}

inline hipError_t launch_unified_sparse_decode_d512_mfma_split2_qreg_kvprefetch(
    const UnifiedSparseDecodeArgs& args, void* workspace_ptr, hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_mfma_split2_impl<false, true, true>(args,
                                                                                 workspace_ptr,
                                                                                 2,
                                                                                 stream);
}

inline hipError_t launch_unified_sparse_decode_d512_mfma_split4_qreg_kvprefetch(
    const UnifiedSparseDecodeArgs& args, void* workspace_ptr, hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_mfma_split2_impl<false, true, true>(args,
                                                                                 workspace_ptr,
                                                                                 4,
                                                                                 stream);
}

// Candidate integration entry point. It is intentionally separate from the
// workspace-free semantic baseline so callers must opt in explicitly.
inline hipError_t launch_unified_sparse_decode_d512_gfx90a(
    const UnifiedSparseDecodeArgs& args,
    void* workspace_ptr,
    std::size_t workspace_bytes,
    hipStream_t stream = nullptr)
{
    if(ck::get_device_name() != "gfx90a" ||
       workspace_bytes < get_mfma_split_workspace_size(args, 2))
    {
        return hipErrorInvalidValue;
    }
    return launch_unified_sparse_decode_d512_mfma_split2_qreg_kvprefetch(
        args, workspace_ptr, stream);
}

inline hipError_t launch_unified_sparse_decode_d512(const UnifiedSparseDecodeArgs& args,
                                                     hipStream_t stream = nullptr)
{
    return launch_unified_sparse_decode_d512_wave_tiled<kDefaultHeadGroup, kDefaultKvTile>(args,
                                                                                           stream);
}

} // namespace ck_tile::dsv4
