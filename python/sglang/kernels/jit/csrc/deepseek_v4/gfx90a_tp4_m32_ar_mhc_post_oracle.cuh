#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <torch/all.h>

#include "custom_all_reduce.cuh"

#include <cstdint>
#include <stdexcept>

namespace sglang {

constexpr uint32_t kArMhcM = 32;
constexpr uint32_t kArMhcH = 4096;
constexpr uint32_t kArMhcC = 4;
constexpr uint32_t kArMhcWorld = 4;
constexpr uint32_t kArMhcBlocksPerRow = 2;
constexpr uint32_t kArMhcBlocks = kArMhcM * kArMhcBlocksPerRow;
constexpr uint32_t kArMhcThreads = 256;
constexpr uint32_t kArMhcPacksPerRow = kArMhcH / 8;
constexpr uint32_t kArMhcPacksPerBlock =
    kArMhcPacksPerRow / kArMhcBlocksPerRow;
constexpr uint32_t kArMhcPartialsPerBlock = kArMhcThreads / 32;
constexpr uint32_t kArMhcPartialsPerRow =
    kArMhcBlocksPerRow * kArMhcPartialsPerBlock;

// The direct-HIP workspace is IPC registered.  Every local workspace contains
// its own view of all peer arrivals, plus one monotonically increasing epoch
// per block.  This is intentionally separate from AIter's Signal: the oracle
// needs 64 blocks while the SGLang copy of that structure has only 36 slots.
constexpr uint32_t kArMhcFlagOffset = 0;
constexpr uint32_t kArMhcStartOffset = kArMhcFlagOffset + kArMhcBlocks;
constexpr uint32_t kArMhcEndOffset =
    kArMhcStartOffset + kArMhcBlocks * kArMhcWorld;
constexpr uint32_t kArMhcWorkspaceU32 =
    kArMhcEndOffset + kArMhcBlocks * kArMhcWorld;

using ArMhcPack = aiter::array_t<__hip_bfloat16, 8>;

__device__ __forceinline__ ArMhcPack ar_mhc_peer_load(
    const ArMhcPack* ptr) {
  union {
    ArMhcPack pack;
    uint64_t words[2];
  } value;
  const auto* words = reinterpret_cast<const uint64_t*>(ptr);
  value.words[0] = __builtin_nontemporal_load(words);
  value.words[1] = __builtin_nontemporal_load(words + 1);
  return value.pack;
}

__device__ __forceinline__ uint32_t ar_mhc_epoch_load(
    const uint32_t* ptr, int order) {
  return __scoped_atomic_load_n(ptr, order, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void ar_mhc_epoch_store(
    uint32_t* ptr, uint32_t value, int order) {
  __scoped_atomic_store_n(ptr, value, order, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t ar_mhc_start(
    const aiter::RankData& sync_rd, uint32_t* local_sync, int rank) {
  const uint32_t block = blockIdx.x;
  const uint32_t epoch =
      ar_mhc_epoch_load(local_sync + kArMhcFlagOffset + block,
                        __ATOMIC_RELAXED) +
      1;
  // Order the graph producer's stores before publishing readiness to peers.
  __threadfence_system();
  if (threadIdx.x < kArMhcWorld) {
    const uint32_t peer = threadIdx.x;
    auto* peer_sync = reinterpret_cast<uint32_t*>(
        const_cast<void*>(sync_rd.ptrs[peer]));
    ar_mhc_epoch_store(
        peer_sync + kArMhcStartOffset + block * kArMhcWorld + rank,
        epoch, __ATOMIC_RELEASE);
    while (ar_mhc_epoch_load(
               local_sync + kArMhcStartOffset + block * kArMhcWorld + peer,
               __ATOMIC_ACQUIRE) < epoch) {
    }
  }
  __syncthreads();
  return epoch;
}

__device__ __forceinline__ void ar_mhc_end(
    const aiter::RankData& sync_rd, uint32_t* local_sync, int rank,
    uint32_t epoch) {
  __syncthreads();
  __threadfence_system();
  const uint32_t block = blockIdx.x;
  if (threadIdx.x < kArMhcWorld) {
    const uint32_t peer = threadIdx.x;
    auto* peer_sync = reinterpret_cast<uint32_t*>(
        const_cast<void*>(sync_rd.ptrs[peer]));
    ar_mhc_epoch_store(
        peer_sync + kArMhcEndOffset + block * kArMhcWorld + rank,
        epoch, __ATOMIC_RELEASE);
    while (ar_mhc_epoch_load(
               local_sync + kArMhcEndOffset + block * kArMhcWorld + peer,
               __ATOMIC_ACQUIRE) < epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ar_mhc_epoch_store(local_sync + kArMhcFlagOffset + block, epoch,
                       __ATOMIC_RELAXED);
  }
}

template <bool WriteReduced>
__global__ void __launch_bounds__(kArMhcThreads, 1)
    gfx90a_tp4_m32_ar_mhc_post_kernel(
        aiter::RankData* input_rd_ptr, aiter::RankData* sync_rd_ptr,
        uint32_t* local_sync, const __hip_bfloat16* __restrict__ residual,
        const float* __restrict__ post, const float* __restrict__ comb,
        __hip_bfloat16* __restrict__ output,
        float* __restrict__ rms_partials,
        __hip_bfloat16* __restrict__ reduced_debug, int rank) {
  const aiter::RankData input_rd = *input_rd_ptr;
  const aiter::RankData sync_rd = *sync_rd_ptr;
  const uint32_t epoch = ar_mhc_start(sync_rd, local_sync, rank);

  const uint32_t token = blockIdx.x / kArMhcBlocksPerRow;
  const uint32_t half = blockIdx.x % kArMhcBlocksPerRow;
  const uint32_t pack_in_row = half * kArMhcPacksPerBlock + threadIdx.x;
  const uint32_t pack_index = token * kArMhcPacksPerRow + pack_in_row;

  // Production selects AIter's two-stage AR at 256 KiB.  Its flattened owner
  // chunks are eight complete token rows, and every owner accumulates peers in
  // rotated order owner, owner+1, owner+2, owner+3.  Preserve that order before
  // the production BF16 all-reduce rounding boundary.
  const auto* p0 = reinterpret_cast<const ArMhcPack*>(input_rd.ptrs[0]);
  const auto* p1 = reinterpret_cast<const ArMhcPack*>(input_rd.ptrs[1]);
  const auto* p2 = reinterpret_cast<const ArMhcPack*>(input_rd.ptrs[2]);
  const auto* p3 = reinterpret_cast<const ArMhcPack*>(input_rd.ptrs[3]);
  const ArMhcPack v0 = ar_mhc_peer_load(p0 + pack_index);
  const ArMhcPack v1 = ar_mhc_peer_load(p1 + pack_index);
  const ArMhcPack v2 = ar_mhc_peer_load(p2 + pack_index);
  const ArMhcPack v3 = ar_mhc_peer_load(p3 + pack_index);
  ArMhcPack reduced;
  const uint32_t owner = token / 8;
#pragma unroll
  for (uint32_t j = 0; j < 8; ++j) {
    float value;
    if (owner == 0) {
      value = static_cast<float>(v0.data[j]);
      value += static_cast<float>(v1.data[j]);
      value += static_cast<float>(v2.data[j]);
      value += static_cast<float>(v3.data[j]);
    } else if (owner == 1) {
      value = static_cast<float>(v1.data[j]);
      value += static_cast<float>(v2.data[j]);
      value += static_cast<float>(v3.data[j]);
      value += static_cast<float>(v0.data[j]);
    } else if (owner == 2) {
      value = static_cast<float>(v2.data[j]);
      value += static_cast<float>(v3.data[j]);
      value += static_cast<float>(v0.data[j]);
      value += static_cast<float>(v1.data[j]);
    } else {
      value = static_cast<float>(v3.data[j]);
      value += static_cast<float>(v0.data[j]);
      value += static_cast<float>(v1.data[j]);
      value += static_cast<float>(v2.data[j]);
    }
    reduced.data[j] = static_cast<__hip_bfloat16>(value);
  }
  if constexpr (WriteReduced) {
    reinterpret_cast<ArMhcPack*>(reduced_debug)[pack_index] = reduced;
  }

  const uint32_t hidden0 = pack_in_row * 8;
  float square_sum[kArMhcC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (uint32_t channel = 0; channel < kArMhcC; ++channel) {
    ArMhcPack out_pack;
    const float post_value = post[token * kArMhcC + channel];
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      float value = post_value * static_cast<float>(reduced.data[j]);
      const uint32_t residual_base = token * kArMhcC * kArMhcH + hidden0 + j;
      float residual_value =
          comb[(token * 16) + channel] *
          static_cast<float>(residual[residual_base]);
      residual_value = fmaf(
          comb[(token * 16) + 4 + channel],
          static_cast<float>(residual[residual_base + kArMhcH]),
          residual_value);
      residual_value = fmaf(
          comb[(token * 16) + 8 + channel],
          static_cast<float>(residual[residual_base + 2 * kArMhcH]),
          residual_value);
      residual_value = fmaf(
          comb[(token * 16) + 12 + channel],
          static_cast<float>(residual[residual_base + 3 * kArMhcH]),
          residual_value);
      value += residual_value;
      out_pack.data[j] = static_cast<__hip_bfloat16>(value);
      const float rounded = static_cast<float>(out_pack.data[j]);
      square_sum[channel] += rounded * rounded;
    }
    const uint32_t output_pack =
        (token * kArMhcC + channel) * kArMhcPacksPerRow + pack_in_row;
    reinterpret_cast<ArMhcPack*>(output)[output_pack] = out_pack;
  }

  // Match the production BLOCK_H=256 partial layout.  Every wave32 owns 32
  // adjacent 16-byte packs, i.e. exactly 256 contiguous hidden elements.
#pragma unroll
  for (uint32_t channel = 0; channel < kArMhcC; ++channel) {
    float partial = square_sum[channel];
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down(partial, offset, 32);
    }
    if ((threadIdx.x & 31) == 0) {
      const uint32_t partial_id =
          half * kArMhcPartialsPerBlock + threadIdx.x / 32;
      rms_partials[(token * kArMhcC + channel) *
                       kArMhcPartialsPerRow +
                   partial_id] = partial;
    }
  }

  // This exit handshake is mandatory: a fast rank must not let the next graph
  // epoch overwrite its registered input while a slower peer is still reading.
  ar_mhc_end(sync_rd, local_sync, rank, epoch);
}

struct Gfx90aTp4M32ArMhcPostOracle {
  static hipStream_t stream_for(DLDevice device) {
    return sglang::host::LaunchKernel::resolve_device(device);
  }

  template <bool WriteReduced>
  static void launch(int64_t fa, const tvm::ffi::TensorView input,
                     const tvm::ffi::TensorView sync_workspace,
                     const tvm::ffi::TensorView residual,
                     const tvm::ffi::TensorView post,
                     const tvm::ffi::TensorView comb,
                     const tvm::ffi::TensorView output,
                     const tvm::ffi::TensorView rms_partials,
                     const tvm::ffi::TensorView reduced_debug, int64_t rank) {
    if (rank < 0 || rank >= kArMhcWorld) {
      throw std::runtime_error("TP4 oracle rank must be in [0,4)");
    }
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(input.device());
    auto* input_rd = comm->get_buffer_RD(stream, input.data_ptr());
    auto* sync_rd = comm->get_buffer_RD(stream, sync_workspace.data_ptr());
    sglang::host::LaunchKernel(kArMhcBlocks, kArMhcThreads, stream)(
        gfx90a_tp4_m32_ar_mhc_post_kernel<WriteReduced>, input_rd, sync_rd,
        static_cast<uint32_t*>(sync_workspace.data_ptr()),
        static_cast<const __hip_bfloat16*>(residual.data_ptr()),
        static_cast<const float*>(post.data_ptr()),
        static_cast<const float*>(comb.data_ptr()),
        static_cast<__hip_bfloat16*>(output.data_ptr()),
        static_cast<float*>(rms_partials.data_ptr()),
        static_cast<__hip_bfloat16*>(reduced_debug.data_ptr()),
        static_cast<int>(rank));
  }

  static void run(int64_t fa, const tvm::ffi::TensorView input,
                  const tvm::ffi::TensorView sync_workspace,
                  const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView post,
                  const tvm::ffi::TensorView comb,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView rms_partials,
                  const tvm::ffi::TensorView reduced_debug, int64_t rank) {
    launch<false>(fa, input, sync_workspace, residual, post, comb, output,
                  rms_partials, reduced_debug, rank);
  }

  static void run_debug(int64_t fa, const tvm::ffi::TensorView input,
                        const tvm::ffi::TensorView sync_workspace,
                        const tvm::ffi::TensorView residual,
                        const tvm::ffi::TensorView post,
                        const tvm::ffi::TensorView comb,
                        const tvm::ffi::TensorView output,
                        const tvm::ffi::TensorView rms_partials,
                        const tvm::ffi::TensorView reduced_debug,
                        int64_t rank) {
    launch<true>(fa, input, sync_workspace, residual, post, comb, output,
                 rms_partials, reduced_debug, rank);
  }
};

}  // namespace sglang
