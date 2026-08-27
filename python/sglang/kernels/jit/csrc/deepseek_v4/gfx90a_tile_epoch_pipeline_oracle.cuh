#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <torch/all.h>

#include "custom_all_reduce.cuh"
#include "quick_all_reduce_base.h"

#include <cstdint>

namespace sglang {

constexpr uint32_t kTileEpochM = 32;
constexpr uint32_t kTileEpochH = 4096;
constexpr uint32_t kTileEpochWidth = 256;
constexpr uint32_t kTileEpochTiles = kTileEpochH / kTileEpochWidth;
constexpr uint32_t kTileEpochWorld = 8;
constexpr uint32_t kTileEpochSlots = 1;
constexpr size_t kTileEpochSlotElements =
    static_cast<size_t>(kTileEpochM) * kTileEpochH;
constexpr size_t kTileEpochDataBytes =
    kTileEpochSlotElements * sizeof(__hip_bfloat16);
constexpr size_t kTileEpochProducedOffset = kTileEpochDataBytes;
constexpr size_t kTileEpochSignalBytes =
    kTileEpochTiles * kTileEpochWorld * sizeof(uint32_t);
constexpr size_t kTileEpochConsumedOffset =
    kTileEpochProducedOffset + kTileEpochSignalBytes;
constexpr size_t kTileEpochEndOffset =
    kTileEpochConsumedOffset + kTileEpochSignalBytes;

union TileEpochBf16x8 {
  aiter::int32x4_t words;
  __hip_bfloat16 values[8];
};

__device__ __forceinline__ TileEpochBf16x8 peer_load_bf16x8(
    const void* base, uint32_t byte_offset) {
  // gfx90a MUBUF_ACQUIRE=GLC.  A plain flat load may retain an old peer line
  // across graph replays even though the epoch flag itself is system-scoped.
  // Reuse AIter's own quick-all-reduce load primitive and coherence setting.
  aiter::BufferResource resource(
      const_cast<void*>(base),
      kTileEpochSlotElements * sizeof(__hip_bfloat16));
  TileEpochBf16x8 result;
  // MUBUF_ACQUIRE is defined as 1 for gfx90a only during the device pass;
  // keep the same literal here so HIP's host pass can parse this helper too.
  result.words =
      aiter::buffer_load_dwordx4(resource.descriptor, byte_offset, 0, 1);
  return result;
}

__device__ __forceinline__ uint32_t epoch_load(const uint32_t* ptr) {
  return __scoped_atomic_load_n(
      ptr, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void epoch_store(uint32_t* ptr, uint32_t value) {
  __scoped_atomic_store_n(
      ptr, value, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
}

// Each CTA owns one H tile. Signals are inboxes: producer rank r writes
// produced[tile,r] in every destination rank, which then polls local memory.
__global__ void tile_epoch_producer_kernel(
    aiter::RankData* data_rd, aiter::RankData* produced_rd,
    aiter::RankData* consumed_rd, __hip_bfloat16* local_data,
    uint32_t* local_produced, const uint32_t* local_consumed, int rank) {
  const uint32_t tile = blockIdx.x;
  const uint32_t own_slot = tile * kTileEpochWorld + rank;
  const uint32_t epoch = epoch_load(local_produced + own_slot) + 1;
  const uint32_t data_slot = 0;

  if (threadIdx.x < kTileEpochWorld) {
    const uint32_t peer_rank = threadIdx.x;
    const auto* peer_consumed = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(consumed_rd->ptrs[peer_rank]) +
        kTileEpochConsumedOffset);
    while (epoch_load(peer_consumed + tile * kTileEpochWorld + peer_rank) + 1 <
           epoch) {
    }
  }
  __syncthreads();

  constexpr uint32_t kElementsPerStore = 8;
  constexpr uint32_t kVectorsPerTile =
      kTileEpochM * kTileEpochWidth / kElementsPerStore;
  for (uint32_t vector_linear = threadIdx.x; vector_linear < kVectorsPerTile;
       vector_linear += blockDim.x) {
    const uint32_t tile_element = vector_linear * kElementsPerStore;
    const uint32_t row = tile_element / kTileEpochWidth;
    const uint32_t col_base =
        tile * kTileEpochWidth + tile_element % kTileEpochWidth;
    TileEpochBf16x8 values;
#pragma unroll
    for (int element = 0; element < kElementsPerStore; ++element) {
      const uint32_t col = col_base + element;
      const uint32_t code =
          (col * 17u + row * 29u + epoch * 13u + rank * 37u) & 255u;
      const float value =
          (static_cast<int32_t>(code) - 128) * (1.0f / 64.0f) + rank * 0.25f;
      values.values[element] = __float2bfloat16(value);
    }
    const size_t offset = data_slot * kTileEpochSlotElements +
                          static_cast<size_t>(row) * kTileEpochH + col_base;
    __builtin_nontemporal_store(
        values.words,
        reinterpret_cast<aiter::int32x4_t*>(local_data) +
            offset / kElementsPerStore);
  }
  // Every writer must publish its own global stores. A release performed only
  // by thread 0 does not order stores issued by the other CTA threads.
  __threadfence_system();
  __syncthreads();

  if (threadIdx.x == 0) epoch_store(local_produced + own_slot, epoch);
}

__global__ void tile_epoch_reduce_kernel(
    aiter::RankData* data_rd, aiter::RankData* produced_rd,
    aiter::RankData* consumed_rd, const uint32_t* local_produced,
    uint32_t* local_consumed, __hip_bfloat16* output, int rank) {
  const uint32_t tile = blockIdx.x;
  const uint32_t own_slot = tile * kTileEpochWorld + rank;
  const uint32_t epoch = epoch_load(local_consumed + own_slot) + 1;
  const uint32_t data_slot = 0;

  if (threadIdx.x < kTileEpochWorld) {
    const uint32_t peer_rank = threadIdx.x;
    const auto* peer_produced = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(produced_rd->ptrs[peer_rank]) +
        kTileEpochProducedOffset);
    while (epoch_load(peer_produced + tile * kTileEpochWorld + peer_rank) < epoch) {
    }
  }
  __syncthreads();

  constexpr uint32_t kElementsPerLoad = 8;
  constexpr uint32_t kVectorsPerTile =
      kTileEpochM * kTileEpochWidth / kElementsPerLoad;
  for (uint32_t vector_linear = threadIdx.x; vector_linear < kVectorsPerTile;
       vector_linear += blockDim.x) {
    const uint32_t tile_element = vector_linear * kElementsPerLoad;
    const uint32_t row = tile_element / kTileEpochWidth;
    const uint32_t col =
        tile * kTileEpochWidth + tile_element % kTileEpochWidth;
    const size_t output_offset = static_cast<size_t>(row) * kTileEpochH + col;
    TileEpochBf16x8 peer_values[kTileEpochWorld];
#pragma unroll
    for (int peer = 0; peer < kTileEpochWorld; ++peer) {
      const auto* peer_slot =
          reinterpret_cast<const uint8_t*>(data_rd->ptrs[peer]) +
          data_slot * kTileEpochSlotElements * sizeof(__hip_bfloat16);
      // Keep the raw-buffer voffset within one slot.  Using the allocation
      // base plus a slot-1 offset lands exactly at 0x40000 on this shape and
      // caused gfx90a reducers to stall; rebasing the descriptor avoids that
      // address-encoding boundary without changing the registered allocation.
      peer_values[peer] = peer_load_bf16x8(
          peer_slot, output_offset * sizeof(__hip_bfloat16));
    }
#pragma unroll
    for (int element = 0; element < kElementsPerLoad; ++element) {
      float sum = __bfloat162float(peer_values[0].values[element]);
#pragma unroll
      for (int peer = 1; peer < kTileEpochWorld; ++peer) {
        sum += __bfloat162float(peer_values[peer].values[element]);
      }
      output[output_offset + element] = __float2bfloat16(sum);
    }
  }
  // Complete every reader's peer loads before the single ack publisher makes
  // this tile reusable by the next graph replay.
  __threadfence_system();
  __syncthreads();

  if (threadIdx.x == 0) epoch_store(local_consumed + own_slot, epoch);
}

// Diagnostics kept separate from the production-shaped reducer so a stall can
// be attributed to flag acquisition or peer payload access.
__global__ void tile_epoch_wait_only_kernel(aiter::RankData* produced_rd,
                                            const uint32_t* local_consumed,
                                            uint32_t* waited, int rank) {
  const uint32_t tile = blockIdx.x;
  const uint32_t own_slot = tile * kTileEpochWorld + rank;
  const uint32_t epoch = epoch_load(local_consumed + own_slot) + 1;
  if (threadIdx.x < kTileEpochWorld) {
    const uint32_t peer_rank = threadIdx.x;
    const auto* peer_produced = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(produced_rd->ptrs[peer_rank]) +
        kTileEpochProducedOffset);
    while (epoch_load(peer_produced + tile * kTileEpochWorld + peer_rank) <
           epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) waited[tile] = epoch;
}

__global__ void tile_epoch_ack_kernel(uint32_t* local_consumed, int rank) {
  const uint32_t tile = blockIdx.x;
  const uint32_t own_slot = tile * kTileEpochWorld + rank;
  const uint32_t epoch = epoch_load(local_consumed + own_slot) + 1;
  if (threadIdx.x == 0) epoch_store(local_consumed + own_slot, epoch);
}

__global__ void tile_epoch_load_only_kernel(aiter::RankData* data_rd,
                                            __hip_bfloat16* output,
                                            uint32_t epoch) {
  const uint32_t tile = blockIdx.x;
  const uint32_t data_slot = 0;
  constexpr uint32_t kElementsPerLoad = 8;
  constexpr uint32_t kVectorsPerTile =
      kTileEpochM * kTileEpochWidth / kElementsPerLoad;
  for (uint32_t vector_linear = threadIdx.x; vector_linear < kVectorsPerTile;
       vector_linear += blockDim.x) {
    const uint32_t tile_element = vector_linear * kElementsPerLoad;
    const uint32_t row = tile_element / kTileEpochWidth;
    const uint32_t col =
        tile * kTileEpochWidth + tile_element % kTileEpochWidth;
    const size_t output_offset = static_cast<size_t>(row) * kTileEpochH + col;
    TileEpochBf16x8 peer_values[kTileEpochWorld];
#pragma unroll
    for (int peer = 0; peer < kTileEpochWorld; ++peer) {
      const auto* peer_slot =
          reinterpret_cast<const uint8_t*>(data_rd->ptrs[peer]) +
          data_slot * kTileEpochSlotElements * sizeof(__hip_bfloat16);
      peer_values[peer] = peer_load_bf16x8(
          peer_slot, output_offset * sizeof(__hip_bfloat16));
    }
#pragma unroll
    for (int element = 0; element < kElementsPerLoad; ++element) {
      float sum = __bfloat162float(peer_values[0].values[element]);
#pragma unroll
      for (int peer = 1; peer < kTileEpochWorld; ++peer) {
        sum += __bfloat162float(peer_values[peer].values[element]);
      }
      output[output_offset + element] = __float2bfloat16(sum);
    }
  }
}

__global__ void tile_epoch_end_kernel(aiter::RankData* end_rd,
                                      uint32_t* local_end, int rank) {
  const uint32_t epoch = epoch_load(local_end + rank) + 1;
  if (threadIdx.x == 0) epoch_store(local_end + rank, epoch);
  __syncthreads();
  if (threadIdx.x < kTileEpochWorld) {
    const uint32_t peer_rank = threadIdx.x;
    const auto* peer_end = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(end_rd->ptrs[peer_rank]) +
        kTileEpochEndOffset);
    while (epoch_load(peer_end + peer_rank) < epoch) {
    }
  }
}

struct Gfx90aTileEpochPipelineOracle {
  static hipStream_t stream_for(DLDevice device) {
    return sglang::host::LaunchKernel::resolve_device(device);
  }

  static void producer(int64_t fa, const tvm::ffi::TensorView workspace,
                       const tvm::ffi::TensorView data,
                       const tvm::ffi::TensorView produced,
                       const tvm::ffi::TensorView consumed, int64_t rank) {
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    const auto device = data.device();
    auto stream = stream_for(device);
    auto* workspace_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(kTileEpochTiles, 256, stream)(
        tile_epoch_producer_kernel, workspace_rd, workspace_rd, workspace_rd,
        static_cast<__hip_bfloat16*>(data.data_ptr()),
        static_cast<uint32_t*>(produced.data_ptr()),
        static_cast<const uint32_t*>(consumed.data_ptr()), static_cast<int>(rank));
  }

  static void reduce(int64_t fa, const tvm::ffi::TensorView workspace,
                     const tvm::ffi::TensorView data,
                     const tvm::ffi::TensorView produced,
                     const tvm::ffi::TensorView consumed,
                     const tvm::ffi::TensorView output, int64_t rank) {
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    const auto device = data.device();
    auto stream = stream_for(device);
    auto* workspace_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(kTileEpochTiles, 256, stream)(
        tile_epoch_reduce_kernel, workspace_rd, workspace_rd, workspace_rd,
        static_cast<const uint32_t*>(produced.data_ptr()),
        static_cast<uint32_t*>(consumed.data_ptr()),
        static_cast<__hip_bfloat16*>(output.data_ptr()), static_cast<int>(rank));
  }

  static void end(int64_t fa, const tvm::ffi::TensorView workspace,
                  const tvm::ffi::TensorView end_epoch,
                  int64_t rank) {
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(end_epoch.device());
    auto* end_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(1, 64, stream)(
        tile_epoch_end_kernel, end_rd,
        static_cast<uint32_t*>(end_epoch.data_ptr()), static_cast<int>(rank));
  }

  static void wait_only(int64_t fa, const tvm::ffi::TensorView workspace,
                        const tvm::ffi::TensorView produced,
                        const tvm::ffi::TensorView consumed,
                        const tvm::ffi::TensorView waited, int64_t rank) {
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(produced.device());
    auto* produced_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(kTileEpochTiles, 64, stream)(
        tile_epoch_wait_only_kernel, produced_rd,
        static_cast<const uint32_t*>(consumed.data_ptr()),
        static_cast<uint32_t*>(waited.data_ptr()), static_cast<int>(rank));
  }

  static void load_only(int64_t fa, const tvm::ffi::TensorView workspace,
                        const tvm::ffi::TensorView data,
                        const tvm::ffi::TensorView output, int64_t epoch) {
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(data.device());
    auto* data_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(kTileEpochTiles, 256, stream)(
        tile_epoch_load_only_kernel, data_rd,
        static_cast<__hip_bfloat16*>(output.data_ptr()),
        static_cast<uint32_t>(epoch));
  }


  static void ack(const tvm::ffi::TensorView consumed, int64_t rank) {
    auto stream = stream_for(consumed.device());
    sglang::host::LaunchKernel(kTileEpochTiles, 64, stream)(
        tile_epoch_ack_kernel, static_cast<uint32_t*>(consumed.data_ptr()),
        static_cast<int>(rank));
  }
};

}  // namespace sglang
