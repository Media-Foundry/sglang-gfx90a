#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <torch/all.h>

#include "custom_all_reduce.cuh"

#include <cstdint>
#include <stdexcept>

namespace sglang {

constexpr uint32_t kProgArRows = 128;
constexpr uint32_t kProgArHidden = 4096;
constexpr uint32_t kProgArWorld = 4;
constexpr uint32_t kProgArBlocks = 12;
constexpr uint32_t kProgArThreads = 512;
constexpr uint32_t kProgArDraftBlocks = 9;
constexpr uint32_t kProgArAnchorBlocks = 3;
constexpr uint32_t kProgArPack = 8;
constexpr uint32_t kProgArPacksPerRow = kProgArHidden / kProgArPack;
constexpr uint32_t kProgArTotalPacks = kProgArRows * kProgArPacksPerRow;

// The first three regions form an independent TP4 entry/exit epoch protocol.
// The final three words coordinate the two local HIP streams without changing
// the rank-global collective sequence.
constexpr uint32_t kProgArFlagOffset = 0;
constexpr uint32_t kProgArStartOffset = kProgArFlagOffset + kProgArBlocks;
constexpr uint32_t kProgArEndOffset =
    kProgArStartOffset + kProgArBlocks * kProgArWorld;
constexpr uint32_t kProgArArmedOffset =
    kProgArEndOffset + kProgArBlocks * kProgArWorld;
constexpr uint32_t kProgArReadyOffset = kProgArArmedOffset + 1;
constexpr uint32_t kProgArArrivalsOffset = kProgArReadyOffset + 1;
constexpr uint32_t kProgArWorkspaceU32 = kProgArArrivalsOffset + 1;

using ProgArPack = aiter::array_t<__hip_bfloat16, kProgArPack>;

__device__ __forceinline__ uint32_t prog_ar_system_load(
    const uint32_t* ptr, int order) {
  return __scoped_atomic_load_n(ptr, order, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void prog_ar_system_store(
    uint32_t* ptr, uint32_t value, int order) {
  __scoped_atomic_store_n(ptr, value, order, __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t prog_ar_device_load(
    const uint32_t* ptr, int order) {
  return __scoped_atomic_load_n(ptr, order, __MEMORY_SCOPE_DEVICE);
}

__device__ __forceinline__ void prog_ar_device_store(
    uint32_t* ptr, uint32_t value, int order) {
  __scoped_atomic_store_n(ptr, value, order, __MEMORY_SCOPE_DEVICE);
}

__device__ __forceinline__ uint32_t prog_ar_start(
    const aiter::RankData& sync_rd, uint32_t* local_sync, int rank) {
  const uint32_t block = blockIdx.x;
  const uint32_t epoch =
      prog_ar_system_load(local_sync + kProgArFlagOffset + block,
                          __ATOMIC_RELAXED) +
      1;
  __threadfence_system();
  if (threadIdx.x < kProgArWorld) {
    const uint32_t peer = threadIdx.x;
    auto* peer_sync = reinterpret_cast<uint32_t*>(
        const_cast<void*>(sync_rd.ptrs[peer]));
    prog_ar_system_store(
        peer_sync + kProgArStartOffset + block * kProgArWorld + rank, epoch,
        __ATOMIC_RELEASE);
    while (prog_ar_system_load(
               local_sync + kProgArStartOffset + block * kProgArWorld + peer,
               __ATOMIC_ACQUIRE) < epoch) {
    }
  }
  __syncthreads();
  return epoch;
}

__device__ __forceinline__ void prog_ar_end(
    const aiter::RankData& sync_rd, uint32_t* local_sync, int rank,
    uint32_t epoch) {
  __syncthreads();
  __threadfence_system();
  const uint32_t block = blockIdx.x;
  if (threadIdx.x < kProgArWorld) {
    const uint32_t peer = threadIdx.x;
    auto* peer_sync = reinterpret_cast<uint32_t*>(
        const_cast<void*>(sync_rd.ptrs[peer]));
    prog_ar_system_store(
        peer_sync + kProgArEndOffset + block * kProgArWorld + rank, epoch,
        __ATOMIC_RELEASE);
    while (prog_ar_system_load(
               local_sync + kProgArEndOffset + block * kProgArWorld + peer,
               __ATOMIC_ACQUIRE) < epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    prog_ar_system_store(local_sync + kProgArFlagOffset + block, epoch,
                         __ATOMIC_RELAXED);
  }
}

__device__ __forceinline__ ProgArPack prog_ar_peer_load(
    const ProgArPack* ptr) {
  union {
    ProgArPack pack;
    uint64_t words[2];
  } value;
  const auto* words = reinterpret_cast<const uint64_t*>(ptr);
  value.words[0] = __builtin_nontemporal_load(words);
  value.words[1] = __builtin_nontemporal_load(words + 1);
  return value.pack;
}

__device__ __forceinline__ ProgArPack prog_ar_reduce_pack(
    const aiter::RankData& input_rd, uint32_t pack_index, uint32_t owner) {
  const auto* p0 = reinterpret_cast<const ProgArPack*>(input_rd.ptrs[0]);
  const auto* p1 = reinterpret_cast<const ProgArPack*>(input_rd.ptrs[1]);
  const auto* p2 = reinterpret_cast<const ProgArPack*>(input_rd.ptrs[2]);
  const auto* p3 = reinterpret_cast<const ProgArPack*>(input_rd.ptrs[3]);
  const ProgArPack v0 = prog_ar_peer_load(p0 + pack_index);
  const ProgArPack v1 = prog_ar_peer_load(p1 + pack_index);
  const ProgArPack v2 = prog_ar_peer_load(p2 + pack_index);
  const ProgArPack v3 = prog_ar_peer_load(p3 + pack_index);
  ProgArPack out;
#pragma unroll
  for (uint32_t j = 0; j < kProgArPack; ++j) {
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
    out.data[j] = static_cast<__hip_bfloat16>(value);
  }
  return out;
}

__global__ void gfx90a_tp4_m128_progressive_begin_kernel(
    aiter::RankData* sync_rd_ptr, uint32_t* local_sync, int rank) {
  const aiter::RankData sync_rd = *sync_rd_ptr;
  prog_ar_start(sync_rd, local_sync, rank);
}

__global__ void __launch_bounds__(kProgArThreads, 1)
    gfx90a_tp4_m128_progressive_draft_kernel(
        aiter::RankData* input_rd_ptr, uint32_t* local_sync,
        __hip_bfloat16* __restrict__ output) {
  const aiter::RankData input_rd = *input_rd_ptr;
  const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;

  constexpr uint32_t kDraftRows = 96;
  constexpr uint32_t kDraftPacks = kDraftRows * kProgArPacksPerRow;
  for (uint32_t compact_pack = first; compact_pack < kDraftPacks;
       compact_pack += stride) {
    const uint32_t compact_row = compact_pack / kProgArPacksPerRow;
    const uint32_t pack_in_row = compact_pack % kProgArPacksPerRow;
    const uint32_t row = (compact_row / 3) * 4 + (compact_row % 3) + 1;
    const uint32_t pack = row * kProgArPacksPerRow + pack_in_row;
    const uint32_t owner = row / 32;
    reinterpret_cast<ProgArPack*>(output)[pack] =
        prog_ar_reduce_pack(input_rd, pack, owner);
  }

  __syncthreads();
  __threadfence();
  if (threadIdx.x == 0) {
    const uint32_t ticket =
        atomicAdd(local_sync + kProgArArrivalsOffset, 1u);
    if ((ticket % kProgArDraftBlocks) == kProgArDraftBlocks - 1) {
      uint32_t armed;
      uint32_t ready;
      do {
        armed = prog_ar_device_load(local_sync + kProgArArmedOffset,
                                    __ATOMIC_ACQUIRE);
        ready = prog_ar_device_load(local_sync + kProgArReadyOffset,
                                    __ATOMIC_RELAXED);
      } while (armed <= ready);
      __threadfence();
      prog_ar_device_store(local_sync + kProgArReadyOffset, armed,
                           __ATOMIC_RELEASE);
    }
  }
}

__global__ void __launch_bounds__(kProgArThreads, 1)
    gfx90a_tp4_m128_progressive_anchor_kernel(
        aiter::RankData* input_rd_ptr, __hip_bfloat16* __restrict__ output) {
  const aiter::RankData input_rd = *input_rd_ptr;
  const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  constexpr uint32_t kAnchorRows = 32;
  constexpr uint32_t kAnchorPacks = kAnchorRows * kProgArPacksPerRow;
  for (uint32_t compact_pack = first; compact_pack < kAnchorPacks;
       compact_pack += stride) {
    const uint32_t compact_row = compact_pack / kProgArPacksPerRow;
    const uint32_t pack_in_row = compact_pack % kProgArPacksPerRow;
    const uint32_t row = compact_row * 4;
    const uint32_t pack = row * kProgArPacksPerRow + pack_in_row;
    const uint32_t owner = row / 32;
    reinterpret_cast<ProgArPack*>(output)[pack] =
        prog_ar_reduce_pack(input_rd, pack, owner);
  }
}

__global__ void gfx90a_tp4_m128_progressive_end_kernel(
    aiter::RankData* sync_rd_ptr, uint32_t* local_sync, int rank) {
  const aiter::RankData sync_rd = *sync_rd_ptr;
  const uint32_t epoch =
      prog_ar_system_load(local_sync + kProgArFlagOffset, __ATOMIC_RELAXED) + 1;
  prog_ar_end(sync_rd, local_sync, rank, epoch);
}

__global__ void gfx90a_tp4_m128_progressive_wait_kernel(uint32_t* sync) {
  if (threadIdx.x == 0) {
    const uint32_t epoch = atomicAdd(sync + kProgArArmedOffset, 1u) + 1u;
    while (prog_ar_device_load(sync + kProgArReadyOffset,
                               __ATOMIC_ACQUIRE) < epoch) {
    }
  }
}

__global__ void gfx90a_tp4_m128_progressive_arm_kernel(uint32_t* sync) {
  if (threadIdx.x == 0) atomicAdd(sync + kProgArArmedOffset, 1u);
}

struct Gfx90aTp4M128ProgressiveArOracle {
  static hipStream_t stream_for(DLDevice device) {
    return sglang::host::LaunchKernel::resolve_device(device);
  }

  static void progressive(int64_t fa, const tvm::ffi::TensorView input,
                          const tvm::ffi::TensorView sync_workspace,
                          const tvm::ffi::TensorView output, int64_t rank) {
    if (rank < 0 || rank >= kProgArWorld) {
      throw std::runtime_error("TP4 progressive oracle rank must be in [0,4)");
    }
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(input.device());
    auto* input_rd = comm->get_buffer_RD(stream, input.data_ptr());
    auto* sync_rd = comm->get_buffer_RD(stream, sync_workspace.data_ptr());
    // One rank-global entry and exit epoch surround two compact compute grids.
    // This removes eleven duplicate signal barriers while preserving one
    // logical M128 collective and the original owner/reduction association.
    sglang::host::LaunchKernel(1, 64, stream)(
        gfx90a_tp4_m128_progressive_begin_kernel, sync_rd,
        static_cast<uint32_t*>(sync_workspace.data_ptr()),
        static_cast<int>(rank));
    sglang::host::LaunchKernel(kProgArDraftBlocks, kProgArThreads, stream)(
        gfx90a_tp4_m128_progressive_draft_kernel, input_rd,
        static_cast<uint32_t*>(sync_workspace.data_ptr()),
        static_cast<__hip_bfloat16*>(output.data_ptr()));
    sglang::host::LaunchKernel(kProgArAnchorBlocks, kProgArThreads, stream)(
        gfx90a_tp4_m128_progressive_anchor_kernel, input_rd,
        static_cast<__hip_bfloat16*>(output.data_ptr()));
    sglang::host::LaunchKernel(1, 64, stream)(
        gfx90a_tp4_m128_progressive_end_kernel, sync_rd,
        static_cast<uint32_t*>(sync_workspace.data_ptr()),
        static_cast<int>(rank));
  }

  static void wait_draft(const tvm::ffi::TensorView sync_workspace) {
    auto stream = stream_for(sync_workspace.device());
    sglang::host::LaunchKernel(1, 64, stream)(
        gfx90a_tp4_m128_progressive_wait_kernel,
        static_cast<uint32_t*>(sync_workspace.data_ptr()));
  }

  static void arm(const tvm::ffi::TensorView sync_workspace) {
    auto stream = stream_for(sync_workspace.device());
    sglang::host::LaunchKernel(1, 64, stream)(
        gfx90a_tp4_m128_progressive_arm_kernel,
        static_cast<uint32_t*>(sync_workspace.data_ptr()));
  }
};

}  // namespace sglang
