#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <torch/all.h>

#include "custom_all_reduce.cuh"
#include "quick_all_reduce_base.h"

#include <cstdint>
#include <stdexcept>

namespace sglang {

constexpr uint32_t kOwnerM = 32;
constexpr uint32_t kOwnerPadN = 2048;
constexpr uint32_t kOwnerOutN = 4160;
constexpr uint32_t kOwnerWorld = 8;
constexpr size_t kOwnerDataElements =
    static_cast<size_t>(kOwnerM) * kOwnerPadN;
constexpr size_t kOwnerDataBytes =
    kOwnerDataElements * sizeof(__hip_bfloat16);
constexpr size_t kOwnerProducedOffset = kOwnerDataBytes;
constexpr size_t kOwnerSignalBytes = kOwnerWorld * sizeof(uint32_t);
constexpr size_t kOwnerConsumedOffset =
    kOwnerProducedOffset + kOwnerSignalBytes;
constexpr size_t kOwnerEndOffset = kOwnerConsumedOffset + kOwnerSignalBytes;

constexpr uint32_t kOwnerWidths[4] = {1536, 2048, 512, 64};
constexpr uint32_t kOwnerDstOffsets[4] = {0, 1536, 3584, 4096};

union OwnerBf16x8 {
  aiter::int32x4_t words;
  __hip_bfloat16 values[8];
};

__device__ __forceinline__ uint32_t owner_epoch_load(const uint32_t* ptr) {
  return __scoped_atomic_load_n(ptr, __ATOMIC_ACQUIRE,
                                __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void owner_epoch_store(uint32_t* ptr,
                                                   uint32_t value) {
  __scoped_atomic_store_n(ptr, value, __ATOMIC_RELEASE,
                          __MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ OwnerBf16x8 owner_peer_load_bf16x8(
    const aiter::BufferResource& resource, uint32_t byte_offset) {
  OwnerBf16x8 result;
  // gfx90a's MUBUF acquire is GLC.  This is the same peer-load primitive used
  // by AIter quick-all-reduce and avoids retaining an old peer cache line
  // across graph epochs.
  result.words =
      aiter::buffer_load_dwordx4(resource.descriptor, byte_offset, 0, 1);
  return result;
}

// Only owner ranks 0..3 launch this kernel.  One CTA first waits until every
// consumer has acknowledged the previous epoch, then copies the original
// production-shaped GEMM output into the registered local slot and publishes
// the new epoch with a system-scope release.
__global__ void projection_owner_publish_kernel(
    aiter::RankData* consumed_rd, const __hip_bfloat16* input,
    __hip_bfloat16* local_data, uint32_t* local_produced, int rank,
    uint32_t width) {
  const uint32_t epoch = owner_epoch_load(local_produced + rank) + 1;
  if (threadIdx.x < kOwnerWorld) {
    const uint32_t peer = threadIdx.x;
    const auto* peer_consumed = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(consumed_rd->ptrs[peer]) +
        kOwnerConsumedOffset);
    while (owner_epoch_load(peer_consumed + peer) + 1 < epoch) {
    }
  }
  __syncthreads();

  constexpr uint32_t kVector = 8;
  const uint32_t vectors = kOwnerM * width / kVector;
  for (uint32_t linear = threadIdx.x; linear < vectors;
       linear += blockDim.x) {
    const uint32_t element = linear * kVector;
    const uint32_t row = element / width;
    const uint32_t col = element - row * width;
    const size_t src = static_cast<size_t>(row) * width + col;
    const size_t dst = static_cast<size_t>(row) * kOwnerPadN + col;
    *reinterpret_cast<aiter::int32x4_t*>(local_data + dst) =
        *reinterpret_cast<const aiter::int32x4_t*>(input + src);
  }
  // All payload stores are issued by this CTA.  Publish them before the flag.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) owner_epoch_store(local_produced + rank, epoch);
}

__global__ void projection_owner_wait_ready_kernel(
    aiter::RankData* produced_rd, const uint32_t* local_consumed, int rank) {
  const uint32_t epoch = owner_epoch_load(local_consumed + rank) + 1;
  if (threadIdx.x < 4) {
    const uint32_t owner = threadIdx.x;
    const auto* peer_produced = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(produced_rd->ptrs[owner]) +
        kOwnerProducedOffset);
    while (owner_epoch_load(peer_produced + owner) < epoch) {
    }
  }
  __syncthreads();
}

// wait/copy/ack are three consecutive kernels on one stream.  Kernel ordering
// is the global completion barrier, allowing this payload stage to use enough
// CTAs to fill XGMI without an atomic task queue.
__global__ void projection_owner_peer_pack_kernel(
    aiter::RankData* data_rd, __hip_bfloat16* output) {
  aiter::BufferResource owner0(const_cast<void*>(data_rd->ptrs[0]), kOwnerDataBytes);
  aiter::BufferResource owner1(const_cast<void*>(data_rd->ptrs[1]), kOwnerDataBytes);
  aiter::BufferResource owner2(const_cast<void*>(data_rd->ptrs[2]), kOwnerDataBytes);
  aiter::BufferResource owner3(const_cast<void*>(data_rd->ptrs[3]), kOwnerDataBytes);
  constexpr uint32_t kVector = 8;
  constexpr uint32_t kVectors = kOwnerM * kOwnerOutN / kVector;
  for (uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < kVectors; linear += gridDim.x * blockDim.x) {
    const uint32_t element = linear * kVector;
    const uint32_t row = element / kOwnerOutN;
    const uint32_t col = element - row * kOwnerOutN;
    uint32_t owner;
    uint32_t local_col;
    if (col < kOwnerDstOffsets[1]) {
      owner = 0;
      local_col = col;
    } else if (col < kOwnerDstOffsets[2]) {
      owner = 1;
      local_col = col - kOwnerDstOffsets[1];
    } else if (col < kOwnerDstOffsets[3]) {
      owner = 2;
      local_col = col - kOwnerDstOffsets[2];
    } else {
      owner = 3;
      local_col = col - kOwnerDstOffsets[3];
    }
    const uint32_t byte_offset =
        (static_cast<size_t>(row) * kOwnerPadN + local_col) *
        sizeof(__hip_bfloat16);
    OwnerBf16x8 values;
    if (owner == 0) {
      values = owner_peer_load_bf16x8(owner0, byte_offset);
    } else if (owner == 1) {
      values = owner_peer_load_bf16x8(owner1, byte_offset);
    } else if (owner == 2) {
      values = owner_peer_load_bf16x8(owner2, byte_offset);
    } else {
      values = owner_peer_load_bf16x8(owner3, byte_offset);
    }
    *reinterpret_cast<aiter::int32x4_t*>(output + element) = values.words;
  }
}

__global__ void projection_owner_ack_kernel(uint32_t* local_consumed,
                                             int rank) {
  __threadfence_system();
  if (threadIdx.x == 0) {
    const uint32_t epoch = owner_epoch_load(local_consumed + rank) + 1;
    owner_epoch_store(local_consumed + rank, epoch);
  }
}

__global__ void projection_owner_end_kernel(aiter::RankData* end_rd,
                                            uint32_t* local_end, int rank) {
  const uint32_t epoch = owner_epoch_load(local_end + rank) + 1;
  if (threadIdx.x == 0) owner_epoch_store(local_end + rank, epoch);
  __syncthreads();
  if (threadIdx.x < kOwnerWorld) {
    const uint32_t peer = threadIdx.x;
    const auto* peer_end = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(end_rd->ptrs[peer]) + kOwnerEndOffset);
    while (owner_epoch_load(peer_end + peer) < epoch) {
    }
  }
}

struct Gfx90aProjectionOwnerPeerOracle {
  static hipStream_t stream_for(DLDevice device) {
    return sglang::host::LaunchKernel::resolve_device(device);
  }

  static void publish(int64_t fa, const tvm::ffi::TensorView workspace,
                      const tvm::ffi::TensorView input,
                      const tvm::ffi::TensorView data,
                      const tvm::ffi::TensorView produced, int64_t rank) {
    if (rank < 0 || rank >= 4) {
      throw std::runtime_error("projection owner rank must be in [0,4)");
    }
    const uint32_t width = kOwnerWidths[rank];
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(data.device());
    auto* workspace_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(1, 256, stream)(
        projection_owner_publish_kernel, workspace_rd,
        static_cast<const __hip_bfloat16*>(input.data_ptr()),
        static_cast<__hip_bfloat16*>(data.data_ptr()),
        static_cast<uint32_t*>(produced.data_ptr()), static_cast<int>(rank),
        width);
  }

  static void pack(int64_t fa, const tvm::ffi::TensorView workspace,
                   const tvm::ffi::TensorView consumed,
                   const tvm::ffi::TensorView output, int64_t rank) {
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(output.device());
    auto* workspace_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(1, 64, stream)(
        projection_owner_wait_ready_kernel, workspace_rd,
        static_cast<const uint32_t*>(consumed.data_ptr()),
        static_cast<int>(rank));
    constexpr uint32_t kThreads = 256;
    constexpr uint32_t kVectors = kOwnerM * kOwnerOutN / 8;
    sglang::host::LaunchKernel((kVectors + kThreads - 1) / kThreads,
                               kThreads, stream)(
        projection_owner_peer_pack_kernel, workspace_rd,
        static_cast<__hip_bfloat16*>(output.data_ptr()));
    sglang::host::LaunchKernel(1, 64, stream)(
        projection_owner_ack_kernel,
        static_cast<uint32_t*>(consumed.data_ptr()), static_cast<int>(rank));
  }

  static void end(int64_t fa, const tvm::ffi::TensorView workspace,
                  const tvm::ffi::TensorView end_epoch, int64_t rank) {
    auto* comm = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    auto stream = stream_for(end_epoch.device());
    auto* workspace_rd = comm->get_buffer_RD(stream, workspace.data_ptr());
    sglang::host::LaunchKernel(1, 64, stream)(
        projection_owner_end_kernel, workspace_rd,
        static_cast<uint32_t*>(end_epoch.data_ptr()), static_cast<int>(rank));
  }
};

}  // namespace sglang
