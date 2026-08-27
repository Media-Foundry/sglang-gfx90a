#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <stdexcept>

namespace sglang {

using namespace device;

constexpr uint32_t kGroupedProjectionM = 32;
constexpr uint32_t kGroupedProjectionGroups = 3;
constexpr uint32_t kGroupedProjectionPaddedN = 256;
constexpr uint32_t kGroupedProjectionN = 520;
constexpr uint32_t kGroupedProjectionVec = 8;

// Pack a rank's three consecutive chunks from grouped_mm's padded
// [3, M, 256] output into one contiguous [M, 520] rank-local buffer.  Chunk 0
// is always 256 rows.  Chunk 1 is 256/240/208/200 rows depending on where the
// rank's global N slice meets an original projection boundary; chunk 2 fills
// the remainder.  Every boundary is 16-byte aligned.
__global__ void gfx90a_grouped_projection_pack_kernel(
    const bf16_t* __restrict__ grouped, bf16_t* __restrict__ packed,
    uint32_t rank) {
  constexpr uint32_t kVecsPerRow =
      kGroupedProjectionN / kGroupedProjectionVec;
  const uint32_t linear_vec = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_vec >= kGroupedProjectionM * kVecsPerRow) return;

  const uint32_t row = linear_vec / kVecsPerRow;
  const uint32_t out_vec = linear_vec % kVecsPerRow;
  uint32_t middle_vecs = 32;
  if (rank == 2) {
    middle_vecs = 30;
  } else if (rank == 6) {
    middle_vecs = 26;
  } else if (rank == 7) {
    middle_vecs = 25;
  }
  uint32_t group;
  uint32_t local_vec;
  if (out_vec < 32) {
    group = 0;
    local_vec = out_vec;
  } else if (out_vec < 32 + middle_vecs) {
    group = 1;
    local_vec = out_vec - 32;
  } else {
    group = 2;
    local_vec = out_vec - 32 - middle_vecs;
  }

  const size_t src =
      (static_cast<size_t>(group) * kGroupedProjectionM + row) *
          kGroupedProjectionPaddedN +
      local_vec * kGroupedProjectionVec;
  const size_t dst =
      static_cast<size_t>(row) * kGroupedProjectionN +
      out_vec * kGroupedProjectionVec;
  *reinterpret_cast<float4*>(packed + dst) =
      *reinterpret_cast<const float4*>(grouped + src);
}

struct Gfx90aGroupedProjectionPackKernel {
  static void run(const tvm::ffi::TensorView grouped,
                  const tvm::ffi::TensorView packed, int64_t rank) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({kGroupedProjectionGroups, kGroupedProjectionM,
                   kGroupedProjectionPaddedN})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(grouped);
    TensorMatcher({kGroupedProjectionM, kGroupedProjectionN})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(packed);
    if (rank < 0 || rank >= 8) {
      throw std::runtime_error("TP8 projection pack rank must be in [0, 8)");
    }
    constexpr uint32_t kThreads = 256;
    constexpr uint32_t kTotalVecs =
        kGroupedProjectionM * kGroupedProjectionN / kGroupedProjectionVec;
    constexpr uint32_t kBlocks = (kTotalVecs + kThreads - 1) / kThreads;
    LaunchKernel(kBlocks, kThreads, device.unwrap())(
        gfx90a_grouped_projection_pack_kernel,
        static_cast<const bf16_t*>(grouped.data_ptr()),
        static_cast<bf16_t*>(packed.data_ptr()), static_cast<uint32_t>(rank));
  }
};

}  // namespace sglang
