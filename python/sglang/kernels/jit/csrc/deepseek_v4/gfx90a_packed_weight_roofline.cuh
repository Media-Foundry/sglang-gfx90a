#pragma once

// Standalone diagnostic only: stream one TP4 rank's packed routed-expert
// bytes and retain an order-independent xor checksum so the reads cannot be
// eliminated.  It deliberately does not decode FP4 or enter production.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

constexpr int kRoofExperts = 256;
constexpr int kRoofBytesPerExpert = 3 * 1024 * 1024;
constexpr int kRoofChunkBytes = 64 * 1024;
constexpr int kRoofChunksPerExpert = kRoofBytesPerExpert / kRoofChunkBytes;
constexpr int kRoofBlocks = 2080;

__global__ void gfx90a_packed_weight_roofline_kernel(
    const uint8_t* __restrict__ weights,
    const int32_t* __restrict__ order,
    const int32_t* __restrict__ order_len,
    unsigned long long* __restrict__ checksum) {
  __shared__ unsigned long long wave_checksum[4];
  const int n = max(0, min(order_len[0], kRoofExperts));
  const int tasks = n * kRoofChunksPerExpert;
  unsigned long long local = 0;
  for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int order_index = task / kRoofChunksPerExpert;
    const int chunk = task - order_index * kRoofChunksPerExpert;
    const int expert = order[order_index];
    if (expert < 0 || expert >= kRoofExperts) continue;
    const size_t byte_offset =
        static_cast<size_t>(expert) * kRoofBytesPerExpert +
        static_cast<size_t>(chunk) * kRoofChunkBytes;
    const unsigned long long* src = reinterpret_cast<const unsigned long long*>(
        weights + byte_offset);
    constexpr int kWords = kRoofChunkBytes / sizeof(unsigned long long);
    for (int word = threadIdx.x; word < kWords; word += blockDim.x) {
      local ^= src[word];
    }
  }
  for (int offset = 32; offset > 0; offset >>= 1) {
    local ^= __shfl_down(local, offset, 64);
  }
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  if (lane == 0) {
    wave_checksum[wave] = local;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    checksum[blockIdx.x] = wave_checksum[0] ^ wave_checksum[1] ^
                           wave_checksum[2] ^ wave_checksum[3];
  }
}

struct Gfx90aPackedWeightRoofline {
  static void run(const tvm::ffi::TensorView weights,
                  const tvm::ffi::TensorView order,
                  const tvm::ffi::TensorView order_len,
                  const tvm::ffi::TensorView checksum) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({kRoofExperts, kRoofBytesPerExpert})
        .with_dtype<uint8_t>().with_device(device).verify(weights);
    TensorMatcher({kRoofExperts}).with_dtype<int32_t>().with_device(device).verify(order);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(order_len);
    TensorMatcher({kRoofBlocks}).with_dtype<int64_t>().with_device(device).verify(checksum);
    LaunchKernel(kRoofBlocks, 256, weights.device())(
        gfx90a_packed_weight_roofline_kernel,
        static_cast<const uint8_t*>(weights.data_ptr()),
        static_cast<const int32_t*>(order.data_ptr()),
        static_cast<const int32_t*>(order_len.data_ptr()),
        reinterpret_cast<unsigned long long*>(checksum.data_ptr()));
  }
};

}  // namespace sglang
