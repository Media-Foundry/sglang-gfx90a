#pragma once

#include <cstdint>
#include <stdexcept>

#include "sgl_kernel/tensor.h"

namespace sglang {

using namespace host;

// Convert checkpoint-order packed DSV4 FP4 weights to CKTile A16W4 order.
// One thread moves one complete 16-byte KPack, keeping both sides vectorized.
template <bool kGateUp>
__global__ void gfx90a_dsv4_fp4_weight_preshuffle_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint32_t experts, uint32_t rows, uint32_t packed_k) {
  constexpr uint32_t kNLane = 16;
  constexpr uint32_t kKLane = 4;
  constexpr uint32_t kKPack = 16;
  const uint32_t logical_rows = kGateUp ? rows / 2 : rows;
  const uint32_t n0_count = logical_rows / kNLane;
  const uint32_t k0_count = packed_k / (kKLane * kKPack);
  const uint64_t units = static_cast<uint64_t>(experts) * rows * packed_k / kKPack;

  for (uint64_t unit = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       unit < units; unit += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    uint64_t q = unit;
    const uint32_t nlane = q % kNLane;
    q /= kNLane;
    const uint32_t klane = q % kKLane;
    q /= kKLane;
    const uint32_t k0 = q % k0_count;
    q /= k0_count;
    const uint32_t gate_up = kGateUp ? q % 2 : 0;
    if constexpr (kGateUp) q /= 2;
    const uint32_t n0 = q % n0_count;
    const uint32_t expert = q / n0_count;
    uint32_t row = n0 * kNLane + nlane;
    if constexpr (kGateUp) {
      row += gate_up * logical_rows;
    } else {
      // CKTile gfx90a stage-2 emits the eight 16-row blocks in
      // [0,2,4,6,1,3,5,7] order.  Apply the established inverse row fix while
      // reading raw W2 so the temporary CKTile buffer remains logical.
      constexpr uint32_t inverse[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      const uint32_t group = row / 128;
      const uint32_t block = (row % 128) / 16;
      const uint32_t within = row % 16;
      row = group * 128 + inverse[block] * 16 + within;
    }
    const uint64_t src_offset =
        (static_cast<uint64_t>(expert) * rows + row) * packed_k +
        (k0 * kKLane + klane) * kKPack;
    reinterpret_cast<uint4*>(dst)[unit] =
        *reinterpret_cast<const uint4*>(src + src_offset);
  }
}

template <bool kGateUp>
__global__ void gfx90a_dsv4_fp4_scale_preshuffle_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint32_t experts, uint32_t rows, uint32_t groups) {
  constexpr uint32_t kNPack = 2;
  constexpr uint32_t kNLane = 16;
  constexpr uint32_t kKPack = 2;
  constexpr uint32_t kKLane = 4;
  const uint32_t logical_rows = kGateUp ? rows / 2 : rows;
  // NPack is part of the physical row factorization.  Gate/up has two
  // logical halves, but that is exactly the NPack=2 dimension rather than an
  // additional division of the row count.
  const uint32_t n1_count = rows / (kNLane * kNPack);
  const uint32_t k1_count = groups / (kKPack * kKLane);
  const uint64_t count = static_cast<uint64_t>(experts) * rows * groups;

  for (uint64_t out = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       out < count; out += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    uint64_t q = out;
    const uint32_t npack = q % kNPack;
    q /= kNPack;
    const uint32_t kpack = q % kKPack;
    q /= kKPack;
    const uint32_t nlane = q % kNLane;
    q /= kNLane;
    const uint32_t klane = q % kKLane;
    q /= kKLane;
    const uint32_t k1 = q % k1_count;
    q /= k1_count;
    const uint32_t n1 = q % n1_count;
    const uint32_t expert = q / n1_count;

    uint32_t row;
    if constexpr (kGateUp) {
      row = npack * logical_rows + (n1 * kNLane + nlane);
    } else {
      row = (n1 * kNPack + npack) * kNLane + nlane;
      constexpr uint32_t inverse[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      const uint32_t group = row / 128;
      const uint32_t block = (row % 128) / 16;
      const uint32_t within = row % 16;
      row = group * 128 + inverse[block] * 16 + within;
    }
    const uint32_t k_group = (k1 * kKPack + kpack) * kKLane + klane;
    dst[out] = src[(static_cast<uint64_t>(expert) * rows + row) * groups + k_group];
  }
}

struct Gfx90aDsv4Fp4Preshuffle {
  static void run(const tvm::ffi::TensorView w13,
                  const tvm::ffi::TensorView s13,
                  const tvm::ffi::TensorView w2,
                  const tvm::ffi::TensorView s2,
                  const tvm::ffi::TensorView out_w13,
                  const tvm::ffi::TensorView out_s13,
                  const tvm::ffi::TensorView out_w2,
                  const tvm::ffi::TensorView out_s2,
                  int64_t requested_blocks) {
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({256, 1024, 2048}).with_dtype<uint8_t>().with_device(device).verify(w13);
    TensorMatcher({256, 1024, 128}).with_dtype<uint8_t>().with_device(device).verify(s13);
    TensorMatcher({256, 4096, 256}).with_dtype<uint8_t>().with_device(device).verify(w2);
    TensorMatcher({256, 4096, 16}).with_dtype<uint8_t>().with_device(device).verify(s2);
    TensorMatcher({256, 1024, 2048}).with_dtype<uint8_t>().with_device(device).verify(out_w13);
    TensorMatcher({256, 1024, 128}).with_dtype<uint8_t>().with_device(device).verify(out_s13);
    TensorMatcher({256, 4096, 256}).with_dtype<uint8_t>().with_device(device).verify(out_w2);
    TensorMatcher({256, 4096, 16}).with_dtype<uint8_t>().with_device(device).verify(out_s2);
    constexpr uint32_t threads = 256;
    const uint32_t blocks = static_cast<uint32_t>(requested_blocks);
    if (blocks == 0 || blocks > 4096) {
      throw std::runtime_error("gfx90a DSV4 preshuffle blocks must be in [1,4096]");
    }
    LaunchKernel(blocks, threads, device.unwrap())(
        gfx90a_dsv4_fp4_weight_preshuffle_kernel<true>,
        static_cast<const uint8_t*>(w13.data_ptr()), static_cast<uint8_t*>(out_w13.data_ptr()),
        256u, 1024u, 2048u);
    LaunchKernel(blocks, threads, device.unwrap())(
        gfx90a_dsv4_fp4_scale_preshuffle_kernel<true>,
        static_cast<const uint8_t*>(s13.data_ptr()), static_cast<uint8_t*>(out_s13.data_ptr()),
        256u, 1024u, 128u);
    LaunchKernel(blocks, threads, device.unwrap())(
        gfx90a_dsv4_fp4_weight_preshuffle_kernel<false>,
        static_cast<const uint8_t*>(w2.data_ptr()), static_cast<uint8_t*>(out_w2.data_ptr()),
        256u, 4096u, 256u);
    LaunchKernel(blocks, threads, device.unwrap())(
        gfx90a_dsv4_fp4_scale_preshuffle_kernel<false>,
        static_cast<const uint8_t*>(s2.data_ptr()), static_cast<uint8_t*>(out_s2.data_ptr()),
        256u, 4096u, 16u);
  }
};

}  // namespace sglang
