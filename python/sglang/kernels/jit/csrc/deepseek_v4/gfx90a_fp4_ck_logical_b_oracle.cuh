#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

// Reconstruct the logical BF16 B vectors consumed by CK's gfx90a
// preshuffled MoE pipeline directly from checkpoint-order E2M1/E8M0 storage.
// This is an oracle for the future CK DynamicBuffer::Get implementation; it
// deliberately handles arbitrary logical offsets rather than assuming that
// the caller always requests an aligned bhalf8 vector.
template <uint32_t E, uint32_t N, uint32_t K>
__global__ void gfx90a_fp4_ck_logical_b_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    const int64_t* __restrict__ logical_offsets,
    bf16_t* __restrict__ out,
    uint32_t count) {
  constexpr uint32_t kNLane = 16;
  constexpr uint32_t kKLane = 4;
  constexpr uint32_t kKPack = 8;
  constexpr uint32_t kK0 = K / (kKLane * kKPack);
  constexpr uint32_t kTileValues = kNLane * kKLane * kKPack;
  constexpr uint64_t kExpertValues = static_cast<uint64_t>(N) * K;

  const uint32_t vector_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (vector_id >= count) {
    return;
  }

  const uint64_t vector_base = logical_offsets[vector_id];
#pragma unroll
  for (uint32_t item = 0; item < 8; ++item) {
    const uint64_t logical = vector_base + item;
    const uint32_t expert = logical / kExpertValues;
    const uint64_t within_expert = logical -
        static_cast<uint64_t>(expert) * kExpertValues;
    const uint32_t tile = within_expert / kTileValues;
    const uint32_t inner = within_expert -
        static_cast<uint64_t>(tile) * kTileValues;
    const uint32_t n0 = tile / kK0;
    const uint32_t k0 = tile - n0 * kK0;
    const uint32_t klane = inner / (kNLane * kKPack);
    const uint32_t rem = inner - klane * (kNLane * kKPack);
    const uint32_t nlane = rem / kKPack;
    const uint32_t kpack = rem - nlane * kKPack;
    const uint32_t n = n0 * kNLane + nlane;
    const uint32_t k = k0 * (kKLane * kKPack) + klane * kKPack + kpack;

    const uint8_t packed = weight[
        (static_cast<uint64_t>(expert) * N + n) * (K / 2) + k / 2];
    const uint8_t nibble = (k & 1) ? packed >> 4 : packed & 0x0f;
    const float s = gfx90a_e8m0_value(scale[
        (static_cast<uint64_t>(expert) * N + n) * (K / 32) + k / 32]) * 0.5f;
    out[static_cast<uint64_t>(vector_id) * 8 + item] = cast<bf16_t>(
        static_cast<float>(gfx90a_fp4_i8_code(nibble)) * s);
  }
}

template <uint32_t E, uint32_t N, uint32_t K>
struct Gfx90aFp4CkLogicalBOracle {
  static void run(const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView scale,
                  const tvm::ffi::TensorView logical_offsets,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(scale);
    const uint32_t count = logical_offsets.size(0);
    TensorMatcher({count}).with_dtype<int64_t>().with_device(device).verify(logical_offsets);
    TensorMatcher({count, 8}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel((count + 255) / 256, 256, weight.device())(
        gfx90a_fp4_ck_logical_b_kernel<E, N, K>,
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()),
        static_cast<const int64_t*>(logical_offsets.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()), count);
  }
};

}  // namespace sglang
