#pragma once

#include "ck/utility/dsv4_fp4_logical_b_buffer.hpp"
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

namespace sglang {

template <uint32_t E, uint32_t N, uint32_t K>
__global__ void gfx90a_ck_logical_b_runtime_kernel(
    const uint8_t* weight, const uint8_t* scale, const int64_t* offsets,
    ck::bhalf_t* out, uint32_t count) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= count) return;
  constexpr uint64_t expert_values = static_cast<uint64_t>(N) * K;
  const uint64_t logical = offsets[id];
  const uint32_t expert = logical / expert_values;
  constexpr uint32_t ProjectionN = N / 2;
  constexpr uint32_t TileValues = 16 * 4 * 8;
  constexpr uint32_t K0Count = K / 32;
  const uint64_t within_expert = logical - expert * expert_values;
  const uint32_t tile = within_expert / TileValues;
  const uint32_t inner = within_expert - static_cast<uint64_t>(tile) * TileValues;
  const uint32_t n0 = tile / K0Count;
  const uint32_t k0 = tile - n0 * K0Count;
  const uint32_t klane = inner / (16 * 8);
  const uint32_t rem = inner - klane * (16 * 8);
  const uint32_t nlane = rem / 8;
  const uint32_t kpack = rem - nlane * 8;
  const uint32_t n = n0 * 16 + nlane;
  const uint32_t projection = n / ProjectionN;
  const uint32_t local_n = n - projection * ProjectionN;
  const uint32_t local_offset =
      (((local_n / 16) * K0Count + k0) * 4 + klane) * (16 * 8) +
      (local_n % 16) * 8 + kpack;
  ck::Dsv4Fp4LogicalBBuffer<> buffer(
      weight + static_cast<uint64_t>(expert) * N * (K / 2),
      scale + static_cast<uint64_t>(expert) * N * (K / 32),
      ProjectionN,
      K,
      projection);
  using BHalf8 = ck::vector_type<ck::bhalf_t, 8>::type;
  const BHalf8 value = buffer.template Get<BHalf8>(local_offset, true);
  ck::vector_type<ck::bhalf_t, 8> unpacked{value};
  ck::static_for<0, 8, 1>{}([&](auto i) {
    out[static_cast<uint64_t>(id) * 8 + i] =
        unpacked.template AsType<ck::bhalf_t>()[i];
  });
}

template <uint32_t E, uint32_t N, uint32_t K>
struct Gfx90aCkLogicalBRuntimeOracle {
  static void run(const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView scale,
                  const tvm::ffi::TensorView offsets,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    const uint32_t count = offsets.size(0);
    LaunchKernel((count + 255) / 256, 256, weight.device())(
        gfx90a_ck_logical_b_runtime_kernel<E, N, K>,
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()),
        static_cast<const int64_t*>(offsets.data_ptr()),
        static_cast<ck::bhalf_t*>(out.data_ptr()), count);
  }
};

}  // namespace sglang
