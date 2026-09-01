#pragma once

#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {

template <uint32_t E, uint32_t N, uint32_t K>
__global__ void gfx90a_fp4_to_bf16_kernel(
    bf16_t* __restrict__ out, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale) {
  constexpr uint32_t kValuesPerThread = 16;
  constexpr uint32_t kChunksPerRow = K / kValuesPerThread;
  const uint32_t chunks = E * N * kChunksPerRow;
  for (uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
       chunk < chunks; chunk += blockDim.x * gridDim.x) {
    const uint32_t row = chunk / kChunksPerRow;
    const uint32_t k_chunk = chunk - row * kChunksPerRow;
    const uint32_t k0 = k_chunk * kValuesPerThread;
    const float s = gfx90a_e8m0_value(
        scale[static_cast<size_t>(row) * (K / 32) + k0 / 32]) * 0.5f;
    const uint64_t packed = *reinterpret_cast<const uint64_t*>(
        weight + static_cast<size_t>(row) * (K / 2) + k0 / 2);
#pragma unroll
    for (uint32_t byte_index = 0; byte_index < 8; ++byte_index) {
      const uint8_t byte = static_cast<uint8_t>(packed >> (byte_index * 8));
      out[static_cast<size_t>(row) * K + k0 + byte_index * 2] =
          cast<bf16_t>(static_cast<float>(gfx90a_fp4_i8_code(byte & 0x0f)) * s);
      out[static_cast<size_t>(row) * K + k0 + byte_index * 2 + 1] =
          cast<bf16_t>(static_cast<float>(gfx90a_fp4_i8_code(byte >> 4)) * s);
    }
  }
}

template <uint32_t E, uint32_t N, uint32_t K, uint32_t kBlocks>
struct Gfx90aFp4ToBf16Oracle {
  static void run(const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView scale,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({E, N, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E, N, K / 32}).with_dtype<uint8_t>().with_device(device).verify(scale);
    TensorMatcher({E, N, K}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(kBlocks, 256, weight.device())(
        gfx90a_fp4_to_bf16_kernel<E, N, K>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()));
  }
};

}  // namespace sglang
