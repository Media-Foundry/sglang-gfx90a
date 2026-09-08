#pragma once
#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {
// Local-only oracle: no peer publication, atomics, or changed slot ordering.
template <uint32_t M>
__global__ void gfx90a_finalize_shared_oracle(
    const float* __restrict__ partial, const bf16_t* __restrict__ shared,
    bf16_t* __restrict__ out) {
  constexpr uint32_t N = 4096;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  const size_t base = static_cast<size_t>(idx / N) * 6 * N + idx % N;
  const float s0 = partial[base] + partial[base + 4 * N];
  const float s1 = partial[base + N] + partial[base + 5 * N];
  const float sum = s0 + s1 + partial[base + 2 * N] + partial[base + 3 * N];
  const bf16_t rounded = cast<bf16_t>(sum);
  out[idx] = cast<bf16_t>(cast<float>(rounded) + cast<float>(shared[idx]));
}

template <uint32_t M>
struct Gfx90aFinalizeSharedOracle {
  static void run(tvm::ffi::TensorView partial, tvm::ffi::TensorView shared,
                  tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, 6, 4096}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({M, 4096}).with_dtype<bf16_t>().with_device(device).verify(shared);
    TensorMatcher({M, 4096}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel(M * 4096 / 256, 256, out.device())(
        gfx90a_finalize_shared_oracle<M>, static_cast<const float*>(partial.data_ptr()),
        static_cast<const bf16_t*>(shared.data_ptr()), static_cast<bf16_t*>(out.data_ptr()));
  }
};
}  // namespace sglang
