#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// DeepSeek-V4 MHC decode pre-mix specialized for CDNA2 wave64:
//   mixes[t, 0, :] = fn @ (residual[t].flatten() * rms_scale)
//
// One native wave owns three output rows. Eight independent CTAs per token
// provide enough CU parallelism without the Mori scheduling instability seen
// when all 24 rows were launched as separate CTAs.
constexpr uint32_t kGfx90aMhcK = 4 * 4096;
constexpr uint32_t kGfx90aMhcN = 24;
constexpr uint32_t kGfx90aMhcWave = 64;

__device__ __forceinline__ float gfx90a_wave64_sum(float value) {
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, kGfx90aMhcWave);
  }
  return value;
}

template <uint32_t kRowsPerBlock>
__global__ void __launch_bounds__(kGfx90aMhcWave)
    gfx90a_mhc_pre_mix_kernel(const bf16_t* __restrict__ residual,
                              const float* __restrict__ fn,
                              float* __restrict__ mixes,
                              float rms_eps) {
  constexpr uint32_t kBlocksPerToken = kGfx90aMhcN / kRowsPerBlock;
  const uint32_t token = blockIdx.x / kBlocksPerToken;
  const uint32_t row0 =
      (blockIdx.x % kBlocksPerToken) * kRowsPerBlock;
  const uint32_t lane = threadIdx.x;
  const bf16_t* x = residual + static_cast<size_t>(token) * kGfx90aMhcK;

  float sumsq = 0.0f;
  float acc[kRowsPerBlock] = {};

  // Two BF16 values per lane load: 128 iterations, no CTA barrier. Re-reading
  // the small 32 KiB activation per row buys enough parallelism to hide the
  // much larger FP32 weight stream.
  for (uint32_t pair = lane; pair < kGfx90aMhcK / 2;
       pair += kGfx90aMhcWave) {
    const uint32_t k = pair * 2;
    const auto [x0, x1] =
        cast<fp32x2_t>(*reinterpret_cast<const bf16x2_t*>(x + k));
    sumsq = fmaf(x0, x0, sumsq);
    sumsq = fmaf(x1, x1, sumsq);
 #pragma unroll
    for (uint32_t r = 0; r < kRowsPerBlock; ++r) {
      const float* w = fn + static_cast<size_t>(row0 + r) * kGfx90aMhcK;
      acc[r] = fmaf(w[k], x0, acc[r]);
      acc[r] = fmaf(w[k + 1], x1, acc[r]);
    }
  }

  sumsq = gfx90a_wave64_sum(sumsq);
#pragma unroll
  for (uint32_t r = 0; r < kRowsPerBlock; ++r) {
    acc[r] = gfx90a_wave64_sum(acc[r]);
  }

  if (lane == 0) {
    const float scale = rsqrtf(sumsq / static_cast<float>(kGfx90aMhcK) + rms_eps);
    float* out = mixes + static_cast<size_t>(token) * kGfx90aMhcN + row0;
#pragma unroll
    for (uint32_t r = 0; r < kRowsPerBlock; ++r) {
      out[r] = acc[r] * scale;
    }
  }
}

struct Gfx90aMhcPreMixKernel {
  template <uint32_t kRowsPerBlock>
  static void run_impl(const tvm::ffi::TensorView residual,
                       const tvm::ffi::TensorView fn,
                       const tvm::ffi::TensorView mixes,
                       float rms_eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T, 4, 4096})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({24, 16384})
        .with_dtype<float>()
        .with_device(device)
        .verify(fn);
    TensorMatcher({T, 1, 24})
        .with_dtype<float>()
        .with_device(device)
        .verify(mixes);

    constexpr uint32_t kBlocksPerToken = kGfx90aMhcN / kRowsPerBlock;
    LaunchKernel(static_cast<uint32_t>(T.unwrap()) * kBlocksPerToken,
                 kGfx90aMhcWave,
                 device.unwrap())(gfx90a_mhc_pre_mix_kernel<kRowsPerBlock>,
                                  static_cast<const bf16_t*>(residual.data_ptr()),
                                  static_cast<const float*>(fn.data_ptr()),
                                  static_cast<float*>(mixes.data_ptr()),
                                  rms_eps);
  }

  static void run(const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView fn,
                  const tvm::ffi::TensorView mixes,
                  float rms_eps) {
    run_impl<3>(residual, fn, mixes, rms_eps);
  }

  static void run_m64(const tvm::ffi::TensorView residual,
                      const tvm::ffi::TensorView fn,
                      const tvm::ffi::TensorView mixes,
                      float rms_eps) {
    run_impl<2>(residual, fn, mixes, rms_eps);
  }
};

}  // namespace sglang
