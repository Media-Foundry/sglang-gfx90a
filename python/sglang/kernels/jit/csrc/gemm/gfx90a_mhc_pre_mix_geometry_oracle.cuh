#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <cstdint>

namespace sglang {
using namespace device;
constexpr uint32_t kMhcGeomK = 4 * 4096;
constexpr uint32_t kMhcGeomN = 24;
constexpr uint32_t kMhcGeomWave = 64;

__device__ __forceinline__ float mhc_geom_wave_sum(float value) {
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, kMhcGeomWave);
  }
  return value;
}

template <uint32_t kRows>
__global__ void __launch_bounds__(kMhcGeomWave)
mhc_pre_mix_geometry_kernel(const bf16_t* __restrict__ residual,
                            const float* __restrict__ fn,
                            float* __restrict__ mixes, float rms_eps) {
  static_assert(kMhcGeomN % kRows == 0);
  constexpr uint32_t kBlocksPerToken = kMhcGeomN / kRows;
  const uint32_t token = blockIdx.x / kBlocksPerToken;
  const uint32_t row0 = (blockIdx.x % kBlocksPerToken) * kRows;
  const uint32_t lane = threadIdx.x;
  const bf16_t* x = residual + static_cast<size_t>(token) * kMhcGeomK;
  float sumsq = 0.0f;
  float acc[kRows] = {};
  for (uint32_t pair = lane; pair < kMhcGeomK / 2; pair += kMhcGeomWave) {
    const uint32_t k = pair * 2;
    const auto [x0, x1] =
        cast<fp32x2_t>(*reinterpret_cast<const bf16x2_t*>(x + k));
    sumsq = fmaf(x0, x0, sumsq);
    sumsq = fmaf(x1, x1, sumsq);
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      const float* w = fn + static_cast<size_t>(row0 + r) * kMhcGeomK;
      acc[r] = fmaf(w[k], x0, acc[r]);
      acc[r] = fmaf(w[k + 1], x1, acc[r]);
    }
  }
  sumsq = mhc_geom_wave_sum(sumsq);
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) acc[r] = mhc_geom_wave_sum(acc[r]);
  if (lane == 0) {
    const float scale =
        rsqrtf(sumsq / static_cast<float>(kMhcGeomK) + rms_eps);
    float* out = mixes + static_cast<size_t>(token) * kMhcGeomN + row0;
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) out[r] = acc[r] * scale;
  }
}

template <uint32_t kRows>
struct Gfx90aMhcPreMixGeometryOracle {
  static void run(const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView fn,
                  const tvm::ffi::TensorView mixes, float rms_eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({T, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({24, 16384}).with_dtype<float>().with_device(device).verify(fn);
    TensorMatcher({T, 1, 24}).with_dtype<float>().with_device(device).verify(mixes);
    constexpr uint32_t kBlocksPerToken = kMhcGeomN / kRows;
    LaunchKernel(static_cast<uint32_t>(T.unwrap()) * kBlocksPerToken,
                 kMhcGeomWave, device.unwrap())(
        mhc_pre_mix_geometry_kernel<kRows>,
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(fn.data_ptr()),
        static_cast<float*>(mixes.data_ptr()), rms_eps);
  }
};
}  // namespace sglang

