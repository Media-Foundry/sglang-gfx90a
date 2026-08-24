#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kInt8QuantWave = 64;
constexpr uint32_t kInt8QuantSubgroup = 16;
constexpr uint32_t kInt8QuantGroupsPerWave = 4;
constexpr uint32_t kInt8QuantWavesPerBlock = 4;
constexpr uint32_t kInt8QuantGroupsPerBlock =
    kInt8QuantGroupsPerWave * kInt8QuantWavesPerBlock;

__global__ void __launch_bounds__(
    kInt8QuantWave * kInt8QuantWavesPerBlock, 2)
    gfx90a_int8_group32_quant_kernel(
        const bf16_t* __restrict__ input,
        int8_t* __restrict__ output,
        float* __restrict__ scales,
        uint32_t num_groups) {
  const uint32_t lane = threadIdx.x & (kInt8QuantWave - 1);
  const uint32_t wave = threadIdx.x / kInt8QuantWave;
  const uint32_t subgroup = lane / kInt8QuantSubgroup;
  const uint32_t subgroup_lane = lane & (kInt8QuantSubgroup - 1);
  const uint32_t group =
      blockIdx.x * kInt8QuantGroupsPerBlock +
      wave * kInt8QuantGroupsPerWave + subgroup;
  if (group >= num_groups) return;

  const size_t base = static_cast<size_t>(group) * 32;
  const float x0 = cast<float>(input[base + subgroup_lane]);
  const float x1 = cast<float>(input[base + 16 + subgroup_lane]);
  float absmax = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
  for (uint32_t offset = 8; offset > 0; offset >>= 1) {
    absmax = fmaxf(
        absmax, __shfl_xor(absmax, offset, kInt8QuantSubgroup));
  }
  const float scale = fmaxf(absmax, 1.0e-10f) / 127.0f;
  const float q0 = fmaxf(-128.0f, fminf(127.0f, x0 / scale));
  const float q1 = fmaxf(-128.0f, fminf(127.0f, x1 / scale));
  output[base + subgroup_lane] = static_cast<int8_t>(q0);
  output[base + 16 + subgroup_lane] = static_cast<int8_t>(q1);
  if (subgroup_lane == 0) scales[group] = scale;
}

struct Gfx90aInt8Group32QuantKernel {
  static void run(const tvm::ffi::TensorView input,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView scales) {
    using namespace host;
    auto groups = SymbolicSize{"num_groups"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({groups, 32})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({groups, 32})
        .with_dtype<int8_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({groups})
        .with_dtype<float>()
        .with_device(device)
        .verify(scales);
    const uint32_t num_groups = static_cast<uint32_t>(groups.unwrap());
    LaunchKernel(
        (num_groups + kInt8QuantGroupsPerBlock - 1) /
            kInt8QuantGroupsPerBlock,
        kInt8QuantWave * kInt8QuantWavesPerBlock,
        input.device())(
        gfx90a_int8_group32_quant_kernel,
        static_cast<const bf16_t*>(input.data_ptr()),
        static_cast<int8_t*>(output.data_ptr()),
        static_cast<float*>(scales.data_ptr()),
        num_groups);
  }
};

}  // namespace sglang
