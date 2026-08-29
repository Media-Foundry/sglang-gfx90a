#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

namespace sglang {
using namespace device;
using CkBf16 = ck::bhalf_t;
using CkRow = ck::tensor_layout::gemm::RowMajor;
using CkCol = ck::tensor_layout::gemm::ColumnMajor;
using CkPass = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... Is> using CkS = ck::Sequence<Is...>;

using QwenHcUpM32 = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
    CkRow, CkCol, CkRow, CkBf16, CkBf16, CkBf16, float, CkBf16,
    CkPass, CkPass, CkPass,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    1, 128, 32, 128, 32, 8, 8, 32, 32, 1, 2,
    CkS<4, 32, 1>, CkS<1, 0, 2>, CkS<1, 0, 2>, 2, 8, 8, 1,
    CkS<4, 32, 1>, CkS<1, 0, 2>, CkS<1, 0, 2>, 2, 8, 8, 1,
    1, 1, CkS<1, 16, 1, 8>, 8, ck::LoopScheduler::Default,
    ck::PipelineVersion::v2>;

__global__ void qwen_hc_m32_epilogue(
    const bf16_t* __restrict__ gates, const bf16_t* __restrict__ x,
    bf16_t* __restrict__ out) {
  constexpr uint32_t M = 32, HC = 4, HS = 2560;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * HS) return;
  const uint32_t m = i / HS;
  const uint32_t j = i - m * HS;
  float value = 0.0f;
#pragma unroll
  for (uint32_t g = 0; g < HC; ++g) {
    const float gate = cast<float>(gates[m * (HC * HS) + g * HS + j]);
    const float residual = cast<float>(x[m * (HC * HS) + g * HS + j]);
    value += residual / (1.0f + expf(-gate));
  }
  out[i] = cast<bf16_t>(value * 0.25f);
}

struct Gfx90aCkHcMixM32 {
  static void up(const tvm::ffi::TensorView x,
                 const tvm::ffi::TensorView weight,
                 const tvm::ffi::TensorView out) {
    using namespace host;
    constexpr ck::index_t M = 32, N = 10240, K = 320;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    QwenHcUpM32 gemm;
    auto argument = gemm.MakeArgument(
        static_cast<CkBf16*>(x.data_ptr()),
        static_cast<CkBf16*>(weight.data_ptr()),
        static_cast<CkBf16*>(out.data_ptr()),
        M, N, K, K, K, N, CkPass{}, CkPass{}, CkPass{});
    // Do not call CK's IsSupportedArgument here: this entry point is first
    // reached while SGLang captures BS32, and CK's device-capability query is
    // not capture-safe. The fixed gfx90a selector and tensor matchers above
    // encode the complete support predicate for this prevalidated instance.
    auto invoker = gemm.MakeInvoker();
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    invoker.Run(argument, StreamConfig{stream, false});
  }

  static void epilogue(const tvm::ffi::TensorView gates,
                       const tvm::ffi::TensorView x,
                       const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({32, 10240}).with_dtype<bf16_t>().with_device(device).verify(gates);
    TensorMatcher({32, 10240}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({32, 2560}).with_dtype<bf16_t>().with_device(device).verify(out);
    LaunchKernel((32 * 2560 + 255) / 256, 256, device.unwrap())(
        qwen_hc_m32_epilogue, static_cast<const bf16_t*>(gates.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()), static_cast<bf16_t*>(out.data_ptr()));
  }
};
}  // namespace sglang
