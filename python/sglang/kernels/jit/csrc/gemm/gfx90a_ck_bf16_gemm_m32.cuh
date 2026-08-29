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
using CkM32Bf16 = ck::bhalf_t;
using CkM32Row = ck::tensor_layout::gemm::RowMajor;
using CkM32Col = ck::tensor_layout::gemm::ColumnMajor;
using CkM32Pass = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... Is> using CkM32S = ck::Sequence<Is...>;

using Gfx90aCkBf16M32Instance =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
        CkM32Row, CkM32Col, CkM32Row,
        CkM32Bf16, CkM32Bf16, CkM32Bf16, float, CkM32Bf16,
        CkM32Pass, CkM32Pass, CkM32Pass,
        ck::tensor_operation::device::GemmSpecialization::MNKPadding,
        1, 128, 32, 128, 32, 8, 8, 32, 32, 1, 2,
        CkM32S<4, 32, 1>, CkM32S<1, 0, 2>, CkM32S<1, 0, 2>, 2, 8, 8, 1,
        CkM32S<4, 32, 1>, CkM32S<1, 0, 2>, CkM32S<1, 0, 2>, 2, 8, 8, 1,
        1, 1, CkM32S<1, 16, 1, 8>, 8,
        ck::LoopScheduler::Default, ck::PipelineVersion::v2>;

template <uint32_t N, uint32_t K>
struct Gfx90aCkBf16GemmM32 {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    constexpr ck::index_t M = 32;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    Gfx90aCkBf16M32Instance gemm;
    auto argument = gemm.MakeArgument(
        static_cast<CkM32Bf16*>(x.data_ptr()),
        static_cast<CkM32Bf16*>(weight.data_ptr()),
        static_cast<CkM32Bf16*>(out.data_ptr()),
        M, N, K, K, K, N, CkM32Pass{}, CkM32Pass{}, CkM32Pass{});
    auto invoker = gemm.MakeInvoker();
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    invoker.Run(argument, StreamConfig{stream, false});
  }
};
}  // namespace sglang
