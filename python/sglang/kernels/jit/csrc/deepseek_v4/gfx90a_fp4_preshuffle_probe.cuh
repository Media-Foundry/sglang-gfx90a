#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

template <uint32_t E, uint32_t R, uint32_t K, bool kGateUp>
__global__ void gfx90a_fp4_preshuffle_probe_kernel(
    uint8_t* __restrict__ weight_out, uint8_t* __restrict__ scale_out,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    const int32_t* __restrict__ query, uint32_t q) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= q) return;
  const uint32_t expert = query[i * 3];
  const uint32_t row = query[i * 3 + 1];
  const uint32_t k_index = query[i * 3 + 2];
  if constexpr (kGateUp) {
    weight_out[i] = weight[
        gfx90a_gate_up_preshuffled_weight_offset<E, R / 2, K>(
            expert, row, k_index)];
    scale_out[i] = scale[gfx90a_gate_up_scale_offset<E, R / 2, K>(
        expert, row, k_index / 16)];
  } else {
    weight_out[i] = weight[
        gfx90a_down_preshuffled_weight_offset<E, R, K>(
            expert, row, k_index)];
    scale_out[i] = scale[gfx90a_down_scale_offset<E, R, K>(
        expert, row, k_index / 16)];
  }
}

template <uint32_t E, uint32_t R, uint32_t K, bool kGateUp>
struct Gfx90aFp4PreshuffleProbeKernel {
  static void run(const tvm::ffi::TensorView weight_out,
                  const tvm::ffi::TensorView scale_out,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView scale,
                  const tvm::ffi::TensorView query) {
    using namespace host;
    auto q = SymbolicSize{"queries"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({q}).with_dtype<uint8_t>().with_device(device).verify(weight_out).verify(scale_out);
    TensorMatcher({E, R, K / 2}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({E * R, K / 32}).with_dtype<uint8_t>().with_device(device).verify(scale);
    TensorMatcher({q, 3}).with_dtype<int32_t>().with_device(device).verify(query);
    constexpr uint32_t threads = 256;
    LaunchKernel((q.unwrap() + threads - 1) / threads, threads, device.unwrap())(
        gfx90a_fp4_preshuffle_probe_kernel<E, R, K, kGateUp>,
        static_cast<uint8_t*>(weight_out.data_ptr()),
        static_cast<uint8_t*>(scale_out.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()),
        static_cast<const int32_t*>(query.data_ptr()),
        static_cast<uint32_t>(q.unwrap()));
  }
};

}  // namespace sglang
