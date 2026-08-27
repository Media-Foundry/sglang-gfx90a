#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

namespace sglang {

using namespace device;

__global__ __launch_bounds__(64) void gfx90a_qwen_topk_kernel(
    const bf16_t* __restrict__ logits, float* __restrict__ weights,
    int* __restrict__ ids) {
  constexpr int E = 512;
  constexpr int K = 10;
  const int lane = threadIdx.x;
  float values[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    values[i] = cast<float>(logits[lane * 8 + i]);

  float row_max = values[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) row_max = fmaxf(row_max, values[i]);
#pragma unroll
  for (int mask = 32; mask > 0; mask >>= 1)
    row_max = fmaxf(row_max, __shfl_xor(row_max, mask, 64));

  float row_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = expf(values[i] - row_max);
    row_sum += values[i];
  }
#pragma unroll
  for (int mask = 32; mask > 0; mask >>= 1)
    row_sum += __shfl_xor(row_sum, mask, 64);
  const float inv_sum = 1.0f / row_sum;
#pragma unroll
  for (int i = 0; i < 8; ++i) values[i] *= inv_sum;

  float selected_sum = 0.0f;
#pragma unroll
  for (int pick = 0; pick < K; ++pick) {
    float best = values[0];
    int expert = lane * 8;
#pragma unroll
    for (int i = 1; i < 8; ++i) {
      if (values[i] > best) {
        best = values[i];
        expert = lane * 8 + i;
      }
    }
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
      const float other = __shfl_xor(best, mask, 64);
      const int other_id = __shfl_xor(expert, mask, 64);
      if (other > best || (other == best && other_id < expert)) {
        best = other;
        expert = other_id;
      }
    }
    if (lane == 0) {
      weights[pick] = best;
      ids[pick] = expert;
      selected_sum += best;
    }
    if (lane == expert / 8) values[expert % 8] = -10000.0f;
  }
  if (lane == 0) {
    const float inv_selected = 1.0f / selected_sum;
#pragma unroll
    for (int i = 0; i < K; ++i) weights[i] *= inv_selected;
  }
}

struct Gfx90aQwenTopk {
  static void run(const tvm::ffi::TensorView logits,
                  const tvm::ffi::TensorView weights,
                  const tvm::ffi::TensorView ids) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({1, 512}).with_dtype<bf16_t>().with_device(device).verify(logits);
    TensorMatcher({1, 10}).with_dtype<float>().with_device(device).verify(weights);
    TensorMatcher({1, 10}).with_dtype<int>().with_device(device).verify(ids);
    LaunchKernel(1, 64, logits.device())(
        gfx90a_qwen_topk_kernel,
        static_cast<const bf16_t*>(logits.data_ptr()),
        static_cast<float*>(weights.data_ptr()),
        static_cast<int*>(ids.data_ptr()));
  }
};

}  // namespace sglang
