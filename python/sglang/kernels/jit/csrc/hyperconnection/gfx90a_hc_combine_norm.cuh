#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

namespace sglang {

using namespace device;

struct QwenHcCombineNormParams {
  const bf16_t* block_output;
  const bf16_t* residual;
  const float* gate_partials;
  const bf16_t* norm_weight;
  bf16_t* combined;
  bf16_t* normed;
  float eps;
  uint32_t batch;
};

// Exact Qwen4 BS1 boundary:
//   combined[c,j] = residual[c,j] + gate[c] * block_output[j]
//   normed[c,:]   = GemmaRMSNorm(combined[c,:])
//
// The 160-thread/two-vector mapping deliberately matches the retained
// grouped_gemma_rmsnorm<2560> reduction.  Combined values are rounded to BF16
// before the sum of squares, just as the two-kernel reference materializes
// them to memory first.
__global__ __launch_bounds__(160) void gfx90a_qwen_hc_combine_norm_kernel(
    const QwenHcCombineNormParams params) {
  using Float2 = packed_t<bf16_t>;
  using Storage = AlignedVector<Float2, 4>;  // eight BF16 values
  constexpr uint32_t HC = 4;
  constexpr uint32_t H = 2560;
  constexpr uint32_t Vec = 8;
  constexpr uint32_t Threads = 160;
  constexpr uint32_t Vecs = H / Vec;
  constexpr uint32_t Loads = 2;
  constexpr uint32_t Warps = Threads / 32;

  const uint32_t branch = blockIdx.x % HC;
  const uint32_t batch = blockIdx.x / HC;
  const uint32_t tid = threadIdx.x;
  float total = 0.0f;
#pragma unroll
  for (uint32_t split = 0; split < 8; ++split)
    total += params.gate_partials[(batch * 8 + split) * HC + branch];
  const float gate = 2.0f / (1.0f + math::exp(-total * 0.25f));

  Storage combined_vec[Loads];
  float sum_of_squares = 0.0f;
#pragma unroll
  for (uint32_t load = 0; load < Loads; ++load) {
    const uint32_t vec = tid + load * Threads;
    Storage r;
    Storage y;
    r.load(params.residual + batch * HC * H + branch * H, vec);
    y.load(params.block_output + batch * H, vec);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y[i]);
      combined_vec[load][i] =
          cast<Float2>(fp32x2_t{rx + gate * yx, ry + gate * yy});
      const auto [cx, cy] = cast<fp32x2_t>(combined_vec[load][i]);
      sum_of_squares += cx * cx + cy * cy;
    }
    combined_vec[load].store(
        params.combined + batch * HC * H + branch * H, vec);
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  __shared__ float smem[32];
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  if (lane == 0) smem[warp] = sum_of_squares;
  __syncthreads();
  if (warp == 0) {
    float local = lane < Warps ? smem[lane] : 0.0f;
    local = warp::reduce_sum(local);
    smem[lane] = math::rsqrt(local / H + params.eps);
  }
  __syncthreads();
  const float norm_factor = smem[warp];

#pragma unroll
  for (uint32_t load = 0; load < Loads; ++load) {
    const uint32_t vec = tid + load * Threads;
    Storage w;
    Storage out;
    w.load(params.norm_weight + branch * H, vec);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto [cx, cy] = cast<fp32x2_t>(combined_vec[load][i]);
      const auto [wx, wy] = cast<fp32x2_t>(w[i]);
      out[i] = cast<Float2>(fp32x2_t{
          cx * norm_factor * (1.0f + wx),
          cy * norm_factor * (1.0f + wy)});
    }
    out.store(params.normed + batch * HC * H + branch * H, vec);
  }
}

struct Gfx90aQwenHcCombineNorm {
  static void run(const tvm::ffi::TensorView block_output,
                  const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView gate_partials,
                  const tvm::ffi::TensorView norm_weight,
                  const tvm::ffi::TensorView combined,
                  const tvm::ffi::TensorView normed,
                  double eps) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({-1, 2560}).with_dtype<bf16_t>().with_device(device).verify(block_output);
    TensorMatcher({-1, 10240}).with_dtype<bf16_t>().with_device(device).verify(residual).verify(combined).verify(normed);
    TensorMatcher({-1, 8, 4}).with_dtype<float>().with_device(device).verify(gate_partials);
    TensorMatcher({10240}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    const QwenHcCombineNormParams params{
        static_cast<const bf16_t*>(block_output.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(gate_partials.data_ptr()),
        static_cast<const bf16_t*>(norm_weight.data_ptr()),
        static_cast<bf16_t*>(combined.data_ptr()),
        static_cast<bf16_t*>(normed.data_ptr()),
        static_cast<float>(eps),
        static_cast<uint32_t>(block_output.size(0))};
    RuntimeCheck(residual.size(0) == block_output.size(0));
    RuntimeCheck(gate_partials.size(0) == block_output.size(0));
    RuntimeCheck(block_output.size(0) > 0 && block_output.size(0) <= 32);
    LaunchKernel(params.batch * 4, 160, device.unwrap())(
        gfx90a_qwen_hc_combine_norm_kernel, params);
  }
};

}  // namespace sglang
