#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kGfx90aRouterExperts = 256;
constexpr uint32_t kGfx90aRouterGroups = 8;
constexpr uint32_t kGfx90aRouterExpertsPerGroup = 32;
constexpr uint32_t kGfx90aRouterTopGroups = 4;
constexpr uint32_t kGfx90aRouterTopK = 6;
constexpr uint32_t kGfx90aRouterWave = 64;
constexpr uint32_t kGfx90aRouterWaves = 4;

__device__ __forceinline__ void gfx90a_router_wave_argmax(
    float& value, int& expert) {
#pragma unroll
  for (uint32_t offset = 32; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down(value, offset, kGfx90aRouterWave);
    const int other_expert = __shfl_down(expert, offset, kGfx90aRouterWave);
    if (other_value > value ||
        (other_value == value && other_expert >= 0 &&
         (expert < 0 || other_expert < expert))) {
      value = other_value;
      expert = other_expert;
    }
  }
}

template <typename BiasT>
__global__ void __launch_bounds__(kGfx90aRouterExperts, 2)
    gfx90a_grouped_router_kernel(
        const float* __restrict__ scores,
        const BiasT* __restrict__ bias,
        float* __restrict__ output,
        int32_t* __restrict__ indices,
        float routed_scaling_factor,
        bool apply_scale) {
  __shared__ float activated[kGfx90aRouterExperts];
  __shared__ float ranked[kGfx90aRouterExperts];
  __shared__ float group_score[kGfx90aRouterGroups];
  __shared__ uint8_t keep_group[kGfx90aRouterGroups];
  __shared__ float wave_max[kGfx90aRouterWaves];
  __shared__ int wave_expert[kGfx90aRouterWaves];
  __shared__ int selected[kGfx90aRouterTopK];
  __shared__ float selected_sum;

  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kGfx90aRouterWave;
  const uint32_t lane = tid % kGfx90aRouterWave;
  const float a = 1.0f / (1.0f + expf(-scores[tid]));
  activated[tid] = a;
  ranked[tid] = a + cast<float>(bias[tid]);
  if (tid < kGfx90aRouterGroups) keep_group[tid] = 0;
  __syncthreads();

  // Eight threads independently score one 32-expert group.  This short,
  // register-only scan is cheaper on CDNA2 than materializing two reduction
  // passes and preserves DeepSeek's top-2-sum group definition.
  if (tid < kGfx90aRouterGroups) {
    const uint32_t begin = tid * kGfx90aRouterExpertsPerGroup;
    float first = -FLT_MAX;
    float second = -FLT_MAX;
#pragma unroll
    for (uint32_t i = 0; i < kGfx90aRouterExpertsPerGroup; ++i) {
      const float v = ranked[begin + i];
      if (v > first) {
        second = first;
        first = v;
      } else if (v > second) {
        second = v;
      }
    }
    group_score[tid] = first + second;
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (uint32_t pick = 0; pick < kGfx90aRouterTopGroups; ++pick) {
      float best = -FLT_MAX;
      int best_group = -1;
#pragma unroll
      for (uint32_t g = 0; g < kGfx90aRouterGroups; ++g) {
        const float v = keep_group[g] ? -FLT_MAX : group_score[g];
        if (v > best || (v == best && static_cast<int>(g) < best_group)) {
          best = v;
          best_group = static_cast<int>(g);
        }
      }
      keep_group[best_group] = 1;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t pick = 0; pick < kGfx90aRouterTopK; ++pick) {
    float value = keep_group[tid / kGfx90aRouterExpertsPerGroup]
                      ? ranked[tid]
                      : -FLT_MAX;
    int expert = static_cast<int>(tid);
    gfx90a_router_wave_argmax(value, expert);
    if (lane == 0) {
      wave_max[wave] = value;
      wave_expert[wave] = expert;
    }
    __syncthreads();
    if (tid == 0) {
      float best = -FLT_MAX;
      int best_expert = -1;
#pragma unroll
      for (uint32_t w = 0; w < kGfx90aRouterWaves; ++w) {
        const float v = wave_max[w];
        const int e = wave_expert[w];
        if (v > best || (v == best && e >= 0 && (best_expert < 0 || e < best_expert))) {
          best = v;
          best_expert = e;
        }
      }
      selected[pick] = best_expert;
      ranked[best_expert] = -FLT_MAX;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kGfx90aRouterTopK; ++k) {
      sum += activated[selected[k]];
    }
    selected_sum = sum;
  }
  __syncthreads();
  if (tid < kGfx90aRouterTopK) {
    const int expert = selected[tid];
    const float scale = apply_scale ? routed_scaling_factor : 1.0f;
    output[tid] = activated[expert] / selected_sum * scale;
    indices[tid] = expert;
  }
}

template <typename ScoreT, typename BiasT>
__global__ void __launch_bounds__(kGfx90aRouterExperts, 2)
    gfx90a_sqrtsoftplus_router_kernel(
        const ScoreT* __restrict__ scores,
        const BiasT* __restrict__ bias,
        float* __restrict__ output,
        int32_t* __restrict__ indices,
        float routed_scaling_factor,
        bool apply_scale) {
  __shared__ float activated[kGfx90aRouterExperts];
  __shared__ float ranked[kGfx90aRouterExperts];
  __shared__ float wave_max[kGfx90aRouterWaves];
  __shared__ int wave_expert[kGfx90aRouterWaves];
  __shared__ int selected[kGfx90aRouterTopK];
  __shared__ float selected_sum;

  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / kGfx90aRouterWave;
  const uint32_t lane = tid % kGfx90aRouterWave;
  const float logit = cast<float>(scores[tid]);
  const float softplus =
      logit > 20.0f ? logit : log1pf(expf(logit));
  const float a = sqrtf(softplus);
  activated[tid] = a;
  ranked[tid] = a + cast<float>(bias[tid]);
  __syncthreads();

#pragma unroll
  for (uint32_t pick = 0; pick < kGfx90aRouterTopK; ++pick) {
    float value = ranked[tid];
    int expert = static_cast<int>(tid);
    gfx90a_router_wave_argmax(value, expert);
    if (lane == 0) {
      wave_max[wave] = value;
      wave_expert[wave] = expert;
    }
    __syncthreads();
    if (tid == 0) {
      float best = -FLT_MAX;
      int best_expert = -1;
#pragma unroll
      for (uint32_t w = 0; w < kGfx90aRouterWaves; ++w) {
        const float v = wave_max[w];
        const int e = wave_expert[w];
        if (v > best ||
            (v == best && e >= 0 && (best_expert < 0 || e < best_expert))) {
          best = v;
          best_expert = e;
        }
      }
      selected[pick] = best_expert;
      ranked[best_expert] = -FLT_MAX;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kGfx90aRouterTopK; ++k) {
      sum += activated[selected[k]];
    }
    selected_sum = sum;
  }
  __syncthreads();
  if (tid < kGfx90aRouterTopK) {
    const int expert = selected[tid];
    const float scale = apply_scale ? routed_scaling_factor : 1.0f;
    output[tid] = activated[expert] / selected_sum * scale;
    indices[tid] = expert;
  }
}

struct Gfx90aGroupedRouterKernel {
  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView bias,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView indices,
                  float routed_scaling_factor,
                  bool apply_scale) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, 256}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({256}).with_dtype<bf16_t>().with_device(device).verify(bias);
    TensorMatcher({1, 6}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({1, 6}).with_dtype<int32_t>().with_device(device).verify(indices);
    LaunchKernel(1, kGfx90aRouterExperts, device.unwrap())(
        gfx90a_grouped_router_kernel<bf16_t>,
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const bf16_t*>(bias.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        routed_scaling_factor, apply_scale);
  }
};

struct Gfx90aGroupedRouterFp32BiasKernel {
  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView bias,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView indices,
                  float routed_scaling_factor,
                  bool apply_scale) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, 256}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({256}).with_dtype<float>().with_device(device).verify(bias);
    TensorMatcher({1, 6}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({1, 6}).with_dtype<int32_t>().with_device(device).verify(indices);
    LaunchKernel(1, kGfx90aRouterExperts, device.unwrap())(
        gfx90a_grouped_router_kernel<float>,
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const float*>(bias.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        routed_scaling_factor, apply_scale);
  }
};

struct Gfx90aSqrtSoftplusRouterKernel {
  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView bias,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView indices,
                  float routed_scaling_factor,
                  bool apply_scale) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, 256}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({256}).with_dtype<bf16_t>().with_device(device).verify(bias);
    TensorMatcher({1, 6}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({1, 6}).with_dtype<int32_t>().with_device(device).verify(indices);
    LaunchKernel(1, kGfx90aRouterExperts, device.unwrap())(
        gfx90a_sqrtsoftplus_router_kernel<float, bf16_t>,
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const bf16_t*>(bias.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        routed_scaling_factor, apply_scale);
  }
};

struct Gfx90aSqrtSoftplusRouterFp32BiasKernel {
  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView bias,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView indices,
                  float routed_scaling_factor,
                  bool apply_scale) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, 256}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({256}).with_dtype<float>().with_device(device).verify(bias);
    TensorMatcher({1, 6}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({1, 6}).with_dtype<int32_t>().with_device(device).verify(indices);
    LaunchKernel(1, kGfx90aRouterExperts, device.unwrap())(
        gfx90a_sqrtsoftplus_router_kernel<float, float>,
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const float*>(bias.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        routed_scaling_factor, apply_scale);
  }
};

struct Gfx90aSqrtSoftplusRouterBf16Kernel {
  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView bias,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView indices,
                  float routed_scaling_factor,
                  bool apply_scale) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, 256}).with_dtype<bf16_t>().with_device(device).verify(scores);
    TensorMatcher({256}).with_dtype<bf16_t>().with_device(device).verify(bias);
    TensorMatcher({1, 6}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({1, 6}).with_dtype<int32_t>().with_device(device).verify(indices);
    LaunchKernel(1, kGfx90aRouterExperts, device.unwrap())(
        gfx90a_sqrtsoftplus_router_kernel<bf16_t, bf16_t>,
        static_cast<const bf16_t*>(scores.data_ptr()),
        static_cast<const bf16_t*>(bias.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        routed_scaling_factor, apply_scale);
  }
};

}  // namespace sglang
