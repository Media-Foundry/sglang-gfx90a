#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// Native CDNA2 wave64 GEMV for single-token decode:
//   out[1, N] = x[1, K] @ weight[N, K]^T
//
// A wave owns consecutive output rows.  The activation is staged once per
// workgroup, weights are read as aligned 16-byte vectors, accumulation is
// FP32, and the only reduction is a wave64 shuffle tree.  This deliberately
// avoids a split-K output buffer and its second launch.
constexpr uint32_t kGfx90aGemvVec = 16 / sizeof(bf16_t);
constexpr uint32_t kGfx90aWave = 64;

__device__ __forceinline__ float gfx90a_dot8_f32(const float4 wv, const float4 xv) {
  const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
    const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
    acc = fmaf(w0, x0, acc);
    acc = fmaf(w1, x1, acc);
  }
  return acc;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t kRows,
          uint32_t kUnroll, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aWave)
    gfx90a_bf16_gemv_kernel(bf16_t* __restrict__ out,
                            const bf16_t* __restrict__ x,
                            const bf16_t* __restrict__ weight) {
  constexpr uint32_t kRowsPerBlock = kRows * kNumWaves;
  constexpr uint32_t kBlocksPerToken =
      (N + kRowsPerBlock - 1) / kRowsPerBlock;
  const uint32_t token = blockIdx.x / kBlocksPerToken;
  const uint32_t local_block = blockIdx.x % kBlocksPerToken;
  __shared__ bf16_t sx[K];

  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kGfx90aGemvVec; k < K;
       k += kNumWaves * kGfx90aWave * kGfx90aGemvVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + static_cast<size_t>(token) * K + k);
  }
  __syncthreads();

  const uint32_t wave = tid / kGfx90aWave;
  const uint32_t lane = tid % kGfx90aWave;
  const uint32_t row0 = (local_block * kNumWaves + wave) * kRows;
  if (row0 >= N) return;

  float acc[kRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) acc[r] = 0.0f;

  constexpr uint32_t kStep = kGfx90aWave * kGfx90aGemvVec * kUnroll;
  for (uint32_t k = lane * kGfx90aGemvVec * kUnroll; k < K; k += kStep) {
    float4 xv[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[u] = *reinterpret_cast<const float4*>(sx + k + u * kGfx90aGemvVec);
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        const bf16_t* wr = weight + static_cast<size_t>(row0 + r) * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          const float4 wv =
              *reinterpret_cast<const float4*>(wr + u * kGfx90aGemvVec);
          acc[r] += gfx90a_dot8_f32(wv, xv[u]);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset, kGfx90aWave);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        out[static_cast<size_t>(token) * N + row0 + r] = cast<bf16_t>(acc[r]);
      }
    }
  }
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t kRows,
          uint32_t kUnroll, uint32_t kNumWaves>
struct Gfx90aBf16GemvKernel {
  static_assert(K % kGfx90aGemvVec == 0,
                "K must cover complete 16-byte BF16 vectors");

  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    constexpr uint32_t kRowsPerBlock = kRows * kNumWaves;
    constexpr uint32_t kBlocksPerToken =
        (N + kRowsPerBlock - 1) / kRowsPerBlock;
    constexpr uint32_t kBlocks = M * kBlocksPerToken;
    LaunchKernel(kBlocks, kNumWaves * kGfx90aWave, device.unwrap())(
        gfx90a_bf16_gemv_kernel<M, N, K, kRows, kUnroll, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()));
  }
};

// Experimental Qwen BS1 shared-expert oracle.  The two 32-lane subgroups in
// each native wave compute gate[row] and up[row+I] concurrently, then retain
// the BF16 materialization boundaries of GEMV -> SiLU -> multiply.
template <uint32_t I, uint32_t K, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aWave)
    gfx90a_bf16_gate_up_swiglu_subgroup_kernel(
        bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ weight) {
  __shared__ bf16_t sx[K];
  __shared__ bf16_t pair[kNumWaves][2];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kGfx90aGemvVec; k < K;
       k += kNumWaves * kGfx90aWave * kGfx90aGemvVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  }
  __syncthreads();

  const uint32_t wave = tid / kGfx90aWave;
  const uint32_t wave_lane = tid % kGfx90aWave;
  const uint32_t subgroup = wave_lane >> 5;
  const uint32_t lane = wave_lane & 31;
  const uint32_t row = blockIdx.x * kNumWaves + wave;
  if (row >= I) return;

  float acc = 0.0f;
  constexpr uint32_t kStep = 32 * kGfx90aGemvVec;
  const bf16_t* wr = weight + static_cast<size_t>(row + subgroup * I) * K;
  for (uint32_t k = lane * kGfx90aGemvVec; k < K; k += kStep) {
    acc += gfx90a_dot8_f32(
        *reinterpret_cast<const float4*>(wr + k),
        *reinterpret_cast<const float4*>(sx + k));
  }
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down(acc, offset, 32);
  if (lane == 0) pair[wave][subgroup] = cast<bf16_t>(acc);
  __syncthreads();

  if (wave_lane == 0) {
    const float gate = cast<float>(pair[wave][0]);
    const float up = cast<float>(pair[wave][1]);
    const bf16_t activated =
        cast<bf16_t>(gate / (1.0f + expf(-gate)));
    out[row] = cast<bf16_t>(cast<float>(activated) * up);
  }
}

template <uint32_t I, uint32_t K, uint32_t kNumWaves>
struct Gfx90aBf16GateUpSwigluSubgroupKernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({2 * I, K})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({1, I}).with_dtype<bf16_t>().with_device(device).verify(out);
    constexpr uint32_t kBlocks = (I + kNumWaves - 1) / kNumWaves;
    LaunchKernel(kBlocks, kNumWaves * kGfx90aWave, device.unwrap())(
        gfx90a_bf16_gate_up_swiglu_subgroup_kernel<I, K, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()));
  }
};

// Attention preparation consumes a float tensor but intentionally rounds each
// dot product through BF16 first to match the AIter BF16 GEMM contract.
template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aWave)
    gfx90a_bf16_fp32_gemv_kernel(float* __restrict__ out,
                                 const bf16_t* __restrict__ x,
                                 const bf16_t* __restrict__ weight) {
  __shared__ bf16_t sx[K];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kGfx90aGemvVec; k < K;
       k += kNumWaves * kGfx90aWave * kGfx90aGemvVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(x + k);
  }
  __syncthreads();

  const uint32_t wave = tid / kGfx90aWave;
  const uint32_t lane = tid % kGfx90aWave;
  const uint32_t row0 = (blockIdx.x * kNumWaves + wave) * kRows;
  if (row0 >= N) return;

  float acc[kRows] = {};
  constexpr uint32_t kStep = kGfx90aWave * kGfx90aGemvVec * kUnroll;
  for (uint32_t k = lane * kGfx90aGemvVec * kUnroll; k < K; k += kStep) {
    float4 xv[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[u] = *reinterpret_cast<const float4*>(sx + k + u * kGfx90aGemvVec);
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        const bf16_t* wr = weight + static_cast<size_t>(row0 + r) * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          acc[r] += gfx90a_dot8_f32(
              *reinterpret_cast<const float4*>(wr + u * kGfx90aGemvVec), xv[u]);
        }
      }
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset, kGfx90aWave);
    }
    if (lane == 0 && row0 + r < N) {
      out[row0 + r] = cast<float>(cast<bf16_t>(acc[r]));
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
struct Gfx90aBf16Fp32GemvKernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({1, N}).with_dtype<float>().with_device(device).verify(out);
    constexpr uint32_t kBlocks =
        (N + kRows * kNumWaves - 1) / (kRows * kNumWaves);
    LaunchKernel(kBlocks, kNumWaves * kGfx90aWave, device.unwrap())(
        gfx90a_bf16_fp32_gemv_kernel<N, K, kRows, kUnroll, kNumWaves>,
        static_cast<float*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()));
  }
};

// DSV4 wo_a has two independent [N,K] groups. Keep a single launch while each
// workgroup stages the activation belonging to its own group.
template <uint32_t M, uint32_t G, uint32_t N, uint32_t K, uint32_t kRows,
          uint32_t kUnroll, uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aWave)
    gfx90a_bf16_grouped_gemv_kernel(bf16_t* __restrict__ out,
                                    const bf16_t* __restrict__ x,
                                    const bf16_t* __restrict__ weight) {
  constexpr uint32_t kRowsPerBlock = kRows * kNumWaves;
  constexpr uint32_t kBlocksPerGroup = (N + kRowsPerBlock - 1) / kRowsPerBlock;
  constexpr uint32_t kBlocksPerToken = G * kBlocksPerGroup;
  const uint32_t token = blockIdx.x / kBlocksPerToken;
  const uint32_t token_block = blockIdx.x % kBlocksPerToken;
  const uint32_t group = token_block / kBlocksPerGroup;
  const uint32_t local_block = token_block % kBlocksPerGroup;
  __shared__ bf16_t sx[K];
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = tid * kGfx90aGemvVec; k < K;
       k += kNumWaves * kGfx90aWave * kGfx90aGemvVec) {
    *reinterpret_cast<float4*>(sx + k) =
        *reinterpret_cast<const float4*>(
            x + (static_cast<size_t>(token) * G + group) * K + k);
  }
  __syncthreads();

  const uint32_t wave = tid / kGfx90aWave;
  const uint32_t lane = tid % kGfx90aWave;
  const uint32_t row0 = (local_block * kNumWaves + wave) * kRows;
  if (row0 >= N) return;
  float acc[kRows] = {};
  constexpr uint32_t kStep = kGfx90aWave * kGfx90aGemvVec * kUnroll;
  for (uint32_t k = lane * kGfx90aGemvVec * kUnroll; k < K; k += kStep) {
    float4 xv[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[u] = *reinterpret_cast<const float4*>(sx + k + u * kGfx90aGemvVec);
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        const size_t row = static_cast<size_t>(group) * N + row0 + r;
        const bf16_t* wr = weight + row * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          acc[r] += gfx90a_dot8_f32(
              *reinterpret_cast<const float4*>(wr + u * kGfx90aGemvVec), xv[u]);
        }
      }
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t offset = 32; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset, kGfx90aWave);
    }
    if (lane == 0 && row0 + r < N) {
      out[(static_cast<size_t>(token) * G + group) * N + row0 + r] =
          cast<bf16_t>(acc[r]);
    }
  }
}

template <uint32_t M, uint32_t G, uint32_t N, uint32_t K, uint32_t kRows,
          uint32_t kUnroll, uint32_t kNumWaves>
struct Gfx90aBf16GroupedGemvKernel {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, G, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({G, N, K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, G, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    constexpr uint32_t kBlocksPerGroup =
        (N + kRows * kNumWaves - 1) / (kRows * kNumWaves);
    LaunchKernel(M * G * kBlocksPerGroup, kNumWaves * kGfx90aWave, device.unwrap())(
        gfx90a_bf16_grouped_gemv_kernel<M, G, N, K, kRows, kUnroll, kNumWaves>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()));
  }
};

}  // namespace sglang
