#include "gfx90a_bf16_gemv.cuh"

namespace sglang {

template <uint32_t M, uint32_t G, uint32_t N, uint32_t K,
          uint32_t kMTile, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
__global__ void __launch_bounds__(kNumWaves * kGfx90aWave)
    gfx90a_bf16_grouped_gemv_mtile_oracle_kernel(
        bf16_t* __restrict__ out, const bf16_t* __restrict__ x,
        const bf16_t* __restrict__ weight) {
  constexpr uint32_t kRowsPerBlock = kRows * kNumWaves;
  constexpr uint32_t kBlocksPerGroup = (N + kRowsPerBlock - 1) / kRowsPerBlock;
  constexpr uint32_t kBlocksPerTokenTile = G * kBlocksPerGroup;
  const uint32_t token_tile = blockIdx.x / kBlocksPerTokenTile;
  const uint32_t token_block = blockIdx.x % kBlocksPerTokenTile;
  const uint32_t group = token_block / kBlocksPerGroup;
  const uint32_t local_block = token_block % kBlocksPerGroup;
  const uint32_t token0 = token_tile * kMTile;
  __shared__ bf16_t sx[kMTile][K];
  const uint32_t tid = threadIdx.x;
#pragma unroll
  for (uint32_t mt = 0; mt < kMTile; ++mt) {
    const uint32_t token = token0 + mt;
    if (token < M) {
      for (uint32_t k = tid * kGfx90aGemvVec; k < K;
           k += kNumWaves * kGfx90aWave * kGfx90aGemvVec) {
        *reinterpret_cast<float4*>(&sx[mt][k]) =
            *reinterpret_cast<const float4*>(
                x + (static_cast<size_t>(token) * G + group) * K + k);
      }
    }
  }
  __syncthreads();

  const uint32_t wave = tid / kGfx90aWave;
  const uint32_t lane = tid % kGfx90aWave;
  const uint32_t row0 = (local_block * kNumWaves + wave) * kRows;
  if (row0 >= N) return;
  float acc[kRows][kMTile] = {};
  constexpr uint32_t kStep = kGfx90aWave * kGfx90aGemvVec * kUnroll;
  for (uint32_t k = lane * kGfx90aGemvVec * kUnroll; k < K; k += kStep) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (row0 + r < N) {
        const size_t row = static_cast<size_t>(group) * N + row0 + r;
        const bf16_t* wr = weight + row * K + k;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          const float4 wv =
              *reinterpret_cast<const float4*>(wr + u * kGfx90aGemvVec);
#pragma unroll
          for (uint32_t mt = 0; mt < kMTile; ++mt) {
            if (token0 + mt < M) {
              const float4 xv = *reinterpret_cast<const float4*>(
                  &sx[mt][k + u * kGfx90aGemvVec]);
              acc[r][mt] += gfx90a_dot8_f32(wv, xv);
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t mt = 0; mt < kMTile; ++mt) {
#pragma unroll
      for (uint32_t offset = 32; offset > 0; offset >>= 1)
        acc[r][mt] += __shfl_down(acc[r][mt], offset, kGfx90aWave);
      if (lane == 0 && row0 + r < N && token0 + mt < M) {
        out[(static_cast<size_t>(token0 + mt) * G + group) * N + row0 + r] =
            cast<bf16_t>(acc[r][mt]);
      }
    }
  }
}

template <uint32_t M, uint32_t G, uint32_t N, uint32_t K,
          uint32_t kMTile, uint32_t kRows, uint32_t kUnroll,
          uint32_t kNumWaves>
struct Gfx90aBf16GroupedGemvMtileOracle {
  static void run(const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{}; device.set_options<kDLCUDA>();
    TensorMatcher({M,G,K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({G,N,K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M,G,N}).with_dtype<bf16_t>().with_device(device).verify(out);
    constexpr uint32_t blocks = (N + kRows*kNumWaves - 1)/(kRows*kNumWaves);
    constexpr uint32_t tiles = (M + kMTile - 1)/kMTile;
    LaunchKernel(tiles*G*blocks, kNumWaves*kGfx90aWave, device.unwrap())(
      gfx90a_bf16_grouped_gemv_mtile_oracle_kernel<
        M,G,N,K,kMTile,kRows,kUnroll,kNumWaves>,
      static_cast<bf16_t*>(out.data_ptr()),
      static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const bf16_t*>(weight.data_ptr()));
  }
};

} // namespace sglang
