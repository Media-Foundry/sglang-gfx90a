#pragma once

// Compiled after gfx90a_fp4_expert_gemv.cuh. Standalone oracle only.

namespace sglang {

using gfx90a_gate_i32x4 = int32_t __attribute__((ext_vector_type(4)));

template <uint32_t kSplit, bool kWriteGroupInt>
__global__ void __launch_bounds__(kSplit * 64)
    gfx90a_mfma_i8_gate_tile_kernel(
        float* __restrict__ out, int32_t* __restrict__ group_int,
        const int8_t* __restrict__ xq, const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale) {
  static_assert(kSplit == 1 || kSplit == 2 || kSplit == 4 || kSplit == 8);
  constexpr uint32_t A = 4;
  constexpr uint32_t N = 64;
  constexpr uint32_t K = 4096;
  constexpr uint32_t G = K / 32;
  __shared__ uint32_t pair_lut[256];
  __shared__ float partial[8][A][N];
  for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
    pair_lut[i] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(i))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t wave = threadIdx.x / 64;
  const uint32_t lane = threadIdx.x % 64;
  const uint32_t assignment_lane = lane & 3u;
  float float_acc[A] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t group = wave; group < G; group += kSplit) {
    gfx90a_gate_i32x4 int_acc = {0, 0, 0, 0};
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      int32_t a = 0;
      if (lane < 4) {
        a = *reinterpret_cast<const int32_t*>(
            xq + assignment_lane * K + group * 32 + j * 4);
      }
      a = __shfl(a, assignment_lane, 64);
      const size_t packed_offset =
          static_cast<size_t>(lane) * (K / 2) + group * 16 + j * 2;
      const int32_t b = gfx90a_fp4_pack4_i8_lds(
          *reinterpret_cast<const uint16_t*>(weight + packed_offset), pair_lut);
      int_acc = __builtin_amdgcn_mfma_i32_4x4x4i8(
          a, b, int_acc, 0, 0, 0);
    }
    const float ws = gfx90a_e8m0_value(weight_scale[lane * G + group]);
#pragma unroll
    for (uint32_t assignment = 0; assignment < A; ++assignment) {
      if constexpr (kWriteGroupInt) {
        group_int[(group * A + assignment) * N + lane] = int_acc[assignment];
      }
      float_acc[assignment] += static_cast<float>(int_acc[assignment]) *
                               x_scale[assignment * G + group] * ws * 0.5f;
    }
  }
#pragma unroll
  for (uint32_t assignment = 0; assignment < A; ++assignment) {
    partial[wave][assignment][lane] = float_acc[assignment];
  }
  __syncthreads();
  if (wave == 0) {
#pragma unroll
    for (uint32_t assignment = 0; assignment < A; ++assignment) {
      float total = partial[0][assignment][lane];
#pragma unroll
      for (uint32_t split = 1; split < kSplit; ++split) {
        total += partial[split][assignment][lane];
      }
      out[assignment * N + lane] = total;
    }
  }
}

template <bool kWriteGroupInt>
__global__ void __launch_bounds__(512)
    gfx90a_sdot_i8_gate_tile_reference_kernel(
        float* __restrict__ out, int32_t* __restrict__ group_int,
        const int8_t* __restrict__ xq, const float* __restrict__ x_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale) {
  constexpr uint32_t A = 4;
  constexpr uint32_t N = 64;
  constexpr uint32_t K = 4096;
  constexpr uint32_t G = K / 32;
  constexpr uint32_t subgroup_width = 16;
  __shared__ uint32_t pair_lut[256];
  if (threadIdx.x < 256) {
    pair_lut[threadIdx.x] = static_cast<uint32_t>(
        gfx90a_fp4_pack4_i8(static_cast<uint16_t>(threadIdx.x))) & 0xffffu;
  }
  __syncthreads();

  const uint32_t wave = threadIdx.x / 64;
  const uint32_t lane = threadIdx.x % 64;
  const uint32_t subgroup = lane / subgroup_width;
  const uint32_t subgroup_lane = lane % subgroup_width;
  const uint32_t row0 = (wave * 4 + subgroup) * 2;
  float acc[A][2] = {};

  for (uint32_t group = subgroup_lane; group < G; group += subgroup_width) {
    int32_t weight_i8[2][8];
    float ws[2];
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      const uint32_t row = row0 + r;
      ws[r] = gfx90a_e8m0_value(weight_scale[row * G + group]);
      const size_t base =
          static_cast<size_t>(row) * (K / 2) + group * 16;
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        weight_i8[r][j] = gfx90a_fp4_pack4_i8_lds(
            *reinterpret_cast<const uint16_t*>(weight + base + j * 2),
            pair_lut);
      }
    }
#pragma unroll
    for (uint32_t assignment = 0; assignment < A; ++assignment) {
      const int8_t* x = xq + assignment * K + group * 32;
#pragma unroll
      for (uint32_t r = 0; r < 2; ++r) {
        int32_t dot = 0;
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
          dot = __builtin_amdgcn_sdot4(
              *reinterpret_cast<const int32_t*>(x + j * 4),
              weight_i8[r][j], dot, false);
        }
        if constexpr (kWriteGroupInt) {
          group_int[(group * A + assignment) * N + row0 + r] = dot;
        }
        acc[assignment][r] += static_cast<float>(dot) *
                              x_scale[assignment * G + group] * ws[r] * 0.5f;
      }
    }
  }

#pragma unroll
  for (uint32_t assignment = 0; assignment < A; ++assignment) {
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
#pragma unroll
      for (uint32_t offset = 8; offset > 0; offset >>= 1) {
        acc[assignment][r] += __shfl_down(acc[assignment][r], offset, 16);
      }
      if (subgroup_lane == 0) {
        out[assignment * N + row0 + r] = acc[assignment][r];
      }
    }
  }
}

template <uint32_t kSplit, bool kWriteGroupInt>
struct Gfx90aMfmaI8GateTileOracleKernel {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView group_int) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({4, 4096}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({4, 128}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({64, 2048}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({64, 128}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({4, 64}).with_dtype<float>().with_device(device).verify(out);
    TensorMatcher({128, 4, 64}).with_dtype<int32_t>().with_device(device).verify(group_int);
    LaunchKernel(1, kSplit * 64, xq.device())(
        gfx90a_mfma_i8_gate_tile_kernel<kSplit, kWriteGroupInt>,
        static_cast<float*>(out.data_ptr()),
        static_cast<int32_t*>(group_int.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()));
  }
};

template <bool kWriteGroupInt>
struct Gfx90aSdotI8GateTileReferenceKernel {
  static void run(const tvm::ffi::TensorView xq,
                  const tvm::ffi::TensorView x_scale,
                  const tvm::ffi::TensorView weight,
                  const tvm::ffi::TensorView weight_scale,
                  const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView group_int) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({4, 4096}).with_dtype<int8_t>().with_device(device).verify(xq);
    TensorMatcher({4, 128}).with_dtype<float>().with_device(device).verify(x_scale);
    TensorMatcher({64, 2048}).with_dtype<uint8_t>().with_device(device).verify(weight);
    TensorMatcher({64, 128}).with_dtype<uint8_t>().with_device(device).verify(weight_scale);
    TensorMatcher({4, 64}).with_dtype<float>().with_device(device).verify(out);
    TensorMatcher({128, 4, 64}).with_dtype<int32_t>().with_device(device).verify(group_int);
    LaunchKernel(1, 512, xq.device())(
        gfx90a_sdot_i8_gate_tile_reference_kernel<kWriteGroupInt>,
        static_cast<float*>(out.data_ptr()),
        static_cast<int32_t*>(group_int.data_ptr()),
        static_cast<const int8_t*>(xq.data_ptr()),
        static_cast<const float*>(x_scale.data_ptr()),
        static_cast<const uint8_t*>(weight.data_ptr()),
        static_cast<const uint8_t*>(weight_scale.data_ptr()));
  }
};

}  // namespace sglang
