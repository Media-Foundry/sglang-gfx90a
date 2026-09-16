#pragma once
#include "gfx90a_fp4_expert_gemv.cuh"

namespace sglang {
__global__ void ck_slot_ids_kernel(const int32_t* ids, const int32_t* valid,
                                   int32_t* out, int count, int rows) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= count) return;
  if (j >= valid[0]) { out[j] = rows * 6; return; }
  const uint32_t id = static_cast<uint32_t>(ids[j]);
  const uint32_t token = id & 0xffffff;
  const uint32_t slot = id >> 24;
  out[j] = token < rows && slot < 6 ? token * 6 + slot : rows * 6;
}

template <typename Out>
__global__ void ck_slot_reduce_kernel(const float* partial, Out* out, int count) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < count;
       j += blockDim.x * gridDim.x) {
    const int token = j / 4096, col = j % 4096;
    const size_t base = static_cast<size_t>(token) * 6 * 4096 + col;
    float sum = partial[base];
    #pragma unroll
    for (int k = 1; k < 6; ++k) sum = sum + partial[base + k * 4096];
    out[j] = cast<Out>(sum);
  }
}

template <typename T>
struct alignas(sizeof(T) * 4) CkSlotVec4 { T v[4]; };

template <typename Out>
__global__ void ck_slot_reduce_vec4_kernel(const float* partial, Out* out, int vectors) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < vectors;
       j += blockDim.x * gridDim.x) {
    const int token = j / 1024, col = (j % 1024) * 4;
    const size_t base = static_cast<size_t>(token) * 6 * 4096 + col;
    auto sum = *reinterpret_cast<const CkSlotVec4<float>*>(partial + base);
    #pragma unroll
    for (int k = 1; k < 6; ++k) {
      const auto x = *reinterpret_cast<const CkSlotVec4<float>*>(partial + base + k * 4096);
      #pragma unroll
      for (int c = 0; c < 4; ++c) sum.v[c] = sum.v[c] + x.v[c];
    }
    CkSlotVec4<Out> y;
    #pragma unroll
    for (int c = 0; c < 4; ++c) y.v[c] = cast<Out>(sum.v[c]);
    reinterpret_cast<CkSlotVec4<Out>*>(out)[j] = y;
  }
}

struct Gfx90aCkFixedSlot {
  static void remap(const tvm::ffi::TensorView ids, const tvm::ffi::TensorView valid,
                    const tvm::ffi::TensorView out, int64_t rows) {
    using namespace host;
    RuntimeCheck(rows > 0 && rows * 6 < (1 << 24), "invalid virtual token count");
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({ids.size(0)}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(valid);
    TensorMatcher({ids.size(0)}).with_dtype<int32_t>().with_device(device).verify(out);
    LaunchKernel((ids.size(0) + 255) / 256, 256, ids.device())(
        ck_slot_ids_kernel, static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<const int32_t*>(valid.data_ptr()), static_cast<int32_t*>(out.data_ptr()),
        static_cast<int>(ids.size(0)), static_cast<int>(rows));
  }
  template <typename Out>
  static void reduce_impl(const tvm::ffi::TensorView partial, const tvm::ffi::TensorView out) {
    using namespace host;
    const int64_t rows = out.size(0);
    RuntimeCheck(rows > 0 && rows <= 36864, "invalid row count");
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, 6, 4096}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({rows, 4096}).with_dtype<Out>().with_device(device).verify(out);
    LaunchKernel(832, 256, out.device())(
        ck_slot_reduce_kernel<Out>, static_cast<const float*>(partial.data_ptr()),
        static_cast<Out*>(out.data_ptr()), static_cast<int>(rows * 4096));
  }
  static void reduce(const tvm::ffi::TensorView partial, const tvm::ffi::TensorView out) {
    reduce_impl<bf16_t>(partial, out);
  }
  static void reduce_float(const tvm::ffi::TensorView partial, const tvm::ffi::TensorView out) {
    reduce_impl<float>(partial, out);
  }
  template <typename Out>
  static void reduce_vec4_impl(const tvm::ffi::TensorView partial, const tvm::ffi::TensorView out) {
    using namespace host;
    const int64_t rows = out.size(0);
    RuntimeCheck(rows > 0 && rows <= 36864, "invalid row count");
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, 6, 4096}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({rows, 4096}).with_dtype<Out>().with_device(device).verify(out);
    RuntimeCheck(reinterpret_cast<uintptr_t>(partial.data_ptr()) % 16 == 0, "unaligned partial");
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr()) % (4 * sizeof(Out)) == 0, "unaligned output");
    LaunchKernel(1664, 256, out.device())(
        ck_slot_reduce_vec4_kernel<Out>, static_cast<const float*>(partial.data_ptr()),
        static_cast<Out*>(out.data_ptr()), static_cast<int>(rows * 1024));
  }
  static void reduce_vec4(const tvm::ffi::TensorView partial, const tvm::ffi::TensorView out) {
    reduce_vec4_impl<bf16_t>(partial, out);
  }
  static void reduce_vec4_float(const tvm::ffi::TensorView partial, const tvm::ffi::TensorView out) {
    reduce_vec4_impl<float>(partial, out);
  }
};
}  // namespace sglang
