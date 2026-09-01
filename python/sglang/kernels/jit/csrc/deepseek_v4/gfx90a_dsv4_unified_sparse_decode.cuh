#pragma once

#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <stdexcept>

#include "dsv4_unified_sparse_decode_ck.cuh"

namespace sglang {

struct Gfx90aDsv4UnifiedSparseDecode {
  static void run(const tvm::ffi::TensorView q,
                  const tvm::ffi::TensorView unified_kv,
                  const tvm::ffi::TensorView kv_indices,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView attn_sink,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView workspace,
                  double softmax_scale) {
    const auto tokens = q.size(0);
    if (q.ndim() != 3 || tokens <= 0 || tokens > 128 || q.size(1) != 16 ||
        q.size(2) != 512 || unified_kv.ndim() != 2 ||
        unified_kv.size(1) != 512 || output.ndim() != 3 ||
        output.size(0) != tokens || output.size(1) != 16 ||
        output.size(2) != 512 || kv_indptr.size(0) != tokens + 1) {
      throw std::runtime_error("gfx90a DSV4 sparse decode shape mismatch");
    }

    ck_tile::dsv4::UnifiedSparseDecodeArgs args{
        static_cast<const ck::bhalf_t*>(q.data_ptr()),
        static_cast<const ck::bhalf_t*>(unified_kv.data_ptr()),
        static_cast<const int32_t*>(kv_indices.data_ptr()),
        static_cast<const int32_t*>(kv_indptr.data_ptr()),
        static_cast<const float*>(attn_sink.data_ptr()),
        static_cast<ck::bhalf_t*>(output.data_ptr()),
        static_cast<int32_t>(tokens),
        16,
        static_cast<int32_t>(unified_kv.size(0)),
        static_cast<float>(softmax_scale)};

    // The Python selector performs the one-time gfx90a gate.  Calling the
    // public convenience entry here would run hipGetDeviceProperties inside
    // CUDA/HIP graph capture, so launch the selected graph-safe instance
    // directly on SGLang's current stream.
    const auto stream = sglang::host::LaunchKernel::resolve_device(q.device());
    const hipError_t status =
        ck_tile::dsv4::launch_unified_sparse_decode_d512_mfma_split2_qreg_kvprefetch(
            args, workspace.data_ptr(), stream);
    if (status != hipSuccess) {
      throw std::runtime_error("gfx90a DSV4 sparse decode launch failed");
    }
  }
};

}  // namespace sglang
