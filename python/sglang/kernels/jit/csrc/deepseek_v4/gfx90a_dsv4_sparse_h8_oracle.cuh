#pragma once
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <stdexcept>
#include "gfx90a_dsv4_unified_sparse_decode.cuh"

namespace sglang {
struct Gfx90aDsv4SparseH8Oracle {
  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView kv,
                  tvm::ffi::TensorView indices, tvm::ffi::TensorView indptr,
                  tvm::ffi::TensorView sink, tvm::ffi::TensorView out,
                  tvm::ffi::TensorView scratch, double scale) {
    const int tokens = q.size(0);
    if(q.ndim()!=3 || tokens<=0 || tokens>192 || q.size(1)!=8 || q.size(2)!=512 ||
       kv.ndim()!=2 || kv.size(1)!=512 || out.ndim()!=3 ||
       out.size(0)!=tokens || out.size(1)!=8 || out.size(2)!=512 ||
       indptr.size(0)!=tokens+1 || sink.size(0)!=8 ||
       scratch.numel() < static_cast<int64_t>(tokens)*2*8*514*4)
      throw std::runtime_error("H8 oracle shape/workspace mismatch");
    ck_tile::dsv4::UnifiedSparseDecodeArgs args{
      static_cast<const ck::bhalf_t*>(q.data_ptr()),
      static_cast<const ck::bhalf_t*>(kv.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<const int32_t*>(indptr.data_ptr()),
      static_cast<const float*>(sink.data_ptr()),
      static_cast<ck::bhalf_t*>(out.data_ptr()), tokens, 8,
      static_cast<int32_t>(kv.size(0)), static_cast<float>(scale)};
    auto stream = sglang::host::LaunchKernel::resolve_device(q.device());
    auto status = ck_tile::dsv4::launch_unified_sparse_decode_d512_mfma_split2_impl<false,true,true,8>(
      args,scratch.data_ptr(),2,stream);
    if(status!=hipSuccess) throw std::runtime_error(
      std::string("H8 oracle HIP launch failed: ") + hipGetErrorString(status));
  }
};
}
