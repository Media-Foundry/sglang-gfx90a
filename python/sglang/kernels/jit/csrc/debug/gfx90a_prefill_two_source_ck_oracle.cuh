#pragma once
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "dsv4_prefill_two_source_core.cuh"

namespace sglang::prefill_two_source_oracle {
namespace ckcore = ck_tile::dsv4_prefill_oracle;

// Sum of the two existing indptrs is already the combined ragged indptr.
// No D2H occupancy read, prefix scan, rectangular padding or KV-bank copy.
__global__ void combine_indices(const int32_t* pi, const int32_t* pp,
                                const int32_t* ei, const int32_t* ep,
                                int32_t* combined, int32_t* ptr,
                                int rows, int prefix_slots, int extend_slots) {
  int row = blockIdx.x;
  int p = pp[row], e = ep[row], start = p + e;
  if(threadIdx.x == 0) ptr[row] = start;
  if(row == rows) return;
  int np = pp[row+1] - p, ne = ep[row+1] - e;
  for(int j = threadIdx.x; j < np + ne; j += blockDim.x) {
    bool extend = j >= np;
    int slot = extend ? ei[e+j-np] : pi[p+j];
    int limit = extend ? extend_slots : prefix_slots;
    combined[start+j] = slot >= 0 && slot < limit
        ? (slot | (extend ? ckcore::kExtendSourceTag : 0)) : -1;
  }
}

struct Entry {
  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView pkv,
                  tvm::ffi::TensorView pi, tvm::ffi::TensorView pp,
                  tvm::ffi::TensorView ekv, tvm::ffi::TensorView ei,
                  tvm::ffi::TensorView ep, tvm::ffi::TensorView sink,
                  tvm::ffi::TensorView out, tvm::ffi::TensorView scratch,
                  tvm::ffi::TensorView combined, tvm::ffi::TensorView ptr,
                  double scale, int64_t splits) {
    if(q.ndim()!=3 || q.size(0)<=0 || q.size(0)>65536 || q.size(1)!=8 || q.size(2)!=512)
      throw std::runtime_error("prefill oracle expects M1..65536 H8 D512");
    int rows = q.size(0);
    if((q.device().device_type!=kDLCUDA && q.device().device_type!=kDLROCM) ||
       splits<1 || splits>2 || !std::isfinite(scale) || scale<=0)
      throw std::runtime_error("prefill oracle device/scale/splits mismatch");
    for(auto t : {q, pkv, pi, pp, ekv, ei, ep, sink, out, scratch, combined, ptr}) {
      if(!t.IsContiguous() || t.device().device_type!=q.device().device_type ||
         t.device().device_id!=q.device().device_id)
        throw std::runtime_error("prefill oracle requires contiguous same-device tensors");
    }
    auto typed = [](tvm::ffi::TensorView t, int code, int bits) {
      if(t.dtype().code!=code || t.dtype().bits!=bits || t.dtype().lanes!=1)
        throw std::runtime_error("prefill oracle dtype mismatch");
    };
    for(auto t : {q, pkv, ekv, out}) typed(t,kDLBfloat,16);
    for(auto t : {pi, pp, ei, ep, combined, ptr}) typed(t,kDLInt,32);
    typed(sink,kDLFloat,32); typed(scratch,kDLUInt,8);
    if(pkv.ndim()!=2 || ekv.ndim()!=2 || pkv.size(1)!=512 || ekv.size(1)!=512 ||
       pkv.size(0)>=ckcore::kExtendSourceTag || ekv.size(0)>=ckcore::kExtendSourceTag ||
       out.ndim()!=3 || out.size(0)!=rows || out.size(1)!=8 || out.size(2)!=512 ||
       pi.ndim()!=1 || ei.ndim()!=1 || pp.ndim()!=1 || ep.ndim()!=1 ||
       pp.size(0)!=rows+1 || ep.size(0)!=rows+1 ||
       sink.ndim()!=1 || sink.size(0)!=8 || ptr.ndim()!=1 || ptr.size(0)!=rows+1 ||
       combined.ndim()!=1 || combined.numel()<pi.numel()+ei.numel() || combined.numel()<1 ||
       combined.numel()>std::numeric_limits<int32_t>::max() ||
       scratch.numel()<static_cast<int64_t>(rows)*splits*8*514*4)
      throw std::runtime_error("prefill oracle shape/workspace mismatch");
    auto stream = sglang::host::LaunchKernel::resolve_device(q.device());
    auto check = [](hipError_t status) {
      if(status!=hipSuccess) throw std::runtime_error(hipGetErrorString(status));
    };
    hipLaunchKernelGGL(combine_indices, dim3(rows+1), dim3(256), 0, stream,
        static_cast<const int32_t*>(pi.data_ptr()), static_cast<const int32_t*>(pp.data_ptr()),
        static_cast<const int32_t*>(ei.data_ptr()), static_cast<const int32_t*>(ep.data_ptr()),
        static_cast<int32_t*>(combined.data_ptr()), static_cast<int32_t*>(ptr.data_ptr()),
        rows, static_cast<int>(pkv.size(0)), static_cast<int>(ekv.size(0)));
    check(hipGetLastError());
    ckcore::UnifiedSparseDecodeArgs args{
        static_cast<const ck::bhalf_t*>(q.data_ptr()), static_cast<const ck::bhalf_t*>(pkv.data_ptr()),
        static_cast<const int32_t*>(combined.data_ptr()), static_cast<const int32_t*>(ptr.data_ptr()),
        static_cast<const float*>(sink.data_ptr()), static_cast<ck::bhalf_t*>(out.data_ptr()),
        rows, 8, static_cast<int>(pkv.size(0)), static_cast<float>(scale),
        static_cast<const ck::bhalf_t*>(ekv.data_ptr()), static_cast<int>(ekv.size(0))};
    auto workspace=ckcore::partition_mfma_split_workspace(scratch.data_ptr(),args,splits);
    // Separate checked prefill entry, not the production decode launcher.
    hipLaunchKernelGGL((ckcore::unified_sparse_decode_d512_mfma_split_core_kernel<false,true,true,8>),
        dim3(rows,splits),dim3(256),0,stream,args,workspace,static_cast<int>(splits));
    check(hipGetLastError());
    hipLaunchKernelGGL(ckcore::unified_sparse_decode_d512_mfma_split_reduce_kernel,
        dim3(rows,8),dim3(256),0,stream,args,workspace,static_cast<int>(splits));
    check(hipGetLastError());
  }
};
}  // namespace sglang::prefill_two_source_oracle
