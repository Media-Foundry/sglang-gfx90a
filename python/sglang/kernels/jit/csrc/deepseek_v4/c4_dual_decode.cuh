#include "c4_v2.cuh"
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

namespace sglang {

using deepseek_v4::fp8::pack_fp8;

struct C4DualDecodeParams {
  float* core_state; const float* core_input; const float* core_ape;
  float* index_state; const float* index_input; const float* index_ape;
  const PlanD* plan; const float* core_norm; const float* index_norm;
  const float* core_freqs; const float* index_freqs;
  const int64_t* core_out_loc; const int64_t* index_out_loc;
  uint8_t* core_cache; uint8_t* index_cache;
  float* core_tmp; float* index_tmp; float core_eps; float index_eps;
  uint32_t batch;
};

__device__ __forceinline__ void c4_dual_index_finish(
    const C4DualDecodeParams& p, uint32_t token, uint32_t lane) {
  const auto plan = p.plan[token];
  if (plan.seq_len % 4) return;

  // Private copy of the decode epilogue used by
  // fused_norm_rope_indexer<float, CompressDecode, 6, false>.  Keeping this
  // here makes the standalone experiment independent of the production
  // fused_norm_rope_v2.cuh ABI.
  using namespace device;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kVecSize = 4;
  constexpr int64_t kRopeDim = 64;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int32_t kPageBits = 6;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  using Storage = AlignedVector<float, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;

  const auto input = p.index_tmp + token * kHeadDim;
  const auto freqs_cis = p.index_freqs + (plan.seq_len - 4) * kRopeDim;
  const auto lane_id = lane;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
  Float4 data, freq;

  {
    Storage input_vec, weight_vec;
    input_vec.load(input, lane_id);
    weight_vec.load(p.index_norm, lane_id);
    if (is_rope_lane)
      freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }
    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor =
        math::rsqrt(sum_of_squares / kHeadDim + p.index_eps);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }

  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }

  {
    const float a0 = data[0], a1 = data[1];
    const float a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    const float a0 = data[0], a1 = data[1];
    const float a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
#ifndef USE_ROCM
      const float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
#else
      const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  const float hadamard_scale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) data[i] *= hadamard_scale;

  float local_max = math::abs(data[0]);
#pragma unroll
  for (int i = 1; i < kVecSize; ++i) {
    local_max = math::max(local_max, math::abs(data[i]));
  }
  const auto abs_max = warp::reduce_max(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
  const auto inv_scale = 1.0f / scale;
  const int64_t out_loc = p.index_out_loc[token];
  const int64_t page = out_loc >> kPageBits;
  const int64_t offset = out_loc & ((1 << kPageBits) - 1);
  auto* page_ptr = p.index_cache + page * kPageBytes;
  OutStorage result;
  result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
  result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
  // Production gfx90a runs AIter's page-64 MQA path.  Its K cache is tiled as
  // 16 tokens x 16 columns within each 64-token page, rather than row-major.
  // Keep this private prototype layout-identical to
  // fused_norm_rope_indexer<..., kPreshuffleSize=16>.
  constexpr int32_t kPreshuffleTile = 16;
  const int32_t dim_base = lane_id * kVecSize;
  const int32_t token_tile_id = offset / kPreshuffleTile;
  const int32_t token_in_tile = offset % kPreshuffleTile;
  const int32_t col_tile_id = dim_base / kPreshuffleTile;
  const int32_t col_in_tile = dim_base % kPreshuffleTile;
  const int32_t value_offset =
      token_tile_id * (kPreshuffleTile * static_cast<int32_t>(kHeadDim)) +
      col_tile_id * (kPreshuffleTile * kPreshuffleTile) +
      token_in_tile * kPreshuffleTile + col_in_tile;
  result.store(page_ptr + value_offset, 0);
  if (lane_id == 0) {
    reinterpret_cast<float*>(
        page_ptr + (kHeadDim << kPageBits) + offset * sizeof(float))[0] =
        scale;
  }
}

__device__ __forceinline__ void c4_dual_core_finish(
    const C4DualDecodeParams& p, uint32_t token) {
  constexpr uint32_t D=512;
  __shared__ float sums[8];
  const uint32_t tx=threadIdx.x, warp=tx/32, lane=tx%32;
  const auto plan=p.plan[token];
  if (plan.seq_len % 4) return;
  device::AlignedVector<float,2> data, weight, freq;
  data.load(p.core_tmp+token*D,tx); weight.load(p.core_norm,tx);
  float ss=data[0]*data[0]+data[1]*data[1]; ss=device::warp::reduce_sum(ss);
  if(lane==0) sums[warp]=ss;
  __syncthreads();
  ss=device::warp::reduce_sum<8>(sums[lane%8]);
  const float inv=device::math::rsqrt(ss/D+p.core_eps);
  data[0]*=inv*weight[0]; data[1]*=inv*weight[1];
  if(warp==7){
    freq.load(p.core_freqs+(plan.seq_len-4)*64,lane);
    const float a=data[0],b=data[1]; data[0]=a*freq[0]-b*freq[1]; data[1]=a*freq[1]+b*freq[0];
  }
  const int64_t loc=p.core_out_loc[token];
  reinterpret_cast<bf16x2_t*>(p.core_cache+loc*1024)[tx]=
      device::cast<bf16x2_t>(fp32x2_t{data[0],data[1]});
}

__global__ __launch_bounds__(256) void c4_dual_decode_kernel(C4DualDecodeParams p) {
  const uint32_t core_blocks=p.batch;
  if(blockIdx.x<core_blocks){
    const uint32_t token=blockIdx.x;
    if(threadIdx.x<128){
      using Trait=C4Trait<512>;
      const uint32_t sid=threadIdx.x/32, off=sid*128;
      const auto plan=p.plan[token];
      auto src=p.core_input+token*Trait::kElementSize+off;
      auto state=p.core_state+off;
      auto dst=state+plan.write_loc*Trait::kElementSize;
      c4_write_decode<Trait,float,float>(dst,src);
      if(plan.seq_len%4==0) c4_forward<Trait,false,float,float,float>(
        state+plan.read_page_0*Trait::kPageElementSize,
        state+plan.read_page_1*Trait::kPageElementSize,src,
        p.core_tmp+token*512+off,p.core_ape+off,plan.seq_len>4,8);
    }
    __syncthreads(); c4_dual_core_finish(p,token); return;
  }
  const uint32_t ib=blockIdx.x-core_blocks, token=ib*8+threadIdx.x/32, lane=threadIdx.x%32;
  if(token<p.batch){
    using Trait=C4Trait<128>; const auto plan=p.plan[token];
    auto src=p.index_input+token*Trait::kElementSize;
    auto state=p.index_state;
    c4_write_decode<Trait,float,float>(state+plan.write_loc*Trait::kElementSize,src);
    if(plan.seq_len%4==0) c4_forward<Trait,false,float,float,float>(
      state+plan.read_page_0*Trait::kPageElementSize,
      state+plan.read_page_1*Trait::kPageElementSize,src,
      p.index_tmp+token*128,p.index_ape,plan.seq_len>4,8);
  }
  __syncthreads();
  if(token<p.batch) c4_dual_index_finish(p,token,lane);
}

struct C4DualDecodeKernel {
  static void run(tvm::ffi::TensorView core_state,tvm::ffi::TensorView core_input,tvm::ffi::TensorView core_ape,
    tvm::ffi::TensorView index_state,tvm::ffi::TensorView index_input,tvm::ffi::TensorView index_ape,
    tvm::ffi::TensorView plan,tvm::ffi::TensorView core_norm,tvm::ffi::TensorView index_norm,
    tvm::ffi::TensorView core_freqs,tvm::ffi::TensorView index_freqs,
    tvm::ffi::TensorView core_out_loc,
    tvm::ffi::TensorView index_out_loc,tvm::ffi::TensorView core_cache,
    tvm::ffi::TensorView index_cache,tvm::ffi::TensorView core_tmp,
    tvm::ffi::TensorView index_tmp,double core_eps,double index_eps){
    using namespace host; auto N=SymbolicSize{"N"}; auto dev=SymbolicDevice{}; dev.set_options<kDLCUDA>();
    TensorMatcher({N,2048}).with_dtype<float>().with_device(dev).verify(core_input);
    TensorMatcher({N,512}).with_dtype<float>().with_device(dev).verify(index_input);
    TensorMatcher({N,16}).with_dtype<uint8_t>().with_device(dev).verify(plan);
    TensorMatcher({N,512}).with_dtype<float>().with_device(dev).verify(core_tmp);
    TensorMatcher({N,128}).with_dtype<float>().with_device(dev).verify(index_tmp);
    const uint32_t n=N.unwrap(); C4DualDecodeParams p{
      (float*)core_state.data_ptr(),(float*)core_input.data_ptr(),(float*)core_ape.data_ptr(),
      (float*)index_state.data_ptr(),(float*)index_input.data_ptr(),(float*)index_ape.data_ptr(),
      (PlanD*)plan.data_ptr(),(float*)core_norm.data_ptr(),(float*)index_norm.data_ptr(),
      (float*)core_freqs.data_ptr(),(float*)index_freqs.data_ptr(),
      (int64_t*)core_out_loc.data_ptr(),
      (int64_t*)index_out_loc.data_ptr(),(uint8_t*)core_cache.data_ptr(),
      (uint8_t*)index_cache.data_ptr(),(float*)core_tmp.data_ptr(),
      (float*)index_tmp.data_ptr(),(float)core_eps,(float)index_eps,n};
    LaunchKernel(n+div_ceil(n,8u),256,dev.unwrap())(c4_dual_decode_kernel,p);
  }
};
}
