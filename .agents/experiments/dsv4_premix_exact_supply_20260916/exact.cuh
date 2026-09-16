#pragma once
#include "deepseek_v4/gfx90a_mhc_post_wave.cuh"
namespace sglang {
__device__ __forceinline__ float exact_sum64(float v) {
  v=post_dpp<0x118>(v)+v;
  v=post_dpp<0x114>(v)+v;
  v=post_dpp<0x112>(v)+v;
  v=post_dpp<0x111>(v)+v;
  v=post_dpp<0x142,10>(v)+v;
  v=post_dpp<0x143>(v)+v;
  return __shfl(v,63,64);
}
template<int Start>
__device__ __forceinline__ float exact_dot16(const float* x,const float* w,int chunk) {
  float v=x[Start]*w[Start];
  v=__builtin_fmaf(x[1-Start],w[1-Start],v);
  #pragma unroll
  for(int i=2;i<16;++i) {
    // Pinned production object uses packed multiply + separate add here.
    if(((chunk==13 || chunk==14) && i>=14) || (chunk==15 && i>=12))v=v+x[i]*w[i];
    else v=__builtin_fmaf(x[i],w[i],v);
  }
  return exact_sum64(v);
}
template<int Waves,bool Shared,int Start>
__global__ void exact_supply_kernel(const bf16_t* x,const float* fn,const float* rms,float* out,int m) {
  __shared__ bf16_t tile[Shared?8*1024:1];
  const int lane=threadIdx.x%64,wave=threadIdx.x/64;
  const int mb=blockIdx.x*8,n0=(blockIdx.y*Waves+wave)*2;
  float total[8][2]={};
  for(int kb=0;kb<16384;kb+=1024) {
    if constexpr(Shared) {
      for(int j=threadIdx.x;j<8*1024/8;j+=Waves*64) {
        const int r=j/128,k=(j%128)*8;
        uint4 v={0,0,0,0};
        if(mb+r<m)v=*reinterpret_cast<const uint4*>(x+static_cast<size_t>(mb+r)*16384+kb+k);
        *reinterpret_cast<uint4*>(tile+r*1024+k)=v;
      }
      __syncthreads();
    }
    float w0[16],w1[16];
    #pragma unroll
    for(int q=0;q<4;++q) {
      const int k=kb+lane*4+q*256;
      const float4 a=*reinterpret_cast<const float4*>(fn+n0*16384+k);
      const float4 b=*reinterpret_cast<const float4*>(fn+(n0+1)*16384+k);
      w0[q*4]=a.x;w0[q*4+1]=a.y;w0[q*4+2]=a.z;w0[q*4+3]=a.w;
      w1[q*4]=b.x;w1[q*4+1]=b.y;w1[q*4+2]=b.z;w1[q*4+3]=b.w;
    }
    #pragma unroll
    for(int r=0;r<8;++r) {
      float a[16];
      #pragma unroll
      for(int i=0;i<16;++i) {
        const int k=lane*4+(i/4)*256+i%4;
        if constexpr(Shared)a[i]=static_cast<float>(tile[r*1024+k]);
        else a[i]=mb+r<m?static_cast<float>(x[static_cast<size_t>(mb+r)*16384+kb+k]):0.0f;
      }
      total[r][0]=exact_dot16<Start>(a,w0,kb/1024)+total[r][0];
      total[r][1]=exact_dot16<Start>(a,w1,kb/1024)+total[r][1];
    }
    if constexpr(Shared)__syncthreads();
  }
  #pragma unroll
  for(int r=0;r<8;++r) {
    float sq=mb+r<m?rms[(mb+r)*64+lane]:0.0f;
    sq=exact_sum64(sq);
    const float den=__builtin_fmaf(sq,1.0f/16384.0f,1.e-6f);
    float inv;asm("v_rsq_f32 %0, %1" : "=v"(inv):"v"(den));
    if(lane==0 && mb+r<m) {
      out[(mb+r)*24+n0]=total[r][0]*inv;
      out[(mb+r)*24+n0+1]=total[r][1]*inv;
    }
  }
}
struct ExactSupplyScreen {
  template<int Waves,bool Shared,int Start>
  static void run(const tvm::ffi::TensorView x,const tvm::ffi::TensorView fn,const tvm::ffi::TensorView rms,const tvm::ffi::TensorView out) {
    using namespace host;
    const int64_t m=x.size(0);RuntimeCheck(m>0 && m<=65536,"invalid M");
    auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();
    TensorMatcher({m,16384}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({24,16384}).with_dtype<float>().with_device(dev).verify(fn);
    TensorMatcher({m,64}).with_dtype<float>().with_device(dev).verify(rms);
    TensorMatcher({m,24}).with_dtype<float>().with_device(dev).verify(out);
    LaunchKernel(dim3((m+7)/8,12/Waves),Waves*64,x.device())(exact_supply_kernel<Waves,Shared,Start>,
      static_cast<const bf16_t*>(x.data_ptr()),static_cast<const float*>(fn.data_ptr()),static_cast<const float*>(rms.data_ptr()),static_cast<float*>(out.data_ptr()),static_cast<int>(m));
  }
  static void d0(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c,const tvm::ffi::TensorView d){run<1,false,0>(a,b,c,d);}
  static void d1(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c,const tvm::ffi::TensorView d){run<1,false,1>(a,b,c,d);}
  static void s2(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c,const tvm::ffi::TensorView d){run<2,true,1>(a,b,c,d);}
  static void s4(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c,const tvm::ffi::TensorView d){run<4,true,1>(a,b,c,d);}
  static void s6(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c,const tvm::ffi::TensorView d){run<6,true,1>(a,b,c,d);}
  static void s12(const tvm::ffi::TensorView a,const tvm::ffi::TensorView b,const tvm::ffi::TensorView c,const tvm::ffi::TensorView d){run<12,true,1>(a,b,c,d);}
};
}
