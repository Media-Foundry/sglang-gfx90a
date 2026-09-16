// Independent experiment: fixed full-expert stride, bounded output N window.
#define DeviceMoeGemm Dsv4StripeMoeGemm
#define GridwiseMoeGemm Dsv4StripeGridwiseMoeGemm
#define ck_moe_stage2_gemm dsv4_stripe_stage2_gemm
#include "gemm_moe_ck2stages_common.cuh"
#include <c10/hip/HIPGuard.h>

void stage2(torch::Tensor inter, torch::Tensor w2, torch::Tensor ids,
            torch::Tensor experts, torch::Tensor valid, torch::Tensor weights,
            torch::Tensor out, int64_t n0) {
    c10::cuda::CUDAGuard guard(inter.device());
    for(auto t:{inter,w2,ids,experts,valid,weights,out})
        TORCH_CHECK(t.is_cuda() && t.device()==inter.device() && t.is_contiguous());
    TORCH_CHECK(inter.scalar_type()==at::kBFloat16 && w2.scalar_type()==at::kBFloat16);
    TORCH_CHECK(ids.scalar_type()==at::kInt && experts.scalar_type()==at::kInt && valid.scalar_type()==at::kInt);
    TORCH_CHECK(weights.scalar_type()==at::kFloat && out.scalar_type()==at::kFloat);
    TORCH_CHECK(inter.dim()==3 && inter.size(1)==1 && inter.size(2)==256);
    TORCH_CHECK(w2.dim()==3 && w2.size(0)==256 && w2.size(1)==4096 && w2.size(2)==256);
    TORCH_CHECK(out.dim()==2 && out.size(0)==inter.size(0));
    int n=out.size(1);
    TORCH_CHECK((n==128 || n==256 || n==512 || n==1024 || n==2048 || n==4096)
                && n0>=0 && n0+n<=4096 && n0%128==0);
    TORCH_CHECK(ids.dim()==1 && ids.numel()==weights.numel() && valid.numel()>=1);
    int tokens=inter.size(0), sorted_size=std::min(int64_t(tokens)*64,ids.numel());
    // BF16 preshuffle consists of complete contiguous N64 blocks. n0 is N128
    // aligned. The private Gridwise header retains FULL 4096*K expert stride;
    // N passed below controls only stripe descriptors, grid and output stride.
    void *a=inter.data_ptr(), *unused=nullptr;
    void *b=static_cast<char*>(w2.data_ptr())+n0*256*2;
    void *si=ids.data_ptr(), *se=experts.data_ptr(), *sw=weights.data_ptr();
    void *nv=valid.data_ptr(), *o=out.data_ptr();
    dsv4_stripe_stage2_gemm<B16,B16,F32,F32,TypeCastExpertWeight,V3,
        256,64,128,64,1,4,false,false,true,0>(at::hip::getCurrentHIPStream(),
        tokens,sorted_size,n,256,1,a,unused,b,si,se,sw,nv,o,
        std::nullopt,std::nullopt,1,true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("stage2",&stage2,"Trusted unique-slot CK N-stripe stage2");
}
