// Isolated oracle entry, not a production backend. Geometry matches the
// current explicit 256x64x64x128 / 1x4 / v1 / DSV4 bounded-SiLU instance.
#if DSV4_ROUTE_MAJOR_STAGE1
#define DeviceMoeGemm Dsv4RouteProducerDevice
#define GridwiseMoeGemm Dsv4RouteProducerGrid
#else
#define DeviceMoeGemm Dsv4TokenProducerDevice
#define GridwiseMoeGemm Dsv4TokenProducerGrid
#endif
#define ck_moe_stage1_gemm dsv4_producer_stage1_gemm
#include "gemm_moe_ck2stages_common.cuh"
#include <c10/hip/HIPGuard.h>

void stage1(torch::Tensor hidden, torch::Tensor w13, torch::Tensor ids,
            torch::Tensor experts, torch::Tensor valid, torch::Tensor out) {
    c10::cuda::CUDAGuard guard(hidden.device());
    for (auto t : {hidden,w13,ids,experts,valid,out}) {
        TORCH_CHECK(t.is_cuda() && t.device()==hidden.device() && t.is_contiguous());
    }
    TORCH_CHECK(hidden.scalar_type()==at::kBFloat16 && w13.scalar_type()==at::kBFloat16);
    TORCH_CHECK(out.scalar_type()==at::kBFloat16);
    TORCH_CHECK(ids.scalar_type()==at::kInt && experts.scalar_type()==at::kInt && valid.scalar_type()==at::kInt);
    TORCH_CHECK(hidden.dim()==2 && hidden.size(1)==4096);
    TORCH_CHECK(hidden.size(0)>=8192 && hidden.size(0)<=36864);
    TORCH_CHECK(w13.dim()==3 && w13.size(0)==256 && w13.size(1)==512 && w13.size(2)==4096);
    TORCH_CHECK(ids.dim()==1 && ids.numel()>=hidden.size(0)*6 && ids.numel()<(1<<24));
    TORCH_CHECK(experts.dim()==1 && experts.numel()>=(ids.numel()+63)/64);
    TORCH_CHECK(valid.numel()>=1 && out.dim()==2 && out.size(1)==256);
    TORCH_CHECK(out.size(0)==(DSV4_ROUTE_MAJOR_STAGE1 ? ids.numel() : hidden.size(0)*6));
    TORCH_CHECK(out.numel()*out.element_size()<(int64_t(1)<<31));
    int tokens=hidden.size(0), sorted_size=ids.numel();
    void *a=hidden.data_ptr(), *b=w13.data_ptr(), *unused=nullptr;
    void *si=ids.data_ptr(), *se=experts.data_ptr(), *nv=valid.data_ptr(), *o=out.data_ptr();
    // ActivationType::Dsv4Silu is3 at the API, but CK's internal ActOP is2.
    // No routed weights in stage1; preserve BF16 output rounding and splitK1.
    dsv4_producer_stage1_gemm<B16,B16,F32,B16,TypeCast,V1,
        256,64,64,128,1,4,false,false,false,2,false>(at::hip::getCurrentHIPStream(),
        tokens,sorted_size,256,4096,6,a,b,unused,si,se,unused,nv,o,
        std::nullopt,std::nullopt,1,false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("stage1",&stage1,"Isolated stage1 output-layout oracle (trusted real sorter IDs)");
}
