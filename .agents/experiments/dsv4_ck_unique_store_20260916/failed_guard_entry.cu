// Experimental CK unique-slot stage2. Never accepts ordinary shared token IDs.
#define DeviceMoeGemm Dsv4UniqueMoeGemm
#define ck_moe_stage2_gemm dsv4_unique_stage2_gemm
#include "gemm_moe_ck2stages_common.cuh"
#include <c10/hip/HIPGuard.h>

void stage2(torch::Tensor inter, torch::Tensor w2, torch::Tensor ids,
            torch::Tensor experts, torch::Tensor valid, torch::Tensor weights,
            torch::Tensor out) {
    c10::hip::HIPGuard guard(inter.device());
    for (auto t : {inter, w2, ids, experts, valid, weights, out}) {
        TORCH_CHECK(t.is_cuda() && t.device() == inter.device() && t.is_contiguous());
    }
    TORCH_CHECK(inter.scalar_type() == at::kBFloat16 && w2.scalar_type() == at::kBFloat16);
    TORCH_CHECK(ids.scalar_type() == at::kInt && experts.scalar_type() == at::kInt && valid.scalar_type() == at::kInt);
    TORCH_CHECK(weights.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat);
    TORCH_CHECK(inter.dim() == 3 && inter.size(1) == 1 && inter.size(2) == 256);
    TORCH_CHECK(w2.dim() == 3 && w2.size(0) == 256 && w2.size(1) == 4096 && w2.size(2) == 256);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == inter.size(0) && out.size(1) == 4096);
    TORCH_CHECK(ids.dim() == 1 && ids.numel() == weights.numel() && valid.numel() >= 1);
    int tokens=inter.size(0), sorted_size=std::min(int64_t(tokens)*64,ids.numel());
    void *a=inter.data_ptr(), *unused=nullptr, *b=w2.data_ptr(), *si=ids.data_ptr();
    void *se=experts.data_ptr(), *sw=weights.data_ptr(), *nv=valid.data_ptr(), *o=out.data_ptr();
    // Matches materialized BF16 / FP32 stage2 heuristic blockM64 / K256.
    dsv4_unique_stage2_gemm<B16,B16,F32,F32,TypeCastExpertWeight,V3,
        256,64,128,64,1,4,false,false,true,0>(at::hip::getCurrentHIPStream(),
        tokens,sorted_size,4096,256,1,a,unused,b,si,se,sw,nv,o,
        std::nullopt,std::nullopt,1,true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stage2", &stage2, "Unique-slot CK stage2 (trusted remapped IDs only)");
}
