// A narrow C ABI over the installed hipBLASLt C/C++ APIs. No global backend flag.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation)+": hipBLASLt status "+std::to_string(status));
}
struct Plan {
    hipblasLtHandle_t handle=nullptr;
    hipblasLtMatmulDesc_t desc=nullptr;
    hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    hipblasLtMatmulAlgo_t algo{};
    size_t bytes=0;
    int device=-1;
    float alpha=1.0f,beta=0.0f;
    ~Plan() {
        if(c) hipblasLtMatrixLayoutDestroy(c);
        if(b) hipblasLtMatrixLayoutDestroy(b);
        if(a) hipblasLtMatrixLayoutDestroy(a);
        if(desc) hipblasLtMatmulDescDestroy(desc);
        if(handle) hipblasLtDestroy(handle);
    }
};
}
extern "C" const char* dsv4_lt_error() { return error.c_str(); }
extern "C" int dsv4_lt_create(int m,int n,int k,int index,void** result,size_t* bytes) {
    try {
        if(m<=0 || n<=0 || k<=0 || index<0 || !result || !bytes)
            throw std::runtime_error("invalid plan dimensions or outputs");
        auto p=std::make_unique<Plan>();
        if(hipGetDevice(&p->device)!=hipSuccess) throw std::runtime_error("hipGetDevice failed");
        check(hipblasLtCreate(&p->handle),"create");
        check(hipblasLtMatmulDescCreate(&p->desc,HIPBLAS_COMPUTE_32F,HIP_R_32F),"descriptor");
        hipblasOperation_t trans=HIPBLAS_OP_T;
        check(hipblasLtMatmulDescSetAttribute(p->desc,HIPBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)),"transpose");
        // Row-major Y=X W^T is column-major Y^T=W X^T.
        check(hipblasLtMatrixLayoutCreate(&p->a,HIP_R_16BF,k,n,k),"layout A");
        check(hipblasLtMatrixLayoutCreate(&p->b,HIP_R_16BF,k,m,k),"layout B");
        check(hipblasLtMatrixLayoutCreate(&p->c,HIP_R_16BF,n,m,n),"layout C/D");
        std::vector<int> indices{index};
        std::vector<hipblasLtMatmulHeuristicResult_t> found;
        check(hipblaslt_ext::getAlgosFromIndex(p->handle,indices,found),"algorithm index");
        if(found.size()!=1) throw std::runtime_error("algorithm index did not resolve uniquely");
        p->algo=found[0].algo;
        check(hipblaslt_ext::matmulIsAlgoSupported(p->handle,p->desc,&p->alpha,p->a,p->b,
              &p->beta,p->c,p->c,p->algo,p->bytes),"algorithm support");
        *bytes=p->bytes;*result=p.release();return 0;
    } catch(const std::exception& e) { error=e.what();return -1; }
}
extern "C" int dsv4_lt_run(void* opaque,const void* x,const void* w,void* y,
                            void* workspace,size_t bytes,void* stream) {
    try {
        auto* p=static_cast<Plan*>(opaque);
        if(!p || !x || !w || !y || bytes<p->bytes || (p->bytes && !workspace))
            throw std::runtime_error("invalid pointers or workspace");
        int device=-1;
        if(hipGetDevice(&device)!=hipSuccess || device!=p->device)
            throw std::runtime_error("plan device mismatch");
        check(hipblasLtMatmul(p->handle,p->desc,&p->alpha,w,p->a,x,p->b,&p->beta,y,p->c,y,p->c,
              &p->algo,workspace,bytes,static_cast<hipStream_t>(stream)),"matmul");
        return 0;
    } catch(const std::exception& e) { error=e.what();return -1; }
}
extern "C" void dsv4_lt_destroy(void* opaque) { delete static_cast<Plan*>(opaque); }
