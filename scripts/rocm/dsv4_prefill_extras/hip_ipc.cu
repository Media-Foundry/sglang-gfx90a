#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

static void check(hipError_t e) { TORCH_CHECK(e==hipSuccess,hipGetErrorString(e)); }
torch::Tensor allocate(int64_t bytes) {
    TORCH_CHECK(bytes>0);int device;check(hipGetDevice(&device));void* p=nullptr;
    check(hipMalloc(&p,bytes));
    return torch::from_blob(p,{bytes},[device](void* ptr) noexcept {
        int prior=-1;hipGetDevice(&prior);hipSetDevice(device);hipFree(ptr);
        if(prior>=0)hipSetDevice(prior);
    },torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA,device));
}
pybind11::bytes handle(torch::Tensor buffer) {
    TORCH_CHECK(buffer.is_cuda() && buffer.is_contiguous() && buffer.storage_offset()==0);
    c10::cuda::CUDAGuard guard(buffer.device());hipIpcMemHandle_t h;
    check(hipIpcGetMemHandle(&h,buffer.data_ptr()));
    return pybind11::bytes(reinterpret_cast<const char*>(&h),sizeof(h));
}
torch::Tensor open_handle(pybind11::bytes raw,int64_t bytes) {
    std::string s=raw;TORCH_CHECK(s.size()==sizeof(hipIpcMemHandle_t)&&bytes>0);
    hipIpcMemHandle_t h;std::memcpy(&h,s.data(),sizeof(h));int device;
    check(hipGetDevice(&device));void* p=nullptr;
    check(hipIpcOpenMemHandle(&p,h,hipIpcMemLazyEnablePeerAccess));
    return torch::from_blob(p,{bytes},[device](void* ptr) noexcept {
        int prior=-1;hipGetDevice(&prior);hipSetDevice(device);hipIpcCloseMemHandle(ptr);
        if(prior>=0)hipSetDevice(prior);
    },torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA,device));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("allocate",&allocate);m.def("handle",&handle);m.def("open",&open_handle);
}
