#include "deepseek_v4/gfx90a_fp4_cta_weight_multicast_oracle.cuh"

// Compile-only translation unit used to force gfx90a code-object generation
// for the oracle kernels.  It is never linked into SGLang production.
extern "C" void instantiate_dsv4_m64_cta_multicast_gate(
    sglang::bf16_t* out, const int8_t* xq, const float* x_scale,
    const uint8_t* weight, const uint8_t* weight_scale,
    const int32_t* sorted_ids, const int32_t* descriptor_experts,
    const int32_t* descriptor_starts, const int32_t* descriptor_counts,
    const int32_t* num_descriptors, float limit) {
  hipLaunchKernelGGL(
      (sglang::gfx90a_fp4_cta_multicast_gate_kernel<
          256, 64, 6, 512, 4096, 2080>),
      dim3(2080), dim3(256), 0, nullptr, out, xq, x_scale, weight,
      weight_scale, sorted_ids, descriptor_experts, descriptor_starts,
      descriptor_counts, num_descriptors, limit);
}

extern "C" void instantiate_dsv4_m64_cta_multicast_down(
    float* partial, const int8_t* xq, const float* x_scale,
    const uint8_t* weight, const uint8_t* weight_scale,
    const int32_t* sorted_ids, const int32_t* descriptor_experts,
    const int32_t* descriptor_starts, const int32_t* descriptor_counts,
    const int32_t* num_descriptors, const float* topk_weights) {
  hipLaunchKernelGGL(
      (sglang::gfx90a_fp4_cta_multicast_down_kernel<
          256, 64, 6, 4096, 512, 832>),
      dim3(832), dim3(256), 0, nullptr, partial, xq, x_scale, weight,
      weight_scale, sorted_ids, descriptor_experts, descriptor_starts,
      descriptor_counts, num_descriptors, topk_weights);
}
