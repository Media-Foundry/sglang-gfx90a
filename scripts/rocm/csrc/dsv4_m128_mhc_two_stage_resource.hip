#include "deepseek_v4/gfx90a_m128_mhc_two_stage_oracle.cuh"

// Compile-only TU: forces both standalone oracle kernels into a gfx90a code
// object so VGPR/SGPR/LDS/scratch can be audited without launching a GPU job.
extern "C" void instantiate_dsv4_m128_mhc_producer(
    const sglang::bf16_t* x, const sglang::bf16_t* residual,
    const float* previous_post, const float* previous_comb,
    const half* fn, sglang::bf16_t* residual_out,
    float* rms_partial, float* dot_partial) {
  hipLaunchKernelGGL(
      (sglang::gfx90a_m128_mhc_producer_kernel),
      dim3(128), dim3(128), 0, nullptr, x, residual, previous_post,
      previous_comb, fn, residual_out, rms_partial, dot_partial);
}

extern "C" void instantiate_dsv4_m128_mhc_consumer(
    const sglang::bf16_t* residual, const float* rms_partial,
    const float* dot_partial, const float* hc_scale, const float* hc_base,
    const sglang::bf16_t* norm_weight, float* mixes_out, float* post_out,
    float* comb_out, sglang::bf16_t* layer_input_out) {
  hipLaunchKernelGGL(
      (sglang::gfx90a_m128_mhc_consumer_kernel),
      dim3(128), dim3(512), 0, nullptr, residual, rms_partial, dot_partial,
      hc_scale, hc_base, norm_weight, mixes_out, post_out, comb_out,
      layer_input_out, 1e-6f, 1e-6f, 2.0f, 1e-6f);
}
