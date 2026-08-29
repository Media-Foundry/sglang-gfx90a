#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

namespace sglang {

template <uint32_t kExperts, uint32_t kCtasPerExpert, uint32_t kIters>
__global__ void readiness_producer_kernel(
    uint32_t* counters, uint32_t* ready, uint32_t* scratch) {
  const uint32_t expert = blockIdx.x / kCtasPerExpert;
  uint32_t value = blockIdx.x * 1664525u + threadIdx.x + 1013904223u;
#pragma unroll 1
  for (uint32_t i = 0; i < kIters; ++i) {
    value = value * 1664525u + 1013904223u;
    value ^= value >> 13;
  }
  if (threadIdx.x == 0) {
    scratch[blockIdx.x] = value;
    // Every RMW acquires the preceding release sequence. The final RMW thus
    // transitively publishes every producer CTA's scratch write.
    const uint32_t old = __scoped_atomic_fetch_add(
        counters + expert, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_SYSTEM);
    if (old % kCtasPerExpert == kCtasPerExpert - 1) {
      __scoped_atomic_store_n(
          ready + expert, old / kCtasPerExpert + 1,
          __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
  }
}

template <uint32_t kExperts, uint32_t kConsumerCtas, uint32_t kIters>
__global__ void readiness_consumer_kernel(
    const uint32_t* ready, uint32_t* consumed, uint32_t* queue,
    uint32_t* output) {
  __shared__ uint32_t expert_shared;
  __shared__ uint32_t epoch_shared;
  while (true) {
    if (threadIdx.x == 0) {
      expert_shared = atomicAdd(queue, 1u);
    }
    __syncthreads();
    const uint32_t expert = expert_shared;
    if (expert >= kExperts) return;
    if (threadIdx.x == 0) {
      const uint32_t previous = consumed[expert];
      uint32_t epoch;
      do {
        epoch = __scoped_atomic_load_n(
            ready + expert, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      } while (epoch <= previous);
      epoch_shared = epoch;
    }
    __syncthreads();
    uint32_t value = expert * 747796405u + threadIdx.x + epoch_shared;
#pragma unroll 1
    for (uint32_t i = 0; i < kIters; ++i) {
      value = value * 2891336453u + 1u;
      value ^= value >> 16;
    }
    if (threadIdx.x == 0) {
      output[expert] = value;
      __scoped_atomic_store_n(
          consumed + expert, epoch_shared,
          __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
  }
}

template <uint32_t kExperts, uint32_t kCtasPerExpert, uint32_t kIters>
struct ReadinessProducerOracle {
  static void run(const tvm::ffi::TensorView counters,
                  const tvm::ffi::TensorView ready,
                  const tvm::ffi::TensorView scratch) {
    using namespace host;
    LaunchKernel(kExperts * kCtasPerExpert, 64, counters.device())(
        readiness_producer_kernel<kExperts, kCtasPerExpert, kIters>,
        static_cast<uint32_t*>(counters.data_ptr()),
        static_cast<uint32_t*>(ready.data_ptr()),
        static_cast<uint32_t*>(scratch.data_ptr()));
  }
};

template <uint32_t kExperts, uint32_t kConsumerCtas, uint32_t kIters>
struct ReadinessConsumerOracle {
  static void run(const tvm::ffi::TensorView ready,
                  const tvm::ffi::TensorView consumed,
                  const tvm::ffi::TensorView queue,
                  const tvm::ffi::TensorView output) {
    using namespace host;
    LaunchKernel(kConsumerCtas, 64, ready.device())(
        readiness_consumer_kernel<kExperts, kConsumerCtas, kIters>,
        static_cast<const uint32_t*>(ready.data_ptr()),
        static_cast<uint32_t*>(consumed.data_ptr()),
        static_cast<uint32_t*>(queue.data_ptr()),
        static_cast<uint32_t*>(output.data_ptr()));
  }
};

template <uint32_t kCtas, uint32_t kIters>
__global__ void consumer_pressure_kernel(uint32_t* output) {
  uint32_t value = blockIdx.x * 747796405u + threadIdx.x + 1u;
#pragma unroll 1
  for (uint32_t i = 0; i < kIters; ++i) {
    value = value * 2891336453u + 1u;
    value ^= value >> 16;
  }
  if (threadIdx.x == 0) output[blockIdx.x] = value;
}

template <uint32_t kCtas, uint32_t kIters>
struct ConsumerPressureOracle {
  static void run(const tvm::ffi::TensorView output) {
    using namespace host;
    LaunchKernel(kCtas, 64, output.device())(
        consumer_pressure_kernel<kCtas, kIters>,
        static_cast<uint32_t*>(output.data_ptr()));
  }
};

}  // namespace sglang
