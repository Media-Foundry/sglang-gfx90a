# DSV4 TP4 prefill chunk-2304 checkpoint (2026-09-02)

The native TP4/EP1, no-A2A prefill profile now defaults to a 2304-token chunk.
For the standard single-request 4604-token context probe this changes the
schedule from `2048 + 2048 + 508` into two near-full MFMA64 chunks.

All GPU runs used physical GCDs 4--7 after `amd-smi process` reported no active
processes. The model weights and numerical kernels were unchanged.

Steady HTTP TTFT medians (first request excluded):

```text
B1 chunk2304: 1.861 s  (six samples)
A  chunk2048: 1.978 s  (six samples)
B2 chunk2304: 1.852 s  (six samples)
```

The returned B arm is 6.4% faster than A. A 4096-token chunk was previously
about 1.99 s and therefore did not reproduce this benefit. The win comes from
removing the small tail traversal without moving all work to the less efficient
M4096 routed-MoE shape.

Correctness gates on B2:

- OpenAI chat: `What is the capital of France?` returned exactly
  `The capital of France is Paris.`
- Three native-AR 256-token probes all completed with `finish=length`, identical
  completion SHA256 prefix `38c3d431e7c1dd65`, and coherent text.
- Decode selectors are unchanged; this checkpoint only changes the prefill
  scheduler chunk size.

The first request still pays roughly 20+ seconds at this new M2304 shape because
its JIT/modules are cold. Cold prewarm is a separate latency task and must not be
reported as steady kernel throughput.
