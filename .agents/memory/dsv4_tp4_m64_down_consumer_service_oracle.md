# TP4 M64 down-consumer service oracle

Date: 2026-08-30

The existing exact M32 HIP down-consumer kernel was instantiated for M64. It
quantizes each A4 gate intermediate directly into LDS and consumes it in the
FP4 down projection, while preserving the BF16 intermediate, group-32
quantization, FP32 partial tensor, router weights, and fixed-slot reduction.

Profile: TP4/EP1/no-A2A, original weights, native AR, graph tiers 1 and 64,
64 real heterogeneous prompts, 128 generated tokens.

```text
A current logical-W2 path: 993.25 resident tok/s
B M64 down consumer:       993.63 resident tok/s
delta:                       +0.038%
```

Correctness was exact against the accepted A teacher for all 64 requests:

- output token IDs: 64/64 exact;
- token logprobs: 64/64 exact;
- top-5 logprobs: 64/64 exact;
- France sentinel passed;
- all requests reached `finish=length`.

Decision: service-neutral, keep the experimental selector default-off and do
not enable it in the production M64 profile. Like fused quant+sort, this shows
that removing a standalone intermediate quantization launch does not shorten
the current rank-max critical path.

Artifacts:

- `/tmp/dsv4_tp4_bs64_down_consumer_teacher.json`
- `/tmp/dsv4_tp4_bs64_down_consumer_b.json`

