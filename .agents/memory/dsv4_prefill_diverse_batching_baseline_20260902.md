# DSV4 diverse prefill batching baseline (2026-09-02)

A deterministic workload now covers 32 different real SGLang/ROCm code-review
requests. Each prompt is encoded with the checkpoint's official DSV4 chat
formatter and contains 2303--2304 input tokens. No prompt or filler body is
shared. The harness reports aggregate prefill as the sum of server-audited
prompt tokens divided by the interval from the first HTTP send to the last
request's first streamed token.

All GPU runs used original checkpoint precision and physical GCDs 4--7 after
`amd-smi process` reported no active process.

## Global chunk-budget bottleneck

With the accepted `chunked_prefill_size=2304`, medians were approximately:

```text
C1    2355 input tok/s
C4    2321 input tok/s
C8    2366 input tok/s
C16   2375 input tok/s
C32   2375 input tok/s
```

Concurrency does not scale because `PrefillAdder.rem_chunk_tokens` is a global
per-forward budget. The first 2304-token request consumes it and admission stops,
so requests execute as separate prefill forwards.

## Large-batch routed-MoE A/B

Raising the global budget to 16384 admits about seven requests per forward.
The standard single-request 4604-token probe then runs in one forward and its
warm TTFT is 1.838--1.851 s (about 2.50k input tok/s), but the raw-weight custom
MFMA64 routed path performs poorly at M around 16K:

```text
custom raw-FP4 MFMA64 C32 steady: about 1413 input tok/s
```

The same 16384-token budget with `SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE=0`
(AIter/CK preshuffled grouped MoE) reached:

```text
C32: 2969 / 3106 input tok/s
```

This confirms that high-occupancy grouped work decomposition matters, but 3.1K
is still far from the 10K target. The two paths cannot be selected per forward
without retaining both raw and preshuffled routed weights (or adding a new
layout-compatible kernel), so this is an oracle rather than a production
checkpoint. It also does not justify changing the accepted C1/decode layout.

Artifacts:

- `.agents/memory/dsv4_prefill_diverse_32_input_ids.json`
- `scripts/rocm/build_dsv4_prefill_diverse_manifest.py`
- `scripts/rocm/bench_dsv4_prefill_diverse_concurrent.py`
- `/tmp/dsv4_prefill_c{1,4,8,16,32}_baseline.json`
- `/tmp/dsv4_prefill_c32_aiter_chunk16384.json`
