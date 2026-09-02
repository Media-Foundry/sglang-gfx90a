# DSV4 gfx90a C32 prefill request-12 checkpoint (2026-09-02)

## Configuration

- Four physical gfx90a GCDs 4--7, original model weights.
- TP4/EP1/no-A2A, native AR.
- 83,968-token pool, prefill budget/chunk ceiling 36,864.
- Correct large-M BF16 CK path with `block_m=32` and FP32 stage-2 workspace.
- Token-row MHC plus large-M BF16 GEMM pre-mix.
- 32 fixed heterogeneous code prompts, 73,724 total input tokens.

The only tested scheduling variable was `prefill_max_requests`:

```text
16 (16+16):       prior warm median 4865.5 input tok/s
 8 (8+8+8+8):    5008 / 5009, median 5008.3
12 (12+12+8):    5514 / 5359, median 5359.4
```

The request-12 profile is about 10.1% above the prior request-16 checkpoint
and about 7.0% above request-8. Its cold/JIT round was 4665 tok/s.

## Correctness

The official token-ID France oracle was run twice after the scheduling change.
Both rounds matched the historical first nine tokens exactly and contained
the semantic Paris answer. Completion hashes were identical across rounds.

The 32-request, 64-token greedy code suite still showed repetitive outputs in
several prompts. The same symptom also appears on the previously accepted
large-M FP32-CK path and is not introduced by the request-count-only scheduling
change; do not use that suite alone as a semantic quality oracle. The profile
does not change any kernel or arithmetic path.

## Productization

`SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE=1` selects the measured
default-off profile. The FP32 stage-2 workaround now has a production-facing
`SGLANG_DSV4_GFX90A_BF16_CK_STAGE2_FP32=1` switch instead of requiring the
old `AITER_DSV4_DEBUG_ACTIVATION=dsv4-m32-stage2-fp32` alias.

