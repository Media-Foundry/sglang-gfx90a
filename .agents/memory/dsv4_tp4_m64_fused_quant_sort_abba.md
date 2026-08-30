# TP4 M64 fused quant + stable A4 sort ABBA

Date: 2026-08-30

## Candidate

The candidate combines BF16-to-group32-INT8 quantization with the A4 expert
histogram/scan/sort in one HIP launch for exact M64 (`64 x 4096`, top-6).
Unlike the old M32 prototype, same-expert assignments are written in stable
token-major/slot-major order rather than by a race-ordered atomic cursor.

Standalone on one gfx90a GCD:

```text
separate quant + AIter sort: 127.20 us median
fused stable kernel:          51.92 us median
subchain reduction:           59.2% / 75.3 us
```

M32 and M64 each passed 100 repeated mutations/replays. The valid sorted IDs
matched a CPU stable token/slot oracle exactly and replay was deterministic.

## Service ABBA

Profile: TP4/EP1/no-A2A, original weights, native AR, graph tiers 1 and 64,
64 real heterogeneous prompts, 128 generated tokens.

```text
A1 current preprocess: 992.83 resident tok/s
B  fused preprocess:   993.47 resident tok/s
A2 current preprocess: 993.25 resident tok/s
```

The candidate is only about +0.09% relative to the A midpoint and is therefore
service-neutral. The original preprocess is evidently overlapped or outside
the rank-max critical path; the candidate's standalone win does not shorten
the model step.

Correctness:

- 64/64 next-token IDs exact against A;
- 64/64 token logprobs exact;
- 64/64 top-5 logprobs exact;
- France sentinel passed;
- all requests completed with `finish=length`.

Decision: keep the experimental selector default-off, do not enable it in the
production profile, and do not revisit preprocess-only fusion without a trace
showing it on the critical path.

Artifacts:

- `/tmp/dsv4_tp4_bs64_fused_qsort_teacher.json`
- `/tmp/dsv4_tp4_bs64_fused_qsort_clean.json`
- `/tmp/dsv4_tp4_bs64_fused_qsort_a2.json`

