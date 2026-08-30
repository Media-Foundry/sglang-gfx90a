# TP4 M64 grouped-gate row-prefetch service A--B--A

Date: 2026-08-30

## Scope

This experiment tested the production-wired
`SGLANG_DSV4_GFX90A_M64_GATE_ROW_PREFETCH` selector without changing any
production source during the run.  The profile was:

```text
GPU 0--3
TP4 / EP1 / no A2A
native autoregressive decoding, no speculative settings
original checkpoint weights
32K raw-token KV pool
decode graph tiers 1 and 64 only
64 real heterogeneous token-ID prompts
128 generated tokens per request, four rounds per service
```

The input manifest was
`.agents/memory/dsv4_tp8_diverse_64_input_ids.json`.  `amd-smi process` was
captured before every service/test run.  Small BIO contexts which appeared
during the experiment were left untouched; only experiment-owned service and
client PIDs were stopped.

The accepted M64 stack was verified from `/proc/<service-pid>/environ` before
traffic was sent:

```text
A4, gate blocks 2080, down blocks 832, gate/down rows 2
LDS E2M1 unpack
M64 DPP gate
logical W2 scale cache and four-wave W4 down
C128 attention multistream and CK sparse decode
ROCm multistream / single-batch-overlap profile
```

An initial unprofiled run was discarded before completing the A--B--A.  It
mistakenly used the shell variable `GFX90A_TP4_BS32_PROFILE=1` instead of the
script's actual `SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE=1`; `/proc` showed A8,
208-block kernels, LDS disabled and SBO disabled, and its approximately
691 tok/s result is not part of the measurements below.

## Candidate reachability

The B service did not rely on the environment variable as proof of selection.
All four TP workers' live `/proc/<pid>/maps` contained:

```text
sgl_kernel_jit_gfx90a_fp4_expert_gate_row_prefetch_
256_64_6_512_4096_4_2_8_2080_2.so
```

The complete map evidence is in
`/tmp/dsv4_gate_rowprefetch_b_proc_maps.txt`.  Thus graph capture and replay
loaded the M64 row-prefetch specialization rather than the ordinary M64 DPP
module.

## Correctness

Every independently started service ran the 64-row teacher next-token oracle.
B and A2 were byte-for-byte identical to A1, including:

```text
output token IDs:       64 / 64 exact
output token logprobs:  64 / 64 exact
complete top-5 rows:    64 / 64 exact
```

Every throughput round passed the France first-nine-token sentinel, returned
128 completion tokens for all 64 requests, and ended with `finish=length`.
Greedy continuation hashes are not used as the cross-service numerical oracle:
as previously observed for the asynchronous stack, only 4/32 tracked requests
were identical across all four independently salted rounds within each arm.
The fixed-token teacher comparison above is the strict arithmetic check.

## Resident decode A--B--A

The primary metric is the common interval during which all 64 heterogeneous
requests were resident at M64.  Long and variable prefill time is excluded.

```text
A1 row-prefetch=0:
  996.623, 996.484, 996.954, 997.348 tok/s
  median = trimmed mean = 996.788 tok/s

B row-prefetch=1:
  1002.548, 1002.519, 1002.707, 1003.294 tok/s
  median = trimmed mean = 1002.627 tok/s

A2 row-prefetch=0:
  999.949, 1000.071, 999.975, 999.792 tok/s
  median = trimmed mean = 999.962 tok/s
```

Against the median of all eight A samples, B improved resident decode by:

```text
998.570 -> 1002.627 tok/s: +0.4063%
```

Scheduler/model-rate samples exclude each service's first round because the
endpoint did not provide a delta for that round:

```text
A1: 1009.471, 1012.508, 1013.543 tok/s
B:  1019.057, 1018.958, 1017.996 tok/s
A2: 1016.566, 1016.382, 1012.637 tok/s

combined-A median 1013.090 -> B median 1018.958 tok/s: +0.5793%
```

## Decision

The candidate is exact and exceeds the requested `+0.2%` continuation gate in
both resident HTTP decode and scheduler/model rate.  Recommend enabling it by
default only under the existing strict TP4 M64 A4/R2/W8/DPP/LDS profile.  It
must remain disabled for M32 and generic/fallback shapes.

## Artifacts

```text
/tmp/dsv4_gate_rowprefetch_a1_valid_teacher.json
/tmp/dsv4_gate_rowprefetch_b_teacher.json
/tmp/dsv4_gate_rowprefetch_a2_teacher.json
/tmp/dsv4_gate_rowprefetch_a1_valid_bench.json
/tmp/dsv4_gate_rowprefetch_b_bench.json
/tmp/dsv4_gate_rowprefetch_a2_bench.json
/tmp/dsv4_gate_rowprefetch_a1_valid.log
/tmp/dsv4_gate_rowprefetch_b.log
/tmp/dsv4_gate_rowprefetch_a2.log
/tmp/dsv4_gate_rowprefetch_b_proc_maps.txt
/tmp/dsv4_gate_rowprefetch_*_environ.txt
/tmp/dsv4_gate_rowprefetch_amd_*.txt
```

The final A2 service was stopped after measurement.
