# TP4 M64 router hipBLASLt solution 4358 service A--B--A (2026-08-30)

## Scope

This experiment tested the default-off
`SGLANG_DSV4_GFX90A_M64_ROUTER_HIPBLASLT` wrapper after the standalone
router oracle found hipBLASLt solution 4358 at about 14.29 us versus the
current 21.25 us projection.  No model weights or routing semantics were
changed intentionally.

The service contract was identical for all three independently loaded arms:

```text
physical GPU 0--3 only
TP4 / EP1 / no A2A
native autoregressive decode; no speculative environment
original safetensors checkpoint
SGLANG_DSV4_GFX90A_TP4_BS32_PROFILE=1
M64 row-prefetch accepted stack enabled
CUDA graph tiers exactly 1 and 64
32K raw-token KV pool
64 distinct real heterogeneous input-ID prompts
128 generated tokens/request, four rounds/service
```

The static M64 safety capacities were explicitly set to 384 routed rows for
the AIter quantization and general dispatch grids and 128 for the decode
dispatch floor.  `/proc/<pid>/environ` was saved for the server and live TP
workers in every arm.  It confirms the correct profile variable (not the
similarly named obsolete shell variable), A4/R2, LDS unpack, M64 DPP gate,
gate row-prefetch, logical W2 scale, four-wave W4 down, ROCm multistream and
the candidate flag value.

Before every service/test, `amd-smi process` was saved.  BIO processes that
appeared late in the run were not touched; they had only approximately
0.6 MiB idle ROCm contexts and zero observed GFX occupancy on GPU 0--3.  The
experiment stopped only its own service PID.

## Candidate reachability

For the B service only, a temporary once-per-process diagnostic was placed
immediately before:

```python
hipb_gemm(hidden_states, weight.detach(), 4358)
```

The graph-capture log contained exactly four lines:

```text
DSV4_M64_ROUTER_HIPBLASLT_HIT solution=4358 hidden=(64, 4096) weight=(256, 4096)
```

one from each TP worker.  This proves that the captured M64 graph used
solution 4358 and does not infer selection from the environment variable.
The temporary logging code was removed after the experiment and is not part
of the retained production diff.

## Correctness

Each independently loaded service ran the 64-row teacher-forced next-token
oracle.  Relative to A1, both B and A2 were exact:

| comparison | output IDs | token logprobs | complete top-5 rows |
|---|---:|---:|---:|
| B versus A1 | 64/64 | 64/64 | 64/64 |
| A2 versus A1 | 64/64 | 64/64 | 64/64 |

Thus the standalone router projection's non-bitwise BF16 values did not
change the first-token chosen ID, its reported logprob, or any complete top-5
row in this heterogeneous 64-request service oracle.  A direct internal
router expert-ID capture was not added; the full-model teacher result is the
stronger downstream check available in the existing harness.

All 12 throughput rounds also passed:

- France first-nine-token sentinel;
- 64/64 completion lengths equal to 128;
- every finish reason equal to `length`.

## Resident M64 throughput

The primary metric is the common wall-time interval in which all 64 requests
were resident at M64.  Variable heterogeneous prefill is excluded.

```text
A1 flag=0:
  1006.6140, 1006.9276, 1005.2443, 1006.6367 tok/s
  median = 1006.6253 tok/s

B flag=1, solution 4358 proven:
  1008.4110, 1009.1860, 1009.1533, 1009.2190 tok/s
  median = 1009.1697 tok/s

A2 flag=0:
  1006.6784, 1005.7112, 1006.1197, 1005.6313 tok/s
  median = 1005.9155 tok/s
```

Using the median of all eight A samples:

```text
1006.3669 -> 1009.1697 tok/s: +0.2785%
```

Removing one global minimum and maximum from the eight A samples and one
minimum and maximum from the four B samples gives:

```text
1006.2319 -> 1009.1697 tok/s: +0.2920%
```

Scheduler/model rates omit each service's first round because that endpoint
does not provide a delta for the first interval:

```text
A1: 1023.0806, 1019.3580, 1023.0662 tok/s
B:  1024.9249, 1025.3852, 1023.0850 tok/s
A2: 1022.1885, 1022.6493, 1022.0260 tok/s

combined-A median 1022.4189 -> B median 1024.9249 tok/s: +0.2451%
```

The measured service gain is consistent with only part of the standalone
6.97 us/layer saving surviving queue overlap.  It is small but exceeds the
predeclared 0.2% service continuation threshold in both resident and
scheduler/model metrics.

## Decision

The candidate passed the real heterogeneous service correctness gate and
showed a reproducible approximately 0.25--0.29% gain.  It is reasonable to
enable solution 4358 by default only inside the existing strict
TP4/M64/N256/K4096 BF16 profile guard.  It must remain disabled for M32,
non-TP4 layouts, other shapes and generic AIter callers.

This result does not make solution 4358 bitwise-equivalent as a standalone
matrix projection: the prior mutation oracle remains 0/100 bitwise exact.
The acceptance rationale is specifically that the complete downstream
teacher oracle was exact for all 64 heterogeneous rows and every service
quality sentinel passed.

## Artifacts

```text
/tmp/dsv4_router4358_a1_teacher.json
/tmp/dsv4_router4358_b_teacher.json
/tmp/dsv4_router4358_a2_teacher.json
/tmp/dsv4_router4358_a1_bench.json
/tmp/dsv4_router4358_b_bench.json
/tmp/dsv4_router4358_a2_bench.json
/tmp/dsv4_router4358_{a1,b,a2}_environ.txt
/tmp/dsv4_router4358_{a1,b,a2}.log
/tmp/dsv4_router4358_amd_*.txt
```
