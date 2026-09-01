# DSpark clean BS32 segment profile and FP16 screen (2026-09-01)

## Accepted E2E control

- Physical GCDs: 4--7, TP4/EP1/no-A2A, original checkpoint weights.
- Static gamma-three DSpark and the accepted TP4 BS32 profile.
- Frozen 32-request heterogeneous workload SHA256:
  `f74de67a93a660cde060991df71c9e2972a05d82c3ba3f9fe7c144b1f066a152`.
- 1024 output tokens/request, `stream_interval=1`.
- Separate BS1 France request passed the historical first-nine-token oracle and
  semantic Paris check.

The clean run produced:

```text
resident BS32:       1573.187 tok/s
aggregate HTTP:      1275.433 tok/s
mean accepted length:   3.53518
all 32 lengths:      1024
all finish reasons:  length
```

Artifact: `/tmp/dsv4_observer_fresh_bs32.json`.

## Steady-state segment observer

The observer enabled only `core`, `step_cpu_time`, `step_gpu_time`,
`draft_gpu_time`, and `target_verify_gpu_time`; no request-level D2H recorder
was enabled.  There were 258 BS32 records.  After dropping the first and last
16, medians and 10% trimmed means were:

| Segment | Median (ms) | 10% trimmed mean (ms) |
|---|---:|---:|
| CPU step interval | 69.5722 | 69.5978 |
| GPU step | 69.5332 | 69.5447 |
| draft | 10.2054 | 10.2104 |
| target verify | 56.8569 | 56.8574 |
| GPU residual | 2.4579 | 2.4578 |
| max(0, CPU-GPU) upper bound | 0.0476 | 0.1084 |

Consequences:

- Host/scheduler submission is not the current bottleneck.
- Target verification is about 81.8% of the GPU step and is the only segment
  with enough budget to approach 2k.
- Draft is about 14.7%; all remaining GPU work is only about 3.5%.
- The independently rejected progressive M128 draft/anchor overlap cannot be
  rescued by host run-ahead because the host slack is essentially zero.

Artifact: `/tmp/dsv4_observer_fresh_server_info.json`.

## Untuned BF16 GEMM screen

Startup logs showed no tuned AIter entry for M128/K4096 BF16 projections at
N512/N1024/N2048.  A one-GCD hipBLASLt screen found:

- N512: best screened solution about 38.50 us versus torch about 33.79 us;
- N1024: solution 3729 about 40.63 us versus torch about 42.67 us;
- N2048: best stable candidate about 50.99 us versus torch about 51.22 us.

Only N1024 wins, by about 2.0 us per call, and all hipBLASLt outputs use a
different reduction association (max difference 0.5 in the random screen).
This is below the service-integration gate and agrees with the earlier
compressor hipBLASLt service rejection.  Do not add these globally to the
tuned CSV.

## FP16 shared-expert compute screen

To test whether CDNA2's FP16 path could improve the dense shared branch while
leaving stored weights unchanged, graph ABBA compared BF16 with FP16 compute
and BF16 output:

| Shape M,N,K | BF16 (us) | FP16 (us) | Result |
|---|---:|---:|---:|
| 128,1024,4096 | 42.874 | 41.087 | +4.35% |
| 128,4096,512 | 17.546 | 25.189 | -30.34% |
| 96,1024,4096 | 37.007 | 40.847 | -9.40% |
| 96,4096,512 | 14.995 | 22.088 | -32.11% |

Relative L2 was about 1.4e-3.  The down projection regresses decisively and
the M96 draft gate/up also regresses, so do not wire FP16 shared-expert compute.

## Next action

Capture a current target-verify kernel trace.  Older marker decompositions
predate several accepted M128 attention and all-reduce changes; further work
must target the current rank-max critical kernels rather than another isolated
GEMM/small-elementwise substitution.
