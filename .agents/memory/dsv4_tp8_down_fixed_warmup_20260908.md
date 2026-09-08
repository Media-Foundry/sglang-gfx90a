# TP8 down fixed-first-use startup diagnostic (2026-09-08)

## Purpose and status

Default-off `SGLANG_DSV4_GFX90A_TP8_M32_DOWN_FIXED_WARMUP` runs baseline then
uniform down using the existing full-model M32 warmup inputs, before the usual
selected-arm warmups and single graph capture. No new graphs, weight caches,
or persistent tensors. Native TP8/EP1 DSV4, 1M KV, no speculative; dense M32
only. Sparse M32 and C1 retain their ordinary capture. Paired-graph mode is
incompatible and rejected. Neither this flag nor down-uniform is promoted.

This is a confound-control experiment, not a new faster kernel. Previous
single-graph candidate PID3542331 measured C1 82.7854, repeated 82.8039 after
C32. Standalone first-launch evidence showed each down module added 2 MiB
outside Torch reserved memory; loading the modules alone added none.

## Implementation and initial failed attempt

Both eager warmups drain device work and restore Raw attention metadata.
Selector contexts restore on exceptions; an unsuccessful forward aborts
startup rather than retrying a possibly poisoned context. Normal selected
warmup/capture remains intact. No extra replay work is introduced.

First attempt PID3558833 exited at the explicit diagnostic memory check,
not an illegal-address fault. The check incorrectly charged the ordinary
first model forward allocation to the experiment. Corrected measurement:
snapshot after baseline forward, then charge the alternate forward delta.
Keep a 256 MiB/GCD incremental limit and 2 GiB free reserve, with CPU-group
all-rank admission. Never reduce KV to satisfy this diagnostic.

Retry PID3564536 succeeded. Every rank logged:
- baseline first warmup: 482344960 bytes (460 MiB);
- alternate first use: 2097152 bytes (2 MiB).

Graph capture 12.42–12.50 seconds; graph memory 0.65 GB/GCD, available
15.58–15.64 GB. Runtime reports max_total_num_tokens=context_len=1048576.
This is memory-pool admission evidence, not a full 1M live-context benchmark.
Exact previous owned launch cmd/env was privately saved before stopping;
the failed process was confirmed gone and AMD GPU PID list empty before retry.
Private environment snapshots must never be committed or printed.

## First candidate process A1 (not yet ABBA)

Fixed diverse request corpus, original weights, native AR:
- C1 four measured rounds of three code tasks, per-task min/max trimmed,
  geometric mean: **84.18316169213634 tok/s**.
- C32 six waves, drop first, median HTTP: **990.1398332769456 tok/s**.
- C32 warm resident median: **1036.124244390715 tok/s**.
- France 32/32 exact through reference EOS.
- All 12 C1 complete 256-token outputs match the existing baseline reference;
  LE uint32 hashes recomputed.
- All 192 C32 requests have 256 tokens, finish=length, recomputed completion
  hashes, and no speculative acceptance statistic.
- Only 10/32 C32 requests match across all six rounds: **not bitwise stable**.
  Hash integrity/France do not establish general semantic correctness.

C1 is higher than the preceding slow process, but one restart is not causal
proof. C32 is effectively unchanged versus the earlier single-graph candidate.
Next: fresh baseline with the same fixed warmup, then complete ABBA before
any promotion. Do not count startup first-use timings as steady throughput.

## Reproducibility

CPU checks: 7 fixed-warmup tests (ordering, metadata, failure cleanup,
non-target tiers, memory bounds, topology/spec/KV scope), 11 paired graph
checks, 7 capture-transaction checks, down-uniform scope checker, py_compile.
These complement, not replace, the full-model checks above.

Artifact prefix:
`/tmp/dsv4_tp8_down_fixed_warmup_A1_retry_validation_20260908.block0`
- `.france.json` SHA256 `88d06a102f75e87060464b75107ddb8f5ece992ec02154d71bcb63a2f2a2776f`
- `.c1.json` SHA256 `ad5d507283261f88a122bcfa0f77257180829d2faf8da53de3930369ac86c608`
- `.c32.json` SHA256 `a71243ef47be19a65d8de71b0fd93c93fde68e0d82a907d895c4dbad55927c7e`

Workload SHA256:
`4d7f83aa48de5df30bf819fe33800bb7f12ecfd4e02ce9215188f62bfd16a424`.
Startup log: `/tmp/dsv4_tp8_down_fixed_warmup_A1_retry_20260908.service.log`.
First failed log: `/tmp/dsv4_tp8_down_fixed_warmup_A1_20260908.service.log`.
