# DSV4 TP4 prefill planner and MFMA issue audit (2026-09-02)

## Accepted baseline and test contract

- Four physical gfx90a GCDs 4--7, original DeepSeek-V4-Flash checkpoint.
- TP4 / EP1 / no A2A, queue-aware M2304/M4608 chunks, token-row MHC.
- C1 4604-token warm prefill remains about 2.50--2.52k input tok/s.
- C32 uses 32 distinct code-review prompts (73,724 audited tokens) and remains
  about 2.74k aggregate input tok/s.
- Every GPU experiment was preceded by `amd-smi process --general --sort-by-pid`.

## Attention-ahead raw-FP4 preshuffle rejection

The exact HIP raw-to-CKTile preshuffle primitive is about 50% faster than the
AIter Python transform and can overlap a standalone M4608 compute stream.  A
true attention-ahead service integration nevertheless measured:

```text
2566.38 / 2669.58 / 2697.04 input tok/s
median 2669.58
```

This is about 2.6% below the accepted C32 control.  The production integration
was removed; only the already committed standalone transform oracle remains.

## C1 chunk-length cost table

Real code tokens were concatenated in a fixed order to form deterministic C1
inputs.  Each point used a cold request plus three warm requests with unique
cache salts and one generated token.

With the accepted base chunk M2304:

| length | warm median | input tok/s |
|---:|---:|---:|
| 1024 | 0.5399 s | 1896.6 |
| 2304 | 0.9299 s | 2477.7 |
| 4604 | 1.8405 s | 2501.5 |
| 8192 | 3.5628 s | 2299.3 |
| 16384 | 7.1219 s | 2300.5 |

M2048 removes the M1280 tail at 8192 and improves it to 2355.9 tok/s (+2.5%),
but 16384 falls to 2262.6 tok/s (-1.7%).  All per-length output hashes were
stable.  This is not enough for the 4% planner gate, so no global chunk change
or narrow length special case was retained.  France returned exactly `The
capital of France is Paris.` after the experiment.

## Paired-I32 gate epilogue quantization

The older exact gate-to-group32-INT8 oracle serialized two adjacent I16 tiles
inside one 4-wave CTA and lost 25--27%.  A follow-up used an 8-wave CTA: two
independent production-order split-K4 wave groups compute the I16 halves in
parallel; a 32-KiB partial buffer is reused for gate and up, with separate 8-KiB
FP32 gate and 4-KiB BF16 boundary storage.  An initial alias of BF16 output with
the partial buffer exposed a cross-wave race (`max_abs=1.43e-4`) and was fixed
before formal timing.

Formal physical-GCD4 correctness:

```text
M2048: 100/100 input/weight mutations exact; 1000 graph replays exact
M2304: 100/100 input/weight mutations exact; 1000 graph replays exact
```

Seven-round ABBA complete routed-stage results:

| shape | control | paired gate+quant | delta |
|---:|---:|---:|---:|
| M2048 | 20953.89 us | 21293.48 us | +1.62% slower |
| M2304 | 23776.62 us | 23756.64 us | 0.08% faster |

The exact epilogue is retained only as a standalone oracle.  It does not meet
the 5% gate and must not be wired to production.

## Production M4608 MFMA hardware counters

ROCm Compute Profiler 3.7 collected counters for the raw-FP4 MFMA64 gate/up
kernel (`832 x 256`, LDS 32 KiB, scratch 0, VGPR178, SGPR56).  Representative
values were:

```text
duration                  about 14.3--14.7 ms
L2 hit rate               128.935M / (128.935M + 12.183M) = 91.36%
mean resident waves/CU    16.829G / (22.334M * 104) = 7.25
MFMA utilization          1.073744G / (22.334M * 416) = 11.55%
Int8 MFMA operations      536.8712M * 512 = 274.9 GOP/kernel
estimated achieved rate   about 18.7--19.5 TOP/s/GCD
HBM read request scale    about 23.5M requests/kernel (roughly 50--60 GB/s)
```

This kernel is neither near HBM bandwidth nor MFMA peak.  The principal limit
is low matrix-pipe issue/occupancy plus FP4 decode/scale and accumulator
lifetimes, not cold HBM bandwidth.

Two exact instruction-order oracles were tested without changing production:

- four assignment halves kept live across the two dependent MFMAs: 8.3% slower;
- shallow two-half version: 3.3% slower.

The compiler resources did not spill (the four-half candidate even reported
VGPR168 versus reference VGPR178), so the regression is instruction scheduling
and longer live dependency structure rather than scratch traffic.  The source
and benchmark wiring were removed.  Do not repeat manual cross-half MFMA
pipelining; a future kernel must raise issue utilization without extending
accumulator lifetime, for example by changing the work decomposition or using
an asynchronous weight/decode producer.

## Structural closures reconfirmed

- TP4/EP4 no-A2A prefill: C1 1776, C32 1766 tok/s; rejected.
- PP4/TP1 filled pipeline: best C32 2669 tok/s; rejected.
- One BS32/M73728 forward: client 2968, server 2390 tok/s; batching alone does
  not create weight reuse.
- MFMA128 LDS multicast: M13824 29.6% slower; assignment-parallel A128 22.1%
  slower; do not repeat per-K-group barriers.
- Expert-persistent M4608 is exact and 7--8% faster standalone, but 9.1% slower
  in service.  A new grid sweep reconfirmed gate 832 and down 416 as the local
  optima already used by the rejected service integration.

The 10k C32 objective therefore requires a different routed-MoE execution
architecture, not another chunk size, grid count, EP switch, or synchronous
LDS multicast around the current kernel.

## Split-K2/4/8 gate sweep and service rejection

The profiler's low occupancy motivated a real M4608 raw-FP4 gate/up sweep.
All shapes used the same A64 sorter metadata and physical GCD4:

```text
split2 best (1664 blocks): 17.125 ms
split4 best (416 blocks):  13.891 ms
split8 best (624 blocks):  12.922 ms
```

Split8 was about 7.0% faster than the local split4 control.  Its changed FP32
association was finite over the full output (`max_abs=0.5`, mean absolute
`6.62e-6`).  A default-off selector was then restricted to exact M4608 only;
C1 M2304, decode, and DSpark were untouched.  France returned `The capital of
France is Paris.`

Four C32 heterogeneous service rounds were:

```text
2539.67 / 2621.11 / 2631.09 / 2626.55 input tok/s
warm center about 2624 tok/s
```

This is about 4.2% below the accepted 2.74k control.  More split waves improve
the isolated gate but compete with the rest of the layer and/or increase the
four-rank tail.  The environment variable, runner branch, and shell export were
removed.  Do not enable split8 from the microbenchmark result alone.

The corresponding M4608 down sweep found only a small local candidate:

```text
current split2/624: 13.868 ms
split4/624:         13.494 ms  (2.7% faster)
split8 best:        18.496 ms  (decisive regression)
```

Split4 changed the reduction association (`max_abs=16`, mean absolute
`7.26e-5` under deliberately wide random scales).  Since down is only one part
of the routed stage, its optimistic E2E value is below 1%; it was not connected
to the service.

## MFMA64 in-place LDS pair-LUT rejection

The MFMA kernel normally decodes each packed FP4 `uint16` with three `v_perm`
operations plus selector bit manipulation.  A default-off oracle borrowed the
first 1 KiB of the existing 32-KiB split-partial LDS as a byte-pair LUT during
the K loop, so it added no static LDS.  Correctness exposed two required
ordering barriers: partial publication must wait for all LUT readers, and the
LUT must be rebuilt for every task because the previous task overwrites it.

After both races were fixed, output was bitwise exact.  Seven-round timing was:

```text
direct v_perm decode: 13.940 ms
in-place LDS LUT:     14.328 ms  (+2.7% slower)
```

The per-task LUT initialization and two barriers cost more than the removed
per-packed-value permutations.  The template experiment was removed.  Do not
reuse gate partials as a LUT without accounting for both wave and task lifetime.

## Native reciprocal SwiGLU screen

Disassembly showed the exact `gate / (1 + exp(-gate))` lowering to a corrected
division chain.  Replacing only the reciprocal with CDNA2 `v_rcp_f32` kept all
clamps and exponentials unchanged.  Across 100 pressure mutations the result
was finite (`max_abs=0.5`, mean absolute `2.58e-8`).  Gate ABBA changed only
13.943 to 13.892 ms (0.37% faster), far below even a 1% E2E bound.  The
approximate template was removed.

## LLVM AMDGPU scheduler sweep

The same M4608/split4/blocks416 source was compiled with six machine-scheduler
profiles.  Five mutations were bitwise exact for every profile.  Trimmed gate
times were:

```text
default                         13.941 ms
gcn-max-occupancy               14.009 ms
iterative-max-occupancy         13.950 ms
schedule-metric-bias=0          13.991 ms
schedule-metric-bias=100        13.845 ms
relaxed-occupancy               13.860 ms
disable-mfma-rewrite-stage      14.001 ms
```

The best compiler-only change is roughly 0.7%, below the component gate and
normal run-to-run drift.  Production remains plain `-O3`.
