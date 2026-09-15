# Unique-slot CK Set: deterministic reference at production-equivalent C16 speed

Original DeepSeek-V4-Flash,TP8/EP1/native AR, original checkpoint,1M logical KV,
C16x8K/chunk32768/query producerOFF/accepted stages1 attention. Default remains
unchanged. This is a numerical-repeatability improvement, NOT a throughput win.

## Why a new store helps

Existing fixed-slot remaps each Top-6 assignment to a unique virtual token,
still zeroes6xFP32 scratch and invokes AtomicAdd, then sums slots0..5. The earlier
real M8192 screen measured3.029ms atomic versus4.492ms fixed (48.3% penalty).

Independent CK module retains the exact baseline heuristic: V3,256threads,
M64/N128/K64,1x4 waves, BF16 inputs/FP32 partial, routed weight multiplication.
Only the output MemoryDataOp becomes Set. Each slot is written, so scratch is
allocated empty, then consumed by the unchanged fixed-order HIP reducer.
No installed AIter binary/header was modified; class/function names are isolated.

The wrapper is an experiment, trusted only with the validated remapped-sorter
contract. CPU oracle checks one-to-one coverage of all M*6 slots, matching route
weights and expert IDs. Runtime loader is default-off, guarded by original-V4
TP8 native large-prefill scope, M8192..36864,I256,Top6,blockM64,default CK tactic.
It checks source/build/binary hashes and fails if the artifact changes.

## Important large-buffer failure and repair

Initial Set passed M8192 but failed M32767, max output error0.0396652. Full
partial comparison localized exactly4096 elements: virtual slot65530, actual
token10921/TopK slot4, overwritten with zero. Not an acceptable rounding delta.

CK's buffer-store invalid-lane trick adds0x80000000 to the byte offset. Padding
uses virtual token M*6. At M32767, the FP32 partial is about3GiB; the invalid
offset addition wraps32bits into the live allocation:

    ((32767*6*4096*4 + 2^31) mod 2^32)/(4096*4) = 65530

The original scatter helper computes is_dst_valid correctly but passes the
invalid offset to this trick. AtomicAdd of padded zero hid this specific
overwrite; Set made it visible. An experiment-only header overlay now explicitly
predicates Set stores on is_dst_valid. No speculative access, added CTA barrier,
changed MFMA reduction order or dropped routed assignment.

The CPU address test covers below/above2GiB and the exact observed slot. Full
GPU real-data tests below verify the fix. Do not generalize this into a claim
that ordinary production atomics were overwriting nonzero values.

## Correctness and component timings

`screen-v2.json`: two original-model fixtures, physicalGCD6 only. Every partial
and final BF16 output matches the old fixed-slot reference byte-for-byte across
five mutation cases, then30 poisoned-output eager replays per fixture. Original
real inputs restored before timing. These are not HIP Graph replay claims.

| M | Atomic ms | New Set+fixed sum ms | Old fixed ms | New vs old fixed |
|---|---:|---:|---:|---:|
|8192|3.000326|2.877165|4.463709|~1.55x|
|32767|10.731939|10.684115|16.271606|~1.53x|

Each comparison family runs threeABBA cycles, five complete calls/sample,
median of six samples/arm. Complete stage2 includes allocation/remap/zero/store/
reduction/cast as appropriate. Scratch remains6x:768MiB atM8192,~3GiB atM32767.
No memory-capacity reduction is claimed.

Fresh service PID2061301 uses new Set on the controlled16K request/chunk8192,
with prefix8192. All3440 stage/rank tensors (43layers x8ranks x10stages) match
the previously verified fixed-slot service. Complete inputs/positions/metadata
match; only the expected fixed-slot helper differs among common runtime source
hashes. New module selection witnessed on all8ranks. France and prompt echo pass.
This extends the cached-chunk remedy; it is not arbitrary batch-invariant decode.

## Formal service ABBA

Three real waves/leg,131069 input tokens/wave,16 distinct code prompts,one output
token, zero prefix hits. Client uses perf_counter for all throughput timestamps.

| Leg | input tok/s median |
|---|---:|
|A1 atomic|8399.053516|
|B1 unique|8382.156038|
|B2 unique|8379.820522|
|A2 atomic|8372.178084|

Atomic center8385.615800; unique center8380.988280; change-0.05518%. Control
arm drift itself-0.31998%. Treat this as throughput parity, not a speed victory.
1M logical pool/chunk32768 and actual unique path on all8ranks verified.

Each arm also ran two16-request x128-token quality waves: all input echoes and
token counts checked, within-arm16/16 outputs equal; France passed. Candidate
versus either atomic control is0/16 whole-output exact. Their request IDs/order
were checked directly, so this is not accidental list misalignment. Inspected
answer openings remain coherent/topic-related, but this is not proof every code
claim is correct. Fixed reduction chooses a reproducible different numerical
trajectory; do not promise to reproduce the atomic greedy trajectory.

## Harness interruption and verified resume

A1 PID2068733 completed warmup then the old owned() guard failed because
psutil.create_time() changed from1789493786.07 to1789493785.07. PID/command were
unchanged and the server remained live. Do not report a GPU crash.

Verified exact tmux pane2068727, direct parent, command, cwd and open owned log;
recorded boot ID/start ticks, resumed SAME process after warmup, then completed
A1 and shutdown. Subsequent starts use immutable kernel identity, retaining the
command check. No service restarted solely due to observation failure. Separate
resume provenance and two CPU lifecycle tests retained. Client perf_counter
timing does not use the changing wall-clock-derived birth value.

B PID2078737 and A2 PID2085640 completed/stopped cleanly; final all8GCDs free.
Warmups~7.23–7.29k excluded. A1/B/A2 common29 runtime source hashes match.

## Other implementation notes / next work

- First compile failed only on the local hipified guard name: the header exposes
  c10::cuda::CUDAGuard, not c10::hip::HIPGuard. Failed source/log retained.
- Binding inspection found historical Python callers pass nt in positional
  splitk. Explicit splitk1/non_temporal_load=False reproduced the same partial
  as the historical call in the tested current binary; it does NOT explain the
  large-M Set overwrite. Separate ABI cleanup remains, not silently mixed into ABBA.
- New code requires an explicit manifest plus FIXED_SLOT=1. It is NOT default,
  not a portable packaged kernel yet. Preserve old fixed-slot oracle and atomic
  fallback, harden build/shape coverage before promoting. H16 peer attention
  remains a separate, unintegrated performance opportunity.

Artifacts: `.agents/experiments/dsv4_ck_unique_store_20260916/` and
`.agents/experiments/dsv4_ck_unique_service_20260916/`. Bounded archives retain
raw logs/JSON/build provenance/selected check tensors; full tensor corpora and
DSO/object files remain local. All module hashes recorded, including failures.
