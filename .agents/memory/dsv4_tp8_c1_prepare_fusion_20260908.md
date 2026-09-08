# C1 existing preparation fusion screen, 2026-09-08

## Withdrawn at user request

The user requested withdrawal before the final control completed. Stopped only
experiment controller PID2712721; retained the already restarted control service
PID2727188 with FUSED_ATTN_PREP_GEMV=0 and the validated TP8 M32 overlap=1.
No production kernel or default was changed by this experiment. Preserve the
oracle and raw results for audit; do not promote this candidate. C1B2 medians
were84.547/84.452/84.603tok/s, unlike the slower B1. Without the completed final
control, no stable E2E improvement or regression is established. The pending
sequence descriptions below are historical, not instructions to resume it.

Withdrawal verification: retained service reached application startup complete.
`/tmp/dsv4_tp8_withdraw_correctness_20260908.json` completed 12 fresh-prefix
probes: next IDs, input token logprobs and output top-20 logprobs all exact12/12
against runtime-M reference. This is a bounded prefix oracle, not full C32 parity.

Next-direction audit (no production edits): current topk.py attempts importing
aiter.topk_gating with a silent ImportError fallback. The actual DS installation
at /home/pc/pytorch/third_party/aiter lacks that export (direct import verified),
despite SGLANG_USE_AITER=1. Thus the sqrtsoftplus selector falls through to
sglang.kernels.ops.moe.moe_fused_gate, not AIter topk_gating. Benchmark the actual
fallback before proposing any Top-K replacement; old packed/native negatives
remain relevant. An unavailable optional API is not itself a correctness bug.

Parent488b148e6d; preserve winning TP8 M32 C4 overlap, native AR, original
checkpoint, pool131072, mem0.80, same graph tiers and interleave all.

Existing default-off FUSED_ATTN_PREP_GEMV combines qkv-a1536, core compressor2048,
index compressor512 and index weights64 in one HIP launch without weight concat
or new weight cache. For short KV, the actual indexer skips weights projection;
therefore its fourth launch must NOT be counted in the baseline. Initial
four-producer screen37.289->30.484us is not a valid short-context service estimate.

Corrected oracle compares three actual wave64 producers against fused output
segments0–2; fused still pays for its unnecessary64-row fourth segment. Eight
independent bundles (260MiB transient weights) avoid an all-L2-resident timing.
GPU4 only, amd-smi confirms no external GPU workload. No persistent service
memory change; isolated process exits after test.

Five ABBA cycles,8-node graphs x50 replay: separate34.4458us, fused30.5022us,
~11.45% component improvement /3.944us. 100 mutations, weight mutation every25,
10 graph replays/mutation: qkv BF16 and both compressor FP32 outputs exactly
equal, finite and replay-stable100/100. Current compressor helper intentionally
rounds to BF16 then promotes to FP32; fusion preserves this contract. No claim
about the fourth (currently unused) index weights versus generic GEMM at long KV.

Uninstrumented A0 C1: case medians83.813/83.731/84.094tok/s, output hashes and
fixed-prefix probes match reference. Raw `/tmp/dsv4_tp8_c1_prep_A0_20260908.json`.

E2E ABBA now runs through existing helper with --switch
SGLANG_DSV4_GFX90A_FUSED_ATTN_PREP_GEMV, --arms C1B1:1 C1B2:1 C1A2:0,
prefix `/tmp/dsv4_tp8_c1_prep_abba_20260908` (unique state file).
Root exec session26076; state JSON contains active parent. Per arm: C1 two
rounds/task plus fixed-prefix oracle, C32 six real coding waves,12 transition
probes. The C32 overlap switch stays1 in every arm. Last control stays resident.
No production source/default changed. Wait for results before acceptance;
short C1 micro savings are not an E2E speedup.

First candidate C1B1 measured C1 near83.3tok/s, not above A0 (~83.8–84.1).
C32 six rounds remain near1008tok/s; no new C32 benefit is attributed to this
M1-only switch. Full helper sequence still owns the live service; do not restart
or run concurrent GPU work while it proceeds to C1B2 and C1A2. Await final
state and artifacts before drawing the final E2E conclusion.
