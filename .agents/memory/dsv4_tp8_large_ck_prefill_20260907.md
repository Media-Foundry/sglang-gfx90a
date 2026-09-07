# TP8 large-prefill CK migration (2026-09-07)

Base `86e4bd4a79`; independent opt-in experiment. Original checkpoint,
TP8/EP1/no-A2A native AR, not speculative. Preserve small-chunk TP8 control
(P32 3825 input tok/s) and historical TP4 large-M record (5840.91 tok/s)
as different profiles, not interchangeable measurements.

## Migration and layout findings

1. `gfx90a_bf16_ck_moe` hardcoded intermediate=512 (TP4); TP8 requires256.
   Infer it from raw W13; accept only exact E256/H4096/Top6 TP4/TP8
   weights and scale shapes. Reallocate per-device expanded-weight workspace
   on shape changes so TP4 storage cannot be reused incorrectly for TP8.
2. The explicit prefill profile did not export
   `AITER_DSV4_DEBUG_SHUFFLE_BF16_WEIGHTS=1`. CK BF16 B needs (16,16)
   preshuffle despite the module's `preshuffle_off` filename. Unshuffled
   TP8 AND TP4 synthetic oracles give near-zero output/reference cosine;
   enabling preshuffle restores ~0.99999. Add the explicit default to the
   opt-in profile. Historical service environment is not fully reconstructed;
   this does NOT prove whether that earlier 5841 run omitted the setting.
3. TP4 token-row MHC predicates exclude TP8. Do not simply change `4` to `8`:
   communication ownership/ABI needs its own oracle. This experiment retains
   TP8's existing MHC and leaves the TP4-only selector a documented miss.

No global defaults or decode selector changed. Existing unrelated dirty
CK/FP16/debug code was preserved. Current helper still uses FP32 atomic
stage2 accumulation; same-configuration bitwise stability is not assumed.

## Component screen

`scripts/rocm/bench_dsv4_bf16_ck_tp8.py` reuses the existing variable-M
oracle with I256. Physical GPU0 with resident serving traffic idle; amd-smi
PID check first. Flags: stage2 FP32=1, block64 V1=1, preshuffle=1.
Synthetic random packed weights, E8M0 scales112, BF16 activations,
balanced/skewed Top6. Five full-stage timing samples; sequential Torch
selected-expert first-token reference. Not real-weight quality validation.

M8192 TP8: 9.148/9.905 ms balanced/skewed; cosine 0.9999919/0.9999927.
M36864 TP8: see `/tmp/dsv4_ck_tp8_m36864.log`; cosine
0.9999903/0.9999901. Sub-1e-9 absolute replay differences are observed at
these deliberately small scales, not a proof of E2E harmlessness.
Unshuffled negative logs: `/tmp/dsv4_ck_tp8_m8192.log` and
`/tmp/dsv4_ck_tp4_recheck_m8192.log`; corrected log:
`/tmp/dsv4_ck_tp8_shuffled.log`.

## Initial service screen: REJECTED

Both prefill-throughput and TP8 multi-request profiles enabled. Explicit:
physical0--7, TP8 EP1, chunk/max-prefill36864, request16, delayer20ms,
pool131072, mem.80, graph1/2/4/8/16/24/32, overlap/SBO on,
C4 trivial logits skip=1. Localhost30011.
Log: `/tmp/dsv4_ck_tp8_large_server.log`.

Same fixed 32 real-code requests / 73724 input tokens, 5 P32 rounds;
then fixed32x32 token/logprob oracle and two independent C32x128 semantic
waves. France is short and does not exercise CK, so it is only a regression
sentinel.

P32 measured 5383.86 cold, then 6432.06 / 6441.29 / 6427.29 / 6276.29
input tok/s (warm median6429.67). **Not valid delivery throughput**: first
tokens varied, and round0/request5 repeated "This is a placeholder" through
the long response. Round1/request5 was coherent. Most other responses had
coherent content but some corrupted-looking leading tokens. Do not classify
this as harmless greedy drift. Raw artifacts: `/tmp/dsv4_ck_tp8_large_*.json`.

## Second correctness root cause: shuffled scales passed as logical scales

The production direct runner preserves raw FP4 WEIGHTS, but its E8M0 SCALES
still undergo `shuffle_scale_a16w4` during loading. The CK helper used only
`reshape`, interpreting shuffled bytes as logical [expert,row,K32] scales.
The original synthetic screen used constant scale112, masking this bug.

Added explicit `scales_shuffled` input contract to the helper and set it at
the AIter runner call site according to the existing raw-scale debug flag.
Inverse reshape/permute restores logical scale bytes before dequantization.
No scale values, weights or activation precision are changed by this fix.
CPU test `check_dsv4_ck_scale_layout.py`: TP4/TP8 x gate/up/down, 10 random
all-byte mutations per shape, exact inverse of the actual AIter producer.

Same large-M profile with CK disabled was run as an isolation control:
`/tmp/dsv4_tp8_large_raw_oracle32.json`. Of32 requests,29 first IDs and2
complete32-token sequences match the small-chunk reference. Inspected code
responses, including request5, start coherently. Shape-first-use compilation
dominates its first request; not a valid steady throughput timing.

## Corrected runtime-scale validation

Additional full-stage oracle with nonconstant scales118--120 and actual
runtime-shuffled input scales: cosine0.9999890/0.9999887 (balanced/skewed).
Wide scales118--131: cosine0.9999964/0.9999958; max abs512 with reference
absmax99840/109568. These compare BF16 CK against the sequential Torch
reference, not SDOT bitwise arithmetic. Wrapper timing includes an extra
synthetic scale producer shuffle and is not a production stage timing.
Logs: `/tmp/dsv4_ck_tp8_runtime_scales_{oracle,wide}.log`.

One corrected process stalled on the benchmark's initial `/freeze_gc` RPC
before large CK work. Background loopback freeze requests also logged502.
Root cause was not established (do not blame CK or hardware). Stopped only
that service/process tree and restarted; final C1 harness used the new
`--skip-freeze-gc` option. Default harness behavior is unchanged. This is
an explicit protocol difference, not a claim of strict single-variable ABBA.

Final service: `/tmp/dsv4_ck_tp8_fixed2_server.log`, same GPU/model/profile
settings above. Tests completed:

- P32 input tok/s: **5359.58 cold / 6421.65 / 6420.96 / 6419.83 / 6255.42**.
  Warm median **6420.39 input tok/s**, median wave **11.4828 s** for73724
  input tokens. Distinct real code requests, fresh salts, no cache hits.
- Short C1 HTTP decode smoke: **77.53 / 77.59 / 77.59 output tok/s**.
  France exact twice. BS32 steady decode throughput was not remeasured.
- Two fixed-batch C32x32 oracles: first IDs identical32/32; complete output
  IDs identical31/32. Request26 first diverges at generated index25, where
  the second run has a top-1 tie. Text changes from "is the practice of"
  to "is a technique used"; both remain coherent. Logprobs are not exact.
- Compared to small-chunk SDOT: first ID28/32 and full32 IDs1/32 match.
  Large prefill shape and BF16 CK arithmetic differ; do not call this a
  bitwise-equivalent numerical path.
- Two independent C32x128 waves: all64 completions reached128 tokens,
  cached tokens0. Manual leading/trailing text review of all64 responses
  and the earlier failing request5 found no placeholder/repetition collapse.
  Request5 explains sparse-attention dataflow in both runs. This is a
  semantic smoke check, not proof that every generated code-review claim is
  correct, nor a full model-quality benchmark. Cross-round hashes differ.

Data (including full long-response text, completion hashes, timings and
remaining first-divergence detail): adjacent
`dsv4_tp8_large_ck_prefill_20260907.json`. Original full-ID/logprob artifacts
remain under `/tmp/dsv4_ck_tp8_fixed2_*.json` with SHA256 in that record.

Decision: preserve the corrected TP8 throughput profile as **opt-in,
non-bit-exact**. The severe scale-layout corruption is fixed and the sampled
long-output collapse no longer reproduces; deterministic FP32 expert
reduction and control-RPC reliability remain separate follow-ups. Do not
promote the pre-fix6429.67 result or silently enable the profile globally.

Reproduction launch:

```bash
SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE=1 \
SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE=1 \
SGLANG_DSV4_C4_TRIVIAL_LOGITS_SKIP=1 \
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8 EP_SIZE=1 \
MOE_A2A_BACKEND=none PORT=30011 HOST=127.0.0.1 \
MEM_FRACTION_STATIC=0.80 MAX_TOTAL_TOKENS=131072 \
CHUNKED_PREFILL_SIZE=36864 CUDA_GRAPH_MAX_BS_DECODE=32 \
CUDA_GRAPH_BS_DECODE='1 2 4 8 16 24 32' DISABLE_OVERLAP_SCHEDULE=0 \
scripts/rocm_dsv4_flash.sh serve
```

This is a 131072-token pool validation, not a 1M-pool capacity acceptance.
Use the earlier chunk2304/no-prefill-throughput profile for rollback; that
configuration and its experiment records were not removed.
