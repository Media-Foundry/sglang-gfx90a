# Exact paired pre-mix ownership: 9.975k TP8 C16 prefill, +11.11%

2026-09-16, base0f810add50. Original DeepSeek-V4-Flash, TP8/EP1/no-A2A,
native AR, original checkpoint precision,1048576 logical KV tokens,32K chunk,
C16 x approximately8K diverse real code prompts, zero prefix hits,
131069 input tokens/wave. This is prefill input throughput, not decode/DSpark.

## Formal service ABBA

Three measured waves per leg; warmup excluded. Fresh control processes A1/A2,
two consecutive three-wave legs in the same B process. No concurrent GPU test.

| Leg | Median input tok/s |
|---|---:|
| A1 full paired pre-mix |8982.319256|
| B1 row-owner pre-mix |9973.404172|
| B2 row-owner pre-mix |9977.194785|
| A2 fresh full paired control |8973.376830|

Control8977.848043; candidate9975.299479; **+11.110139%**. Control drift-0.099556%.
Candidate wave time approximately13.13935seconds. Do not round this into a
claim of sustained >10000. Full raw samples are in summary.json.

This supersedes8.965k as the fastest measured exact-reference-equivalent
configuration on this workload. The9.427k cooperative MFMA result is a separate,
non-bit-exact alternative; MFMA is OFF in every owner experiment.
The global launcher default is unchanged. Use the explicit validated launcher
in `.agents/experiments/dsv4_premix_owner_20260916/validated-launcher.sh`.

## What changed

Each rank keeps the complete residual, post, RMS partials, Sinkhorn and norm.
Only the24-coefficient pre-mix is partitioned into contiguous8-row-aligned
regions. Each owner calls the SAME production `premix8_pair` on its slice, then
RCCL allgathers FP32 coefficient values, without a floating-point reduction.

AtM32768 each rank computes4096 rows and contributes384KiB; full gathered
result is3MiB/rank. RaggedM32767 uses the same4096-row send capacity, zeros the
single padded row on the last rank, and exposes only the valid32767 rows.
No hidden/residual ReduceScatter, H4096/H16384 exchange, weight copy, checkpoint
conversion, host occupancy sync or persistent global workspace is introduced.
The local kernel intentionally bypasses the outer8192 minimum: global_M has
already passed admission; local_M is merely the same kernel's bounds argument.

Flag `SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER=1`, default0. Requires existing
original-V4 TP8/EP1/CP1/PP1 ordinary EXTEND mix-pair scope,8192..65536 rows,
validated layout/device and no graph capture. Native decode, draft, target
verify, V4.1, TP4, TBO and unsupported paths do not enter. Simultaneously enabling
MFMA raises an error; these candidates are not silently stacked.
`SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER_CHECK=1` checks every admitted output against
that rank's full-input paired reference; checks are excluded from speed arms.

## Identity was established before assigning ownership

An observation-only service ran the current exact checkpoint first. At12 sampled
boundaries covering early/middle/late positions in each of four real forwards,
all8 ranks hashed COMPLETE residual, Fn, RMS partials and reference mixes.
All96 records matched across ranks, including dtype, shape and epsilon.
Every rank's directly computed local slice matched its full reference bytewise.
No model output was replaced by this diagnostic. No inference from sampled
activation values or normalized-hidden equality was substituted for full hashes.

Actual forward sizes:32767,32767,32767,32768. Calls0/40/84 etc are ordinal
sampling positions, not independently asserted layer identities.

## Eight-rank component oracle

Sampled real layer0 data repeated for occupancy; NOT independent full live input.
M8192/32767/32768/65536,10Fn/input mutations, random row permutation and1000
graph replays all byte-exact across8 ranks. Timing uses the slowest rank for each
matching sample, then ABBA medians; not a sum of unrelated rank medians.

M32768 eager full-chain:5.47321 ->1.11683ms, including local kernel and RCCL
allgather. Graph:5.47955 ->1.09368ms. EagerM32767:5.48705 ->1.11502ms.
This cleared the2ms/boundary gate. Results include extra communication, not
just an ideal1/8 local-compute estimate. No service overlapped the oracle.

## Integrated correctness and drift evidence

- Separate service check:2720/2720 byte-equal comparisons,85boundaries x4forwards
  x8ranks. Importantly, each rank compared gathered output with ITS OWN complete
  input reference, not with rank0's output. Diagnostic7218.87tok/s is not scored.
- France passes in all five services: identity audit, full-check, A1, B, A2.
- Timed arms:16requests x128outputtokens x4qualitywaves x3processes =192answers.
  All repeats AND cross-arm outputs are identical for all16requests.
- Same64-token fixed continuation appended to each original prompt. Each API
  response has one leading null logprob/Top-5 entry: excluded explicitly.
  Actual count16x63=1008. A1/A2 and A1/B both have exactlyzero selected-token
  logprob difference,1008/1008sameTop1 and1008/1008identical returnedTop5 records.
- Input-ID echoes, tokenizer output decoding and zero-prefix hits checked.

This is strong evidence for the tested C16x8K fixed workload and continuations,
not a universal proof for arbitrary batch arrangement, prefixes, lengths or
long-generation quality. No claim that all historical whole-model drift is fixed.
All five process trees stopped, ownership checked by bootID/startticks; GPUs
released. Runtime memory pickle remains untouched/unstaged.

## Artifacts / reproduction

Directory `.agents/experiments/dsv4_premix_owner_20260916/`:

- service_audit.py: observation-only full-hash and local-slice gate.
- oracle.py: eight-rank local+allgather component and graph replay.
- service.py --arm check/A1/B/A2: isolated live check then formalABBA.
- analyze.py: strict response, performance, repeat and teacher-forced validation.
- summary.json / oracle.json: full sample evidence.
- service-evidence.tar.gz + archive-manifest.json: raw logs, launches, input,
  output, source hashes, state/resource snapshots and diagnostic records.
- validated-launcher.sh: explicit measured configuration; common H16/uniqueCK/
  vec4/exactpost settings inherited from the prior validated checkpoint.

Sixteen CPU scope/partition/negative-dispatch tests pass (one pre-existing pytest
asyncio configuration warning). No default promotion beyond explicit launcher.

## Next

Update the closed service profile before spending the old1.77second pre-mix
budget again: this work has already removed most of its replicated computation.
Prioritize wider-input coverage / remaining MoE supply-writeback and retain the
exact full-pre-mix control. MFMA and owner savings overlap and must not be added.
