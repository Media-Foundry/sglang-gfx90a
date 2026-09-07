# C4 trivial-row logits skip (independent, default-off experiment)

Base: `9aee233a57`, TP8/EP1/no-A2A native AR, original weights.

## Contract

`SGLANG_DSV4_C4_TRIVIAL_LOGITS_SKIP=1` enables only gfx90a ordinary
extend/prefill with the full-Triton FP8 paged logits wrapper, SGL Top-512,
and deterministic ordering mode 2/3. The default is **off**. Decode and
other backends retain the existing path.

For each row whose actual C4 length is <=512, the logits kernel writes
defined zero scratch without reading Q/K or evaluating scores. Top-K keeps
the real lengths and produces the same canonical valid indices. Query
projections, compressor updates and KV writes are untouched. The full
FP32 logits buffer is still allocated/written; this is not fused Top-K or
query producer compaction.

## Component oracle

`scripts/rocm/check_dsv4_c4_trivial_skip.py`, physical GPU 0, idle resident
service, resource check with `amd-smi process --general --sort-by-pid`.
Real preshuffled cache is produced by `triton_fused_store_indexer`.
Seed 2709; 1000 HIP graph replays; 100 mutations of Q, weights, cache,
lengths and page maps. Active scores, all logical IDs/order and physical
IDs are bitwise exact in all tested cases. Empty and 511/512/513 edges
are included. Candidate trivial score scratch is zero.

GPU event ABBA, 100 graph replays per arm, **logits + Top-K only**, us:

| Query rows | C4 width | A1 | B1 | B2 | A2 |
|---:|---:|---:|---:|---:|---:|
|32|576|31.7488|24.5888|24.4560|31.3856|
|2304|576|1153.2494|220.8307|220.7587|1153.0398|
|32|4096|142.5938|90.0337|89.9761|142.5218|

M2304 uses causal C4 lengths `(row+1)//4`, clamped to the width.
The ~81% component time reduction is not an E2E or full-indexer claim.
Raw log: `/tmp/dsv4_c4skip_oracle.log`.

Final code was also tested with `SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER=2`:
all three shapes passed 1000 graph replays and 100 mutations, exact active
scores and IDs. Mode-2 ABBA us: M32/L576 55.1323/48.1467/48.0572/54.7627;
M2304/L576 1181.6704/251.6984/251.8232/1181.5216;
M32/L4096 165.4192/113.2261/113.0245/165.0944.
Raw log: `/tmp/dsv4_c4skip_oracle_mode2.log`. The service E2E uses mode 3.

## E2E ABBA

TP8 throughput profile, physical 0–7, native AR, chunk 2304, pool 131072,
memory fraction .80, graph tiers 1/2/4/8/16/24/32, scheduler overlap and SBO.
Only the new flag changes. Fixed real-code manifest:
`.agents/memory/dsv4_prefill_diverse_32_input_ids.json`.
P1 selects request offset 2 (2304 tokens), seven rounds; P32 uses all
32 distinct requests, three rounds; one output token and fresh cache salts.
Discard first round per arm. B1/B2 use one candidate process; A2 is a new
control process. Full prefill-to-decode oracle is required separately.

| Arm | P1 warm median TTFT (s) | P32 warm median wave TTFT (s) |
|---|---:|---:|
|A1|0.648960|20.038924|
|B1|0.625384|19.286430|
|B2|0.625569|19.262028|
|A2|0.649703|20.073259|

Pooled warm medians: P1 **0.649202 -> 0.625505 s** (-3.650% TTFT,
+3.788% input throughput); P32 **20.053920 -> 19.273402 s** (-3.892%
wave latency, +4.050% input throughput). These are HTTP prefill timings
with one generated token, not pure GPU throughput. First rounds are
excluded separately for every arm. B1/B2 are consecutive suites on one
process, not two independent process starts.

The restarted A2 recovers the A1 timing, so the candidate improvement is
not simply the initial service warming up. Logged server-argument dicts
for A1/B differ only in the generated random seed (greedy tests); the
candidate adds the new environment flag.
Both cross-round completion checks passed, as did B2. Every P32 first
token in B1/B2 matches A1, with zero prefix-cache hits.

The fixed-batch 32 distinct long prompts, 32 generated tokens each,
matched `/tmp/dsv4_tp8_pd_final_oracle.json` **in the entire JSON record**:
IDs, text, token logprobs, top-5 logprobs and cache metadata.
Candidate record: `/tmp/dsv4_c4skip_B_oracle32.json`.
The C1 2304-token code prompt followed by 256 generated tokens also
matches A2 in its entire record, including all token/top-5 logprobs:
`/tmp/dsv4_c4skip_{B,A2}_oracle1.json`.

During static scope audit, `ForwardMode.is_extend()` was found to include
TARGET_VERIFY and MIXED. The final guard uses exact `ForwardMode.EXTEND`.
The native-AR B1/B2 trials used ordinary EXTEND already; a fresh-process
candidate smoke will validate the final narrowed guard. Optional
`SGLANG_DSV4_INDEXER_DEBUG=1` prints a one-time per-process actual full
Triton skip hit (static shape/threshold; no D2H).
Raw JSON prefix: `/tmp/dsv4_c4skip_`.

Raw arm times, cache counters, output hashes and oracle artifact hashes
are preserved in the adjacent `dsv4_c4_trivial_logits_skip_20260907.json`.

Final narrowed-guard process passed: all eight ranks logged actual hits
at M2304/C4 width576/Top512; warm P1 TTFT 0.625918 s. Both fixed32x32
and fixed1x256 entire records again matched the controls. France sentinel
passed twice with the expected nine IDs. Final artifacts use the
`/tmp/dsv4_c4skip_final_` prefix and their hashes are in the JSON record.

Status: component, ABBA E2E and independent final-process validation
passed. Keep the switch default-off for this independent experiment.
Do not extrapolate this TP8 raw routed result to M36864 BF16-CK prefill.

## Reproduction

Launch the existing `scripts/rocm_dsv4_flash.sh serve` with the TP8
throughput profile and explicit settings above; add the skip flag only
for B. Use `SGLANG_DSV4_INDEXER_DEBUG=1` for actual-hit evidence.

Component test (with all serving traffic idle):

```bash
amd-smi process --general --sort-by-pid
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=python:python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:python/sglang/kernels/aot/python \
/home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/check_dsv4_c4_trivial_skip.py --replays 1000 --mutations 100
```

P1 timing: `bench_dsv4_prefill_diverse_concurrent.py --request-count 1
--request-offset 2 --tokens 1 --rounds 7`; P32: same helper with
`--request-count 32 --tokens 1 --rounds 3`. Set `--base-url` and `--output`.
Fixed decode correctness: `check_dsv4_tp4_m32_next_token.py --inputs
.agents/memory/dsv4_prefill_diverse_32_input_ids.json --request-count 32
--tokens 32`; C1 uses `--request-count 1 --request-offset 2 --tokens 256`.

Scope limitations: query producer compaction, other context lengths,
prefix-hit workload matrices and BF16-CK large-prefill are **not** tested
by this E2E result. This does not resolve the pre-existing independent
HTTP decode-wave batch-composition drift documented in the TP8 P/D note.
