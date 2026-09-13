# Original V4 Flash TP8 revalidation and P/D matrix

User explicitly requested latest-main regression at a few sizes, followed by
the M128 down-consumer experiment, and then TP8 C1/2/4/8/16/32/64 P/D.
Decode AR and strict full-target DSpark are separate arms. Original weights
and1M logical pool are retained. Candidate acceptance requires correctness
and no demonstrated E2E slowdown; optimize via ABBA, final matrix three rounds.

**Scope update: user subsequently cancelled DSpark measurements.** Proceed
only with TP8 native-AR C1/2/4/8/16/32/64 P/D. The DSpark M128 experiment is
deferred, not rejected by a new measurement. Its untested standalone harness
extensions and DSpark launcher override were removed after the scope update.
No candidate production selector was added. Historical1127DSpark is not an
AR comparison point.

Starting HEAD `a365956778`; backup branch
`backup/pre-v4-tp8-regression-20260913`. V4.1 stays frozen; its service is off.
Before starting, `amd-smi process` showed no processes on GPU0--7.
Unrelated dirty files are preserved, including the graph-memory pickle.

## Plan and measurement contracts

1. Fresh strict gamma3 process on current code, diagnostic tiers1/8/32,
   original `/home/pc/models/modelscope`, pool1048576, chunk2304. Confirm
   Paris, readable real code, actual graph/kernel selection, and short pilot.
2. Revisit M128 down-consumer only after a usable control. Preserve full
   target experts and draft/accept TP synchronization. Component exactness
   alone and cross-wave generated hashes alone are insufficient verdicts.
3. Final C1/2/4/8/16/32/64 matrix with explicit graph tiers and actual row
   counts. P uses8K code input and1 output token; D uses real code and natural
   EOS with bounded output length. Measure common resident windows separately
   from admission/drain/HTTP wall time. Keep inputs fixed across matched arms.
4. Persist inputs, readable outputs, hashes, timestamps, launch configuration
   and summaries outside `/tmp`; prior raw matrix artifacts were lost.

Historical references: AR C32 matrix1033.44resident tok/s; short256-output
AR geometry ABBA1050.80resident/1002.61HTTP tok/s; final strict DSpark MHC
FP32 family1127.48resident mean on512-output ignore_eos protocol. These are
not identical workloads. Old TP4 approximate target1.5k is not a strict goal.

No new performance or correctness result at this initial record.

## Fresh-start regression, first attempt

The first fresh process (PID487458) loaded target and bundled draft weights,
then failed before ready: unified KV initialization registered the upstream
`clear_c4_req_states` callback, but the merged pool implementation lacked it.
No request or speed measurement ran. The failed log is retained separately.

The same merge gap left unified BF16 KV charged with paged-SWA/C4 linear
storage, silently reducing the requested1M pool to745728. Restored request-ring
C4 sizing/clear callback and separated fixed unified SWA/C4 state costs from
per-token compressed KV costs. Retained this fork's actual speculative ring
size and preserved the non-unified V4.1 formula; no attention math changed.

Validation before second launch: C4 lifecycle/budget tests9passed;
`scripts/rocm/check_dsv41_unit.sh`128passed plus16subtests;
new benchmark harness CPU tests3passed. The broader pool-configurator test
file could not collect because the environment lacks `datasets`; it is not
claimed passing. GPU process scan again showed all eight devices idle.
Second fresh startup and E2E are pending.

Second process499788 confirmed full=1048576 on every rank, coefficient6373.86
bytes/token, fixedC1285.05GiB and unified rings2.24GiB; after target/draft
pools about19.7GiB remained per GCD. It then failed at target graph metadata:
V4.1 ratio discovery used `kv_pools`, which is intentionally empty in unified
V4. The backend incorrectly declared C4/C128 absent and dereferenced a None
C4 table. Corrected discovery to use the logical source map only for unified
storage, retaining actual paged allocations for V4.1. Four CPU constructor
regressions plus existing V4.1 suite:132passed and16subtests. No GPU request
has run yet; third fresh startup is next.

Third process506227 captured all target tiers1/8/32, then failed at the same
expression for a different legitimate layout: the bundled DSpark draft is
SWA-only. Added absent-compressed-stream handling (SWA still updated), and
single-ratio/empty-ratio CPU tests. Combined ratio tests, graph-shape audit
tests and V4.1 CPU regressions:144passed plus16subtests. User then narrowed
the task to AR-only, so no further DSpark startup/benchmark is planned.

Next service uses accepted native M32 overlap/gate-prefetch/down-uniform/fixed
warmup/AR4 plus the existing large-prefill throughput profile, pool1M and
explicit graph tiers1/2/4/8/16/32/64. Quant capacity override0 selects the
general grid instead of underallocating C64*Top6 rows. A host-only deduplicated
graph-key audit records actual executed rows separately from request count.

AR process519441 launched on127.0.0.1:30021 in tmux session
`dsv4-tp8-ar-20260913`. Controller `run-ar-pilot.py` waits on this exact PID
and birth time, checks GPU ownership, then runs France and natural-output
C1/C8/C32 warmups plus two short measured waves each. Full seven-tier,
three-round matrix starts only after inspecting that pilot. No new speed yet.

## Native fresh startup passed

At23:42HKT AR process519441 reached ready. Seven target-decode tiers captured,
no target-verify/draft graphs. Requested1M pool retained; before graph capture
about24.6GiB/GCD remained. France returned `The capital of France is **Paris**.`
with completion IDs and no speculative acceptance. Runtime audit confirms
C1 uses key1/executed_rows1/input_rows1, not a padded C32 graph. Initial
request includes cold module/JIT work and is not a speed result.
The existing CK scale-layout CPU oracle also passed all40 mutations across
TP4/TP8 gate/down layouts. Native short pilot and full P/D remain pending.

## Native pilot completed; formal matrix running

Two measured natural-EOS waves (max512 output tokens), excluding warmup:
C1 median81.1169, C8 median335.3845, C32 median1040.9756 resident output tok/s.
C32 individual rates1041.1312/1040.8200. No visible resident regression versus
historical AR. France and all completion-only ID/count checks passed; leading
and trailing text of all32 first-wave code answers was inspected. No obvious
loop/corruption was seen, but truncated512-token source excerpts lead to some
unsupported code-review hypotheses/refusals: this is NOT a factual code-quality
certification or proof of bitwise parity.

C32 whole-request HTTP rates803.75/370.13 differed strongly despite stable
resident rates. TTFTmax4.17/27.54seconds explains the difference; its cause
was not captured and is not assumed to be JIT. Added a read-only CPU compiler
subprocess witness for subsequent work; lack of such subprocesses does not
exclude in-process compilation. Do not mix these HTTP rates with resident D.

Startup repair commitc224e9c1ce; C64 harness/audit commitd111ffa4f7. Both pushed
to gfx90a/sync/media-foundry-gfx90a, plus backup/pre-v4-tp8-regression-20260913.
Full matrix controller started with `--phases prefill decode`, native AR,
seven concurrencies, three measured rounds plus excluded warmup per group,
8K input for P and512input/naturalEOS/max2048output for D. D accumulates30s
of common resident windows per round. PID519441 remains owned service;
artifacts under `.agents/experiments/dsv4_tp8_revalidation_20260913/ar-matrix`.

Completed P medians so far: C14676.39, C24989.25, C45265.43, C85099.23,
C165171.36 input tok/s. C32/C64 and formal D pending. Peak-VRAM samples are
collected at5s cadence (observed samples, not exact allocator peak). Auditor
recomputes throughput from counts/timestamps and rejects count corruption,
cache-hit P, non-native D; benchmark/auditor CPU tests5passed.

### Follow-up evidence for pilot HTTP latency outlier

Retrospective on-disk `.ninja_log` and object timestamps establish four builds
inside the slow second C32 wave (23:44:48--23:45:07HKT): TP8 FP4->BF16 gate
and down dequantizers, plus exactM8190 MFMA gate and down. Build+link times
5.715+5.637+5.863+5.881=23.096seconds, matching the~23.4second TTFT increase.
M8191 variants were compiled one wave earlier. The input manifest contains
511/512-token rows, so dynamic request grouping can expose8190/8191 as new
shapes below the8192 CK threshold even after an8192 warmup. This is supported
evidence of a cold-shape/JIT contribution, not a GPU decode slowdown; no
claim that the later compiler watcher observed those already-finished builds.
Persistent artifacts use build/dependency hashes distinct from old cached
versions. The formal matrix's existing warmup protocol is unchanged; no
measured outlier is silently removed or retimed.

Corpus note: the pilot uses the exact historical32-case decode manifest.
The formal matrix uses the new64-case manifest (first32 cases byte-identical);
P takes the requested leading C rows, while repeated D waves rotate through
the full64 cases. Thus formal C32 D may cover both32-case halves, unlike the
old32-case corpus. Preserve per-case counts and do not interpret that final
table as a strict old/new throughput ABBA; the matched pilot is the regression
screen. All new cases also contain committed public source, not random tokens.

## Formal P completed; long native D in progress (Sep14)

P three-round medians C1/2/4/8/16/32/64:
4676.39/4989.25/5265.43/5099.23/5171.36/5254.98/5250.05 input tok/s.
C64 wave0 took99.86s for524k input tokens; this is a full wave, not a single
M64 prefill kernel. All P cached-token counts are zero.

Formal C1 D median75.7006tok/s. Its longer answers cross the raw2048/C4512
dual-graph boundary. First warmup completed1690tokens naturally; timestamp
segments give81.289tok/s at generated1--512,81.207 at512--1400,81.146 at
1400--1536,54.614 at1536--1690. Launcher log explicitly confirms dense/sparse
graphs dispatch at raw2048. Do not mistake this context-mix effect for the
matched short pilot regressing from81 to75. No output cap/EOS policy changed.
Formal C2 first two waves are96.145/96.119tok/s, with1984/2048 output tokens;
readable conclusions/tests, no evident loop. Remaining groups pending.

VRAM observer switched from CLI watch (PID534376) to append-only JSONL
`watch-vram.py`; serving PID519441 was not interrupted. **Correction at00:33:**
the earlier claim that the old CLI file held only the final observation was
wrong. Although the file is rewritten, its JSON array retains cumulative
history:1272 entries,159 observations/GPU,23:48:31--00:01:46HKT, starting
during P16. Observed old-file peak54100/65520 tool-MB (82.57%, GPU6) at
23:49:31. It covers later prefill, but not P1--P8, and is not an exact
allocator peak. `audit-vram.py` now handles both history and JSONL wrappers;
the original raw history remains preserved. P/D throughput is unaffected.

Formal D completed through C8: C1/2/4/8 medians
75.7006/96.1189/188.1458/333.3611 resident output tok/s. C16 is running;
its runtime audit confirms key16, raw_bs16, executed_rows16, input_rows16.
The JSONL memory observer returns a `gpu_data` wrapper (not a bare list).
Across211 observations so far, maximum used VRAM is54117/65520 tool-MB
(82.596%, GPU3). This is an observed decode sample, not allocator peak or
a reconstructed prefill maximum. Saved130 pilot/early-matrix completion-ID
sequences independently decode to the returned text in130/130 cases.

An explicit post-hoc first-wave context slice (generated32 through at most1400,
input<=512) gives C1/2/4/8 =81.170/109.031/188.862/334.703 tok/s. The C2
same-wave full window was96.145. This supports the context-boundary diagnosis;
the diagnostic does NOT replace the predeclared formal medians or constitute
an independent ABBA. `audit-window-context.py` persists/reproduces that slice.
Tail16-gram repetition screening of88 completed formal C1--C8 answers gave
maximum0.199; the five highest cases were inspected and contain repeated test
scaffolding rather than obvious loops. Their source-based factual assertions
remain unverified. This is not a substitute for a numerical reference oracle.

P C8 three raw rates are4710.94/5099.23/5101.67. The excluded warmup used
chunks36864+28672; the first measured wave used36864+4096+24576. An on-disk
Ninja witness in that first measured interval is
`sgl_kernel_jit_gfx90a_bf16_gemv_3_16160_4096_2_2_8/`
`build-c301d38b774bd4d3/deps-d693b77de2686ee8/.ninja_log` under the persistent
gfx90a JIT cache: compile0--5311ms, link5311--5373ms; object timestamp
2026-09-13T23:47:55.028788HKT and module23:47:55.115173. This establishes a
cold M3 vocab-projection compile during the slow wave, not a steady prefill
kernel regression. Compiler duration is not additive critical-path time;
the measured wave delta is only about1.1s. All three rounds remain recorded.

At00:37HKT formal C16 completed:609.1787resident tok/s median. The C32
warmup was1040.809; its first measured round was1040.874 across four windows
(1040.01/1041.68/1041.53/1040.69). Remaining C32 rounds and C64 still running.
Independent CPU tokenizer verification has passed835 saved completed responses
across25 P/D files; running/incomplete files were excluded. This includes P's
single-token outputs and is an integrity count, not835 long quality judgments.

Provenance: the fresh service launched from patched working-tree source;
startup repairs and graph audit were committed after ready as c224e9c1ce and
d111ffa4f7. The formal controller records d111ffa4f7. Subsequent commits so far
change audit scripts/docs only, not the active model/kernel execution path.
This was not a separate clean-worktree checkout; unrelated existing local
files remain preserved and unstaged.

## C64 exposes a separate long-context cost cliff (baseline unchanged)

Formal C32 three-round median1040.9881resident tok/s. C64 warmup1330.7371
resident versus486.8324whole-wave HTTP tok/s;98976 output tokens over203.306s,
TTFTmax7.772s,43natural stops and21length limits. A5-second timestamp slice
around wave90--120s has34--36fully active requests, generated positions
1554--1640, and only96--105aggregate tok/s. This is more than ordinary batch
drain: input512 plus these generated positions crosses the raw2048 boundary.

Read-only code/process evidence:
- Live INDEXER_MAX_C4_SEQ_LEN=262144, TRITON_INDEXER_FULL=1; default BLOCK_S16.
- PagedIndexerMetadata uses captured page-table capacity (bounded by that cap).
- Full Triton grid is(batch,ceil(max_seq_len/BLOCK_S)):1048576CTAs atM64.
- The original kernel masks invalid K loads but still performs Q loads/dot
  on every non-trivial row's empty tail tiles. AR does not use the prefill-only
  trivial-row optimization. At the boundary only~513--640keys/row are valid.
This is a concrete waste mechanism, but component/causal E2E confirmation
is still pending; do not claim a measured fix or a newly introduced merge bug.

Prepared `scripts/rocm/check_dsv4_indexer_empty_tiles.py`, isolated from serving:
candidate only guards empty tiles, invokes the original kernel for nonempty
tiles, and preserves full-width zero stores. It checks score bits, logical and
physical Top-K, ties/boundaries,100mutations,1000graph replays and component
ABBA. Only Python syntax has been checked. No GPU test or production change
has happened. Finish the unchanged baseline C64 matrix before GPU experiments.

Compiler witness also observed new exact-M7679 MFMA gate/down builds at
01:18:47--01:18:56 during C64 admission. Those are separate cold-prefill/JIT
events; they cannot explain the much later >2048-token sparse decode slowdown.
