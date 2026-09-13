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

## Completed baseline matrix and exact empty-tile candidate

All seven native AR P/D groups completed, three measured rounds each:

| C | P input tok/s (8K) | D resident tok/s (512 input, natural EOS, max2048 output) |
|--:|--:|--:|
|1|4676.39|75.70|
|2|4989.25|96.12|
|4|5265.43|188.15|
|8|5099.23|333.36|
|16|5171.36|609.18|
|32|5254.98|1040.99|
|64|5250.05|1330.46|

All1859 saved warmup/measured responses pass independent ID->text decoding;
this includes P's one-token outputs. Post-matrix France twice answered Paris.
Three8K code prompts,128output cap, repeated twice: readable and on topic,
no visible collapse, but different texts across repeats and no factual/bitwise
whole-model claim. Original1M capacity and seven actual runtime graph tiers
were confirmed. Baseline observed VRAM maximum82.596%/GCD; the observer now
excludes timestamps after the completion-state mtime so later GPU oracles do
not contaminate that peak. `measurement-times.json` records this boundary.

Isolated then integrated empty-tile oracles passed7shapes (short-live/capacity,
ragged577, dense8192, full262144C4keys and trivial-row mode),100mutations each.
The integrated run compares score bits and logical/physical IDs after EVERY
one of1000graph replays/case, with independent output storage. For live640 and
capacity262144, integrated full logits+TopK component medians:
M1 230.88->40.06us; M32 6934.30->837.17us; M64 13903.65->1691.02us.
Fully populated8192/262144 controls were effectively flat. These are single
GPU0 components, not service speedups.

Production opt-in added in ab7739fc06:
`SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP`, defaultFalse. It is gated to
HIP/gfx90a unified original-V4 DECODE with known non-draft/non-DSpark/non-MTP
roles. Prefill, paged V4.1 and speculative paths stay unchanged. The new Triton
wrapper invokes original arithmetic for each nonempty tile; empty tiles still
write full-width zeros. No KV truncation, weight change or Top-K change.
Six CPU selector/wrapper contract tests passed.

Service ABBA started separately using the SAME32 public8K source prompts,
natural EOS, max256output: control A1 resident169.6567tok/s. This directly
exposes the long-context cost, unlike the short-context resident matrix.
After A1 completed, original service519441 and owned children were stopped
cleanly. Candidate B PID576055 (tmux dsv4-empty-tiles-B-20260914) is loading;
same full launcher with only the new flag enabled. No candidate E2E result yet.

B completed: native long8K C32 resident B1=611.1026 and B2=610.8532tok/s,
versus A1=169.6567. A1/B1 each produced8192tokens (all32x256), same8097
resident tokens; resident47.726->13.250s. Whole request wave98.486->63.580s,
HTTP83.179->128.846output tok/s. This is not the short-context1041 profile;
do not multiply the short-context rate by3.60. France passed in B. Sampled
code texts remain coherent but do not have verified factual diagnoses.

B PID576055 was stopped cleanly. Return control A2 PID583304 is ready with
flag0 and no empty-tile hit log; pool1M retained. Its warmup/formal wave are
running, so ABBA acceptance is still pending. No default launcher promotion
has happened yet. The baseline table remains separately preserved.

## Accepted ABBA and final native matrix rerun

Return control A2 completed at169.8429 resident tok/s, reproducing A1.
Final sequence A1/B1/B2/A2=169.6567/611.1026/610.8532/169.8429 on the
same32 public8K source requests. The roughly3.60x gain applies to this
long-context resident window, not the short-context baseline. No KV capacity,
weight precision, selected IDs or nonempty-tile arithmetic was changed.

The validated TP8/EP1/no-A2A multi-request launcher now defaults the exact
empty-tile guard to1, preserving explicit0. Global EnvBool remainsFalse;
runtime guards still exclude prefill/draft/DSpark/MTP/paged-V4.1. Seven CPU
contract tests (including profile default/override), shell syntax, Python
syntax and git diff whitespace checks passed. Final service will be launched
without an explicit flag to verify this actual default. It will repeat all
seven native AR D concurrencies, three rounds each, unchanged real-code
natural-EOS protocol. Completed P measurements are retained because this
selector cannot enter prefill. Final D results are not available yet.

## C1 configuration hole found during final audit (must close before handoff)

While final D proceeds, direct source/process inspection found the historical
accepted TP8 BS1 wo_a GEMV missing from this matrix launcher. It is NOT now a
default: `SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV=EnvBool(False)` and Final
PID591550 has no exported value. BF16_ATTN_LINEAR and WAVE64_GROUPED_GEMV
are1, but single-group TP8 still falls back without the separate authorization.
Current native C1 median79.5746 is thus not yet the best historical configuration.
Historical `dsv4_tp8_bs1_final_20260907.md` reports a~10% candidate gain on a
different short256-token/131072-pool benchmark; do not copy that rate here.

Finish the current matrix without mid-run changes. Then revalidate the existing
component, do a C1-only fresh-service ABBA on fixed real code, and if accepted
repeat C1's same three-round naturalEOS/2048-output protocol. Other resident
tiers and P cannot select this BS1/native-decode-only shape. Keep separate
provenance and do not claim whole-request drain times unchanged by a C1 kernel.
This candidate changes reduction arithmetic (historical tests not bit-exact);
it needs France and long-source smoke in addition to timing. No new kernel or
default promotion has been made for this C1 direction at this point.

## Seven-tier default-indexer matrix completed

Final PID591550 (production launch3535a3d76f; controller edb52d5a8a) completed
all seven D tiers with three measured rounds each. C1/C2/C4/C8/C16/C32/C64:
79.5746/109.9393/188.7103/334.1751/608.2312/1044.3250/1334.2383 resident
tok/s. Whole-wave HTTP rates respectively78.66/102.41/148.50/254.45/422.51/
680.61/848.14. Baseline vs final is sequential, not per-tier ABBA; C1 remains
the GEMV-OFF control pending the explicit configuration-hole recheck above.

All1367 final warmup/measured responses across14 files pass independent
completion-ID/text decoding. Post-matrix France twice correctly answers Paris;
three8K source prompts repeated twice complete at128output cap, coherent and
on topic but not a factual code-audit oracle and not identical wording.
All requested graph tiers observed, including raw_bs64/executed_rows64.
Final D observed VRAM peak45773/65520 tool-MB=69.861%/GCD, sampled5s and
bounded by main-matrix completion time. Large P's separate observed peak stays
82.596%; neither establishes an exact allocator peak or1M-filled-context test.

Final C64 warmup resident1336.861 versus whole-wave HTTP862.635tok/s. Post-hoc
five-second slices at90/100s still show550.0/498.4 aggregate tok/s with30/23
fully active requests at generated1613/1798, unlike the old~100tok/s cliff.
These are drain diagnostics, not replacements for the formal resident table.

C1 existing component recheck on physicalGPU0 after all matrix/quality traffic
ended: einsum30.7465us versus groupedG1 GEMV6.8771us.100mutations finite,
100/100stable repeated-replay checks;70/100fully bit-equal to einsum, largest
absolute difference0.5 on unscaled random weights, max relativeL2 toFP32
0.00178775. Same numerical character as the historical candidate, not exact
einsum arithmetic. Two CPU shape/predicate tests passed on current source.
New C1 A1 control79.5580resident tok/s reproduces main-matrix79.5746.
France passed. Candidate B is next; no C1 E2E speedup claimed yet.

## CLOSED: best validated native AR profile and final measurements

C1 wo_a ABBA A1/B1/B2/A2=79.5580/87.5832/87.5043/79.1513resident tok/s.
Mean control79.3547, mean candidate87.5437, gain10.3195%; control return-0.5112%,
B spread0.0901%. B1/B2 completion IDs identical for both whole natural answers
(1453 and1887tokens). Those source responses are coherent and acknowledge
truncated evidence, but contain speculative/incorrect diagnoses; they are NOT
verified code fixes. Do not execute/apply their proposed patches as evidence.

Candidate C1 separate formal three-round median87.5990; IDs->text validation
passes all7 warmup/measured responses. France all three arms passes. Two
repeats of France plus three8K source prompts complete (128output cap on code),
with readable wording, not whole-model bitwise/factual parity. A subsequent
C32 native transition smoke completes2x32x64outputs and passes64 ID/text
checks; its~1.88s common window is too short to replace the formal C32 result.
No C1 kernel implementation was changed: this recovers an omitted existing
opt-in, while leaving global defaultFalse. `start-best-ar.sh LABEL` is the
resource-guarded reproducible native-only profile with the C1 flag explicitly1.

Final table (three measured rounds per reported cell):

| C | P input tok/s,8K | D native resident tok/s | D whole-wave HTTP tok/s |
|--:|--:|--:|--:|
|1|4676.39|87.60|86.39|
|2|4989.25|109.94|102.41|
|4|5265.43|188.71|148.50|
|8|5099.23|334.18|254.45|
|16|5171.36|608.23|422.51|
|32|5254.98|1044.32|680.61|
|64|5250.05|1334.24|848.14|

P retains the completed main matrix: both new decode guards exclude prefill.
C1 is the accepted GEMV-on supplement, other cells are the seven-tier main
matrix. Their whole-wave HTTP drain rates were NOT retested GEMV-on; no claim
of identical all-tier full-request latency under the combined launcher.
Native-only, original checkpoint, TP8/EP1/no-A2A,1M logical pool, seven graph
tiers. D512input/naturalEOS/max2048output, >=30s resident per measured round.
No DSpark/M128 throughput. Historical6.42k P uses2304inputs/smaller pool and is
not the same workload; no filled1M-context stress or exact whole-model oracle.
Cold exact-M prefill JIT remains a separate latency hole, documented above.

Final test services591550/615112/624081 and owned children stopped cleanly;
memory observers exited. Closing amd-smi reports no running GPU processes on
any of0--7. V4.1 remains frozen; no new V4.1 service was launched.
`RESULTS.md`, `final-report.json`, `c1-woa-acceptance.json` and the verified
raw-evidence archive provide the final provenance. Existing unrelated files,
including cuda_graph_runner_memory_usage.pickle, remain unstaged/preserved.
