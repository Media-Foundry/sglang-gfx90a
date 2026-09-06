# Current-main gfx90a MHC boundary recovery (2026-09-06)

## Scope

Base HEAD `aa73405166`, current `sync/media-foundry-gfx90a` mainline working
tree. NOT an old worktree or a new rebase. Existing dirty attention/model/CK
experiments retained; they were identical across timing arms. No weights,
FP4 kernels, attention kernels, all-reduce modules or scheduler code changed.
Only runtime change: restore the gfx90a exception in the canonical MHC
cross-layer enable predicate. The native/Triton gfx90a boundary does not
require standalone TileLang pre/post to be enabled.

The old local predicate still had the exception but was no longer called
after upstream `d315eb7250` moved consumers to
`is_cross_layer_mhc_fusion_enabled`. This is a lost dispatch connection, not
a reason to revert all later optimizations or enable unsupported standalone
TileLang MFMA kernels.

Existing current-main `apply_mhc_post_pre_boundary` already forwards
`global_batch_size`, `fn_bf16`, `fn_fp16`, norm weight and norm epsilon.
Those later contracts were preserved. No numerical kernel was rewritten.

## ABBA: four independently started services

TP4/EP1/no-A2A, original model `/home/pc/models/modelscope`, physical GCDs4--7,
port30011 (loopback), native AR, graph BS1, pool8192, memory0.80,
chunk2048. Current DS environment and shared AIter/AOT modules unchanged.
AMD-SMI resource check before each service; no competing GPU processes.
Each arm froze GC once and warmed every measured shape before measurement.

A = `SGLANG_OPT_FUSE_MHC_POST_PRE=0` (equivalent disabled boundary to the
pre-fix default); B = `=1` with the repaired canonical selector.
No other SGLang runtime consumer of that flag changes these arms.

Each arm: three real code prompts, one excluded warmup per prompt, then
three rounds of 256 completion tokens per prompt. Fresh cache salts, fixed
input IDs, greedy, ignore_eos=true, sequential C1 HTTP wall timing.

| Arm | Python linked list | SQL duplicate emails | Merge sorted arrays |
|---|---:|---:|---:|
| A1 | 54.28 | 53.49 | 54.25 |
| B1 | 76.59 | 76.32 | 75.06 |
| B2 | 75.96 | 72.44 | 74.82 |
| A2 | 53.71 | 54.78 | 54.74 |

Units: HTTP completion tok/s; entries are three-request medians.
Across 18 measured requests per variant:

- Control median **54.262**, candidate median **75.478 tok/s**: **+39.10%**.
- Drop two lowest and two highest samples: control mean53.992,
  candidate mean75.188 tok/s.
- Additional B1 historical 2+2 probe: 71.251 / 76.790 / 75.842 tok/s,
  hash `f0148c9486ae8e27` stable. This is not the historical old-tree hash;
  current-main numerical paths have evolved.

Raw per-request timings/hashes and fixed-prefix comparisons are committed in
`dsv4_mhc_recovery_abba_20260906.json`. Rates are NOT GPU-only decode,
not speculative accepted-token throughput, and not concurrent aggregate.

## Correctness: what passed and what did not

- Unit suite: 14 passed, 13 subtests passed; two gfx95-only GPU tests skipped.
  New policy table covers gfx90a fuse off/on independently of pre/post flags.
  Existing non-gfx90a and gfx95 policy coverage retained. Updated stale
  mocked-layer fixtures to disable GPU markers and provide batch_size.
  Added explicit test that an already-normalized fused result is not
  normalized twice; fallback still closes the previous deferred post first.
- France nine-token sentinel: twice per independent ABBA service, exact
  `[671,6102,294,8760,344,2619,51119,42499,1]`.
- Every C1 code case has stable completion IDs across rounds AND independent
  restarts within its variant. The B linked-list hash also matches the old
  pre-rebase reproduction. SQL and merge hashes differ from the old tree.
- Six teacher-forced probes: same A1 completion prefixes of32/128 tokens
  appended to the same three code prompts in every arm. Next-token IDs
  agree A1/B1 for6/6; output top20 overlap17--19/20. Input token logprobs
  are finite, but A1/B1 are NOT bit-exact: per-probe mean absolute delta
  0.0149--0.1303, maximum single delta **2.6359**. This is not a claim of
  negligible logit error or full-model numerical parity.
- B1/B2 input token logprobs AND output top20 values are exactly identical
  in all six fixed-prefix probes. This rules out observed cross-restart
  drift for these probes, not for every scheduler/context boundary.
- B2 2304-token real source prefill:1.116 /1.022 /1.028 seconds, same first
  token39111; A2:1.188 /1.097 /1.076 seconds, also39111. Small non-ABBA
  smoke sample only, not a new prefill performance claim.
- Concurrent smoke completes, but full hashes can differ across rounds in
  BOTH A and B with dynamic batch arrivals. Do not claim concurrent bitwise
  parity. Initial B2 smoke forced France beyond EOS; its post-EOS text is
  not a quality oracle. The harness now stops France at nine tokens while
  code requests continue to64, exercising a batch-size drop.

These tests support a C1 native-AR speed recovery and semantic/repeatability
smoke acceptance. They do not constitute a full code-quality evaluation,
long-context oracle, or DSpark/TP8 performance certification.

### Default graph / chunk integration check

A fifth candidate service used the unchanged harness defaults: decode graph
tiers **1/2/4/8**, chunk2304, TP4/EP1, pool8192, memory0.80. All four tiers
captured successfully. Initial one-round code samples were69.25/69.12/71.81;
a subsequent full three-round warm repeat gave medians
**74.92 /74.89 /75.37 tok/s** for the same three cases. All per-case C1
hashes match B1/B2. This integration check is separate from the BS1 ABBA.

France and the short heterogeneous concurrent smoke pass the sentinel and
completion checks with an early France EOS / batch-size drop. Full C4 hashes
still vary, as in the unfused control; no concurrent bitwise claim.

Important outstanding issue: with chunk2304, the repeated source-prefill
first IDs were **201 /39111 /32111**, unlike the chunk2048 smoke above.
The September5 pre-fix audit already recorded first-word changes ("Based",
"I'll", "Looking") for this same2304-token input. Thus nondeterministic
large-chunk prefill was already observed; this test does NOT isolate its
cause or prove it is unaffected in magnitude. Keep it as an open correctness
issue, not a success claim. Do not discard later prefill optimizations as
part of this C1 boundary repair; evaluate that issue separately.

All test services stopped after validation. Isolated historical worktrees
from the prior reproduction remain available and production files were not
replaced with their old versions.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 EP_SIZE=1 MOE_A2A_BACKEND=none \
PORT=30011 MEM_FRACTION_STATIC=0.80 MAX_TOTAL_TOKENS=8192 \
CHUNKED_PREFILL_SIZE=2048 CUDA_GRAPH_MAX_BS_DECODE=1 \
SGLANG_OPT_FUSE_MHC_POST_PRE=1 scripts/rocm_dsv4_flash.sh serve

/home/pc/anaconda3/envs/DS/bin/python scripts/rocm/bench_dsv4_c1_mhc_recovery.py \
  --arm B1 --reference /tmp/dsv4_mhc_restore_A1.json \
  --output /tmp/dsv4_mhc_restore_B1.json
```

For A1, set fuse=0 and omit `--reference` so A1 creates the fixed-prefix
corpus; pass that same A1 JSON to B1/B2/A2. Restart services for each arm.
Use `--smoke-only` for source-prefill and concurrent completion checks.
Local raw JSON/logs: `/tmp/dsv4_mhc_restore_*` (full text, IDs and logprobs).

Checkpoint identity (unchanged): config SHA256
`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`;
tokenizer SHA256
`8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`;
safetensors index SHA256
`98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`.
