# TP8 two-wave decode attention E2E — ABBA rejected

Do not promote. Full ABBA found no E2E benefit. Chronological progress notes
below are retained; the final outcome section supersedes their pending status.

Base426f1d0d3e. Default-off flag:
`SGLANG_DSV4_GFX90A_TP8_DECODE_ATTN_WARPS2`.
Selector requires native DSV4 decode scope, TP8/attentionTP8/EP1, H8/D512,
M1 or M32, BF16 Q, nonquantized unified KV, split>1, blockH16/blockK16,
no oracle override and no fused inverse RoPE. Prefill and speculative excluded.
`check_dsv4_tp8_attention_scope.py` executes the actual selector AST with mocks;
M1/M32 selection, excluded shapes/dtypes/TP/EP, prefill/spec and cleanup pass.
Initial test-source multiline-with typo was fixed before launching any service.

## Important actual-hit finding

Existing service has `SGLANG_DSV4_GFX90A_FUSE_ATTN_INVERSE_ROPE=1`.
M1 does NOT hit this candidate: its fused inverse-RoPE path is deliberately
excluded. Nonfused M1 micro savings cannot be claimed for the actual service.
All8 ranks log M32 two-wave selection with split4. C1 is a non-regression arm.

Candidate process3159536: graph0.65GB/GCD, rank0 free15.64GB,
max_total_num_tokens1048576, unchanged from baseline. First C1 block completed:
France passed, six256-token code outputs exactly matched existing reference.
Case medians84.35154/84.33960/84.18249tok/s, not a final ABBA result.

## Live controller / artifacts

Controller command:
`run_dsv4_rowstable_abba.py --pid 3130584 --candidate-flag
SGLANG_DSV4_GFX90A_TP8_DECODE_ATTN_WARPS2 --output
/tmp/dsv4_tp8_attention_warps2_abba_20260908.json`.

Root exec session75545. Revalidate this handle and state JSON before taking any
action; do not restart on an observation timeout. Controller switches A/B/B/A
using fresh processes for switches, B/B reuses its process. Each block runs C1
reference checks then six32-distinct-code256-token waves. It performs AMD-SMI
ownership checks and preserves NUMA interleave, weights, other flags and1M pool.
At last inspection block0 was running C32, process3159536 alive.

Continuation update: block0 completed, controller switched to baseline
process3166974, block1 completed and block2 started. First pair warm C32
candidate978.82523/resident1023.96636 versus baseline983.87064/resident1030.35199.
All six fixed-prefix output IDs, input logprobs and output top-logprob rows
match across the pair. Candidate is about0.5% slower in this pair; await full
ABBA, no promotion. Root session75545 remains the same controller.

Prepared (not yet GPU-tested) `--inverse-rope` extension to the standalone
attention geometry oracle so future C1 work can test the actual fused reducer.
Do not run it concurrently with the service ABBA. Summary script now accepts
this candidate flag and requires all six teacher-forced comparisons exact;
regression run against old complete shared-gate ABBA artifacts passed.

State JSON is authoritative. Per-block artifacts append `_0_c1.json`,
`_0_c32.json`, etc.; service logs append `_service_0.log` etc. Controller leaves
the last candidate service running if all blocks succeed, and stops on failure
without automatic restart. Needs final review before selecting production state.

## Still required

- Finish ABBA; compare warm C32 and resident rates (discard first wave), C1
  medians and all fixed-prefix logprobs, actual hit logs and memory each service.
- France C32 and long diverse-code correctness; if no E2E gain restore flagoff.
- Do not infer large-context/fused-RoPE correctness from nonfused micro tests.
- Commit/push scoped code and final evidence only after review; unrelated dirty
  worktree changes belong to other work and must remain unstaged.

## Final ABBA outcome

Four blocks completed; root controller75545 exited successfully. Final summary
is in `dsv4_tp8_attention_warps2_abba_summary_20260908.json`.

| Metric | Baseline geometric mean | Candidate geometric mean | Change |
| --- | ---: | ---: | ---: |
| C1 tok/s | 84.24720 | 84.21975 | -0.0326% |
| C32 warm E2E | 984.24028 | 980.89189 | -0.3402% |
| C32 warm resident | 1029.96982 | 1026.87044 | -0.3009% |

Candidate blocks978.82523/982.96292; control983.87064/984.61005.
Discarded first C32 wave of each block; B/B shared a process. No statistical
confidence claim. Both independent candidate services failed to beat controls.

All24 C1 measured256-token sequences matched reference and recorded hashes;
24 teacher-forced fixtures matched output IDs, input logprobs and output-top
logprobs across blocks. All768 C32 requests finished length256 and their saved
completion hashes were recomputed. Cross-round exact requests7/7/6/6 of32,
so this is not full C32 bitwise parity. No long-output promotion test performed:
the candidate already failed the performance gate and stays default-off.

The whole-kernel micro benefit did not survive graph/service scheduling. The
source of that gap is not proven by these results; do not assert a cache or
occupancy mechanism without a matching trace. No workspace was added:
all services retained1M pool and0.65GB/GCD graph allocation.

Restoration job root session95928 stops only final candidate3175344 and its
verified child tree, removes the candidate flag, clones the same parameters
and NUMA interleave, waits readiness without automatic restart, and runs France
C32. Restore log `/tmp/dsv4_tp8_attention_warps2_restore_20260908_service.log`;
result `/tmp/dsv4_tp8_attention_warps2_restore_20260908_france.json`.
Revalidate this job before assuming baseline is ready.

Restoration confirmed: job95928 completed successfully. Baseline3184375 is
ready with flag absent; post-restore France C32 passed32/32. No long-code test
was needed to promote this rejected candidate; it remains a default-off oracle.

## Separate C1 fused inverse-RoPE oracle

After ABBA ended and baseline France finished, AMD-SMI again confirmed only
the baseline service owners. Ran GPU4-only geometry oracle with `--heads 8
--tokens 1 --contexts 128,256,512 --inverse-rope --ragged-check`.
Full100 mutations for all9 profiles and100 ragged/empty/position/Q/KV/sink
mutations plus1000 replay checks for2wave/2stage passed bit-exact.
Synthetic interleaved cos/sin frequencies exercise the fused reducer layout;
this is not a captured model-Q fixture or a new model correctness claim.

| KV/query | 4wave/2stage us | 2wave/2stage us |
| --- | ---: | ---: |
| 128 | 17.080 | 15.111 |
| 256 | 17.140 | 15.137 |
| 512 | 17.380 | 15.367 |

Raw samples in `dsv4_tp8_c1_fused_rope_warps2_20260908.json`.
This warrants a distinct C1-only selector/ABBA next, NOT re-enabling the failed
C32 setting. Current production integration still excludes fused inverse RoPE;
baseline3184375 is unchanged. CPU scope test passed again after these checks.
