# DeepSeek-V4 DSpark correctness audit (2026-08-31)

## Scope

- TP4/EP1/no-A2A on physical GCDs 4,5,6,7.
- Original DeepSeek-V4-Flash checkpoint.
- Greedy, concrete 32-request coding workload with the France prompt as row 0.
- Native target graphs captured at BS1/32; DSpark gamma is 5 and verify width is 6.

## Confirmed CPU sequence-length bug

`DraftBlockProposer._run_forward()` pre-added the five draft query rows to
`seq_lens_cpu`, while the HIP DSV4 TARGET_VERIFY metadata builder also adds its
effective width.  The CPU mirror therefore described prefix+10 while the device
length remained prefix+5.  Passing prefix lengths to the backend and counting
the five query rows only in `seq_lens_sum` changed the real BS32 greedy result:

- old causal draft: mean accepted length about 1.33--1.43, scheduler about
  186--192 tok/s;
- with the CPU-length fix: mean accepted length 1.60--1.83, scheduler up to
  240.8 tok/s.

Unit coverage: `test_dspark_draft_cpu_lens.py`; together with the eager embed
tests, 3 tests pass.  This is a real bug fix, but it does not by itself restore
target-equivalent output.

## Official full-block attention semantics

The checkpoint reference in `inference/model.py::DSparkAttention` makes every
one of the five draft queries attend the same set:

```
all 128 physical SWA ring slots + all five current draft-block KV slots
```

It is not causal inside the draft block.  The SGLang draft runner previously
used generic causal TARGET_VERIFY streams.  A default-off experiment adds a
draft-backend-only full-block SWA stream.  The unified pool already reserves
`sliding_window + gamma = 133` ring slots, so this does not alias the five extra
positions with committed history.

With the CPU-length fix plus full-block draft attention and folded proposal:

- round 0: accepted length 1.957, aggregate 193.3 tok/s, France contains Paris;
- round 1: accepted length 1.442, aggregate 174.2 tok/s, France is not semantic;
- only 1/32 requests are token-exact across the two rounds.

Therefore the full-block semantic correction is promising for the proposal,
but the current run is not a correctness or performance checkpoint.

## Folded accept is not the whole correctness problem

Keeping the two fixes above and setting `SGLANG_DSPARK_FOLDED_PROPOSAL=0` makes
accept/finalize run outside the proposal graph.  A 32-request/64-token run still
fails the France oracle:

- mean accepted length 1.347;
- scheduler 197.3 tok/s;
- resident window 142.0 tok/s;
- France output diverges at token 3--4 and does not contain Paris.

Thus the graph-folded accept/commit path is not the base correctness fault.

## Concurrency audit

The scheduler does batch the 32 requests; it does not execute 32 independent
model forwards serially.  Within one speculative step, proposal, target verify,
and accept/commit are necessarily serial dependencies.  Capturing only BS1 and
BS32 means all raw BS2--31 seams pad to BS32 (draft M160 and target-verify M192),
which wastes substantial work during admission and retirement.  This explains
bad seam throughput but not incorrect stable-BS32 tokens.

The client start barrier is not a server admission barrier.  Logs show the 32
requests can arrive as several prefill batches.  Once correctness is restored,
compare graph tiers `1 32` against `1 4 8 16 24 32` and report a long common
position window rather than whole-request throughput.

## Next decisive oracle

Force `correct_len=0` for every request (the existing
`SGLANG_SIMULATE_ACC_LEN=1.0` diagnostic).  Each step then emits only target
verify row-0's bonus token:

- if it matches native AR, the remaining fault is draft acceptance/commit;
- if it still diverges, target-verify row-0 logits or target-cache metadata is
  already different from native decode.

The oracle was run and selected the second branch.  With graphs enabled it is
two-round deterministic but emits
`[671,6102,294,8760,734,11111,1,...]`; native AR on the same HEAD, prompt,
TP4 profile, and GCDs is two-round deterministic at
`[671,6102,294,8760,344,2619,51119,42499,1,...]`.  Native AR passes the France
oracle and sustains about 58.37 tok/s after the first token.

Disabling all decode CUDA graphs leaves the bonus-only DSpark token sequence
unchanged.  Therefore the base fault is not CUDA-graph replay or its folded
epilogue; it is in eager TARGET_VERIFY semantics/cache handling.  The next A/B
is unified-KV versus the legacy Triton/FlashMLA cache path under this same
bonus-only oracle.

That A/B isolated the backend: legacy Triton/FlashMLA passed the France first
nine tokens, while unified-KV failed.  Static inspection then found the exact
fault in `MQALayer._forward_prepare`: for unified TARGET_VERIFY the fused
Q/K-norm+RoPE kernel intentionally receives no cache destination and leaves the
normalized BF16 candidate KV for the backend store, but the function
unconditionally replaced `kv` with `None`.  The backend consequently treated
the store as already completed and read stale/uninitialized verify ring slots.

Preserving `kv` only for unified TARGET_VERIFY, plus a fail-loud assertion at
the backend handoff, restores the bonus-only eager France oracle in two rounds:

```
671 6102 294 8760 344 2619 51119 42499 1
```

The two runs are token-exact.  This is the first confirmed numerical/correctness
fix in this audit.  Graph-enabled bonus-only and normal-accept service tests are
still required before enabling/committing the full DSpark path.

Both follow-ups pass the France oracle:

- graph-enabled bonus-only: two identical runs, France first nine exact;
- graph-enabled normal acceptance: two identical single-request completions,
  France first nine exact, accepted length 1.68--1.78.

The 32-request, 128-token diverse-code run also passes France in both rounds.
Its first round reaches accepted length 2.23 and scheduler 195.3 tok/s; the
second reaches 1.36 and 185.4 tok/s.  Only 6/32 full completions are bitwise
identical across the two independently admitted batches, so this is a semantic
correctness checkpoint rather than a deterministic-inference claim.  The
remaining batch-shape/arrival-order sensitivity should be handled separately;
it does not invalidate the missing-KV-store fix.

Do not resume performance tuning or commit the full-block selector as enabled
by default until this oracle and France correctness pass.
