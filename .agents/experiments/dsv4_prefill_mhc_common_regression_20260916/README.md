# Common large-prefill MHC: 8K/16K regression

Parent32K checkpoint34fa7b0e9c:9605.065395 input tok/s, exact against the
new full-paired FP32/20 reference. This follow-up tests whether the same policy
preserves the already-accepted shorter-input behavior and speed.

Both arms: originalV4/nativeAR/TP8/EP1/no-A2A, original weights,1M logical KV,
32Kchunk,C16 real heterogeneous code requests, zero prefix hits. Exact HIPpost,
paired pre-mix owner, H16 peer, unique Set/vec4 CK, direct-row dequant and wide
indexer owner enabled. Query producer compaction and MFMApremix disabled.
Only SGLANG_DSV4_PREFILL_MHC_COMMON_FP32 differs (A=0,B=1).

8K131069 and16K262141 input tokens reuse checksum-verified historical fixtures.
Each length independently runs fresh A1/B/A2 processes; B1/B2 share the B
process. One unscored warm wave, three scored waves/leg, four128-token quality
waves/process, and the same A1-frozen64-token teacher continuation per length.
Numerical comparisons never assume equality of empty output arrays. Exclude
exactly one leading null logprob/request (1008 valid positions).

This is a regression test, not a promised speed improvement. Acceptance requires
stable controls, no material speed regression (2% screen threshold), unchanged
observed continuations/logprobs, intended all-rank paths, unchanged1Mpool and
source hashes, owned cleanup. Analyzer records evidence before acceptance; a
complete report alone is not acceptance. No global default promotion or claim
of universal batch invariance, tail/decode precision restoration or model accuracy.

Run with /home/pc/anaconda3/envs/DS/bin/python run.py from repository root.
No concurrent GPU work or runtime/source edits during the trial.
