# Fixed-continuation target logprobs for owner-Q producer

Progress on the active drift/C16 goal. No production changes this turn and no
new throughput result: accepted checkpoint remains7953.956103 input tok/s.
Producer remains default-off, native V4/TP8/EP1/original weights/1M logical KV.

## Fixture and alignment

Six numerical waves in fresh A1/B/A2 processes (two per process), plus one
separate fresh B-check wave. Same16 real code-analysis tasks. Original raw
chat encoding was verified byte-for-token against the saved benchmark:
`<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>`.
Tokenizer has no chat_template; apply_chat_template raised ValueError during
inspection, so the verified original explicit encoding was retained.

Code tails were shortened by whole lines (task and closing code fence retained)
until base prompt <=7936 tokens, then the SAVED original-control256-token
continuation was appended. Combined lengths:
8184/8189/8178/8182/8192/8183/8185/8180/8187/8185/8185/8181/8174/8191/8180/8192;
total130948. These are not the original131069-token performance prompts.
Continuation is off-policy for the slightly shortened source excerpt; this
is a numerical counterfactual, NOT held-out perplexity or accuracy evaluation.

`logprob_start_len=base_len-1` deliberately requests the last base token's
null entry plus256 continuation entries. Inspected scheduler processing
prepends None and removes the sampling-token entry. Each response was checked
for257 entries, exact token-ID alignment, first null,256 finite target values,
and20 finite top-logprobs per scored position. All request echoes exact,
zero prefix-cache hits, exactly one newly sampled token (not used as a label).
96 numerical request echoes/24576 scored positions; another16/4096 in B-check.

## Real candidate hits, not a wider-context fallback

All8ranks hit the respective owner/producer path each wave, with identical
observed full/local-row/width signatures across all A1/B/A2 waves:

32733/3088/2048,32740/3072/2048,32738/3088/2048,32737/3072/2048.

The numerical candidate had check=0, i.e. actual optimized execution without
extra full-Q diagnostic work. A separate B-check process used check=1 and
passed Q/folded-weight/scores/logical+physical-ID equality against SAME-BLAS
full Q on all admitted calls for these irregular shapes. Its values are not
included in numerical A/B statistics. All four processes stopped cleanly.
These logs prove executed sizes, not a complete internal request-row trace.

## Results: candidate drift exceeds baseline noise, but mean NLL change is small

Each comparison has4096 aligned fixed-token positions. NLL increase is right
minus left; positive means worse likelihood of this fixed continuation.

| Comparison | Mean NLL increase (nat/token) | Mean absolute target-logprob delta | Top1 agreement |
|---|---:|---:|---:|
| A1 repeat | +0.00003996 | 0.003168 | 99.8535% |
| A2 repeat | -0.00017312 | 0.001722 | 99.9268% |
| A1.0 → A2.0 | +0.00028573 | 0.002726 | 99.8291% |
| B repeat | +0.00007708 | 0.000899 | 99.9268% |
| A1.0 → B.0 | +0.00121072 | 0.020699 | 98.5840% |
| A1.0 → B.1 | +0.00128781 | 0.020836 | 98.5107% |
| A2.0 → B.0 | +0.00092500 | 0.020617 | 98.5107% |

A1.0 mean NLL0.21581607, B.0=0.21702680. Largest per-case mean increase is
case12:+0.01152218. Effects do not uniformly worsen every request.

For A1.0→B.0, absolute-logprob delta median0.00020048, p95=0.112658,
p99=0.227680, max1.554009. Top20 overlap mean96.4868%. There are58 top1
changes versus7 between independent controls. Largest left-margin flip is
case12 continuation position0: margin1.375 nats, token5→39111. Therefore
DO NOT describe all changes as cutoff ties or insignificant last bits.

Control itself has tail variation: repeat max absolute delta0.59610 and
cross-process max0.46790. It remains non-bitwise even with identical request
tokens and observed batch sizes. Candidate repeat is more stable in this
small fixture, but that is not proof of global model determinism or repair
of previously reproduced CK atomic-stage2 drift.

Top1 metric resolves exact top-logprob ties by token ID; top20 is a limited
view, not full-distribution KL. Shared-token logprob deltas, full flip lists,
per-case NLL and raw API responses are retained. CPU metric test checks NLL
sign, counts, overlap and controlled top1 changes;1 test passed.

## Interpretation and next action

Identical request tokens and fixed continuation eliminate autoregressive
trajectory divergence as the reason for these paired logprob differences.
The separate full-Q checked run found no compact/full mapping discrepancy on
the actual irregular shapes. Results are consistent with the already observed
BLAS arithmetic → FP8 requantization → sparse membership → downstream-logit
changes, not proof that every remaining source of drift has been isolated.

Average fixed-continuation likelihood change is modest, while tail and greedy
choices visibly change. Combined with the prior coherent256-token outputs,
this supports continuing to an EXPLICITLY EXPERIMENTAL throughput screen under
the user's small-drift tolerance. It does not certify equal answer accuracy,
unbiased quality, bit-equivalence or justify default promotion by itself.
Do not silently redefine old output hashes as the target of this candidate.

Next: noninstrumented original131069-token C16 service ABBA to determine
whether any useful end-to-end gain exists. Keep producer default-off pending
that measurement and the numerical/quality tradeoff. Do not use timing of
these logprob-heavy requests (extra LM-head rows/top20 extraction) as P speed.

Directory `.agents/experiments/dsv4_c16_owner_producer_teacher_20260915/`:
build/run/run_check/analyze scripts, fixed input manifest, A1/B/A2/B-check
source manifests, all responses and path-hit logs, analysis.json.

Evidence archive `teacher-evidence.tar.gz`:15142407 bytes, SHA256
`cbeca44aef98812ae467f4afa1648cbc4a2aec85c18dca78377c87ef70b28e4e`.
Includes all four completed process lifecycles and raw API responses. Python
syntax checks and diff checks pass. Final amd-smi process query shows all
eight GCDs free. No inference service or benchmark remains running.
