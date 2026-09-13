# V4.1 resume and fixed-prefix checks

The user explicitly said `continue` after the emergency pause. At15:03 HKT,
all8GCDs had no processes, port30101 was free, and974GiB host RAM was available.
HEAD remained`ae79b6fd8a`; unrelated dirty/untracked files were left untouched.

## Service restored

tmux`dsv41-resume-20260913-1505`, parent207990, TP0scheduler209152, log
`/tmp/dsv41-resume-20260913-1505.log`. Original weights, TP8/EP1/no-A2A,
native AR/eager,32768pool/context,2304chunk, max_running_requests256: same
logged configuration as the earlier service. Startup gate111tests+16subtests
passed; ready at15:07:21, load_weight103.68s, tokenizer startup136.99s.

Storage audit still reports53.987GiB unique named GPU storage/rank,42.188GiB
routed FP4 weights,2.930GiB E8M0 scales, no direct FP32 scale clones,
gpu_engram0GiB and23.604GiB host Engram table storage/rank. This is named
storage accounting, not total process allocation. The service still does
not enable`/v1/responses` because openai_harmony is missing; that is separate
from `/generate` and chat completion serving. Vision/MTP remain unvalidated.

France after restart: `The capital of France is Paris.`,8IDs including EOS,
identical completion hash to prior starts, zero cache hit, HTTP9.337s cold.
Do not turn this cold smoke time into a decode-speed comparison.

## Latent checker issue fixed, not a newly discovered model error

`check_dsv41_cache_prefix.py` previously gated success on equality of the
first item in reported Top-20 lists, not equality of the actual committed
output IDs. Reporting Top-K order at ties is not necessarily argmax tie order.
The corrected gate requires nonempty comparisons and all committed IDs equal;
ranked Top-20 equality remains a separately named diagnostic. Recomputed
prompt_tokens must also match the entire supplied fixed prefix.

Five CPU tests include equal ranked IDs with different commits (must fail),
different ranked IDs with equal commits (must not falsely fail the commit
gate), and empty/partially mismatching comparisons. Full suite116tests plus
16subtests passes (12.82s). CPU audit of7existing reports/48comparisons found
NO historical disagreement between the two gates. Thus this is preventive
verifier hardening, not evidence that old passing results were fabricated
or that a model numerical bug has been repaired.

## C4 generated baseline vs fresh C1 recomputation

Baseline is the first event-sessionization answer from the previous C4 SQL
trial:203prompt IDs,766completion IDs, natural EOS and16/16functional SQL
fixtures passed. Exact prepared input and baseline source SHA are retained.
We feed the baseline's SAME continuation IDs; no independent greedy rollout
is compared after it branches.

Absolute prefix positions203/204/205,255/256/257,511/512/513,767/768/769,
two fresh-cache repeats each:24/24actual committed IDs agree. All repetitions
at each fixed length return matching metrics. Some score margins are small:
at511baseline0.125 vsrecompute0.375; at513baseline0.125 vsrecompute0.0, with
the same committed token. This is stronger than only checking large-margin
France, but is still a small sample.

Floating logprobs do NOT match: Top-20 overlap0.8-1.0 and maximum common
logprob delta2.312498. A difference already exists at203(the first prefill
output), so it cannot all be attributed to a later cache wrap. Batch shape
and prefill/decode kernels are confounded here. Do not claim bitwise parity
or that cache addressing is the source of these numerical differences.

## Fresh C1 control

Started `check_dsv41_cache_prefix.py` without a reused baseline, same fixed
203-token input, max_new_tokens1536/natural EOS, same12positions x2repeats.
Output directory`/tmp/dsv41-resume-c1-prefix-20260913-1510`. Pending at this
recording point. The purpose is to remove the old-C4/new-C1 cohort difference
before drawing conclusions about cached decode versus recomputation.
