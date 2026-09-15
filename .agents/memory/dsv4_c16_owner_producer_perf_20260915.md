# Owner-Q producer: formal C16 prefill ABBA

Tested source commit: `c14b3ba082`. Original DeepSeek-V4-Flash, TP8/EP1,
native AR/no-A2A, original checkpoint weights, 1,048,576 logical KV pool,
32768 prefill chunk. This is prefill input throughput, not decode throughput.

## Controlled service trial

Fresh A1, B, A2 services; B runs consecutive B1/B2 legs. Each leg has three
waves of the same 16 heterogeneous real code requests (131069 input tokens).
Both arms retain accepted query-owner mode. The only experimental switch is
`SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER=0/1`; numerical diagnostics are off.
CPU flag truth-table test: 1 passed. Source manifests match across services.

One warmup wave per process is excluded. Recomputed every rate from raw
client timestamps: total input tokens divided by earliest request begin to
latest first-token arrival. All 192 formal request echoes match, prefix-cache
hits are zero, completion lengths are one. All eight ranks hit the intended
owner/producer mode; recorded full/local/width signatures in each leg include
32766/3072/2048, 32767/3072/2048, 32768/3072/2048. This is a path/shape
witness, not a complete proof of identical internal batch-row scheduling.
No compile markers were found in the formal log windows; this is not a
general proof of freedom from all future cold-shape compilation.

| Leg | Three input tok/s measurements | Median input tok/s | Median wave seconds |
|---|---|---:|---:|
| A1 | 7962.414 / 7960.764 / 7961.966 | 7961.966 | 16.461888 |
| B1 | 8233.670 / 8241.829 / 8235.094 | 8235.094 | 15.915908 |
| B2 | 8228.674 / 8239.316 / 8240.796 | 8239.316 | 15.907753 |
| A2 | 7961.828 / 7952.521 / 7960.720 | 7960.720 | 16.464466 |

Mean of control leg medians: **7961.342995**. Mean of candidate leg medians:
**8237.205004 input tok/s**, **+3.465019%**. The return control reproduces;
this is a measured opt-in performance result, not a single fast wave.
Request TTFT medians: A1 10.463267 s, B1 10.124730 s, B2 10.125350 s,
A2 10.474553 s. These are distinct from the whole-wave duration.

## Quality and numerical limits

Two additional 128-output-token quality waves per service: 96 exact input
echoes, zero cache hits, all completion lengths checked. Candidate repeat
matches 16/16; A1 repeat 16/16, A2 repeat 15/16, independent controls 15/16.
A1 versus candidate matches only 3/16 complete 128-token outputs.
All 32 candidate outputs exactly match the first 128 tokens of previously
read candidate responses in `dsv4_c16_owner_producer_service_20260915/B-check`.
Those responses were coherent and topical; their purported code defects were
not independently certified. This is not a held-out accuracy benchmark.

The preceding fixed-continuation experiment remains the numerical reference:
`dsv4_c16_owner_producer_teacher_20260915.md`. Candidate/control differences
exceed control noise (about 98.5% top1 agreement; mean NLL increase about
0.0012 nat/token in the paired fixture), and include non-small-margin flips.
Same-rocBLAS full/compact Q checks passed, but old hipBLASLt versus rocBLAS
arithmetic is not bit-exact. Neither global determinism nor complete drift
repair is established. Do not describe this as an exact-path 8.24k result.

## Disposition and next work

Keep producer **default-off**, preserving the accepted production checkpoint
7953.956103 input tok/s and its launcher defaults. The opt-in candidate has a
reproducible throughput benefit under the user's tolerance for coherent small
drift, with the numerical tradeoff explicitly retained. No AR/decode, draft,
speculative, V4.1, or target-expert semantics were changed in this trial.

The review's 6.877k budget is historical: runtime-M/query16, MHC reuse changes,
query-owner, and updated profile work have since been recorded. Do not count
their saved time again. Next prioritize broader numerical/length coverage or
an updated candidate critical-path profile, not another copy of empty-tile
or query4. The current accepted profile is documented in
`dsv4_c16_current_owner_profile_20260915.md`.

All three owned services stopped cleanly. A post-run `amd-smi process --json`
reported no running processes on all eight GPUs. Active optimization goal
continues; this trial is complete, not the overall goal.

## Reproduction and evidence

Directory: `.agents/experiments/dsv4_c16_owner_producer_perf_20260915/`.
`run.py`/`sweep.py` preserve source/fixture/launch checks; `analyze.py` recomputes
all formal metrics, and `review.py` validates quality witnesses. Analysis
scripts intentionally refuse to overwrite existing result JSON.

`perf-evidence.tar.gz`: 2,472,954 bytes; includes A1/B/A2 raw responses,
timestamps, inputs, service logs, plans, completion/stop records and derived
analysis. SHA256:
`74493d263cb5103310411a6328f5453564c82049b35ad14bfff19b790c69a596`.
