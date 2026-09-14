# TP8 C16 8K prefill chunk screen — 2026-09-14

Measured HEAD `2d4ac02b16`, original V4 Flash, TP8/EP1/no-A2A, native AR,
allocated 1M logical KV pool. Do not confuse with TP4, V4.1, DSpark, the old
2304-token/131072-pool 6.4k result, or C1 input throughput.

**Workload-specific positive result:** change chunk and max-prefill budget from
36864 to 32768, admission still 16. Same 16 distinct public-source ~8K inputs,
131069 actual tokens/wave, fresh salts, one generated token, three measured
waves per ABBA leg, excluded warmup per service.

| Leg | Median input tok/s |
| --- | ---: |
| A1, 36864 | 5177.84 |
| B1, 32768 | 5313.25 |
| B2, 32768 | 5307.00 |
| A2, 36864 | 5171.24 |

Mean leg medians: **5174.54 -> 5310.12, +2.62%**, wave ~25.33 -> 24.68 s.
A2/A1 -0.127%; B1/B2 share one service, so candidate independent-process
replication remains untested. Keep as a C16 x 8K option, **not global default**.

Key finding: do not reason from ceil(total_tokens/chunk) alone. Actual logs:
- A every wave: 36864 x 3 + 4096 + 16384 = five forwards.
- B every wave: 32768 x 4 = four forwards.
- Logged shapes include padding; measured input denominator is not 131072.

The old 32768 rejection was for 2304-token prompts and added a forward. It does
not preclude this win; conversely this win does not invalidate that old rejection.
No kernel, weight precision, graph tier or KV capacity changes in this trial.

Correctness limits: France passes all three services; 336 independently decoded
completion-ID/text pairs pass integrity/length checks. Two C16 x 128 output waves
per service; all 32 candidate snippets inspected, no obvious garbled/repetitive
collapse. Truncated source-analysis prose is not executable-code validation or
logit equivalence. Full repeat IDs A1=3/16, B=7/16, A2=3/16: control itself is
nondeterministic. Do not claim bitwise correctness or precise drift attribution.

Evidence and exact launchers:
`.agents/experiments/dsv4_tp8_c16_prefill_20260914/`
(`RESULTS.md`, `summary.json`, hash-verified `measurement-evidence.tar.gz`).
Driver uses PID/birth/cmd-checked ownership, checks amd-smi before each workload,
and stops owned services in finally. All eight GCDs free at completion.

Next sensible direction: measured chunk policy across lengths/prefixes, or new
C16 x 8K critical-path profiling. Do not reuse obsolete MoE percentage budgets.
The existing formal matrix script hard-sets 36864; outer env overrides alone
cannot change it. Candidate launcher in the archive contains the tested line.
