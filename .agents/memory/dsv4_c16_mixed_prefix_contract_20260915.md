# Mixed-prefix comparison contract tightened (CPU only)

The prepared real-code prefix client now rejects changes in **actual** cached
token counts across waves, rather than merely requiring positive hits bounded
by the requested priming length. A new `--cache-reference <completed-result>`
also enforces the same input-manifest hash and hit vector across configurations.

Why: equal full input IDs do not imply equal prefill work when cache hits differ.
Do not attribute that difference to query reuse or another kernel. A stable
partial hit below the page-aligned requested prefix is allowed and measured;
changing that hit pattern is not allowed to silently enter the timing average.
Raw responses are saved before the gate, so failures retain their evidence.

The client continues to report full-input throughput and newly-computed-token
throughput separately. Priming time is excluded and separately recorded. Full
code prompts remain unchanged; only priming uses exact page-aligned prefixes.

Five CPU tests pass, including zero-hit, changed-hit, malformed-count, input-ID,
request-ID and empty-output rejection. No mixed-prefix GPU/service experiment
has been performed yet. This change does not affect the running zero-prefix
comb-refinement ABBA or any production module.

Files: `.agents/experiments/dsv4_c16_query_wide_20260915/bench_prefix.py`
and `test_prefix.py`.
