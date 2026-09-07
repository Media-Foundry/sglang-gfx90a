# Conservative graph shared-read fallback

When a backend declares `IN_REPLAY` reads but no external graph event is
available, the runner previously published read completion before replay.
That violates the declared lifetime. Use `POST_REPLAY` instead; this does not
change model arithmetic or weights. HIP cannot create the external event in
the current helper, so this fallback matters for DSV4 HIP decode.

Validation (TP8, native AR, M36864 prefill profile, GEMV candidate disabled):

- Existing fence tests plus isolated fallback and GEMV guard tests: 7 passed.
- A3: nine measured 256-token completions exactly match the old baseline;
  six full-prefix teacher probes match IDs and logprobs; France passes.
- Per-task C1 medians: 76.818 / 76.571 / 76.820 tok/s.
- Two fresh-cache 2304-token code-source requests both finish normally with
  482 completion tokens and identical output IDs.

Artifacts: `/tmp/dsv4_tp8_war_post_A3_20260907.json` and
`/tmp/dsv4_tp8_war_post_A3_long_20260907.json`.

Old-fallback services stalled at request transitions both with and without
the GEMV candidate. A3 did not stall, but this limited test does not establish
that the fallback was the sole cause or prove general stress stability.
Generated audit prose has not been independently verified for factual accuracy.
