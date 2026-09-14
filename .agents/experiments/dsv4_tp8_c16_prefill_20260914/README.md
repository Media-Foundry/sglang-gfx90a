# TP8 C16 8K prefill: isolated chunk-budget screen

Original V4 Flash, TP8/EP1, native AR, allocated 1M logical KV pool. Preserve
seven native graph tiers and every accepted compute flag. No DSpark or runtime
kernel edit. Eight GCDs must have no external PID before launching.

A: chunk/max-prefill 36864. B: chunk/max-prefill 32768. Admission remains 16 and
delayer 20 ms. Same first 16 public-source 8K prompts in every wave, fresh cache
salts, one output token. The initial hypothesis was about four forwards each;
actual A1 logs instead show five: 36864 x 3 + 4096 + 16384. B1/B2 show four
32768-token batches. Logged counts include padding; the input-token denominator
is the actual 131069 tokens, not the rounded 131072. The old 32768 rejection on
2304-token prompts added a forward and does not answer this workload.

Order A1 / B1 / B2 / A2. Three measured waves per leg, one excluded warmup per
service; B1/B2 share one candidate service. Independent control starts. Each
service also gets France and two C16 x 128-output code smokes, with independent
completion-ID/text validation. Hash equality is reported, not assumed under
the existing BF16-CK/variable-batching path. Code prose requires manual review.

`run.py --arm A1`, then `--arm B`, then `--arm A2`, in the DS conda env. This
reuses the prior PID/birth/command-checked service lifecycle helper, with an
arm-local generated launcher whose only baseline change is the chunk line.
It refuses existing result/log targets and stops only its own service in finally.
All logs, request outputs, source hashes and observations stay in this directory.

The historic C16 baseline is 5171.36 input tok/s; decisions use this run's ABBA,
not that old point. Run `analyze.py` only after all arms complete, then `pack.py`
to hash-verify the evidence archive without deleting original arm directories.
The archive includes exact launchers, request IDs/texts, GPU ownership checks,
server logs and source identities. Output coherence is a smoke check, not proof
that generated code-review claims are correct or that logits are equivalent.
