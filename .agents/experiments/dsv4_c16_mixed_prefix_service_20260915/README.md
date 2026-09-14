# C16 mixed-prefix wide-query service trial

Original V4 TP8/EP1/no-A2A/native AR; original weights;1M logical KV;32768
prefill budget. Sixteen fixed real-code32K inputs are unchanged across arms.
Prime exact page-aligned0/25/50/75% prefixes with per-request salts, then
send the full requests. Salt matches only its own prefix; every round gets
new salts. Priming is timed separately and excluded from wave TTFT.

A1 -> B1/B2 -> A2, fresh process per arm. All retain accepted query16/runtime-M
and mix8, legacy MHC policy (config20/refinement off). Only wide-C4 changes.
Warmup1 and3 formal waves/leg, followed by2 separate128-token quality waves.
No other GPU jobs or API probes while the trial runs.

The client requires exact full input echoes, nonempty output and actual
positive cache hits on primed requests. **Actual cached-token vector must
match warmup, every formal wave, quality waves and control A1 across arms.**
Failed raw responses are preserved; a hit mismatch stops the trial rather
than reporting a speedup. Report full-input/wave-time and newly-computed
tokens/wave-time separately; neither is comparable directly to zero-prefix
throughput. Quality mode is not a prefill timing leg.

Commands (DS conda Python):

```
python ../dsv4_c16_query_wide_20260915/test_prefix.py
python run.py --arm A1 --validate-only
python sweep.py
```

Source hashes include the runner/client and production code and freeze across
the trial. Services are owned by PID and process birth time; shutdown only
targets those owned trees. Output directories are never overwritten/restarted
automatically. A stopped failed arm requires diagnosis and a new named trial.

Acceptance still needs tokenizer decode, actual rank path hits, timing-log
inspection, full ABBA comparison and bounded manual review; passing the driver
alone is not model correctness or global determinism.
