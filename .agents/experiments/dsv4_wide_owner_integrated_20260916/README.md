# Integrated wide-owner trial: accepted explicit C16x16K profile

Baseline: exact 10.206k **8K** checkpoint `6812bb7994`; this experiment measures
**C16 x 16K**, 262141 real input tokens, zero cache hit, TP8/native AR/original
weights/1M logical pool/32768 chunk. Do not compare these rates as an 8K gain.

Both service arms enable the previously tested full wide query16/runtime-M
fallback. Only `SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE` differs. Producer stays
off. Wide ownership is default-off, original V4 eager-prefill only, globally
8192..65536 rows; the internal compact wrapper accepts query16-aligned local
rows only under an explicit admitted-global-M contract. No cache update,
projection, score math, Top-K rule, collective type or physical-ID mapping is
changed. Width is not clipped; >8192 falls back.

`oracle.py` exercises the production helper on all eight ranks, including
8K, 16K, ragged 32K and mixed prefix, random mutation and exact cutoff ties.
Each rank independently renumbers physical pages. Diagnostic compares full
scores bytewise and logical/physical IDs exactly. Its component timing is
not service throughput and excludes full query production/CPU planning.

`service.py --arm check` must finish before `run.py`. The latter runs fresh
A1, B, A2 services serially; B1/B2 share the B process. Each leg has three
scored waves after warmup, followed by four 128-token quality waves and a
fixed 64-token teacher continuation from A1. Owned service cleanup is in
finally. No script clears compiler caches or modifies GPU clocks.

Completed:1344 real-model byte-exact score/logical/physical-ID comparisons;
formal ABBA8249.638738 ->10024.774291 input tok/s (**+21.517737%**); control
drift0.127126%.192 answers all128 tokens identical; A1/A2 and A1/B fixed
continuation logprobs/Top5 identical at1008 valid positions each.16 CPU tests
pass. All owned services stopped and GPUs idle after the trial.

See `summary.json`, `acceptance.json`, `unit.log`, `service-evidence.tar.gz`
and its per-file manifest. `validated-launcher.sh` reproduces explicit settings;
no global default changed. 32K/mixed prefix only have component proof here;
neither universal determinism nor new8K E2E performance is claimed.
