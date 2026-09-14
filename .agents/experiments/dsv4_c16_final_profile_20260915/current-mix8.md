# Pending current-main mix8 diagnostic (not yet measured)

The historical `capture-v2` used query16/runtime-M but preceded mix8. Its
source hashes no longer describe current main. Do not overwrite it or reuse
its stage totals as the current remaining budget.

`capture.py --current-mix8 --label capture-mix8-current` takes the accepted
mix8 B configuration and original16 real8K inputs, but snapshots **current**
sources. The plan retains the historical hashes and explicitly lists every
source difference. It is a new diagnostic, not reproduction of historical
6959 tok/s. Wide C4 and config20 are explicitly off; query16/runtime-M/mix8
are on, with native AR,1M KV and32K chunk. All eight ranks must actually hit
mix8. No production code is changed by this driver.

CPU-only preflight (no files or service created):

```bash
/home/pc/anaconda3/envs/DS/bin/python \
  .agents/experiments/dsv4_c16_final_profile_20260915/capture.py \
  --current-mix8 --label capture-mix8-current --validate-only
```

Run without `--validate-only` only after the current ABBA and GPU oracle
release all GPUs. This diagnostic uses one warmup and three measured waves,
with full input echoes and zero-prefix checks. The existing analyzer accepts
`--label capture-mix8-current` and selects one longest outer-envelope rank
per forward, retaining all of that rank's stages. Do not add independent
rank-max stages or count overlapping subranges twice. Instrumented timings
are not an uninstrumented throughput checkpoint.
