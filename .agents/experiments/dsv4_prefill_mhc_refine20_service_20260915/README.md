# Pending exact comb-refinement service ABBA

Reference: full20 prefill in every arm. Candidate: same arithmetic contract,
large-M8+12 comb continuation only. Original V4 TP8/EP1, native AR,1M KV,
32K chunk, wide query16/runtime-M, original524286-token C16x32K code manifest.
Each leg has3 timing waves, plus a warmup per process and two128-token quality
waves with full input echoes. A1 -> B1/B2 -> A2; all owned services stop in
finally. No GPU microbench or extra API requests during timing.

The shared `../dsv4_prefill_mhc_config_iters_20260915/run.py --comb-refine`
validates completed integrated GPU tests, source hashes, all-rank policy and
refinement path hits. Outputs are isolated here and old trials are not reused
as the contemporaneous control. `sweep.py` refuses overwrites.

Start with the DS interpreter after checking GPU ownership:

```bash
set -o pipefail
/home/pc/anaconda3/envs/DS/bin/python -u \
 .agents/experiments/dsv4_prefill_mhc_refine20_service_20260915/sweep.py \
 2>&1 | tee .agents/experiments/dsv4_prefill_mhc_refine20_service_20260915/sweep.log
```

Default-off until full service ABBA and input/output review. Component
bit-exactness does not imply globally repeatable model output when other
batch-sensitive/atomic paths remain.
