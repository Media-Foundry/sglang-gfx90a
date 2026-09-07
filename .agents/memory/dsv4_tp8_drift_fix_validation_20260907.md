# TP8 migration after TP4 numerical drift repair

## Preconditions

TP4 is validated first; see `dsv4_tp4_topk_cutoff_tie_fix_20260906.md`.
Fixes retain original checkpoint precision and the optimized native FP4
MFMA32/64 prefill path. They address divergent wave shuffle, preshuffled
indexer cache addressing, and atomic Top-K order/cutoff-tie selection.
This is not a BF16-CK approximation experiment or speculative decoding.

## TP8 configuration

- Physical GCDs 0–7, one model, TP8/EP1/no-A2A.
- Native AR, graph BS1, chunk 2304, token pool 65536, memory fraction .80.
- Scheduler overlap disabled for the controlled numerical oracle.
- MFMA32/64 enabled; deterministic indexer mode 2.
- Fixed real heterogeneous code manifest:
  `.agents/memory/dsv4_prefill_diverse_32_input_ids.json` (first 16 requests).
- New independent cache salt for each request in every round; all recorded
  cached-token counts are zero. Greedy, fixed-length completion and top-5
  logprobs. Test harness keeps its historical `tp4` filename but runs against
  the TP8 service on port 30011.

## Completed gates

1. C16 x 32 tokens, three rounds: all 16 requests' complete output IDs,
   per-token logprobs, top-5 logprobs, and text exactly match across rounds.
   `/tmp/dsv4_tp8_topk2_c16_{1,2,3}.json`.
2. C16 x 256 tokens, two rounds: all 16 requests match on every above field.
   `/tmp/dsv4_tp8_topk2_c16long_{1,2}.json`.
3. Official France input IDs, normal EOS: `The capital of France is **Paris**.`
   Output `[671,6102,294,8760,344,2619,51119,42499,1]`, 9 tokens, EOS=1.
4. Direct gfx90a wrapper auto-selection (no environment override): synthetic
   random/negative/tied/ragged/32K/64K rows and the saved real cutoff-tie row
   all match CPU stable selection and pass 1000 HIP Graph replays each.
   `/tmp/dsv4_topk_default_fixture_test.log`.
5. Independent-process TP8 C16 x 32 passes: no explicit MFMA/Top-K override
   on restart; new harness defaults produce all 16 complete IDs/logprobs/
   top-5/text exactly equal to the earlier process. Cached tokens are zero.
   `/tmp/dsv4_tp8_topk2_restart_c16.json`.
6. Isolated formerly drifting request index 2: C1 x 256 tokens, two fresh-cache
   runs match complete IDs, token logprobs, top-5 and text exactly.
   `/tmp/dsv4_tp8_topk2_c1_{1,2}.json`.

## Reproducible small regression fixture

`scripts/rocm/fixtures/dsv4_topk_layer8_row2144.json` holds the real 536
valid indexer scores from the layer-8 row which previously changed selected
KV membership with identical scores. No full model or temporary capture is
needed to test the repaired kernel:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=python \
  /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/check_dsv4_deterministic_topk.py --default-mode --replays 1000
```

The gfx90a direct wrapper now selects mode 2 by default; other GPU
architectures keep their old defaults. The gfx90a harness enables it too
and restores MFMA32/64 for TP4/TP8 EP1/no-A2A. Existing explicit diagnostic
overrides remain available. No unrelated local modifications are reverted.

These gates establish repeatability under the stated configurations, not bitwise equality between
TP4 and TP8 (different shard/reduction geometry), arbitrary scheduling,
all context lengths, or semantic correctness of every model answer.
