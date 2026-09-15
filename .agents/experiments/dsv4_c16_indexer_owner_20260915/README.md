# Real C16 prefill indexer ownership: component gate passed

Original V4, original weights, TP8/EP1/native AR, 1M logical KV and 32K
prefill budget. **Accepted service throughput remains 7367.6734 input tok/s.**
No query-owner production path or service speedup is claimed in this experiment.

## Evidence and correction

Two fresh diagnostic processes ran the accepted paired-MHC/query16/runtime-M
configuration. Each sent the same 16 real 8K code requests, checked exact input
echoes and zero cache hits, generated one token/request, and passed France.
Only the first large forward was captured: M32767, four requests (cases0–3),
at C4 layers2,20,42, on all eight ranks. Both owned services stopped cleanly.

The first capture helper's CPU preshuffle decoder was wrong: it treated the
runtime layout as a simple transpose rather than a tiled row/column layout.
Its synthetic inverse test repeated the same mistaken assumption. This affected
diagnostic logical-KV hashes, **not the runtime cache or model arithmetic**.
`capture/` and `analysis.json` are retained as superseded diagnostic evidence;
do not rely on their logical-KV interpretation. The first `owner-rccl.json`
screen consumed unmodified runtime bytes, but is preliminary, superseded below.

Version2 uses the actual production element address:

```
(row/tile)*(tile*128) + (col/tile)*tile² + (row%tile)*tile + col%tile
```

Its CPU test independently constructs pages using this formula, including
ragged valid lengths0/1/63/64/65/127/128 and tiles0/8/16/32/64. The corrected
capture also saves every rank's referenced packed pages and their SHA256, so
logical hashes can be recomputed offline rather than trusted as log assertions.
Three capture tests and one ownership-plan test pass.

Authoritative capture is `capture-v2/`, analysis is `analysis-v2.json`.
Full x/q_lora/Q/weights hashes match across all eight ranks at each sampled
layer, and all valid logical KV values/scales match with the corrected decoder.
This is one first-forward snapshot, not proof for every layer/context/run.

## Cross-process drift remains

`cross-process.json` compares the two fresh diagnostic processes. Input IDs,
positions, lengths, request identities, and row placement match. Full captured
tensor hashes at layers2 and20 match; the corresponding layer42 hashes differ.
Layer42 sampled normalized x relative-L2 is0.0046581, max-abs0.0087891;
sample metrics are not full-tensor bounds. Raw quantized Q has different units
and its absolute difference must not be interpreted as an unscaled activation
error. First completion text is16/16 identical.

Thus this observed difference cannot be explained by different input rows.
It is present between the captured layer20 and42 points, but the first divergent
operator is **not** identified. Diagnostic synchronization itself can affect
scheduling. No claim of global determinism or repaired model numerics is made.

## Real layer20 eight-rank oracle

`owner_oracle.py` leaves full-M query projection and compressor unchanged and
outside both comparisons. It uses the freshly captured full Q/weights/cache.
Original query16 groups containing active rows (C4 length>512) are round-robin
assigned to owners. Each owner has3072 padded rows; trivial rows are regenerated
locally in the existing descending-logical-ID order. No score perturbation,
expert removal, KV truncation, or shared representative selection is used.

The candidate includes GPU Q/weight/length/page packing, existing exact
query16 logits, deterministic Top-K, **48MiB RCCL integer all-gather**, and
full logical/physical result reconstruction. It exchanges logical IDs only:
every rank deliberately uses different physical page numbering in the oracle.
The query planner is offline CPU fixture analysis, NOT a production no-D2H
implementation. Any eventual service integration must count planning/metadata
cost and preserve collective ordering and scope.

`owner-rccl-v2.json`: three ABBA cycles, five calls per event sample, all eight
ranks participating; rank-max per sample, then median:

| Complete compared chain | Median ms |
|---|---:|
| Replicated logits + Top-K |18.60018|
| Packed owner + exchange + reconstruction |3.13217|

Component speedup5.9384x. This is **not** a model/TTFT speedup. Separate
diagnostic medians: full logits17.70472ms, full Top-K1.25415ms, pack0.07363ms,
owner logits2.09050ms, owner Top-K0.19578ms, exchange0.66682ms,
reconstruction0.17853ms. Do not sum independently measured component medians
to replace complete-chain timings.

Every rank compared all valid owned scores bitwise and all reconstructed
logical/physical IDs exactly. Original input plus nonuniform head-weight and Q
mutations passed;25 repeated eager calls passed. This is ordinary eager-prefill
testing, not graph-replay coverage. No service E2E owner path has run.

Scope remains a single real layer20 M32767/K2048 snapshot. Wider histories,
prefix hits, varied admission shapes, service stream/collective interference,
long-output quality and 1M-pool workspace must be validated before promotion.
The measured candidate is worth an opt-in service oracle; production defaults
are unchanged.

## Reproduction and artifact policy

```
PYTHONPATH=python /home/pc/anaconda3/envs/DS/bin/python \
  .agents/experiments/dsv4_c16_indexer_owner_20260915/capture.py --label <fresh-directory>

PYTHONPATH=python /home/pc/anaconda3/envs/DS/bin/python \
  .agents/experiments/dsv4_c16_indexer_owner_20260915/analyze.py \
  --capture <fresh-directory> --output <fresh-analysis.json>

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 PYTHONPATH=python \
  /home/pc/anaconda3/envs/DS/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  .agents/experiments/dsv4_c16_indexer_owner_20260915/owner_oracle.py \
  --capture <fresh-directory> --analysis <fresh-analysis.json> --output <new-result.json>
```

Do not run the component job concurrently with the model service/other GPU work.
All commands refuse output reuse. The 270MiB full-Q fixture remains local,
identified by SHA256 in capture records and oracle results; do not commit it.
The evidence archive includes small raw-page/sample tensors, corrected audit
records, lifecycle logs, source hashes, and source snapshots, sufficient for
independent audit without publishing the large fixture.
