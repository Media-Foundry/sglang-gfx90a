# Current query-owner C16 prefill profile

Fresh diagnostic on e23a66f0bf; original V4/TP8/EP1/native AR, original weights,
1,048,576 logical KV and32768 chunk. Exact accepted owner/query16/runtime-M/
mix8/paired-column configuration, all intrusive tensor probes disabled. This
is a profile of the7953.956103 input tok/s checkpoint, not a new peak benchmark.

Four waves (one warmup, three measured) produced128 rank-forward snapshots and
64 exact input echoes, zero prefix hits. Eight-rank owner and paired-MHC hits
were checked from actual logs. Client measured waves:16.504609727,
16.508118482,16.510648686seconds. Owned service1677866 shut down cleanly.

Per forward, select the rank with the longest complete GPU envelope and use ALL
stages from that rank. No independent rank-max summation; this is not a full
synchronized cross-rank timeline reconstruction. Mean wave GPU envelope:
**16.417479040seconds**. CUDA/HIP-event versus realtime-clock envelope ratio
0.99999815..1.00002548; maximum rank envelope spread4.5312ms.

Coarse, mutually exclusive measured spans:

| Stage | Seconds/wave |
| --- | ---: |
| Attention-side MHC/norm |1.872327|
| Attention entry gap |0.000381|
| Attention preparation |2.854766|
| Main sparse attention |2.761814|
| Attention output projection/collective |1.548325|
| FFN-side MHC/norm |1.890757|
| MoE/collective |5.373752|

Interlayer/outer gaps close the remaining envelope exactly in the JSON analysis.

Nested subdivisions, **do not add these to the coarse parents**:

- Routed MoE4.377516s; MoE output collective0.747365s.
- Both pre-mix1.730311s; both post1.368826s; weighted norm0.571192s;
  Sinkhorn0.065478s.
- Indexer query0.627263s; weights0.027952s; index compressor0.117081s.
- **Owner chain0.301702s**, including pack/logits/Top-K/integer all-gather/
  reconstruction. The adjacent0.000184s is only the tail marker interval,
  **not** the Top-K cost. Analysis labels now follow the actual path marker.
- Main QKV projection0.576575s; core compressor0.640911s.

The pre-owner17.726924s envelope is superseded for remaining-budget decisions.
Owner has reduced the combined logits/selection/metadata area to~0.30s; it is
no longer a major budget pool. Routed MoE, main attention and MHC remain large.
The separate real CK fixture proves a small atomic stage2 drift, but its
deterministic fixed-slot alternative costs52% more in standalone stage2, so it
was not enabled here.

Next bounded component screen: independent pre-mix columns3/4 versus accepted
columns2, preserving per-column FP32 K1024 reductions and eight-row reuse.
It is experimental only and must pass exactness/resources/full-component
timing before any service trial. No additional speedup is claimed by this report.

Artifacts: capture_owner.py, analyze.py, profile_labels.py, raw marker frames,
analysis.json and details-analysis.json under capture-owner-current. Seven
owner/label CPU tests passed. Source hashes were frozen during collection.
