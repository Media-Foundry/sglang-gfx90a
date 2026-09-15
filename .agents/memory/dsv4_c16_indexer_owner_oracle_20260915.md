# C16 indexer query-owner component passes; no new service peak yet

Accepted original-V4 TP8 native-AR C16x8K remains7367.6734 input tok/s with
original weights,32K chunk,1M logical KV. Reviewer runtime-M/query16/MHC8
priorities were completed; paired-column MHC subsequently gave the7.37k peak.
Latest profile leaves indexer logits/metadata1.405s and Top-K.137s/wave,
not the pre-query-reuse2.64s budget.

New `.agents/experiments/dsv4_c16_indexer_owner_20260915/` audit captures actual
first-forward M32767 (cases0–3) at layers2/20/42 on8ranks. Corrected capture-v2
confirms complete x/qlora/Q/weight hashes and valid logical KV identical across
ranks. Referenced raw pages saved for independent reanalysis. An initial CPU
inverse-preshuffle diagnostic bug was fixed and recaptured; its old logical-KV
hash interpretation is superseded, not silently accepted.

Cross fresh processes: same input IDs/positions/row placement; layers2/20
captured tensor hashes same, layer42 different. Sampled normalized-x relative
L2=.0046581, max_abs=.0087891; not full bounds. First-token text16/16 same.
This drift is not explained by differing inputs; first divergent operator
remains unknown. Neither diagnostic used the new owner execution path.

Real layer20 owner oracle,8ranks/3ABBA cycles/5calls per event:
replicated logits+Top-K18.60018ms → owner full chain3.13217ms (5.9384x).
Includes GPU row packing, owner logits+Top-K,48MiB RCCL logical-ID all-gather,
full-row reconstruction/local physical mapping. Query projection/compressor
unchanged, no shared selection/approximation. Every rank intentionally uses
different physical page numbering. All scores bits/logical IDs/physical IDs
exact across3 input variants and25 repeated eager calls.

Offline planner retains original query16 groups, active lengths>512,3072
rows/owner. It is not yet production metadata planning; no E2E owner result,
no graph coverage, and no change to AR/prefill defaults. Do not multiply5.94x
into total service throughput. Next: opt-in real service integration only
after scope/metadata/collective ordering and workspace validation; retain
full-M query production first. Also narrow the existing layer20→42 drift.

Both diagnostic services and both component jobs terminated; all8GCDs free
at closure. Full270MiB fixture stays local; hashes and smaller evidence archived.
