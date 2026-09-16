# K32 shorter-length regression: complete

OriginalV4,TP8/EP1,C16,nativeAR,originalcheckpoint,1M logical KV,32K prefill
budget, real public-code fixtures and zero prefix hits. CommonFP32/20 MHC and
the rest of the accepted configuration remain fixed. Only OWNER_K32 varies.

| Length | K16 control input tok/s | K32 input tok/s | Gain | Control drift |
|---|---:|---:|---:|---:|
|8K|10217.398796|10282.763437|0.639739%|0.185086%|
|16K|10018.886498|10151.650740|1.325140%|0.095036%|

Each length uses A1/B1/B2/A2,three scored waves/leg,B1/B2 sharing a process.
Both controls start fresh. Each of three processes generates four quality
waves:192128-token answers agree per length. Teacher forcing compares1008
positions between controls,candidate,and the preceding accepted checkpoint;
logprobs and Top5 records match exactly. Candidate timing logs show all8 K32
hits; control logs show none. France, actual input echoes and owned cleanup
pass. No universal batch-invariance claim or global default promotion.

`summary-*.json` reconstructs throughput from raw client timestamps and real
token counts (131069/262141). `acceptance-*.json` additionally gates numerical
evidence and gain versus observed range/control drift; this is not a confidence
interval. The2688 live score/ID comparisons are from the earlier32K diagnostic,
not from new8K/16K duplicate-reference runs. `service-evidence.tar.gz` preserves
169files including startup/source hashes,raw outputs,timestamps and cleanup.

The explicit launcher is in sibling `dsv4_owner_k32_20260916/validated-launcher.sh`.
All processes from this regression stopped; session18439 exited0. No other
model or decode optimization was introduced by this experiment.
