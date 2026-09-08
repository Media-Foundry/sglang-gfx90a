# TP8 M32 wo_b hipBLASLt exhaustive supported-instance screen: stop

2026-09-08, parent `9ab6845b07`. No production code/selector changes.
This closes the explicit gap left by the earlier ordinary-API wo_b screen;
it is not a repetition of the wo_a solution-4429 experiment.

Shape: M32/N4096/K1024, current TP8 cached-BF16 wo_b versus F.linear.
Eight real checkpoint TP8 rank-0 shards (layers 0/6/12/18/20/24/36/42),
64 MiB explicit temporary weight storage total. Original checkpoint unchanged.
Input is the existing real prefill wo_a dump's first 32 rows, not a fresh
diverse-decode fixture. Component evidence only, not a service speed claim.

Physical GPU4; AMD PID ownership audit before GPU initialization. Retained
service PID3589516 was not benchmarked concurrently. Probe PID3599003 exited;
post-test AMD audit showed only 18 distinct owned service GPU PIDs. No JIT
dependency installation, global tuning database edits, NUMA changes, or
persistent model cache. 1M KV service remains unchanged.

## Measurements

`hipb_findallsols` returned 2226 supported instances. Each screened across
eight weights with three graph timing samples. Fastest three validated with
100 input mutations, 1000 replay stability, seven multi-weight graph ABBA
cycles. Raw sample arrays and top-20 screen retained in adjacent JSON.

| Solution | F.linear us | Candidate us | Exact mutations | Max abs | Max relative L2 |
|---|---:|---:|---:|---:|---:|
|3729|16.832519|16.462105|0/100|0.015625|6.616e-5|
|3924|16.835318|16.475307|0/100|0.0078125|4.212e-5|
|5031|16.811317|16.513909|0/100|0.015625|6.619e-5|

All three finite and bitwise stable on 1000 unchanged-input graph replays.
Fastest initially exact screen solution: 3728, 16.785755 us; initial exactness
is one input observation, not 100-mutation evidence. It has no meaningful
screen advantage and was not promoted to a second tuning round.

## Decision

Best saving 0.370414 us/projection (~2.2%). Even assuming all 43 projections
sit wholly on the critical path, ~15.93 us/token is only ~0.052% of a C32
step at resident 1035 tok/s (32/1035 seconds). This is an optimistic budget,
not a measured end-to-end gain. Non-bitwise candidates do not justify service
integration for this ceiling. No E2E run or production selector added.

Do not rescan these same shape/dtype instances without a changed contract
or a concrete new algorithm. This does not reject fused projection-consumer
work or different shapes, but they require their own measured budget.

Command:
```sh
HIP_VISIBLE_DEVICES=4 PYTHONPATH=/home/pc/Code/sglang/python:/home/pc/Code/sglang \
  /home/pc/anaconda3/envs/DS/bin/python scripts/rocm/bench_dsv4_tp8_woa_gemm_screen.py \
  --service-pid 3589516 --projection wo_b --hipblaslt \
  --output /tmp/dsv4_tp8_wob_hipblaslt_screen_20260908.json
```

Full raw 2226-instance screen SHA256:
`5e54f643662e3d5621d22c033d090fbc656eb8e3c895c82366c330b01f550293`.
Portable JSON retains all finalist ABBA samples, top20, fastest initially
exact candidate, source shapes, versions, and raw path/digest.
