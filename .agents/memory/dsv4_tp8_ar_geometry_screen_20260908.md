# TP8 C32 legacy all-reduce geometry oracle

2026-09-08. Service PID3611667 unchanged, native TP8/EP1, 1M KV.
No AIter library rebuild or production selector modification. All isolated
eight-rank jobs audited via amd-smi before launch; no concurrent service
benchmark. Existing direct-HIP registered inputs avoid IPC suballocation
offset ambiguity. Communicator max_size1MiB; no weight or KV allocations.

## Concrete gap and contract

Current BF16[32,4096] is exactly256KiB. Legacy AIter dispatches two-stage at
this boundary, with512threads and block_limit16. Existing TP8 old/new screen
did not vary this limit; TP4 512KiB/1MiB knobs do not apply here.

Independent JIT shim includes the local AIter header, validates shared Signal
size against the loaded library, requires world8/full-peer topology and exact
BF16 shape, and invokes the existing public allreduce with use_new=false.
Only block_limit changes. Existing rank ordering, pack width, buffer lookup,
two-stage selection, system-scope synchronization and output are preserved.
It is not a new one-stage implementation or a production import.

First compare original library16 versus shim16 to check interface and numerics.
Subsequent comparisons use shim16 on both sides, so different compilation of
the same header is not mistaken for a grid gain.

## Results, eight-rank slowest sample

Five ABBA cycles,200 graph replays/sample,20warmups; trim one min/max from10
samples/arm.100 independent normal/integer rank-local input mutations; every
mutation also tests10 additional unchanged-input candidate replays.

| Comparison | Baseline us | Candidate us | All-rank exact mutations |
|---|---:|---:|---|
|library16 -> shim16|27.960959|27.721331|100/100 on each rank|
|shim16 -> shim24|27.788367|30.297224|100/100 on each rank|
|shim16 -> shim8|27.707602|25.667846|100/100 on each rank|

All max_abs0 and replay-stable. Raw rank-max samples, all-rank witnesses,
log digests and source-header digest are in the adjacent JSON. These are
bounded component checks, not model teacher-forced or E2E correctness proof.

8blocks saves2.039756us (~7.36%);24blocks loses2.508857us. In this two-stage
kernel,16384 packed elements /8 ranks gives2048 elements per rank; at512
threads only the first4 blocks perform reduction/gather arithmetic. Additional
blocks still execute synchronization. This is a source-based explanation
consistent with results, not hardware-counter proof of the entire cost.

Next bounded step: extend the oracle to4blocks and test longer consecutive
collective replay sequences before service integration. Any service hook must
be default-off and restricted to native TP8 M32 BF16; C1/prefill/spec retain
their existing paths. Preserve1M KV. Do not infer an E2E gain by multiplying
the per-collective saving by86; overlap and rank arrival must be measured.

Run pattern:
```sh
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
PYTHONPATH=/home/pc/Code/sglang/python:/home/pc/Code/sglang \
numactl --interleave=all /home/pc/anaconda3/envs/DS/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=8 \
  scripts/rocm/bench_dsv4_tp8_decode_ar_variants.py \
  --candidate-blocks 8 --shim-baseline
```

Python syntax checks and actual shim compilation passed. Production service
was not restarted, library source was not edited, and model defaults are intact.
