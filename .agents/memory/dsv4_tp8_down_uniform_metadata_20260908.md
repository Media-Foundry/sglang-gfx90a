# TP8 M32 down uniform metadata: component candidate

Baseline fd366a15ff. Production service PID3391501 and its 1,048,576-token
KV pool stayed unchanged. No service selector yet. Physical GPU4 only; AMD-SMI
ownership checks before component runs found no foreign PIDs. Transient
synthetic gate/down weights are released when the oracle exits, no persistent
weight cache or workspace is added.

## Why the down transformation is legal

Current TP8 K256 down uses eight-lane subgroups and eight subgroups per wave.
With N4096/R2, each sorted A4 expert block has2048 row tasks, divisible by8.
Every wave starts at a multiple of8, and the grid stride832*64 is divisible
by8. Thus all active lanes of a wave share the same expert block, though the
subgroups compute different rows. The terminating task count is also divisible
by8. CPU enumeration checked1..192 A4 scan counts. A compile-time divisibility
assert prevents applying the oracle to a mapping that violates this contract.

Standalone macro `SGLANG_FP4_DOWN_UNIFORM_METADATA_ORACLE` applies
readfirstlane only to expert ID and each of four encoded token/slot IDs.
Weight loading, SDOT, FP32 scaling, subgroup8 shuffle reduction and fixed-slot
reducer stay unchanged. Gate remains the accepted prefetched implementation,
NOT the previous rejected gate-uniform experiment. No production macro enable.

## Five-cycle ABBA results

Ten samples/arm,100 graph replays/sample, drop min/max per arm. Timed chain:
gate + intermediate quant + down partial + fixed reducer. Initial input quant,
sorter and communication excluded; inherited `full_stage=true` is not a full
MoE/model timing claim.

| Synthetic active/scans | Baseline us | Candidate us | Saved us |
|---|---:|---:|---:|
|133/133|275.450|273.164|2.286|
|104/105|242.110|237.608|4.503|
|32/61|183.344|179.198|4.146|

Follow-up uses current real-code C32 recorder pass80, layers0/20/40. TP-summed
counts divided by8, total192 validated. Reconstructed TopK exactly preserves
per-expert histogram but NOT original token/expert correlations. Weights and
activations remain synthetic; these are not full-model numerical fixtures.

| Layer / active / scans | Baseline us | Candidate us | Saved us |
|---|---:|---:|---:|
|0 /136 /136|279.701|275.600|4.100|
|20 /83 /97|221.358|217.783|3.575|
|40 /79 /97|227.474|221.820|5.654|

Do not extrapolate these three isolated layer savings directly to E2E tok/s.
They justify a narrow, default-off native TP8 M32 service candidate and ABBA.
The old down row-prefetch service experiment was withdrawn by the user; this
does not re-enable it or change its status.

## Correctness, resources and test hardening

All six cases passed100 activation/scale mutations: gate BF16, down FP32
partials and final BF16 compared with torch.equal, max final error0. Each
mutation repeated the candidate graph ten times with stable output (1000
total). This is tensor equality, not an explicit signed-zero raw-bit test.
The benchmark now aborts before timing if exact or stable checks fail, including
the FP32 partial check. First synthetic run preceded this hard assert addition;
its saved witnesses satisfy the same assertions. Recorded-histogram run used
the new checks. No claim of complete E2E determinism or semantic validation.

Fresh ROCm7.14 code object notes: down VGPR48 ->40, SGPR36 ->42,
LDS1024 bytes unchanged, scratch0/no spills. Fixed reducer remains VGPR13/SGPR9.
No hardware counter attribution; unlike gate, reduced registers accompany a
positive component result here.

The first recorder run failed importing `scripts` before tests. Retried after
adding repository root as well as python/ to PYTHONPATH; completed normally.
No dependency/environment installation.

```sh
HIP_VISIBLE_DEVICES=4 PYTHONPATH=/home/pc/Code/sglang/python:/home/pc/Code/sglang \
  /home/pc/anaconda3/envs/DS/bin/python \
  scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py --down-uniform \
  --recorder /tmp/expert_distribution_recorder_1788860446.3842516.pt
```

Without --recorder the original synthetic route sequence is preserved. Adjacent
JSON retains all raw samples. Next: native TP8 M32-only selector, exclude C1,
prefill, speculative and unsupported geometry; then correctness and real-code
C32 service ABBA at unchanged1M pool. Production remains baseline until tested.
