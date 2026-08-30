# TP4 M64 structural closures: CK WQ-B, EP4, and gate/down pipeline

Date: 2026-08-30

All service tests used original checkpoint weights, native AR, four gfx90a
GCDs, and 64 real heterogeneous requests. GPU availability was checked with
`amd-smi process` before every run.

## CK WQ-B oracle

The strict M64 TP4 shape was `[64,1024] @ [8192,1024]^T`. The candidate was a
Composable Kernel XDL/CShuffle specialization using the same cached BF16
weight as production.

```text
production F.linear median: 33.398 us
CK candidate median:        51.828 us
candidate regression:       55.17%
```

The numerical result was finite with max absolute error 0.001953125 and
cosine at least 0.99999982 over 100 mutations, but performance fails the gate.
Do not connect the CK selector.

Artifact: `/tmp/dsv4_tp4_m64_ck_wqb_abba.json`.

## TP4/EP4 + Mori M64 oracle

Mori's generic capacity must remain at least the 2048-token prefill chunk;
decode capacity 128 is sufficient for the M64 graph. The service loaded and
captured graph tiers 1 and 64.

Correctness was not acceptable as a production replacement: only 56/64
next-token IDs matched the accepted EP1 teacher and no complete logprob/top-5
row was bitwise identical. The France sentinel still passed.

Performance was decisively worse:

```text
TP4/EP1 resident reference: about 993 tok/s
TP4/EP4 + Mori resident:     454.44 tok/s
```

Thus M64 does not amortize per-layer dispatch/combine, and the EP4 full-expert
K2048 compute path is much slower than the EP1 expert-TP4 K512 sdot path.

Artifacts:

- `/tmp/dsv4_tp4_ep4_bs64_teacher.json`
- `/tmp/dsv4_tp4_ep4_bs64_b.json`

## Two-stream gate/down producer-consumer pipeline

The real pass20/layer34 M64 route has 146 active experts and 174 A4 blocks.
The stable expert-block list was split 87/87 at an expert boundary. The main
stream produced gate chunks; a side stream consumed the first ready chunk with
the exact LDS-quant down kernel. Both streams joined before the unchanged
fixed-slot reduction.

Ten input mutations were bitwise exact at the FP32 partial and final BF16
boundaries. Timings:

| gate blocks | down CTAs/expert | A serial | B pipeline | regression |
|---:|---:|---:|---:|---:|
| 1040 | 8  | 753.384 us | 795.797 us | +5.63% |
| 1040 | 16 | 753.332 us | 787.701 us | +4.56% |
| 2080 | 8  | 753.200 us | 793.201 us | +5.31% |
| 2080 | 16 | 753.320 us | 781.133 us | +3.69% |

Concurrent gate and down scans contend for HBM/L2/CU resources; their overlap
does not shorten the critical path. Do not add a production stream protocol or
extend this to more chunks.

Reusable standalone oracle:
`scripts/rocm/bench_dsv4_tp4_m64_gate_down_pipeline.py`.

