# DSpark gamma-3 semantic-lane overlap oracle (2026-09-01)

## Motivation

After the M128 CK sparse-decode checkpoint, resident throughput is about
1.56k tok/s. Reaching 2k still needs a multi-millisecond structural reduction.
A naive concurrent replay of complete draft and target graphs deadlocked
because their TP collectives interleaved across streams. This oracle therefore
contains compute kernels only and models the dependency-correct gamma-3 split:

```text
draft lane:  shared expert M96 at layer L
             -> CK attention M96 at layer L+1

anchor lane: routed expert M32 at layer L
```

The two lanes are independent between ordered collective boundaries. Unlike an
earlier M32-attention/M32-MoE oracle, it uses the new CK/MFMA sparse attention
and the actual gamma-3 M96/M32 semantic row split.

## Oracle

`scripts/rocm/bench_dsv4_tp4_m32_attn_moe_overlap_oracle.py` now supports:

- `--attention-tokens 32|96|128`;
- `--shared-tokens 32|96|128`;
- `--shared-with-attention`;
- `--moe-routed-only`;
- `--synthetic` for fixed-seed activations/routes when the old `/tmp` tensor
  dump is unavailable.

Synthetic mode preserves production shapes, actual checkpoint BF16 attention
tail/shared-expert weights, current packed-SDOT routed kernels, and CK sparse
attention. It replaces only activations, projection weights, and route IDs;
this is sufficient for the CU/HBM contention continuation gate. Physical GCD
4 was used after `amd-smi` showed no workload on GCDs 4--7.

All arms used context 512, seven-round symmetric timing, and produced bitwise
identical attention/routed/shared outputs between serial and overlapped orders.

| compute lanes | attention-side us | MoE-side us | serial us | overlap us | hidden us/layer |
|---|---:|---:|---:|---:|---:|
| M128 CK attention vs routed M32 | 469.821 | 492.678 | 982.637 | 860.029 | 122.608 |
| M128 CK attention vs routed+shared M32 | 474.095 | 579.750 | 1082.386 | 955.711 | 126.675 |
| M128 CK attention vs routed M32 + shared M128 | 470.826 | 607.957 | 1091.286 | 953.515 | 137.771 |
| M96 CK attention vs routed M32 | 394.101 | 502.504 | 910.799 | 762.142 | 148.657 |
| **shared M96 -> CK attention M96 vs routed M32** | **526.081** | **503.122** | **1039.647** | **847.907** | **191.740** |

The dependency-correct semantic-lane case hides about 8.25 ms over 43 layers
before MHC and collectives. Overlap efficiency is only 37.3%, so the result is
not a promise, but it clears the 120-us/layer continuation floor with useful
margin.

## Next implementation boundary

Do not overlap complete CUDA/HIP graphs. Implement a strict gamma-3 M128 path
with two long-lived row lanes and one ordered communication sequence:

```text
compute stream A: anchor routed M32
compute stream B: draft shared M96 -> draft attention M96
communication:    every TP collective issued in the same rank-global order
join:             only at the row-state boundary required by final logits
```

The first four-rank prototype must remain target-verify-only, preserve original
weights, and validate:

1. identical row mapping `[anchor,d0,d1,d2]`;
2. per-lane residual/post/comb state survives all 43 layers;
3. no collective is launched concurrently or reordered across ranks;
4. BS1 France exact and fixed heterogeneous BS32 completion;
5. at least 100 us/layer net saving after MHC and communication, otherwise
   stop before scheduler integration.

No production model path was changed by this oracle.
