# TP8 DSpark strict M192 C128 CK attention checkpoint (2026-09-10)

## Scope

- Original DeepSeek-V4-Flash weights, TP8/EP1/no-A2A, 1M-token pool.
- Strict gamma-five C32 target verification (`M192`, width six).
- Only the C128 HCA attention path selects the CK/HIP H8 MFMA kernel.
- C4 refined attention, gamma-three M128, draft, prefill and native AR remain
  unchanged.

## Standalone oracle

The former M128 wrapper cap was extended to M192; the underlying kernel already
uses runtime token grids. On physical GCD 4, 1000 HIP Graph replays were stable
and the output remained within the existing BF16 attention tolerance.

| visible keys | Triton | CK/HIP | ratio |
|---:|---:|---:|---:|
| 17 | 33.67 us | 49.35 us | 0.68x (CK loses) |
| 128 | 105.79 us | 87.73 us | 1.21x |
| 512 | 391.41 us | 218.61 us | 1.79x |

The production selector is C128-only. Short synthetic rows lose, but the real
service result determines whether the actual HCA length distribution amortizes
the two-launch split/reduce path.

Evidence: `/tmp/dsv4_sparse_h8_m192_oracle_v3.json`.

## Service comparison

Same 32 fixed heterogeneous code requests, 1024 output tokens/request:

| arm | resident rounds | median | mean acceptance |
|---|---|---:|---:|
| M192 MoE checkpoint | 968.06 / 972.21 | 970.13 tok/s | 3.319 / 3.329 |
| + M192 C128 CK | 982.68 / 981.05 | 981.87 tok/s | 3.367 / 3.340 |

The incremental median gain is **1.21%**. France was exact for 32/32 requests.
All code requests completed 1024 tokens. Candidate repeated-8gram statistics:
median 0.01967, p95 0.07080, maximum 0.42970, zero above 0.8.

Artifact: `/tmp/dsv4_tp8_gamma5_m192_ck_B_c32x1024.json`.

## Decision

Accepted as a small exact-execution improvement and enabled only under the
explicit TP8 full-target gamma-five profile. It is not a major step toward 2k;
the next priority is M192 all-reduce, which still misses the tuned M128
collective selector.
