# DSV4 prefill TP4/EP4 no-A2A rejection (2026-09-02)

## Question

Test whether full-expert EP4 placement can improve large-M prefill utilization
without paying Mori dispatch/combine.  Attention remained TP4; routed experts
were divided across four ranks and combined by the normal TP reduction.

## Configuration and correctness

- Physical gfx90a GCDs 4--7, original checkpoint weights.
- TP4, EP4, no A2A, queue-aware 2304/4608 chunk policy.
- Raw-FP4 MFMA64 prefill enabled; token-row owner MHC disabled because its
  strict contract requires EP1.
- France returned `The capital of France is Paris.`
- Five C1 rounds produced the same one-token completion hash.

## Results

| workload | throughput |
|---|---:|
| C1, 4604 tokens | 1742 / 1777 / 1777 / 1776 / 1747 tok/s |
| C1 median | 1776 tok/s |
| C32, 73,724 heterogeneous code tokens | 1799 / 1734 tok/s |
| C32 median | 1766 tok/s |

The accepted EP1 path is about 2.5k C1 and 2.5--2.8k C32.  EP4 therefore
regresses both regimes by roughly 29--36%, even after removing Mori entirely.
The full-expert local K2048 path does not recover enough MFMA utilization to
offset losing the EP1 expert-TP4 K512 shard.

## Decision

Reject TP4/EP4 for prefill.  This also strengthens the old M64+Mori result:
Mori was not the only cause of the EP4 loss.  Do not spend time rebuilding Mori
for this experiment and do not retain an EP4 production profile.
