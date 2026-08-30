# DSV4 TP4 BS64 native-AR diverse-request probe

Date: 2026-08-30

## Scope

This is a four-GCD TP4, EP1/no-A2A, native autoregressive decode probe at
batch size 64. Speculative decoding and DSpark were not enabled. The workload
contains 64 distinct token-ID prompts and generates exactly 256 tokens per
request with greedy decoding.

The first 32 prompts are the established diverse-request oracle. The remaining
32 are deterministic compound prompts formed from different pairs in that
oracle. All 64 prompt strings and all 64 token-ID sequences are unique. The
tokenizer SHA256 recorded by the manifest is:

```text
8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf
```

## Service configuration

```text
TP=4
EP=1 / no A2A
CUDA graph tiers=1,32,64
max total tokens=65536
MXFP4 quant rows=384
Mori max dispatch tokens/rank=384
Mori decode max dispatch tokens/rank=128
M32 logical W2 scale cache=off
```

The M64 path safely falls back to the generic grouped FP4 implementation. The
M32-only DPP, row-prefetch, and logical-scale specializations are guarded by
the exact `(32,4096)` shape and are not selected. Disabling the unused logical
W2 scale copy avoids roughly 688 MiB/GCD of dead storage for this M64 probe.

## Results

Two independent, clean common-resident measurements were obtained:

| repeat | common-resident decode throughput |
|---:|---:|
| 1 | 949.930 tok/s |
| 2 | 949.915 tok/s |
| center | **949.923 tok/s** |

The scheduler decode moments after the accepted runs were:

```text
decode steps       = 765
sum batch          = 48960 = 765 * 64
sum step time      = 51160453 us
mean decode step   = 66.876 ms
model decode rate  = 956.989 tok/s
```

The harness retains the historical JSON field name
`resident_bs32_tok_s`; with `request_count=64`, it denotes the window in which
all 64 requests are resident, not a BS32 measurement.

The full HTTP group-wall aggregate varied with heterogeneous prompt prefill and
admission (one clean repeat was 291.76 tok/s over 56.16 s). It is not used as
the steady decode result. The common-resident HTTP value and scheduler decode
moments agree within 0.8%, making approximately 950--957 tok/s the defensible
BS64 decode range.

## Correctness

For both accepted measurements:

- all 64 requests completed;
- every completion contained exactly 256 token IDs;
- every finish reason was `length`;
- the fixed France oracle prefix matched exactly;
- all selected input token sequences were distinct.

One attempted multi-round invocation was discarded after later rounds were
started before the service had drained the previous round, causing the running
request count to exceed 64. Its first clean round produced 949.930 tok/s and is
retained; polluted later rounds are not reported.

## Interpretation and next experiment

Relative to the established TP4 BS32 service result (about 629 tok/s by the
same HTTP resident style and about 695--707 tok/s by scheduler/model timing),
BS64 raises model throughput by roughly 35--38%, but remains well below the
1500 tok/s stretch target.

The current M64 gate grid (`G2080`) is inherited from the M32-specialized
profile rather than selected by an M64 oracle. The smallest justified next
sweep is therefore A4/R2/W8/LDS with gate blocks 832/1248/1664/2080, fixed
down D832, and a captured real diverse M64 route. Existing TP8-shaped A8 data
showed only a 1.05% improvement and does not justify changing assignment size.
