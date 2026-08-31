# DSV4 DSpark M128 anchor routed compaction checkpoint

Date: 2026-08-31

## Finding

The gamma-three anchor-only checkpoint initially used `-1` sentinels for the
three draft rows of every four-row request window. A layer-20 graph marker
showed that this preserved correctness but did not contract the routed runner:

```text
M128 coarse layer span: about 1.80 ms
M128 MoE span:          about 0.98 ms
router:                 45--48 us
top-k:                  15--17 us
routed FP4:             842--856 us
TP4 AR/tail:            55--65 us
```

The runner still quantized/sorted/launched as M128 even though only 32 anchor
rows had valid expert IDs. Sentinel masking reduced useful assignments but not
the dominant shape-dependent work.

## Implementation

Under the already strict gamma-three parent selector, the optimized path now:

```text
hidden_states[0::4].contiguous()             -> [32,4096]
topk weights/IDs/router logits [0::4]        -> contiguous M32 TopK output
existing original-weight routed MoE          -> [32,4096]
zeros_like([128,4096]); scatter output[0::4] -> full target layout
```

Shared experts and all non-routed model paths still run on M128. No weight is
modified or repacked. The compaction helper fails loudly unless TopK output is
STANDARD, rather than silently applying an incompatible backend format.

Reachability requires the parent selector:

```text
gfx90a + TARGET_VERIFY + BS32 + width4 + hidden [128,4096]
```

and the child switch:

```text
SGLANG_DSV4_GFX90A_DSPARK_M128_COMPACT_ANCHOR_ROUTED=1
```

The child is short-circuited after the parent guard, so native AR cannot read
or enter it. Setting the child switch to zero restores the same-code M128
sentinel-only control.

## Correctness

Unit tests verify selector isolation and that compaction selects exactly row
zero of each four-row request window for weights, IDs, and router logits while
returning contiguous M32 tensors.

Across both candidate services:

```text
France first nine IDs: exact in all six rounds
France semantic answer: Paris in all six rounds
32 varied code requests: every request generated 256 tokens
finish reason: length for every request
```

The native-AR negative control forced both parent and child environment
variables to one. It still passed France exact/Paris and 32 varied 64-token
requests, with no speculative acceptance fields. Its resident rate was
`713.864626 tok/s`, recorded only as reachability evidence.

## B-A-B result

Physical GCDs 0--3, 32 different concrete code prompts, 256 generated tokens,
and `stream_interval=1`:

```text
B1 compact M32 anchors:
  879.910247, 900.911365, 937.815547
  median 900.911365 tok/s

A sentinel-only M128:
  828.747456, 832.898228, 929.416857
  median 832.898228 tok/s

B2 compact M32 anchors:
  1014.921952, 800.082523, 901.845324
  median 901.845324 tok/s

center(B1,B2) = 901.378345 tok/s
gain vs A      = +8.222% 
```

The wide per-round spread correlates with accepted length, but both independent
candidate medians are near 901 tok/s and both exceed the control median. One
candidate round crossed 1,000 resident tok/s; this is not yet a stable 1k
checkpoint and must not be reported as the center.

## Decision

Enable physical M32 anchor compaction by default inside the TP4 BS32 gamma-three
DSpark profile. It stacks on the gamma-three checkpoint and raises the stable
resident center from roughly 876--888 to roughly 901 tok/s. It remains far from
the 1.5k objective; the new default M128 attention/indexer/compressor path is
now the largest remaining budget.

Rollback:

```text
SGLANG_DSV4_GFX90A_DSPARK_M128_COMPACT_ANCHOR_ROUTED=0
```

Artifacts:

```text
/tmp/dsv4_m128_compact_probe.json
/tmp/dsv4_m128_compact_b1.json
/tmp/dsv4_m128_sentinel_a.json
/tmp/dsv4_m128_compact_b2.json
/tmp/dsv4_ar_m128_compact_negative.json
/tmp/sglang_dsv4_dspark_gamma3_trace.log
```

