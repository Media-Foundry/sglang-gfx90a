# TP8 C16 8K prefill: 32768 chunk-budget screen

2026-09-14. Measured implementation HEAD: `2d4ac02b16` on
`sync/media-foundry-gfx90a`. Original DeepSeek-V4-Flash, not V4.1 or DSpark.

## Result and decision

Retain **32768 as a workload-specific option**, not a global launcher default.
The mean of the two control leg medians is **5174.54 input tok/s**; the mean
candidate median is **5310.12 input tok/s**, a **2.62% improvement**.
The corresponding wave times are approximately **25.33 -> 24.68 s** for
131069 actual input tokens, saving about **0.65 s**. These are concurrent-wave
prefill rates, not C1, resident decode, or isolated GPU-kernel throughput.

| Leg | Chunk budget | Three waves, input tok/s | Median | Median wave, s |
| --- | ---: | --- | ---: | ---: |
| A1 | 36864 | 5177.84 / 5179.50 / 5176.98 | 5177.84 | 25.3134 |
| B1 | 32768 | 5313.25 / 5314.16 / 5306.88 | 5313.25 | 24.6683 |
| B2 | 32768 | 5307.00 / 5309.57 / 5303.90 | 5307.00 | 24.6974 |
| A2 | 36864 | 5171.47 / 5171.24 / 5166.71 | 5171.24 | 25.3457 |

A2 returned within -0.1274% of A1. Within-leg full ranges are 0.049%, 0.137%,
0.107%, and 0.092%, respectively. This is a bounded ABBA screen: A1 and A2
are independent service starts; B1/B2 share one candidate process. It is not
an independent multi-process replication of the candidate or a confidence
interval over arbitrary production traffic.

## Unchanged contract

- TP8 / EP1 / no-A2A, eight gfx90a GCDs, original stored weights.
- Allocated 1048576-token logical KV pool, static fraction 0.96.
- Graph tiers 1/2/4/8/16/32/64 retained; native AR only.
- Existing large-prefill BF16-CK path and FP32 stage-2 accumulation retained.
- Admission 16, existing 20 ms delayer, identical compute flags.
- Same 16 distinct public-source code requests, roughly 8192 input tokens each.
- Fresh cache salts; every recorded cached-token count is zero.
- One output token per timed request; rate is actual input tokens divided by
  earliest request start to latest first token. One excluded warmup per process.
- Only `CHUNKED_PREFILL_SIZE` and `MAX_PREFILL_TOKENS` change together.

The historical 5171.36 C16 matrix point is reproduced in scale, but was not
substituted for the contemporaneous controls. The old roughly 6.4k TP8 result
used roughly 2304-token requests and a smaller pool; it is not this workload.

## Actual execution, not the initial arithmetic assumption

The initial expectation was four large forwards for both budgets. Logs show:

```
A, every measured wave: 36864 + 36864 + 36864 + 4096 + 16384  (5 forwards)
B, every measured wave: 32768 + 32768 + 32768 + 32768          (4 forwards)
```

Log counts include padding; their sum is 131072, whereas the throughput
denominator correctly remains 131069 actual input tokens. The aligned budget
eliminates the observed separate 4096-token tail. This is a plausible explanation
of the E2E improvement, not a profiler attribution of all saved time to one kernel.
The historical 32768 rejection on 2304-token requests added a forward instead;
that negative result remains valid for its different workload.

## Correctness scope

All three fresh services answer France with "The capital of France is **Paris**."
In addition to all one-token waves, each service completed two C16 x 128-output
code waves. In total **336 completion ID/text pairs** passed independent local
tokenizer decoding and length checks. The 32 candidate output snippets were
manually inspected: coherent source-related prose, no obvious garbling or
repetitive collapse. This does not certify the factual accuracy of generated
code-review claims, complete executable answers, or long-generation quality.
The quality harness forces 128 output tokens (`ignore_eos=True`).

Full 128-token repeat equality is **3/16 for A1, 7/16 for B, 3/16 for A2**.
Thus even control is not a bitwise reference. We do not claim this chunk change
preserves exact logits or attribute all differences to one source. Existing
BF16-CK atomic accumulation and changed batch/chunk shapes remain relevant.
No new precision approximation was introduced by this experiment.

## Reproduction and next step

`run.py` creates an exact arm-local launcher, checks GPU ownership, records
server info and source hashes, and stops only its PID/birth/command-verified
service. The candidate invocation and all inherited flags are preserved in
`B/P16-B.state.json` and `B/start-ar-matrix.sh` in the evidence archive.
For this workload the only budget overrides are:

```bash
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
PREFILL_MAX_REQUESTS=16
```

Keep the remaining tested TP8 configuration unchanged. The existing matrix
launcher hard-sets the old budget, so merely exporting these ahead of that old
script will not work; use the saved candidate launcher or edit the exact budget
line as `run.py` does. This report does not change the global launcher.

Before generalizing, cover other input lengths, prefix hits and mixed decode
load. A measured length/prefix chunk policy is a better next scheduling direction
than universally replacing 36864 with 32768. Deeper kernel work needs a fresh
C16 x 8K profile; old 2304-token MoE proportions are not a valid cost budget.

`summary.json` contains the raw rates, log lines and SHA256 inventory;
`measurement-evidence.tar.gz` contains all three arm directories. `pack.py`
verifies each archive member against its recorded hash and keeps originals.
All owned services were stopped; final `amd-smi process --json` showed no GPU
processes on GCD0-7. No production model/kernel code was changed.
