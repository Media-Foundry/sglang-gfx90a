# TP8 DSpark M128 down-consumer rejection (2026-09-10)

Extended the standalone exact down-consumer quantization oracle to M128/I256
and screened a real TP8 recorder route (pass 37, layer 20; 138 active experts,
258 A4 blocks).  The candidate keeps gate, sorter, fixed FP32 reduction and
weights unchanged, but quantizes the BF16 intermediate inside the down CTA.

## Component result

On physical GCD 4, all CTA choices 4/6/8/10/12/14/16 passed 100 input
mutations and 1000 HIP graph replays bitwise.  Best full chain was CTA8/10:

```text
baseline 691.3 us -> candidate 630.1 us  (-8.9%, -61.2 us)
quant+down 307.7 us -> 239.9 us          (-22.1%)
```

This is component evidence only.

## Service result

The candidate was temporarily wired behind a new speculative-only M128 flag
(`SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_DOWN_CONSUMER`) and tested with TP8/EP1,
strict gamma=3, 32 heterogeneous requests.  Code-only workload rounds reached
1170/1179 tok/s versus baseline rounds 1124/1150, but this was not sufficient
without correctness gates.  A mixed workload containing the France sentinel
and varied real prompts produced:

- France first-nine exact and semantic Paris: passed;
- resident: 1039.6 / 1048.8 tok/s;
- severe 8-gram repetition (>0.8): 3 and 4 of 32 requests;
- cross-round all-output exact: false.

The production selector and environment variable were removed immediately.
The standalone oracle and M128 wrapper support remain for future diagnosis.

## Decision

Do not enable this path.  Exact intermediate/partial component output does not
prove full speculative-chain correctness; the failure likely involves stream,
buffer ownership, or interaction with the mixed prompt/acceptance schedule.
Any revisit must first capture first-divergence hidden/logit state before timing.

