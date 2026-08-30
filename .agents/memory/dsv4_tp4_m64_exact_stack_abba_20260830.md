# TP4 M64 exact-stack A--B--A and down-consumer reachability audit

Date: 2026-08-30

## Scope

This was a read-only production-code experiment on GPU 0--3.  It used the
accepted TP4/EP1/no-A2A native-AR profile, original checkpoint weights, graph
tiers 1 and 64, a 32K raw-token pool, and 64 genuinely distinct prompts from:

```text
.agents/memory/dsv4_tp8_diverse_64_input_ids.json
```

The accepted profile retained all current defaults: M64 DPP gate, logical W2
scale cache, four-wave W4 down, C128 attention multistream and the CK-style
sparse decode kernel.  A disabled both candidate environment selectors.  B
requested both selectors:

```text
SGLANG_DSV4_GFX90A_M64_FUSED_QUANT_SORT=1
SGLANG_DSV4_GFX90A_M64_DOWN_CONSUMER=1
```

GPU process state was captured with `amd-smi process` before every service or
test run.  The BIO processes had only sub-MiB shared contexts on GPU 0--3 and
no GFX work; they were not stopped or modified.  No `cmake`, `ninja`, `hipcc`
or `clang++` build was active during the measured sequence.

## Important reachability correction

The M64 down-consumer selector is **not reachable** in the accepted logical-W2
profile.  The two production guards are mutually exclusive:

```text
accepted M64 logical-W2:
  quant_info.w2_weight.shape == (256, 4096, 256)

M32/M64 down-consumer:
  quant_info.w2_weight.shape == (256, 4096, 128)
```

Consequently B below is a valid A/B for the reachable fused-quant-sort path,
but it is not a combined fused-sort + down-consumer measurement.  The earlier
record `dsv4_tp4_m64_down_consumer_service_oracle.md` cannot prove a service
effect for M64 down-consumer on the accepted logical-W2 stack: its environment
toggle was set, but the selector was skipped by the incompatible shape guard.
That old result must be treated as a false A/B rather than evidence of a
service-neutral down-consumer.

Testing the true combination requires a production-code adaptation of the
consumer to the `(256,4096,256)` logical-W2 representation and a new component
correctness oracle; this experiment intentionally made no such code change.

## Correctness

Each independently started service ran the 64-row next-token oracle.  B and A2
were compared with A1, including the complete returned structures:

```text
output token IDs:       64 / 64 exact
output token logprobs:  64 / 64 exact
output top-5 logprobs:  64 / 64 exact
```

Every throughput round passed the France first-nine-token sentinel, returned
128 completion tokens for all 64 requests, and ended with `finish=length`.

## Resident decode A--B--A

The real heterogeneous prompts have substantial prefill, so aggregate request
wall time is not used as decode throughput.  The common interval in which all
64 requests were resident at M64 is the primary measurement.  Four rounds were
run per independently started service:

```text
A1 current accepted stack:
  999.410, 999.759, 999.678, 999.748 tok/s
  median = trimmed mean = 999.713 tok/s

B reachable fused quant-sort (down-consumer requested but unreachable):
  1000.020, 1000.684, 1000.008, 999.849 tok/s
  median = trimmed mean = 1000.014 tok/s

A2 current accepted stack:
  1000.679, 999.202, 999.226, 1000.242 tok/s
  median = trimmed mean = 999.734 tok/s
```

Relative to the median of all eight adjacent A samples, B is only:

```text
resident decode: +0.0301%
```

The scheduler model-rate samples (excluding each service's first round, where
the endpoint did not provide a delta) were:

```text
A1: 1014.732, 1015.714, 1014.362 tok/s
B:  1016.624, 1015.602, 1015.445 tok/s
A2: 1015.944, 1014.248, 1013.791 tok/s

B versus combined-A median: +0.1040%
```

Both deltas are well below a useful continuation/default-on threshold and are
smaller than service-to-service and scheduler sampling variation.  Therefore
keep `SGLANG_DSV4_GFX90A_M64_FUSED_QUANT_SORT=0` by default.  Do not claim that
stacking the two switches improves service throughput, because the second
switch never entered its kernel.

## Artifacts

```text
/tmp/dsv4_exact_stack_a1_teacher.json
/tmp/dsv4_exact_stack_b1_teacher.json
/tmp/dsv4_exact_stack_a2_teacher.json
/tmp/dsv4_exact_stack_a1_bench.json
/tmp/dsv4_exact_stack_b1_bench.json
/tmp/dsv4_exact_stack_a2_bench.json
/tmp/dsv4_exact_stack_a1.log
/tmp/dsv4_exact_stack_b1.log
/tmp/dsv4_exact_stack_a2.log
/tmp/dsv4_exact_stack_amd_before_a1_tests.txt
/tmp/dsv4_exact_stack_amd_before_b1_tests.txt
/tmp/dsv4_exact_stack_amd_before_a2_tests.txt
/tmp/dsv4_exact_stack_amd_after_stop.txt
```

The service owned by this experiment was stopped after A2.  BIO processes were
left untouched.
