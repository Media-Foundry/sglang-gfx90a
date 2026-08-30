# DSV4 TP4/M64 CK-style unified sparse decode

Date: 2026-08-30

## Scope

- Model: DeepSeek-V4-Flash, original checkpoint weights.
- Runtime: native AR, TP4/EP1, no speculative decoding.
- Hardware: four gfx90a GCDs (0--3).
- Workload: 64 heterogeneous input-ID requests driving the resident BS32
  scheduler regime, 256 generated tokens/request.
- Candidate only matches the C128 unified-KV layers at graph tier M64:
  BF16 `q[64,16,512]`, BF16 page-size-one KV, ragged int32 indices/indptr,
  FP32 attention sink.

## Kernel

The source candidate lives in `/home/pc/Code/DSops/composable_kernel` and uses:

- split-K=2;
- four wave64s per CTA;
- native `M16N16K16` BF16 MFMA;
- cooperative QK and D128-per-wave PV;
- register-preloaded Q;
- padded LDS KV tiles with register prefetch;
- a caller-owned 4,210,688-byte workspace per rank.

SGLang calls the graph-safe instance directly. The public CK convenience
entry is deliberately not used because its per-call device-property query is
not legal in HIP graph capture.

## Standalone ABBA

At 64 tokens and 640 KV entries/token on physical GCD0:

| backend | median |
|---|---:|
| CK-style MFMA | 155.73 us |
| production Triton split/reduce | 264.48 us |

The candidate is 41.1% faster. Maximum absolute BF16 output difference on
the fixed full-length case was 0.001953125, mean absolute difference was
0.00018763. One thousand wrapper-level HIP graph replays were bitwise stable.
A separate 100-round graph test mutated ragged lengths across
0/1/7/16/128/512/640 and stayed finite; the largest Triton-relative BF16
difference was 0.015625.

## Service correctness

- 64/64 teacher-forced next-token IDs exactly matched the accepted baseline.
- France check returned `Paris.`.
- Every measured request generated the requested token count and retained
  `france_first9_exact=true`.

## Service A/B/A

Resident decode throughput:

| run | backend | tok/s |
|---|---|---:|
| A1 | CK | 992.77 |
| A1 repeat | CK | 993.12 |
| B | Triton | 982.83 |
| A2 | CK | 994.12 |

The service-level gain is approximately 1.0--1.15%. Scheduler-side counters
were available in the first A service and reached 1001.38/1001.67 tok/s.
Most of the standalone win is hidden by the existing attention branch
overlap, but the residual gain is stable and correctness-qualified, so the
candidate remains as an opt-in building block.

## Related negative result

Changing the C128 attention issue order from production order 3 to order 0
gave 984.73 resident and 992.63 scheduler tok/s versus 983.91/992.02 for the
accepted center. The +0.08%/+0.06% delta is noise, so order 3 remains the
default.
