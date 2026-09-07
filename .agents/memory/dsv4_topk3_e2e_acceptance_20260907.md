# Guarded exact Top-K E2E acceptance (2026-09-07)

Supersedes the default-off status in `dsv4_topk_fast_exact_20260907.md`.
Mode 3 is now the gfx90a automatic and launch-script default. Mode 2 remains
an explicit correctness/performance reference. Other architectures unchanged.
The shape guard is unchanged: fast kernel for M<=32 OR score width<=576;
otherwise mode 2 kernel. No score arithmetic, tie policy, or output ordering
changes; model weights and precision unchanged.

## Four independent TP4 service ABBA

Command:
`python scripts/rocm/run_dsv4_topk_speed_abba.py --output-dir /tmp/dsv4_topk23_e2e_20260907 --long-validation`

A1/A2 mode 2; B1/B2 mode 3. Physical GCD4--7, TP4/EP1/no-A2A,
native AR, graph BS1, pool65536, chunk2304, mem0.80, scheduler overlap OFF.
GC frozen once per service. Fresh salts and identical real code input IDs;
first/warmup samples excluded. amd-smi PID records retained per arm.
Both native MFMA prefill selectors enabled; no BF16-CK throughput profile.

| Metric (pooled warm median) | A | B | Change |
|---|---:|---:|---:|
| Short C1 HTTP output tok/s | 66.670 | 67.185 | +0.77% |
| 2304-token C1 TTFT seconds | 0.95089 | 0.93755 | -1.40% |
| C16 prefill input tok/s | 2434.27 | 2415.06 | -0.79% |
| 2304 input +256 output, streaming decode tok/s | 59.541 | 62.122 | +4.33% |
| Same long C1, full HTTP output tok/s | 48.986 | 50.670 | +3.44% |

C16 prefill comprises 16 real different 2304-token inputs, M2304 GPU chunks,
not one M36864 GPU forward. Decode is client streaming time after first token,
not pure GPU timing. Short C1 skips indexer Top-K and is only a regression check.

Service-order variation is material: long decode A1/A2 medians 57.29/59.98,
B1/B2 62.21/60.86; B2 contains a 55.11 outlier. C16 prefill A1/A2 medians
2479.71/2357.97, B1/B2 2425.74/2377.74. Thus no confident prefill win/loss
claim from the -0.79% pooled difference. Keep the negative raw large-M kernel
result and guard; do not claim the standalone 75% translates into E2E.
Full samples/checks: `dsv4_topk23_e2e_abba_20260907.json`.

Correctness, all four arms:
- France exact nine tokens twice per service;
- all short-code 256-token completions match A1;
- six fixed-continuation token-logprob/top20 probes exact;
- long C1 four 256-token runs exact across rounds and arms;
- sixteen different long inputs x32 output tokens: IDs, logprobs, top5,
  and text exact across arms;
- C16 prefill first-token vectors exact; all audited cached-token counts zero.

## Scheduler overlap performance-profile regression check

Separate TP4 service, mode3, otherwise identical but overlap ON:
- short C1 warm median 76.454 HTTP output tok/s;
- long C1 post-first-token decode median 69.958 tok/s;
- long C1 full HTTP output median 55.934 tok/s;
- France, short teacher-forced probes, long C1 completions, and C16 long
  IDs/logprobs/top5/text all exact versus the no-overlap mode2 control.

This is a regression check, NOT overlap-on mode2/mode3 ABBA. Historical
76.84 mode2 short C1 result remains valid; this task does not claim a new
short-context record. Data: `dsv4_topk3_overlap_acceptance_20260907.json`.

## Chunk clarification and remaining scope

2304 is the numerical validation baseline, NOT universally fastest chunk.
The distinct opt-in throughput profile used ceiling36864, request16 and a
20ms delayer, reaching about5841 input tok/s on Sep5. That large-M BF16-CK
record retained first-token drift. It has not been reaccepted by this Top-K
test, nor have its original-precision equivalence questions been resolved.
Do not enable it implicitly or infer its performance from these M2304 tests.

After switching defaults, automatic-selector component validation passed:
1000 graph replays, 1000 score/length/page mutations, and M2304 fast/fallback
CPU stable-sort oracles. TP8 restoration/acceptance is recorded below.

## TP8 default-selector migration

Restored original resident topology/configuration: TP8/EP1/no-A2A on0--7,
pool65536, chunk2304, graphBS1, overlap OFF, localhost:30011. No explicit
Top-K environment override at launch; process environment confirms mode3.
PID1841527, log `/tmp/dsv4_topk3_tp8_restored.log`.

- `/tmp/dsv4_topk3_tp8_c16.json`: sixteen real2304-token inputs x32 outputs;
  IDs/logprobs/top5/text exactly equal `/tmp/dsv4_tp8_topk2_c16_1.json`.
- `/tmp/dsv4_topk3_tp8_c1.json`: real2304-token prompt index2 x256 outputs;
  IDs/logprobs/top5/text exactly equal `/tmp/dsv4_tp8_topk2_c1_1.json`.
- Input manifest digests match and all cache counts are zero.

This is a migration correctness check, not TP8 performance ABBA. The
76.45 tok/s observation belongs to TP4 overlap ON, not this resident TP8
no-overlap validation service. No speculative settings changed.
