# DeepSeek V4 Flash C1 audit and candidate screen (2026-09-05)

## Scope and reproducibility

Read-only production audit; no runtime selectors or kernels changed.
HEAD e12bb84055, with existing dirty model/backend/kernel files retained.
Consequently these measurements describe this working tree and dependency
build, not a clean checkout of the commit alone.

Service: TP4/EP1/no-A2A, original checkpoint, native AR, physical GCDs 4--7,
port 30011, memory fraction 0.90, 8192-token pool, default chunk 2304 and
decode graph tiers 1/2/4/8. Large-prefill and DSpark profiles were not enabled.
Command: `HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 EP_SIZE=1 MOE_A2A_BACKEND=none
PORT=30011 MEM_FRACTION_STATIC=0.90 scripts/rocm_dsv4_flash.sh serve`.
AIter module_custom_all_reduce.so SHA256:
`0854436bd273f015e3df4fc1ccd60858cd9ce5925dd0e8a03be43542326d6948`.
AMD-SMI checked before GPU work; no competing processes. Service stopped after
tests; final AMD-SMI showed no running processes.

## Current C1 performance

Three rounds per case, sequential requests, 256 completion tokens, greedy,
ignore_eos=true, distinct cache salts. Streaming measured TTFT separately and
decode as the token-count delta divided by the first-to-last token interval.
Short cases use existing dsv4_tp8_diverse_32_input_ids.json; the long case is
request 0 in dsv4_prefill_diverse_32_input_ids.json (real server_args.py source).

| Task | HTTP tok/s median | Decode tok/s median | TTFT median seconds | Distinct hashes / 3 |
|---|---:|---:|---:|---:|
| Python linked-list reversal | 53.20 | 55.95 | 0.204 | 1 |
| SQL duplicate email query | 54.24 | 56.24 | 0.185 | 1 |
| Merge sorted arrays pseudocode | 54.37 | 56.53 | 0.183 | 1 |
| Configuration audit, 2304 input tokens | 43.01 | 51.55 | 1.005 | 3 |

Independent France sentinel: 2/2 exact historical nine-token sequence.
Inspected output text is coherent code/explanation. This is a semantic smoke
test, not a broad code-quality evaluation or proof of long-context numerical
parity. The long-case variation has not been localized or bounded by logits.

Historical harness prompt (2+2, 256 tokens) measured
54.136 / 54.582 / 54.705 HTTP tok/s, all hash `38c3d431e7c1dd65`.
Thus the difference from August's approximately 74--75 tok/s cannot be
explained solely by using code prompts. Historical builds/configs and output
trajectories differ; the cause is NOT established by this audit. Compare
current and historical configuration/backend hits and rank-max stage timings
before blaming communication, clocks, or a specific commit.

### Historical topology and speculative-mode verification

The user specifically questioned whether 74.5 tok/s was TP8 or DSpark. Git
history supports a distinct TP4 native-AR result:

- cc4691b542 (author date Aug 24): records native AR
  74.50/72.07/74.39/68.63/74.08/74.53/73.88, hash 51e2ac132057ead3.
- 7953551a301 (author date Aug 24): adds
  74.53/74.95/74.87 tok/s, explicitly marked native AR.
- experimental_switches.md lines 908--911 explicitly label the MFMA64
  prefill ABBA service TP4/EP1/no-A2A and record its short native-AR negative
  control at 74.37/74.69 tok/s, with the same completion hash.
- The script at 7953551a301 defaults to four visible GCDs and TP4; speculative
  arguments are added only for the explicitly separate DSpark command mode.
- The subsequent TP8/EP1 bring-up section is dated Aug 26 and initially
  reports about 65.5 tok/s with a different completion hash.

This establishes the historical report's claimed topology/mode, not a fresh
reproduction: original process logs for the 74.5 runs were not found during
this audit. Do not conflate the separate TP8/DP2 76-tok/s records with TP4.

## M1 attention geometry screen

Isolated on physical GCD4 after stopping the service. Existing Triton paged
decode wrapper, BF16 Q[1,16,512], BF16 KV, synthetic 128/640-entry selections,
FP32 sink; inverse-RoPE fusion not included. Capture warmed up each candidate.
Five forward/reverse timing rounds, 100 graph replays per sample; 20 mutations
of Q/KV/sink, comparing against split64/warps4. These are component results,
not a production performance change.

| KV entries | splits / waves | median microseconds | max abs vs control |
|---|---|---:|---:|
| 128 | 64 / 4 | 18.928 | 0 |
| 128 | 32 / 4 | 18.664 | 0 |
| 128 | 16 / 4 | 18.664 | 3.05e-5 |
| 128 | 8 / 4 | 18.184 | 3.05e-5 |
| 128 | 64 / 2 | 16.936 | 0 |
| 128 | 16 / 2 | 16.676 | 3.05e-5 |
| 640 | 64 / 4 | 19.445 | 0 |
| 640 | 32 / 4 | 24.161 | 0.001953125 |
| 640 | 16 / 4 | 30.559 | 0.001953125 |
| 640 | 8 / 4 | 42.561 | 0.001953125 |
| 640 | 64 / 2 | 17.395 | 0 |
| 640 | 16 / 2 | 24.049 | 0.001953125 |

All samples finite. Candidate split64/warps2 is exact in these tests and saves
about 2 us/call. Even 43 exposed calls would save only about 86 us/token, around
0.5% at the measured C1 latency. Before wiring: test fused inverse RoPE, ragged
and empty selections, then France and real-code service ABBA. Reduced split
counts are not a general win and change floating-point reduction ordering.

## Priorities from the memory review

1. Recover the current-vs-historical C1 cost breakdown. Historical markers and
   microbenchmarks often included different TP8/split layouts or allocation
   costs; do not reuse their percentages as current TP4 evidence. Existing
   GPU realtime markers are preferable to capture-time Python timing events.
2. C1 attention split64/warps2 is a concrete small exact candidate. CK/MFMA
   attention selectors currently specialize larger M64/96/128 shapes; their
   wins do not establish a C1 win. Attention has 16 local heads, unlike M1
   projection GEMV, so its matrix utilization must be evaluated separately.
3. Long-context sparse graphs: the 2048 raw-token activation threshold and
   pool-based C4 width cap are already fixed. Length-bucketed graphs could
   reduce the remaining pool-capacity scan without shrinking valid context.
   Measure C1 2K/4K/8K with the same pool before choosing buckets. Full logits
   plus TopK remains, but TopK already fuses physical-slot conversion; do not
   budget a nonexistent separate slot kernel. Scan/local-select fusion remains
   research, and its candidate traffic is not a win at every length.
4. Consider wqkv_a plus segmented q_lora RMS only after a C1 full-boundary
   lower-bound oracle. The dot output spans CTAs, so a fusion needing an extra
   global barrier could erase the benefit. Existing Q/K norm/RoPE/cache-store
   fusion and local-head q_out allocation are already present.

Do not repeat without new evidence: M1 BF16 MFMA projection (far slower than
wave64 GEMV), FP16 router (micro win, service ABBA neutral), single-kernel HIP
Q/K norm replacement (E2E slower), MHC v_dot2 stage0 (micro win mostly hidden),
or AIter-preshuffled M1 MoE (52.69 -> 69.76 us full routed stage).

Artifacts: /tmp/dsv4_c1_audit_20260905_server.log,
/tmp/dsv4_c1_audit_france_20260905.json,
/tmp/dsv4_c1_audit_measurements_20260905.jsonl,
/tmp/dsv4_c1_audit_legacy_prompt_20260905.log,
/tmp/dsv4_c1_attention_screen_20260905.log.
