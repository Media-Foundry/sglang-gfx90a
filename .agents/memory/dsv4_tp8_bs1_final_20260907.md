# TP8 M36864 migration and first BS1 optimization: validated profile

## Outcome and measurement scope

- Original DeepSeek-V4-Flash safetensors, TP8/EP1/no-A2A, native AR.
- M36864 BF16-CK prefill migrated with corrected shuffled-scale interpretation.
  Latest five P32 waves:5332.84 cold;6424.01/6416.69/6414.85/6277.46 warm
  input tok/s. Warm median6415.77, preserving the earlier6420.39 checkpoint.
  Each wave has32 distinct code-source requests,73724 total input tokens,
  admitted up to16 at once; this is aggregate prefill, not C1 throughput.
- BS1 wave64 wo_a: fresh control A4 median76.48495 versus fresh candidate B4
  84.22330 output tok/s, **+10.1175%**. Trimmed means76.45137/84.22429.
  Three fixed, distinct code tasks, five measured repetitions each,256 output
  tokens, HTTP wall time including short prefill. Warmup excluded explicitly.
  This is repeated/trimmed independent-service A/B, not a claimed full ABBA.
- Prior independent candidate D1:median83.85771. Reproduction agrees in scale.

## Correctness and reliability evidence

- Current combined unit regression:15 tests passed; scale-layout oracle:
  TP4/TP8 x gate/up/down,10 random byte mutations each, all exact.
- A4:15/15 completions equal original control; France and6 teacher probes pass.
- B4:15/15 completions equal prior candidate D1; France and6 full-prefix
  teacher probes match reference IDs/logprobs. The latter are prefill probes,
  not proof of cached-decode arithmetic equivalence.
- D1:60 probes, five P32 waves, another60 probes and two long code responses
  all complete. B4:another60 probes and two long responses all complete.
  All180 stress probes match next IDs and input/output logprobs to reference.
- Each long response has460 tokens with normal EOS and matches prior candidate
  output. Coherent prose is a smoke test, not validation of every audit claim.
- The delayer's local wall-clock deadline could split admission decisions across
  TP ranks. The fixed six-field existing all-gather publishes the timeout bit;
  all ranks use the same decision. No new collective. Old-method skew oracle
  produces [wait,run]; fixed method gives identical decisions.

## Boundaries (do not overclaim)

- No checkpoint re-quantization or speculative decoding. The BF16-CK prefill
  arithmetic and wave64 decode reduction are **not bit-exact** with the former
  compute paths. Two short candidate trajectories differ from old GEMM; they
  repeat consistently across candidate services. Concurrent BF16-CK outputs
  retain small greedy wording differences; exact expert atomic reduction is
  not claimed. User accepted small numerical/semantic drift.
- This validates a131072-token pool, not a1M-context capacity test.
- No BS32 decode speedup claim; GEMV is strictly TP8,BS1,native-decode only.
- Global defaults remain conservative. The validated launch opts in explicitly.
- Measurements used the shared working tree; unrelated pre-existing kernel/debug
  edits were preserved, not rolled back or silently included in these commits.

## Reproduce the validated profile

Use the DS conda environment through the existing harness. Check amd-smi PIDs
before launching; do not overlap another benchmark on these GPUs.

```bash
NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 \
SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV=1 \
SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE=1 \
SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE=1 \
SGLANG_DSV4_C4_TRIVIAL_LOGITS_SKIP=1 \
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8 EP_SIZE=1 \
MOE_A2A_BACKEND=none PORT=30011 HOST=127.0.0.1 \
MEM_FRACTION_STATIC=0.80 MAX_TOTAL_TOKENS=131072 \
CHUNKED_PREFILL_SIZE=36864 CUDA_GRAPH_MAX_BS_DECODE=32 \
CUDA_GRAPH_BS_DECODE='1 2 4 8 16 24 32' DISABLE_OVERLAP_SCHEDULE=0 \
scripts/rocm_dsv4_flash.sh serve
```

Set only `SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV=0` for the decode control.
Keep the synchronized delayer correction in both arms. No WAR/AG overrides or
periodic stack diagnostics are needed for these accepted runs.

Key commits:137170e367 (TP8 prefill),f922e37908 (wo_a candidate),
a3813c9d72 (delayer consensus). Adjacent JSON contains closing A/B samples,
hashes and artifact checksums. Earlier full outputs/prefill samples are in
`dsv4_tp8_delayer_consensus_20260907.json`; investigation and rejected controls
are in `dsv4_tp8_transition_isolation_20260907.md`.
