# Original V4 TP8 C16: exact post reuse, +6.54% prefill (2026-09-15)

## Accepted outcome

Three fresh owned services, A1-B1-B2-A2, three timed waves per leg:

| Leg | Input tok/s, each wave | Median |
|---|---|---:|
| A1 | 5575.768 / 5571.706 / 5566.019 | 5571.706 |
| B1 | 5935.570 / 5932.604 / 5933.207 | 5933.207 |
| B2 | 5933.523 / 5933.843 / 5933.885 | 5933.843 |
| A2 | 5566.743 / 5568.501 / 5564.736 | 5566.743 |

Mean of leg medians: **5569.2247 -> 5933.5249 input tok/s (+6.5413%)**.
Mean of leg median-request TTFT: **14.80423 -> 13.91251 s (-6.0234%)**.
Control median wave times: 23.52403 / 23.54501 s; candidate: 22.09075 /
22.08838 s. Do not confuse median-request TTFT with whole-wave duration.

Both arms retain the previously accepted prefill C4 empty-tile skip. This is
a new incremental gain, not a remeasurement of its earlier +5.13% result.

## Fixed contract

- Original DeepSeek-V4-Flash checkpoint at `/home/pc/models/modelscope`.
- TP8 / EP1 / no A2A, native AR, original weight precision.
- 16 distinct real source-code tasks, 131,069 input tokens (~8K/request),
  max_new_tokens=1, zero prefix hits.
- 1,048,576 logical KV tokens verified in runtime startup, not merely requested.
- 32K chunk/prefill token budget, prefill request cap16.
- Every timed leg logs exactly 12 forwards with four requests and rounded
  M32768 (four forwards/wave). Actual inner rows may be 32766/32767/32768.
- Same source SHA256s in all three services; diagnostic markers, stable
  projection/FP32-AR diagnostics, and fixed-slot CK remain disabled.
- Every arm has its own warmup, excluded from the reported medians.
- All eight candidate ranks log actual post-fused4 selection. France is sent
  before warmup and must not trigger this large-prefill selector.

## Runtime change

`gfx90a_mhc_post_fused4.py` keeps H256 and four wave64s, loads x and the four
residual channels once per CTA, and computes all four post outputs with the
original expression. `comb` is indexed `[input_channel, output_channel]`.
RMS partials are computed after the same BF16 rounding, with the same H256
reduction and `[M,64]` layout. Output and workspace shapes do not grow.

`dsv4_prefill_experiments.py` provides a separate ContextVar scope at the eager
runner. It requires original V4 (43 layers, H4096), TP8/EP1/CP1/PP1, native
non-speculative extend, no rewritten/TBO batch, M8192..65536, gfx90a, and no
active graph capture. Scope resets on exceptions. It does not infer AR versus
verification from M alone. Unsupported tensor shape/layout/dtype falls back.

The flag is `SGLANG_DSV4_PREFILL_POST_FUSED4`. After acceptance it defaults to1
only when the launcher combines TP8 multi-request and prefill-throughput
profiles with TP8/EP1/no-A2A. Explicit0 is preserved. The runtime scope excludes
decode, speculative execution, V4.1, TP4, CP, and TBO. No other profile is
default-enabled.

## Correctness: local exactness versus whole-service drift

The **integrated dispatch**, not only the original experimental copy, passed
100 mutations at each M128/8192/32767/32768, all BF16 outputs and FP32 RMS
partials bitwise identical to the current control. The asymmetric comb fixture
and M128 1000 HIP graph replays pass. These inputs repeat/scale 19 real sampled
layer-0 rows and mutate x/residual/post/comb, not 32K independent model rows.

Integrated wrapper component medians (GCD4 only, other GPUs idle):

| M | Control ms | Candidate ms |
|---|---:|---:|
| 128 | 0.07042 | 0.07528 |
| 8192 | 2.05707 | 1.02573 |
| 32767 | 8.25503 | 4.04654 |
| 32768 | 8.26205 | 4.04252 |

The M128 eager wrapper overhead loses; it is not admitted in service. Its
forced test scope exercises kernel/graph correctness, not a proposed decode
optimization.

Service checks:

- All three France probes answer Paris.
- Each service runs two 16-request, 128-output-token checks with explicit input
  IDs and fresh salts. All **96 input echoes** match, no prefix hits, completion
  counts and token-ID/text decoding agree.
- All three services repeat **16/16** full 128-token outputs within their own
  two quality waves.
- Cross-process first quality waves: A1/A2 **15/16**, A1/B **15/16**, A2/B
  **16/16** exact. The sole changed case is case8: 40 common output tokens,
  then ID666 versus ID2019. Thus the two control processes themselves exhibit
  the same difference. It cannot be attributed to the post kernel from these
  outputs. This is not proof of global bitwise determinism or localization of
  that remaining cross-process drift.
- All 16 candidate first-wave excerpts were inspected: coherent, task-related
  code analysis without obvious repetition collapse. Case8 is alternate wording
  around `get_available_gpu_memory`/“function”. The 128-token truncation is not
  a complete functional-code or model-quality evaluation.

Do not claim the optimization solves the earlier compressor/projection/CK
determinism problems. It preserves the tested local arithmetic while improving
service throughput; those are separate conclusions.

## Validation and reproducibility

CPU selector/context/launcher/indexer-marker tests: **16 passed**, one unrelated
pytest asyncio_mode configuration warning. `bash -n` and `git diff --check`
pass. Launcher tests execute only the profile assignment block, never a GPU
service. They cover both required profiles, explicit0/1, TP4, EP2 and Mori.

ABBA used explicit0/1 on an immutable launcher. The subsequent default-line
change is intentionally newer than the launcher SHA in the measurement
manifest; arithmetic source files are unchanged. There is no separate claim
of a new post-default fresh-process performance run.

Experiment directory: `.agents/experiments/dsv4_c16_post_fused4_20260915/`.
`sweep.py` refuses implicit restarts/overwrites; `status.py` validates PID birth
before calling a process live. `analyze.py` requires complete arms, clean owned
stops, equal sources/inputs, actual selector hits, runtime capacity and equal
timed-forward shapes before computing the result.

Evidence archive: `evidence.tar.gz`, 82 files, 2,352,019 bytes.
SHA256: `1e367a81775a4a928f259949843bfb0edd3643021e5135e21472f284f818fc97`.
It includes integrated component evidence, all timed and quality responses,
source manifests, startup/hit/stop logs and the summary. No model tensors.

All three owned services exited; their stop records have no remaining children.
AMD-SMI confirmed all eight GCDs free after the run. Next useful pools remain
the FP32 pre-mix (~4.96 s in the previous diagnostic wave) and C4 logits
(~2.64 s); do not mistake this exact post win for a hardware-limit conclusion.
