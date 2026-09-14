# Original V4 query16: runtime-M component and compilation-cache proof

## Motivation and status

The completed query4-vs16 service ABBA improved warm C16 throughput from
6778.9666 to6877.4848 input tok/s (+1.4533%), but two serving-time `reuse`
compile bursts cost up to8.77s and9.33s per slowest rank in the excluded warmup.
Group16 remains opt-in; production defaults remain query4/6773 checkpoint.

`M` is constexpr in the production `reuse` and `emit` functions but only affects
row bounds. The test-local candidate removes that annotation from both functions
and sets `@triton.jit(do_not_specialize=['M'])` on the top-level kernel. All K
loads, FP8 conversion, per-row dot/head reduction, page checks and zero stores
remain the same. No production runtime dispatch has been changed yet.

Previous goal turn was progress (query16 integration and initial ABBA launch).
This turn completed that ABBA, identified the cold-compilation regression,
retained default4, and obtained runtime-M correctness/cache-reuse evidence.

## Validation (physical GCD4 only, no simultaneous service)

Directory `.agents/experiments/dsv4_c16_indexer_runtime_m_20260915/` contains
`candidate.py`, `screen.py`, initial `screen.json`, strengthened `full.json`,
and `cache-proof.json`. All invocations exited0; do not infer completion from
an intermediate fixture JSON alone. The latest driver writes intermediate
results to a separate `.fixture.json`, publishing its final output only after
extra irregular-M checks complete.

Both arms use group16: production exact-M versus test-local runtime-M. They
include logical and physical deterministic Top-K, but exclude query projections
and compressor. Synthetic Q/K/weights and causal8K request metadata, not a
fresh full-model capture. Strong test results:

| M | Exact-M ms | Runtime-M ms |
|---:|---:|---:|
|8192|4.12029|4.10045|
|32768|16.28891|16.30962|
|65536|32.40815|32.42585|

Three ABBA cycles, five graph iterations per observation. Treat this as
essentially unchanged component latency, not a new throughput win.

- 100 mutations and100 fixed graph replays each at M8192/32768/65536: all score
  bits and both Top-K arrays match.
- 100 mixed-page/ragged/tie mutations at M17, width577, page-table stride[13,1]
  match exactly.
- Additional100 mutations each at M32766/32767/65535, width2048, same stride32
  metadata: all score bits and both index arrays match.
- Total strengthened mutation comparisons: **700**, all exact. This is local
  operator evidence, not proof of whole-model deterministic generation.

## Cache reuse: not merely identical-looking assembly

The strengthened test first found the same AMDGCN assembly digest across six
width2048 shapes. A separate cached follow-up (three mutations per case) then
recorded the compiled object identity, Triton cache hash and HSACO bytes too.
For M8192/32768/65536/32766/32767/65535:

- Same compiled Python object **within that process**.
- Same Triton cache hash:
  `c1f3f7d98823fee34df57cd7b8fb9f7cadc780bcfc6aa34e8c467f4abf915aa9`.
- Same HSACO SHA256:
  `5a36b06175940b27d52bed0762627aec3665330f005cfe9e536ad46aa263bd7c`.
- Same AMDGCN SHA256:
  `62fbe647a65b7bd0a7ede75e59c219bf646b0865ff6113041c20513e0324681c`.
- 64 registers, no spills,8192 bytes LDS.

M17/width577/stride13 correctly has a different specialization (63 registers,
no spills). Width, page stride, layout, dtype and query-group size still may
compile separate variants. This does not eliminate initial compilation or
replace startup prewarming; it specifically removes exact-row-count variants.

The legacy `same_large_shape_binary` field compares AMDGCN text; the added
`same_large_shape_hsaco` and `same_large_shape_cached_object` fields establish
the stronger claim above. All three are true in `cache-proof.json`.

## Remaining work

Integrate behind a default-off, native large-prefill selector without changing
the accepted default4 path. Validate the actual integrated entry and a fresh
service using irregular row counts, requiring explicit backend/group/M-mode
hits, unchanged input IDs, teacher-forced or paired output checks, and no
additional exact-M compilation after the first compatible variant. Recheck
warm E2E before making group16/runtime-M the profile default. Do not report
the standalone cached test as a completed cold-service fix.
