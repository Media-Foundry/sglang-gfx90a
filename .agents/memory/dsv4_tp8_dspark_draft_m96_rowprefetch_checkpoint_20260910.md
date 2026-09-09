# TP8 strict DSpark draft-only M96 row-prefetch checkpoint (2026-09-10)

## Scope

- Original DeepSeek-V4-Flash checkpoint weights.
- TP8 / EP1 / no A2A, strict full-target DSpark gamma three.
- Target verification remains M128 and keeps all existing exact target paths.
- The only candidate change is the three-layer draft model's physical M96
  routed MoE: A4/R2/G832/D832 selects the subgroup-8 row-prefetch kernels.
- Native AR, prefill, target M128, gamma-two target M96, and other draft tiers
  are excluded by an explicit draft capture/replay context plus exact shapes.

## Contract repair

The draft graph is captured under `DSparkWorker._draft_context()`.  Publishing
the selector in the generic target forward scope did not reach draft capture.
The accepted implementation therefore uses a ContextVar owned by
`_draft_context()`, which covers initialization, graph capture, and replay.

The actual draft contract observed at capture was M96/H4096/top-6,
W13=[256,512,2048], W2=[256,4096,128], A4/R2, TP8/EP1.  Its generic path had
LDS disabled.  The narrow selector enables the same LDS E2M1 decode contract
as the accepted M128 row-prefetch path.  M96 must use the TP8 subgroup-8 down
module; routing it to the older TP4 subgroup-16 module is rejected at compile
time by design.

## Real-code service ABBA

Every arm used a fresh service, an excluded warm wave, and one measured wave
of 32 heterogeneous code requests with 1024 generated tokens each.  Gamma,
target M128, target kernels, acceptance synchronization, and the one-million
token pool were unchanged.

| arm | resident output tok/s | severe repetition |
|---|---:|---:|
| A1 control | 1079.9808 | none |
| B1 draft M96 row-prefetch | 1093.4372 | none |
| B2 draft M96 row-prefetch | 1082.2507 | none |
| A2 control | 1081.0051 | none |

Means: 1080.4929 -> 1087.8439 tok/s, **+0.6803%**.  All 128 measured
responses passed the severe-repetition gate.  The underlying row-prefetch
arithmetic had already passed 100 input mutations and 1000 graph replays
bitwise against the production grouped kernel.

Decision: enable only in the explicit strict TP8 full-target profile.  This is
a small accepted optimization, not a material step toward 2k by itself.

Artifacts: `/tmp/dsv4_tp8_draft_m96_rowprefetch_abba_retry5_20260910/`.
