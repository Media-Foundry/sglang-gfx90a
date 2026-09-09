# TP8 full-target DSpark transplant (gfx90a)

This opt-in profile preserves the checkpoint and evaluates all target rows.
It is **not** the rejected TP8 anchor-only experiment and is not native AR.
The measured optimizations are the target M128/1-MiB all-reduce grid (80 -> 12
CTAs, through a guarded AIter shim), plus H8 CK attention for C128 layers.

```bash
SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE=1 \
SGLANG_DSV4_GFX90A_DSPARK_TP8_FULL_TARGET_PROFILE=1 \
HOST=127.0.0.1 PORT=30011 \
bash scripts/rocm_dsv4_flash.sh serve-dspark
```

Defaults: eight GCDs (`0,1,2,3,4,5,6,7`), TP8/EP1/no-A2A, 1,048,576-token
pool, memory fraction0.96, DSpark block size3 (target width4), decode graph
tiers1/2/4/8/16/24/32, full indexer graph. The optional prefill profile in this
command matches the measured large-prefill admission configuration. Inspect
`amd-smi process --json` for unrelated GPU owners before starting it.

The profile rejects native-AR commands, TP4/EP2/Mori overrides and anchor-only
profiles instead of silently changing their semantics. It remains default-off.
All runtime optimizations are scoped to DSpark target verification; draft,
native AR and other communication shapes use their existing implementations.

## Evidence and limits

- C32 real-code **single ABBA**: 893.92 /906.09 /900.27 /871.19 tok/s.
  Control mean882.55, tuned mean903.18: observed **+2.34%**.
- Subsequent C128 CK combination ABBA, with grid12 fixed:
  908.05 /939.13 /925.30 /918.52 tok/s. Control mean913.29, combined
  mean932.21: observed **+2.07%**. These are separate comparisons; do not add
  their percentages or claim a directly measured cumulative baseline gain.
- Metric: aggregate output tokens in the common resident decode interval,
  excluding prefill and batch drain; natural EOS, maximum2048 tokens/request.
  One warm wave was excluded for each independent service process.
- Each ABBA's128 measured outputs passed a severe tail-repetition screen. Same-control
  outputs vary across rounds: this is **not whole-model bitwise parity** or a
  comprehensive code-correctness benchmark, nor a statistical confidence claim.
- Component: all eight ranks passed100 exact mutations and independent integer
  sums; alternating M128 tuned/M64 installed communication passed32 collectives
  and1000 graph replays, with all per-step outputs checked after the final replay.
- Native negative smoke: France answered Paris; C32 real-code requests returned
  no speculative statistics and no DSpark H8/communication hit logs even with
  the DSpark attention switches enabled. This is not native performance ABBA.

Raw metrics and SHA256 references:
`.agents/memory/dsv4_tp8_dspark_ar_abba_20260909.json` and
`.agents/memory/dsv4_tp8_dspark_ar_ckc128_abba_20260909.json`.
Full chronology: `.agents/memory/dsv4_tp8_dspark_transplant_20260909.md`.

## Accepted combination and excluded candidates

`SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_SPARSE_DECODE=1` enables the ported
H8 CK C128 attention path and is now the default inside this opt-in profile.
Set it to0 for the communication-only control. The C4 extension and its
refined-probability implementation are enabled by this explicit profile:
`SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_C4_REFINED=1`, together with the parent
C4 switch. It uses a BF16 high/residual-low probability pair for PV, not a
different checkpoint precision. The existing TP8/DSpark-target/M128/C4 guards
still apply; C128 and native AR are unchanged. Real-fixture component tests
passed100 Q/KV/index/sink mutations (including empty/ragged rows), but those
tests do **not** establish whole-model quality or repair the earlier loop.
Its separate C4 ABBA completed: 925.21 /937.05 /938.75 /920.40 tok/s,
control mean922.80 versus refined937.90 (**+1.64% observed**). All128
measured outputs passed the severe-repetition screen, but neither control
nor candidate had cross-round full-output hash parity.

The profile also enables `SGLANG_DSPARK_SYNC_ACCEPT_ACROSS_TP=1`. It
broadcasts rank0's target accept decision before cache commit and therefore
uses eager accept/commit rather than the folded graph epilogue. Six C32x2048
real-code waves passed the severe-repetition gate; this hardens rank
consistency but does not claim cross-run bitwise output identity. See
`.agents/memory/dsv4_tp8_dspark_accept_sync_20260910.md`.
To reproduce the screened candidate:

```bash
SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_C4=1 \
SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_C4_REFINED=1 \
SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE=1 \
SGLANG_DSV4_GFX90A_DSPARK_TP8_FULL_TARGET_PROFILE=1 \
HOST=127.0.0.1 PORT=30011 \
bash scripts/rocm_dsv4_flash.sh serve-dspark
```

Formal metrics and limitations:
`.agents/memory/dsv4_tp8_dspark_refined_c4_abba_20260909.json`.

TP4's M32 gate-prefetch setting depended on anchor-only M128->M32 compaction.
Full-target C32 retains M128, so copying that switch is not a working port.
The rejected `SGLANG_DSV4_GFX90A_DSPARK_TP8_BS32_PROFILE` must not be used for
serving. Historical TP4 ~1.5k numbers used a different target approximation;
they are not a strict full-target TP8 performance promise.

For isolated communication A/B, override
`SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_AR_BLOCKS=80` (same shim, original grid),
or `=0` to use the unmodified installed communicator. M64/C16 communication
has no production CTA override in this profile.
