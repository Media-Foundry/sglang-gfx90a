# Post-owner pre-mix columns3/4 rejected (2026-09-15)

New same-profile phase closure leaves1.7303s/wave in both pre-mix boundaries.
Standalone candidate kept8 rows, independent K1024 FP32 reductions and RMS
consumption; increased activation reuse from2 output columns to3/4. No production
integration. Do not confuse this with the accepted earlier1→2-column optimization.

PhysicalGCD4 only; services stopped first. M17/8192/32767/32768,10 captured/scaled
mutations per M, epsilon cycling, row permutation,1000 graph replays and changed
input replay all output-bit exact. Captured layer0 residual/Fn expanded for
occupancy, not fresh full-model inputs; no universal exactness or E2E claim.

Three ABBA cycles, five calls/event, full pre-mix/RMS component:

- M32767:2-column4.9803ms vs3-column6.1558ms;
  separate control4.9833ms vs4-column5.6537ms.
- M32768:2-column5.0024ms vs3-column6.3625ms;
  separate control5.0034ms vs4-column6.2558ms.
- All smaller tested shapes also lose.
- Registers78→104/128, no reported spills/LDS. Pressure increase is evidence of
  resource cost, not a standalone proof of the slowdown mechanism.

Reject both, keep accepted2-column path. No new prefill speed gain this turn;
production stays7953.956103 input tok/s with original weights/TP8/1M KV. All
experiment code isolated in `.agents/experiments/dsv4_c16_premix_cols34_20260915/`.
README plus cols3/cols4 JSON contain full checks/samples/source and HSACO hashes.

The profile now makes indexer-only tuning lower priority (~0.302s owner chain).
Next speed work needs a genuinely different dataflow/compute schedule in main
attention, MoE, or MHC—not another unbounded increase in resident column count.
