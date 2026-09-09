# TP8 DSpark M128 geometry integration

Default-off `SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_MOE_GEOMETRY` now selects
G832/D832 only inside C32, width-4 DSpark target verification, TP8/EP1,
gfx90a, exact I256 packed weight shapes and A4/R2/LDS geometry.
Native AR, draft and ordinary prefill remain on their existing paths.

Component single-ABBA: balanced +2.687%, skewed +4.684%; these are not
end-to-end gains. Component mutation and graph checks passed (see main
2k goal memory). Integration: syntax checks and 10 unit tests passed.

The service audit found production M128 uses the non-LDS lookup path. A
matched non-LDS component ABBA then measured balanced +5.270% and skewed
+6.621%, again with 100 mutations and 1000 graph replays bitwise exact.
The guard therefore permits both already-validated lookup modes while keeping
the exact A4/R2, shape, topology and DSpark-target constraints.

Real-code C32 E2E single ABBA, same manifest in all arms, excluded warmup per
process, full target verification and 1M pool:

- A1 control: 911.6256 tok/s
- B1 G832/D832: 937.8567 tok/s
- B2 G832/D832: 939.4148 tok/s
- A2 control: 913.7100 tok/s
- arm means: 912.6678 -> 938.6357 tok/s, +2.8453%
- all measured and warm waves passed the severe repetition gate

The full-target TP8 profile now enables the selector by default; users can
still force it off with the dedicated environment variable. This remains far
below the 2k tok/s goal and is only one accepted increment.

Controller: `/tmp/dsv4_tp8_geometry_abba.py`.
Artifacts: `/tmp/dsv4_tp8_dspark_2k_geometry_abba_retry_20260909`.

## Refined C4 stacked ABBA

With G832/D832 enabled in both arms, a second single ABBA tested the already
guarded refined C4 CK path:

- A1 geometry only: 945.0402 tok/s
- B1 geometry + refined C4: 963.5902 tok/s
- B2 geometry + refined C4: 964.5994 tok/s
- A2 geometry only: 945.9084 tok/s
- arm means: 945.4743 -> 964.0948 tok/s, +1.9694%
- all warm and measured waves passed the severe repetition gate

The full-target profile now enables refined C4 as well as its parent CK H8
path. Artifact: `/tmp/dsv4_tp8_dspark_geometry_plus_refined_c4_abba_20260909`.
