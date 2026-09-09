# TP8 DSpark M128 geometry integration

Default-off `SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_MOE_GEOMETRY` now selects
G832/D832 only inside C32, width-4 DSpark target verification, TP8/EP1,
gfx90a, exact I256 packed weight shapes and A4/R2/LDS geometry.
Native AR, draft and ordinary prefill remain on their existing paths.

Component single-ABBA: balanced +2.687%, skewed +4.684%; these are not
end-to-end gains. Component mutation and graph checks passed (see main
2k goal memory). Integration: syntax checks and 10 unit tests passed.

Next: one real-code C32 E2E ABBA, same manifest in all arms, excluded warmup
per process, full target verification and 1M pool. Results pending.
Controller: `/tmp/dsv4_tp8_geometry_abba.py`.
Artifacts: `/tmp/dsv4_tp8_dspark_2k_geometry_abba_20260909`.
