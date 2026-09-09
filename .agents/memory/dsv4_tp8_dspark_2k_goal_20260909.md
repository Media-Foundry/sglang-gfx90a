# TP8 DSpark C32 >=2000 tok/s — active

Requested objective: at least2000 output tok/s at C32 with TP8. Preserve
original checkpoint weights, full-target verification and native AR isolation.
Keep the1M logical-token pool unless a capacity tradeoff is explicitly agreed.
Use real diverse code requests, held fixed across arms; optimization screening
uses one ABBA per candidate per the user's latest instruction. No approximate
anchor-only target path and no fabricated/simulated acceptance results.

Starting evidence: prior TP8 communication+C128 full-target profile ~932 tok/s;
optional refined-C4 screen937.900427 tok/s versus matched control922.802887,
observed+1.636052%. These are common resident decode intervals excluding
prefill/drain, not whole-wave wall time. Historical TP4~1.5k used a rejected
target approximation and is not an equivalent control. New2k target remains
unachieved; no promise that a settings-only change can double throughput.

## First diagnostic, not a performance screen

Revalidated HEAD5e35016f64 and live service165825; amd-smi showed only owned
GPU PIDs. Existing DsparkInfoDumper supports core, step CPU/GPU, draft GPU,
target-verify GPU and request records. Reuse it instead of editing the hot path.
It records only attention TP rank0, so its event durations are NOT independent
eight-rank maxima or a system-wide kernel-duration sum. Diagnostics add event
and copy overhead; their tok/s must not be compared as an optimization win.

Controller `/tmp/dsv4_tp8_dspark_2k_phase_profile.py`, exec session71525:
- Stops only the owned service after another amd-smi ownership check.
- Restarts the conservative full-target profile with observer components on.
- Fixed real-code manifest `/tmp/dsv4_open_code_pd_20260908/decode.json`.
- One C32x256 warm wave, then C32x1024 diagnostic wave, natural EOS.
- Saves only extracted dspark_info_record payloads from server_info, not
  arbitrary server environment/configuration secrets.
- Preserves warm records separately, allowing subtraction by forward_ct.
- Finally restores the profile with observers off, C4/refinement off.

Artifacts `/tmp/dsv4_tp8_dspark_2k_phase_profile_20260909`.
No optimization result yet. Next: close draft/target/host-step budgets for
actual C32/M128 rows, then choose a structural candidate. Exact ragged verify
or draft-budget changes require checking target-verification semantics and
realized acceptance, not merely raising a block-size setting.
