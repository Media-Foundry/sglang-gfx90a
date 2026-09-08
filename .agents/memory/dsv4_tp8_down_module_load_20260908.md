# Down module loading versus first-use device instantiation

Standalone GPU4 probes while owned candidate3542331 was idle. Each probe
checked amd-smi PIDs against the owner tree before creating its GPU context.
No service mutation, model forward, clock change or additional service graph.

Loader-only probe10070: both cached .so modules loaded, zero observed change
in device-free memory, Torch allocated0/reserved0. Host wall baseline~118ms,
candidate~6ms; these are one-time loader timings, not kernel performance.

First-launch probe12955: same two module loads, then bounded random real-shape
fixtures (E256/M32/T6/N4096/K256/A4/R2/W8/G832/LDS). Baseline/candidate public
wrappers produce finite, raw-BF16-bit-identical outputs for this fixture.

Before launch: Torch allocated142667776, reserved157286400 bytes. First
baseline call lowers device-free memory by2097152 bytes (2MiB), and first
candidate call by another2MiB. Each retained output is262144bytes, allocated
within the unchanged Torch reserved arena. First-use device/runtime allocation
is therefore not captured by the earlier .so-only measurement. This does not
prove every2MiB is kernel code or prove it causes the C1 slowdown.

Source/binary SHA256 matches the previous ABBA module audit; no new kernel
mathematics compiled. Temporary fixture allocations are private to the probe
and disappear with process exit. Adjacent JSON retains all device/Torch
snapshots, module paths and SHA256. Raw loader-only artifact:
`/tmp/dsv4_tp8_down_module_load_20260908.json`; first-use artifact:
`/tmp/dsv4_tp8_down_first_launch_20260908.json`.

Next diagnostic: if fixing module-order confounds, execute both variants in a
fixed order during startup warmup with existing model inputs, reset metadata
through the existing hook, then capture the selected single-graph arm. Merely
preloading .so files is insufficient. This diagnostic is not integrated yet;
measure total memory and C1/C32 ABBA before claiming a benefit. No defaults
changed and no new service speed reported.
