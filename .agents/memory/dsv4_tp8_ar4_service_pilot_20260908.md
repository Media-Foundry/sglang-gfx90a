# Native TP8 M32 four-block AR service pilot

2026-09-08, follows1779a9c527. Default-off env
`SGLANG_DSV4_GFX90A_TP8_M32_AR_BLOCKS` (0=existing library,4/8/16=shim).
Requires native TP8/EP1, DP1/PP1,1M KV and existing legacy AR enabled.

The adapter preserves the parent custom_all_reduce wrapper: warmup zeros_like,
unregistered eager copies, graph buffer collection and registration unchanged.
Only active DSV4 native M32, registered contiguous BF16[32,4096], nonquantized,
actual stream capture invokes the shim. C1/prefill/spec/eager remain parent.
Shim loads and Signal ABI validation runs during communicator construction,
before capture. Uses the existing communicator and normal output allocation;
no new registered buffers, weight caches or graph pairs.

Five CPU tests execute actual adapter source with stubs: default identity,
candidate handle/output forwarding, all fallback modes, invalid deployment
scope and ABI failure. Existing down-scope tests and six aggregate-auditor
tests pass; down ABBA now rejects mixed AR geometry as well as issue order.
These are interface tests, not replacements for GPU/model checks.

## First service, not ABBA

PID3632259; exact clone of owned PID3611667 except AR_BLOCKS=4. AMD ownership
audit before replacement. Order0, down-uniform1, fixed down warmup1 preserved.
Eight rank logs confirm actual `registered graph AR selected blocks=4`.
Graph capture12.44–12.55s, graph memory0.65GB/GCD, free15.58–15.64GB.
1M KV unchanged. Original checkpoint weights unchanged; no dependency rebuild.

- C1 trimmed/geomean (three code tasks, four rounds):83.31505759269783tok/s.
- C32 six waves, drop first, warm HTTP median:1003.7412950884815tok/s.
- C32 warm resident median:1052.1441673459176tok/s.
- France32/32 and all12 C1 complete sequences match the existing reference.
-192 C32 outputs length256,finish=length, recomputed completion hashes valid,
  no speculative acceptance statistic.9/32 cross-six-wave exact: not bitwise
  deterministic and not proof of arbitrary generated-code correctness.

Adjacent original-library control was83.6561/985.1527/1031.1351 (C1/HTTP/resident).
Do not attribute that full gap to blocks: a newly loaded compiled implementation
can differ from the library. Next use AR_BLOCKS=16 through the same shim and
fresh-process ABBA before promotion. C1 process variation still needs control.
Both new selector and down-uniform remain default-off.

Artifacts stem `/tmp/dsv4_tp8_ar4_pilot_validation_20260908.block0`:
-`.c1.json`: `c2bea4dec8cd3f12a251160e57955243a9d2ceb3504c4299422273c5a37c4697`
-`.c32.json`: `820c3cc0890d45b873ca1841e0ccbcc31b0a8190884e3b932c7ae525924f90b4`
-`.france.json`: `a67f2f7987c0890563c73e1b0b7910ab156da09c8b6c779767d77d5082cdfd33`
Service log `/tmp/dsv4_tp8_ar4_pilot_20260908.service.log`.
Private launch snapshot is not for publication/version control.
