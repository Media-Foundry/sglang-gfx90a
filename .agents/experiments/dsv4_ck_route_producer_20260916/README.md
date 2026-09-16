# Route-major stage1 output: exact component oracle complete, not integrated

Clean build-v1 completed both modules (session20705 exit0). SingleGCD screen-v1
(session73111 exit0, HIP_VISIBLE_DEVICES5/PCI0000:b3:00.0) compares newly built
token output against BOTH installed production stage1 and captured outputs.
All four mutations including reversed expert-block order pass mapped stage1,
stage2 FP32 partial and final BF16 bit-exact checks;100 graph replays pass.

| Shape | Installed stage1+unique stage2+reducer | Route producer+metadata+stage2+reducer |
|---|---:|---:|
|M8192 eager|4.813390ms|4.818402ms|
|M8192 graph|4.800192ms|4.813816ms|
|M32767 eager|17.208379ms|16.569146ms|
|M32767 graph|17.155668ms|16.561548ms|

Three component ABBA cycles per mode,preallocated buffers,no service timing.
Weight expansion and sorter are outside this measured chain. Large-M saves
0.639ms eager(~3.71% latency); small-M is slightly slower and must not be
promoted from this screen. Additional intermediate8,385,536bytes plus
partial268,337,152bytes(~264MiB/GCD) still needs1M-pool service verification.
Do not interpret3.86% component throughput ratio as an E2E improvement.

Integration trap from current AIter source: after `metadata.stage1`,even
QuantType.No runs a reshape `a2.view(token_num,topk,inter_dim)` before stage2.
Returning a sorter-capacity tensor from only the stage1 hook is invalid.
Next oracle should reuse the same sorter and expanded weights but explicitly
call the two stages,or design an explicit ownership-aware boundary; do not
hide route data behind the old tensor shape or global cached metadata state.

## Full routed-stage oracle completed

`full-stage-v2.json` includes the current helper's weight expansion,scale
inverse layout,shared weight workspace,sorter,per-call allocations and final
output. M32767:20.413024→19.860006ms;M16384:11.910548→11.578140ms. Four
mutation variants pass exact BF16 output and expanded-weight checks. The
boundary sweepv3 also passes M16383,32765,32768,36864;larger fixtures append
real prefix rows and are explicitly labeled,not presented as service captures.

Peak allocator increment is about265MiB/GCD higher,still untested alongside
the full model's1M KV pool. The successful harness uses a real single-rank Gloo
group because production logging accesses TP rank. v1 failed before comparisons
without this initialization;its exact source/report/log remain preserved.
All component tests only used HIP_VISIBLE_DEVICES5/PCI0000:b3:00.0.

`runner.py` demonstrates the explicit same-sorter route path without the old
forced reshape. No production selector is connected yet. Next admission should
start with16384..36864 rows and originalV4TP8 ordinary prefill only;all-layer
service exactness,1M-pool peak and ABBA remain necessary.

## Original preparation and contracts

The earlier route-major stage2 oracle saved only about3% of stage2 time; a
separate BF16 pack consumed about0.23ms atM32767. This experiment tests whether
the current CK stage1 can directly produce the required assignment-major layout.
No production dispatch changes, no installed AIter/CK header edits.

`overlay.py` is pinned to the audited CK header hash. It changes the output
descriptor from token-count×Top6 to sorter capacity and maps output rows to
their original sorted-assignment offset. Original input token gather, expert
weights, MFMA geometry, DSV4 bounded-SiLU and BF16 rounding are retained. Both
Run and Run_2Lds copies are patched. The first CPU test rejected the mistaken
assumption of one copy; source review identified the two variants, and the
corrected test passes. Baseline preprocessing restores the entire original
header byte-for-byte. These are CPU source checks, not GPU correctness evidence.

`stage1.cu` exposes two independent module variants with distinct CK template
names. Geometry is the current explicit256×64×64×128,1×4,v1,splitK1 instance.
CK internal ActOP2 is bounded DSV4SiLU (API enum3 must not be passed directly).
Output is below2GiB for the supported M8192..36864 and local I256 contract.

`build.py --output-dir <new-directory>` clean-builds token and route modules,
reusing the existing source-hashed offline compiler configuration. Do not run
compilation during service ABBA. Both modules are now built and tested above.

Initial gates (1–3 completed in screen-v1; full routed/service gate remains):

1. Compare newly built token-stage1 against captured stage1 output AND the
   currently installed production entry on real M8192/M32767 fixtures.
2. Map route-stage1 output back by original Top6 slot; require exact BF16,
   mutation/permutation tests, poison padding, ragged bounds and graph replay.
3. Replace the old pack with metadata-only identity/inverse construction;
   run unchanged unique-Set stage2 plus fixed-slot reducer.
4. Compare complete stage1+metadata+stage2+reduction, not only removed pack.
   Report extra intermediate/partial capacity, including1M-pool feasibility.

The CPU address audit in the sibling route-major experiment shows thousands
of live routes exceed the old output descriptor. Merely changing store offsets
would be incorrect. Production dispatch remains unchanged.
