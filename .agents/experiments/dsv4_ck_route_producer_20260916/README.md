# Prepared, not yet compiled or GPU-tested: route-major stage1 output

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
compilation during service ABBA. Neither module has been compiled or timed yet.

Required next gates:

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
would be incorrect. No speed or numerical success is claimed by this scaffold.
