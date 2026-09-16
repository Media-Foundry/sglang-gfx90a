# Bounded stage2 route-major storage screen

Production remains unchanged. Reuse the independently rebuilt accepted unique-
Set CK binary. Gather BF16 intermediate into sorted-assignment order, provide
identity virtual token IDs, write route-major FP32 partials, then inverse-map
and reduce the originalTop6 slots in fixed order. All real assignments must
appear exactly once. Padded routes use the same invalid-store guard and sentinel.

This candidate targets producer store locality, NOT fewer total partial bytes.
It adds an intermediate gather, inverse map and padded scratch, and may make
reducer reads worse. Time the complete remap+CK+reduce control against complete
pack+CK+inverse-reduce candidate, with both preallocated/eager and graph ABBA.
No argument about locality alone establishes a performance gain.

Use two checksum-verified historical real layer fixtures, five numeric
mutations including expert-block permutation, FP32 partial byte comparisons,
BF16 outputs and100 graph replays. These are component checks, not full-model
quality or service throughput. Only a stable complete-stage benefit would
justify an isolated integration; no default or precision changes.

Run only after all TP8 services stop, HIP_VISIBLE_DEVICES=5 (resolve actual
PCI via runtime, not rocm-smi card index), OMP_NUM_THREADS=1, DS Python and AOT
PYTHONPATH. Outputs refuse overwrite and record failure rather than a
partial passing result. See screen-v2.json for completed results; screen.json
is the preserved failed initial capacity-alignment assertion, not a passing run.
Both real fixtures passed byte-exact mappedFP32/finalBF16 checks and100 replays.
Complete stage2 latency improved roughly2.6%-3.0%, at additional scratch/pack
cost. breakdown.json contains an independent repeat and isolated substage
diagnostics. No E2E or1Mpool candidate validation and no production integration.
