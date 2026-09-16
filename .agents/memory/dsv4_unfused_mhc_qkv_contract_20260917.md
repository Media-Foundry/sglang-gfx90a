# First/unfused MHC fixed in an opt-in scope; QKV shape drift isolated

This continues 405b81a78b. It is correctness work supporting the TP8 C16 prefill
goal, NOT a new service speed checkpoint. Latest accepted measurements remain
8K10372.9045,16K10216.8948,32K9854.3244 input tok/s. They still refer to the
older first-boundary policy. Do not relabel them as revalidated after this fix.

## Root cause and intervention

Detailed v2 traces on the same request0 positions32768..32831 compare M32576
mixed versus M64 singleton on all8 ranks. The earliest observed difference is
unfused_mhc_mixes, then post/comb; weighted BF16 attention inputs are identical.
scripts/rocm_dsv4_flash.sh:76 defaults MHC_LARGE_M_BF16_GEMM=1 for the throughput
profile. The live serial process2953502 environment confirmed that flag and
WAVE64_MHC_PRE_MIX both1. Thus the large first/unfused pre boundary converts
checkpoint FP32 Fn to BF16 and runs BF16-output GEMM, while small M uses the
native FP32 wave64 dot. The earlier common policy only covered fused post/pre.

first-mhc-oracle.json (session48375 exit0) uses exact live residual suffix64
and hash-checked original layer0 hc_attn_fn. Both original branches separately
reproduce their service mixes byte-for-byte. Turning off the BF16 shortcut
makes M64/65/8192/32576 and reverse row order exact for3 mutations. Native
two-column M64 versus three-column general kernel is exact too. Against FP64,
original live raw-mix max error is ~2.54771 for the BF16 shortcut versus
0.000162935 for FP32. These are raw mixes, NOT normalized logits or quality.
Isolated M32576 first-boundary cost ~4.13ms BF16 versus14.33ms FP32. This occurs
at the unfused boundary, not43 times as a blanket layer multiplier.

New SGLANG_DSV4_PREFILL_MHC_COMMON_UNFUSED=1 is DEFAULT OFF. It requires the
existing originalV4/TP8/native eager common scope (large or explicitly admitted
small); it suppresses only this BF16-Fn shortcut. Other configurations and
decode/draft remain outside scope. No checkpoint mutation or global BLAS knob.
17 CPU tests pass. Integrated oracle repeats3 mutations and all4 sizes with
the real selector; outputs equal the explicit FP32 path.

v3 fresh service pair (run_trace_pair.py session27075 exit0) confirms, at all8
ranks, that all64 corresponding first-layer mixes, post, comb and normalized
activations now agree. Next observed difference is prepare_qkv_a (14/64 rows
at rank0), BEFORE Q norm/RoPE. Full teacher forcing still differs: Top1 equality
993/1008 before ->997/1008 after; max selected-logprob gap .679566->.558391,
mean .02043068->.02042369. This is a partial fix, not whole-model invariance or
an answer-quality proof. Both services return Paris and stop cleanly.

## QKV evidence and bounded negative performance screen

Historical real BF16 QKV weight tensor SHA matches current live layer0 weight.
qkv-shape-oracle.json (session83429 exit0): 3 input mutations, M64/65/8192/32576,
reverse row order, FP64 comparison. Torch hipBLASLt reproduces both corresponding
service outputs (small and large) and exhibits shape AND row-order dependence.
Explicit per-object rocBLAS fixes row-order dependence in this fixture but NOT
cross-M dependence. Do NOT generalize the old indexer rocBLAS result to QKV.
Fixed-K128 Triton passes both tests, but is not bit-identical to old BLAS.

qkv-fixed-screen.json + qkv-fixed-screen-remaining.json cover10 geometries:
8 numerically accepted,2 rejected by compiler LDS limits. All8 passed4 sizes x
3 mutations against the fixed reference and reverse ordering. GPU graph ABBA
includes5 kernels/replay x8 replays; no Python allocation gaps per kernel.
Best large configuration remains128x128,waves8: M32576~3.843ms versus control
3.217ms (~19.5% slower). Best small tested~49.95us versus37.73us; no tested
geometry wins. Keep this as a diagnostic/reference, NOT a production speed win.
The first screen stopped on64x256,waves8 requiring81920B LDS >65536B; its7
completed cases/error/source are archived intact, and a separate remaining
run(session17274 exit0) completed/rejected the last3. No hidden restart or
claim that the failed original run completed.

## Evidence/lifecycle and next action

Experiment root: .agents/experiments/dsv4_prefill_tail_contract_20260917/.
trace-v2-evidence.tar.gz and trace-v3-evidence.tar.gz are LOCAL ONLY (~1.1GB each)
with manifests. They freeze full raw hashes/samples, model source, launchers,
teacher responses and owned-stop proof. Do not add these large files to git.
One premature v2 serial launch was rejected by the GPU-owner guard while mixed
was stopping; no second service started. Both failed-attempt files are retained
under trace-v2-serial-start-guard*. The v3 driver explicitly waits for child exit
and empty owned-stop proof before starting its next arm.

All GPU jobs finished; last amd-smi process check found no live GPU processes.
Do not change the accepted launcher yet. Next: non-diagnostic C16x8K ABBA for
the first-boundary FP32 fix, with current route producer/1M pool preserved;
then decide whether to retain the small expected cost or recover it with an
exact pre-mix implementation. Separately, consider a fixed-library-instance
QKV oracle or a genuinely faster invariant projection. Do not resweep the same
10 slower Triton geometries or claim ordinary rocBLAS closes the drift problem.
