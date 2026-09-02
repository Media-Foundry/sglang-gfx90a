# DSV4 gfx90a BF16 CK atomic rejection (2026-09-02)

## Scope

- Physical GCD4 standalone; GCD4--7 TP4/EP1/no-A2A service.
- Original DeepSeek-V4-Flash weights.
- Large-M BF16 CK routed-expert experiment only; decode selector remained out
  of scope.
- `amd-smi process --general --sort-by-pid` was checked before GPU runs.

## Root causes found

1. AIter/Ninja does not track the included CK headers. Relinking the extension
   after changing `amd_buffer_addressing_builtins.hpp` left stale stage-2
   objects. A clean object rebuild is required for every CK-header experiment.
2. The generated BF16 `block_m=128` stage-1 instance produced an all-zero
   intermediate at M8192. The validated path must force `block_m=32`.
3. BF16 preshuffle is not implemented by the CK two-stage generator. AIter can
   create a module whose name says `preshuffle_on`, but the BF16 kernel list
   ignores the preshuffle argument. Feeding `shuffle_weight(..., (16,16))`
   data to that module produced an all-zero stage-1. Direct tiled shuffle was
   byte-for-byte equal to AIter `shuffle_weight`; the fault is the missing CK
   BF16 preshuffle device variant, not the dequantizer.
4. A minimal gfx90a packed-2xBF16 CAS probe worked and emitted
   `global_atomic_cmpswap`. After a clean rebuild, unshuffled `block_m=32` CK
   stage-2 also wrote nonzero BF16 output, proving the software atomic itself
   was reachable.

## Component performance and determinism

At M8192, BF16 CAS took about 17.3--18.3 ms and was not faster than the
existing FP32-workspace workaround (about 17.37 ms). Replays were non-exact:
maximum absolute delta reached 2048 and mean absolute delta reached 25.7 on
the balanced route.

At M36864:

```text
balanced BF16 CAS median: 80.67 ms
skewed BF16 CAS median:   49.50 ms
skewed FP32 workspace:    54.75 ms (prior accepted oracle)
```

The skewed case was about 9.6% faster and avoided roughly 576 MiB of FP32
workspace, so it cleared the threshold for an E2E semantic experiment.

## E2E result

With the accepted 16+16 admission profile and large-M BF16 MHC GEMM:

```text
C32 cold: 4684 input tok/s
C32 warm: 5365 / 5280 input tok/s
warm median: 5280 input tok/s
```

This was about 8.5% above the previous 4866 tok/s checkpoint. However, the
64-token, 32-request heterogeneous semantic suite showed widespread collapse:
repeated `1. 1. 1...`, repeated identifiers/fields, and short periodic loops.
This is materially worse than the accepted FP32-workspace semantic witness and
is not an acceptable small floating-point drift.

## Decision

Reject BF16 CAS accumulation for production despite its large-M throughput and
memory benefit. Keep the FP32 stage-2 workspace path. Do not enable BF16
preshuffle until CK has a real BF16 preshuffled device implementation. Treat
`block_m=128` for this BF16 CK shape as a correctness bug, not a performance
tactic.

