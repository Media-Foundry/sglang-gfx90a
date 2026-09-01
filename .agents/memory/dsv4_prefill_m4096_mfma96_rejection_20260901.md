# DSV4 prefill M4096 MFMA96 rejection (2026-09-01)

The candidate paired a 4096-token chunk with a 96-assignment expert block so
the average Top-6 occupancy could avoid a second MFMA64 packed-weight scan.
It used the original weights and direct (non-wave-broadcast) metadata/scales,
because wave64 shuffle cannot address lanes 64--95.

Physical GPU 4, balanced E256/Top-6/H4096/I512 gate micro:

```text
MFMA96, split4, 416 blocks, direct scales: 16.401 ms
MFMA64, split4, 416 blocks, broadcast:     12.008 ms
```

MFMA96 regressed 36.6%. Its larger accumulator and 48-KiB LDS footprint reduce
occupancy more than the lower expert scan count helps. The gate stage alone is
decisive, so no down/service test was run. All selector/kernel changes were
removed; do not retry without a different work decomposition that avoids
holding 96 assignments' accumulators concurrently.
