# Bounded FP32 MFMA pre-mix experiment

Outcome: rejected for large-prefill performance; no runtime selector added.
See [memory](../../memory/dsv4_premix_mfma_20260916.md) for numerical limits,
timing scope, ISA evidence and the GPU ordinal/PCI correction.

- `mapping.py`: exact integer layout oracle (alternate map intentionally fails).
- `screen.py`: serial MFMA + RMS finish, full component graph ABBA.
- `screen.py --split`: split-K + fixed partial reduction + RMS finish.
- `analyze.py`: result aggregation and actual device-object evidence.
- `initial-screen.tar.gz`: immutable initial source/result snapshot.

All GPU commands use the DS conda environment and HIP_VISIBLE_DEVICES=5. The
screen records actual PCI `0000:b3:00.0` (rocm-smi card7). Output files are never
silently overwritten. Local real-input fixture paths and hashes are in JSON;
fixture tensors and compiled objects are not part of this source record.

No E2E improvement claimed; accepted checkpoint remains 8964.91 input tok/s.
