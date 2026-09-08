# TP8 H8 attention two-wave oracle: promising, E2E pending

Baseline 7bc60347f7. Production unchanged, 1M KV pool preserved. GPU4 only;
AMD-SMI verified no external GPU owners. Temporary standalone allocations only.

Current paged decode uses blockH16 for H8 (MFMA minimum tile), blockK16,
four waves and two stages. Existing two-wave selector is TP4/H16/M32-only.
We did not broaden it. Extended the geometry oracle to H8, M1/M32 and
production shape-only split heuristic (M1=64, M32=4 on this GPU).

Although server_args says fp8 KV, current HIP unified decode passes its
BF16 unified cache with no `kv_scales` to runtime.decode. Do not substitute
an FP8-dequant microbenchmark based only on the server_args label.

## Full split+reduce graph latency, microseconds

Seven forward/reverse profile-order cycles, trimmed min/max. Nine profiles:
waves2/4/8 x stages1/2/3. Choose2waves/2stages consistently, not per-context best.

| M | KV entries/query | 4wave/2stage | 2wave/2stage |
| --- | ---: | ---: | ---: |
| 1 | 128 | 18.031 | 16.149 |
| 1 | 256 | 18.087 | 16.182 |
| 1 | 512 | 18.254 | 16.307 |
| 32 | 128 | 36.067 | 27.579 |
| 32 | 256 | 64.329 | 44.675 |
| 32 | 512 | 115.179 | 74.337 |

Synthetic BF16 Q/KV and FP32 sink. All nine profiles passed100 input mutations
at every M/context, with KV/sink also mutated periodically. Additional runs
validated2wave/2stage against baseline under100 changing ragged lengths,
indices, Q/KV/sink mutations including empty rows, then1000 fixed-input replay
checks, for both M1 and M32 and all three contexts. Every comparison bit-exact
and finite. No change to split count, mathematical tiles, reduction or KV set.

The full benchmark historically only mutated/checks2wave/1stage. It now checks
all nine profiles and optionally performs the ragged/replay acceptance tests.
Adjacent JSON preserves all measurements; the extra ragged runs use only two
timing cycles and are correctness runs, not the headline timing source.

## Next gate

Worth a default-off native TP8 decode-only M1/M32 selector. Must not hit
prefill, speculative/verify, other heads, FP8 KV dequant or fused inverse-RoPE
variants without explicit validation. Then require C1 reference tokens and
fixed-prefix logits, France/C32 and real diverse long-code output review,
independent-process E2E ABBA, and unchanged1M pool/graph memory accounting.
No model speedup yet: current service remains on the original four-wave core.

Reproduction script: `scripts/rocm/bench_dsv4_tp4_m32_paged_decode_geometry.py`
with `--heads 8 --tokens 1` or `32`, `--contexts 128,256,512`, `--output PATH`.
For additional oracle use `--ragged-check --rounds 2 --mutations 10`.
