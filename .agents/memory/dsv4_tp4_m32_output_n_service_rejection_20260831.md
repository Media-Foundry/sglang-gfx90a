# TP4/M32 output-N service rejection (2026-08-31)

The existing TP8 output-N projection implementation was temporarily enabled
for the exact TP4/M32 native-AR tier.  It concatenated the four original BF16
attention-side projections (`N=1536/2048/512/64`) into `N=4160`, computed a
local `N=1040` shard on each TP4 rank, all-gathered the result, and passed the
four views to their original consumers.  Checkpoint weights and precision were
unchanged.

The component oracle had measured `123.559 -> 67.669 us`, saving `55.890 us`
per layer.  It was previously stopped at a bitwise gate because changing the
hipBLASLt N shape changed floating association (maximum relative L2
`9.2931e-5`).  The experiment was reopened under the explicit policy that a
small association difference may proceed to end-to-end validation when the
France semantic oracle remains correct.

## Four-service ABBA result

All services used physical GCDs `4,5,6,7`, TP4/EP1/no-A2A, native AR, the
current TP4-BS32 profile, graph tiers 1--32, and 32 distinct fixed coding
requests from `dsv4_tp8_diverse_32_input_ids.json`.  Each measured service ran
three 256-token rounds.  France was exact for its first nine completion tokens
and semantically contained Paris in every A and B round; all requests returned
the requested length.

| Service | output-N | Scheduler decode tok/s (measured rounds) | Resident median tok/s |
|---|---:|---:|---:|
| A1 | off | 708.063, 722.182 | 623.982 |
| B1 | on  | 717.113, 715.781 | 629.180 |
| B2 | on  | 704.023, 703.409 | 617.875 |
| A2 | off | 717.413, 719.012 | 631.526 |

Independent-service centers were approximately:

- A: scheduler `716.7 tok/s`, resident `627.8 tok/s`;
- B: scheduler `710.1 tok/s`, resident `623.5 tok/s`.

Thus B regressed about 0.7--0.9% despite its large isolated GEMM saving.

## Decision

Remove the TP4 production selector.  Combining the projections serializes the
qkv, core-compressor and indexer-compressor producers behind one all-gather
completion barrier.  The current multistream schedule hides much of the four
independent projections behind later work; the isolated concat/GEMM result is
therefore not an end-to-end saving.  Do not revisit projection concatenation
unless consumers can retain independent readiness/publication.
