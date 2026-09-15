# Original V4 unified KV: repair TP-local attention sink ownership

Follow-up to the pre-correction H16 experiment8f285b6260. No weights, precision,
attention selection, reduction order or scheduler changed. H16 remainsOFF.

## Root cause, independent of token drift

MQALayer computed attn_sink=self._local_attn_sink(), but its unified-KV call
passed self.attn_sink (the complete64-head checkpoint parameter) instead.
The H8 kernel uses local head offsets0..7, so ranks1..7 consumed rank0's entries.
Legacy padded-Q attention already passed the local-head-first sink correctly.

The old layer20 real fixture compared against the checkpoint confirms all8ranks
consumed exactly entries0..7. Onlyrank0 matches its owned head slice. This is a
deterministic mapping error, NOT proof of the cause of cross-run nondeterminism.
Old H16 attention was exact to the old implementation, including this error;
its8664tok/s result must retain the pre-correction label.

Fix: pass the already-computed local attn_sink variable to the unified backend.
No new copy/allocation in the hot path: _local_attn_sink was already called and
cached its local/padded buffer before the branch. Each per-head sink still
corresponds to the same checkpoint element as the TP-local Q projection.

## Verification

- Four CPU tests execute the actual helper and unified call-site AST for
  TP1/2/4/8. Distinct per-head sinks verify the exact shard and a simple
  analytical attention denominator. Reverting the call reintroduces failures.
-36 combined unit tests pass; diff checks pass.
-Fresh originalV4 TP8/EP1 native AR process,1M logicalKV,chunk32768,H16OFF,
  query-producerOFF and production CK atomics.
-Real M32767 layer20 capture on every rank consumes exactly the corresponding
  checkpoint8-head slice; captured canonical attention output matches runtime.
  Compact observed/expected values are in the summary, full fixtures remainlocal.
-France: "The capital of France is **Paris**."
-Two C16x128-token waves validate32 prompt echoes/counts/decoded text, no cache
  hits. Repeated whole outputs15/16 exact. Reviewed all16 openings remain
  source-related and coherent; not a proof that every generated claim is true.

## Corrected baseline speed

Three warm C16x8K waves,131069 input tokens each,one outputtoken,zero prefixhits:
median **8376.877042 input tok/s**. Capture warmup5722.28 is excluded.
This is near the pre-correction control8374.79, but is not an old/new ABBA and
does not establish a tiny speed change. The mapping fix has no observed material
prefill regression in this check. No new decode throughput claim.

Artifacts: .agents/experiments/dsv4_local_sink_20260916/.
All services stopped and all8GCDs free after the check.

## Remaining work

Do not mark whole-model drift solved: this corrected baseline still has1/16
cross-wave mismatch, and production stage2 atomics remain enabled. The earlier
unique-store fixed-reduction remedy and H16 speed candidate must be revalidated
against this corrected model baseline, not the older sink contract. Next useful
steps are fixed-token first-divergence on the remaining request and corrected-
sink H16 ABBA; no new performance selector should hide this correctness change.
