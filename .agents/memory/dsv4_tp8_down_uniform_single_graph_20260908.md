# Down-uniform single-graph service verification

Following same-process ABBA, exact-cloned PID3527138's launch contract to
PID3542331, changing only paired-graphs1->0 and DOWN_UNIFORM0->1. Private
pre-stop snapshot retained mode0600 outside the repository. Ownership audit
found only the experiment tree. Startup55816 and validation39931 completed.

Native TP8/EP1/no-A2A, original checkpoint, max_total_num_tokens1048576 and
context_len1048576. Graph capture12.09--12.28s; memory0.65GB/GCD,
available15.58--15.64GB: diagnostic extra graph overhead removed.

## Measurements

- C1, three real tasks ×four measured rounds, per-task min/max trimmed then
  geomean: **82.7853557673276 tok/s**.
- C32, six real-code waves ×32 ×256tokens, exclude wave0:
  **989.8091528421774 HTTP tok/s**, **1035.9489381797728 resident tok/s**.
- HTTP/resident raw wave pairs:
  988.888916/1035.814153,
  989.809153/1035.696167,
  989.129820/1035.150978,
  990.023898/1036.212054,
  989.098890/1035.948938,
  990.225175/1036.011956.

France32/32 prefixexact. C1 all12 measured full completion IDs equal historical
reference. All192 C32 lengths256,finish=length and recomputed output hashes
pass;8/32 full outputs exact across waves. Same workload fingerprint
`4d7f83aa48de5df30bf819fe33800bb7f12ecfd4e02ce9215188f62bfd16a424`.

## Acceptance decision

C32 benefit survives removal of paired graph overhead. However this fresh
process C1 is below the paired-process control83.4433. Same-process C1 neutrality
does not rule out capture-layout effects on the separately instantiated C1
graph, nor process-state noise. Do **not** promote to default or claim overall
C1/C32 production non-regression from this run. Keep the flag opt-in and retain
this live candidate for further C1-state investigation. No default changed.

## Artifacts

Startup state `/tmp/dsv4_tp8_down_uniform_single_graph_20260908.json`, validation
state `/tmp/dsv4_tp8_down_uniform_single_validation_20260908.json`.
Startup state's old `candidate_flag` text still names the paired flag because
it was written before the utility's reporting correction; its explicit
`single_graph_candidate=true` plus verified process environment establishes
the actual paired0/down1 configuration. Future launches report it correctly.

Results stem `/tmp/dsv4_tp8_down_uniform_single_20260908`:

- `.c1.json` SHA256 `305c6ffb5b5ff6940138567092374aa44058e0f1d3f7cde43d882b98af4ade0d`
- `.c32.json` SHA256 `fe66fdb37aec3d81e753ef8725e9793650c9624bdc2d8ef6ef0b325abdd4fc4a`
- `.france.json` SHA256 `a970ec76251cf28a471e9fd310701ce8e1498692fe718a2847929b00763d9554`
