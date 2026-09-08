# First paired-graph startup: DSA key guard failure

State: `/tmp/dsv4_tp8_paired_graph_start_20260908.json`.
Log: `/tmp/dsv4_tp8_paired_graph_start_20260908.service.log`.

Owned baseline PID 3468727 and descendants were checked against amd-smi GPU
PIDs, then stopped normally. New PID 3493461 cloned its command/environment
with only the paired-graph flag added. The 1M KV configuration was retained.

The new service loaded weights, then failed at the explicit pair key guard:
`paired down capture accepts only one plain M32 key`. Eight-rank logs confirm
the baseline actually enables DSA dense/sparse dual graphs (cutoff raw length
2048). This configuration was missing from the initial paired-graph assumptions.
No candidate timing or correctness result was produced; this was not an OOM or
observed device kernel fault. The process exited, controller session 7126
finished with exit 1, and amd-smi subsequently reported no running processes on
all eight GPUs. There is currently no experiment server on port 30011.

Fix: retain all baseline captures, capture one alternate for M32 plain/dense
only, and leave sparse M32 unchanged. Unsupported labels still fail loudly.
The real-method CPU test now covers dense then sparse capture and arm selection;
all eight pair tests pass. This fix still needs fresh-process GPU validation.

The launch utility now saves a private mode-0600 command/environment snapshot
before stopping a service, for exact recovery. The first failed launch predates
that addition, so no such snapshot exists for it; do not claim the old process
has been restored. Reconstruct/verify the launch profile before restarting.
Do not print or commit private launch snapshots.
