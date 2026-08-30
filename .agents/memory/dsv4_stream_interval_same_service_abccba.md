# DSV4 same-service stream-interval ABCCBA

## Purpose

Separate three quantities that were previously conflated in TP8 BS32 results:

1. HTTP streaming/drain throughput;
2. scheduler launch-to-launch decode throughput;
3. GPU model-forward time.

Changing `sampling_params.stream_interval` does not require a server restart, so
the comparison can use one resident service and avoid graph/JIT/startup drift.

## Harness

`scripts/rocm/bench_dsv4_tp4_diverse_concurrent.py` supports an ordered interval
sequence and captures `/v1/loads?include=core` plus `/metrics` before and after
each round.

Recommended server instrumentation:

```bash
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
# Add --enable-metrics and --decode-log-interval 64 to the normal TP8 launch.
```

Recommended run:

```bash
python scripts/rocm/bench_dsv4_tp4_diverse_concurrent.py \
  --base-url http://127.0.0.1:30001 \
  --tokens 2048 \
  --stream-interval-sequence 1,8,32,32,8,1 \
  --position-bin-size 256 \
  --resident-time-bins 8 \
  --output /tmp/dsv4_tp8_bs32_stream_interval_abccba.json
```

The filename retains the historical `tp4` name, but the client is topology
agnostic and can target the TP8 service.

## Recorded counters

`decode_moments` is cumulative:

```text
[steps, sum_bs, sum_step_us, sum_bs2, sum_bs_step_us, generated]
```

The harness reports its per-round delta, scheduler decode tok/s, host mean
launch-to-launch step time, and the device-timed decode forward counter delta.
It also retains completion-only SHA256 values and cross-round exactness for all
32 diverse prompts.

## Interpretation

- HTTP throughput changes while scheduler and GPU numbers remain fixed:
  output serialization/network/client drain is responsible.
- Scheduler throughput changes while GPU forward time remains fixed:
  scheduler/detokenizer backpressure or submission gaps are responsible.
- GPU forward time changes:
  the interval affects GPU scheduling/graph execution and is not merely an HTTP
  presentation effect.

The experiment must be run after `amd-smi process --json`; do not run against a
machine with unrelated GPU workloads. Correctness is required on every round.

