#!/usr/bin/env python3
"""Synthetic gfx90a per-expert readiness/release-sequence scheduling oracle."""

import argparse
import statistics

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


EXPERT_BLOCKS = 113
PRODUCER_CTAS = 32


@cache_once
def producer_module(iters: int):
    args = make_cpp_args(EXPERT_BLOCKS, PRODUCER_CTAS, iters)
    return load_jit(
        "gfx90a_readiness_producer_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_readiness_schedule_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::ReadinessProducerOracle<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def consumer_module(ctas: int, iters: int):
    args = make_cpp_args(EXPERT_BLOCKS, ctas, iters)
    return load_jit(
        "gfx90a_readiness_consumer_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_readiness_schedule_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::ReadinessConsumerOracle<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def pressure_module(ctas: int, iters: int):
    total_iters = iters * ((EXPERT_BLOCKS + ctas - 1) // ctas)
    args = make_cpp_args(ctas, total_iters)
    return load_jit(
        "gfx90a_consumer_pressure_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_readiness_schedule_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::ConsumerPressureOracle<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def capture(stream, fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            fn()
    torch.cuda.synchronize()
    return graph


def run_pair(producer_graph, consumer_graph, producer_stream, consumer_stream):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(producer_stream):
        start.record()
        producer_graph.replay()
        end.record()
    with torch.cuda.stream(consumer_stream):
        consumer_graph.replay()
    end.synchronize()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-iters", type=int, default=64)
    parser.add_argument("--consumer-iters", type=int, default=512)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--ctas", default="8,16,24,32,48,64")
    parser.add_argument("--real-gate", action="store_true")
    parser.add_argument(
        "--recorder",
        default="/tmp/expert_distribution_recorder_1787803355.1855972.pt",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    counters = torch.zeros(EXPERT_BLOCKS, dtype=torch.int32, device=device)
    ready = torch.zeros_like(counters)
    consumed = torch.zeros_like(counters)
    queue = torch.zeros(1, dtype=torch.int32, device=device)
    output = torch.zeros_like(counters)
    scratch = torch.zeros(
        EXPERT_BLOCKS * PRODUCER_CTAS, dtype=torch.int32, device=device
    )
    producer = producer_module(args.producer_iters)
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    producer_graph = capture(
        producer_stream, lambda: producer.run(counters, ready, scratch)
    )

    real_gate_graph = None
    if args.real_gate:
        from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
            make_metadata,
            reconstruct_topk_from_counts,
        )
        from scripts.rocm.bench_dsv4_tp4_m32_grouped_oracle import (
            _jit_gate_up_grouped_dpp,
        )
        from sglang.kernels.ops.quantization.int8_kernel import (
            per_token_group_quant_int8,
        )

        payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
        counts = payload["logical_count"][37, 34] // 8
        topk_ids = reconstruct_topk_from_counts(counts).cuda()
        metadata = make_metadata(topk_ids, assignments=4)
        torch.manual_seed(20260830)
        gate_x = torch.randn((32, 4096), dtype=torch.bfloat16, device=device)
        gate_xq, gate_xs = per_token_group_quant_int8(gate_x, 32)
        gate_w = torch.randint(
            0, 256, (256, 1024, 2048), dtype=torch.uint8, device=device
        )
        gate_s = torch.full(
            (256, 1024, 128), 127, dtype=torch.uint8, device=device
        )
        gate_out = torch.empty((32, 6, 512), dtype=torch.bfloat16, device=device)
        gate = _jit_gate_up_grouped_dpp(256, 32, 6, 512, 4096, 4, 2, 8, 2080, 2)
        real_gate_graph = capture(
            producer_stream,
            lambda: gate.run(
                gate_xq, gate_xs, gate_w, gate_s,
                metadata.sorted_ids, metadata.sorted_experts,
                metadata.valid, gate_out, 10.0,
            ),
        )

    for ctas in tuple(int(value) for value in args.ctas.split(",")):
        counters.zero_(); ready.zero_(); consumed.zero_(); queue.zero_()
        torch.cuda.synchronize()
        consumer = consumer_module(ctas, args.consumer_iters)
        # HIP graph capture executes the captured work once. Seed one epoch so
        # the capture-time acquire loop cannot wait forever.
        ready.fill_(1)
        consumed.zero_()
        torch.cuda.synchronize()
        consumer_graph = capture(
            consumer_stream,
            lambda consumer=consumer: (
                queue.zero_(),
                consumer.run(ready, consumed, queue, output),
            ),
        )
        counters.zero_(); ready.zero_(); consumed.zero_(); queue.zero_()
        torch.cuda.synchronize()
        # Capture does not execute the kernels. Start with the same paired
        # replay protocol used by the stress loop.
        for iteration in range(args.replays):
            run_pair(
                producer_graph, consumer_graph, producer_stream, consumer_stream
            )
            if iteration % 100 == 99 or iteration + 1 == args.replays:
                expected = iteration + 1
                if not bool(torch.all(ready == expected)):
                    raise RuntimeError(f"ready stale at replay {expected}, ctas={ctas}")
                if not bool(torch.equal(consumed, ready)):
                    raise RuntimeError(f"consumer stale at replay {expected}, ctas={ctas}")
                if not bool(torch.all(counters == expected * PRODUCER_CTAS)):
                    raise RuntimeError(f"counter stale at replay {expected}, ctas={ctas}")

        # Producer-only graph latency versus paired replay latency.
        solo = []
        paired = []
        for _ in range(args.rounds):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(producer_stream):
                begin.record(); producer_graph.replay(); end.record()
            end.synchronize(); solo.append(begin.elapsed_time(end) * 1000.0)
            paired.append(
                run_pair(
                    producer_graph, consumer_graph,
                    producer_stream, consumer_stream,
                )
            )
        solo_med = statistics.median(solo)
        paired_med = statistics.median(paired)
        print(
            f"RESULT ctas={ctas} replays={args.replays} solo_us={solo_med:.3f} "
            f"paired_us={paired_med:.3f} gate_delta_pct="
            f"{(paired_med / solo_med - 1) * 100:.3f}",
            flush=True,
        )

        if real_gate_graph is not None:
            pressure = pressure_module(ctas, args.consumer_iters)
            pressure_out = torch.empty(ctas, dtype=torch.int32, device=device)
            pressure_graph = capture(
                consumer_stream, lambda: pressure.run(pressure_out)
            )
            solo_gate = []
            paired_gate = []
            for _ in range(args.rounds):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(producer_stream):
                    begin.record(); real_gate_graph.replay(); end.record()
                end.synchronize(); solo_gate.append(begin.elapsed_time(end) * 1000.0)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(producer_stream):
                    begin.record(); real_gate_graph.replay(); end.record()
                with torch.cuda.stream(consumer_stream):
                    pressure_graph.replay()
                end.synchronize(); torch.cuda.synchronize()
                paired_gate.append(begin.elapsed_time(end) * 1000.0)
            solo_value = statistics.median(solo_gate)
            paired_value = statistics.median(paired_gate)
            print(
                f"REAL_GATE ctas={ctas} solo_us={solo_value:.3f} "
                f"paired_us={paired_value:.3f} delta_pct="
                f"{(paired_value / solo_value - 1) * 100:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
