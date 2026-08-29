#!/usr/bin/env python3
"""TP4 gfx90a oracle for the internal HIP custom-AR start barrier."""

import argparse
import multiprocessing as mp
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank, world, port, args, queue):
    os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = "0"
    os.environ["SGLANG_USE_AITER_AR"] = "0"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    cpu_group = dist.new_group(backend="gloo")

    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    comm = CustomAllreduce(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("internal CustomAllreduce disabled")

    shape = (32, 4096)
    graph_inp = torch.empty(shape, dtype=torch.bfloat16, device=device)

    # Capture the registered one-stage path. The input copy/mutation remains a
    # producer before graph replay, which is exactly the start_sync publication case.
    graph = torch.cuda.CUDAGraph()
    with comm.capture():
        with torch.cuda.graph(graph):
            graph_out = comm.custom_all_reduce(graph_inp)
    torch.cuda.synchronize()

    def make_input(iteration):
        gen = torch.Generator(device=device)
        gen.manual_seed(0x5A170000 + rank * 100003 + iteration)
        if iteration & 1:
            return torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
        return torch.randint(-32, 33, shape, generator=gen, dtype=torch.int32, device=device).to(torch.bfloat16)

    failures = 0
    max_abs = 0.0
    for iteration in range(args.mutations):
        inp = make_input(iteration)
        # Fixed 0,1,2,3 reference in FP32, matching packed_reduce's accumulator.
        host = inp.cpu()
        gathered = [None] * world
        dist.all_gather_object(gathered, host, group=cpu_group)
        ref = gathered[0].float()
        for peer in range(1, world):
            ref.add_(gathered[peer].float())
        ref = ref.to(torch.bfloat16).to(device)
        graph_inp.copy_(inp)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(graph_out, ref):
            failures += 1
            max_abs = max(max_abs, float((graph_out.float() - ref.float()).abs().max()))
        dist.barrier(group=cpu_group)

    # Keep replaying mutations without host-side validation to stress flags/ABA.
    for iteration in range(args.mutations, args.replays):
        graph_inp.copy_(make_input(iteration))
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    # Rank-max latency: events measure each rank; Gloo gathers samples so rank 0
    # reports max(rank latency) per round, then a trimmed median.
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.rounds):
        dist.barrier(group=cpu_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.inner):
            graph.replay()
        end.record()
        end.synchronize()
        local_us = start.elapsed_time(end) * 1000.0 / args.inner
        vals = [None] * world
        dist.all_gather_object(vals, local_us, group=cpu_group)
        samples.append(max(vals))

    queue.put((rank, failures, max_abs, samples))
    comm.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", type=int, default=1000)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--inner", type=int, default=200)
    args = parser.parse_args()
    if args.replays < args.mutations:
        parser.error("--replays must be >= --mutations")
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    port = free_port()
    procs = [mp.Process(target=worker, args=(r, 4, port, args, queue)) for r in range(4)]
    for proc in procs:
        proc.start()
    results = [queue.get() for _ in procs]
    for proc in procs:
        proc.join()
        if proc.exitcode:
            raise SystemExit(f"rank process failed: exit={proc.exitcode}")
    results.sort()
    samples = results[0][3]
    ordered = sorted(samples)
    trimmed = ordered[1:-1] if len(ordered) >= 5 else ordered
    print(f"shape=32x4096 dtype=bf16 bytes={32*4096*2} world=4 algo=one-stage")
    print(f"graph_replays={args.replays} validated_mutations={args.mutations}")
    print(f"failures_by_rank={[r[1] for r in results]} max_abs={max(r[2] for r in results)}")
    print(f"rankmax_us_samples={[round(x, 3) for x in samples]}")
    print(f"rankmax_us_trimmed_median={statistics.median(trimmed):.3f}")


if __name__ == "__main__":
    main()
