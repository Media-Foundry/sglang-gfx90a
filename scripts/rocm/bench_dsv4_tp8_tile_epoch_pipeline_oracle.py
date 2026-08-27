#!/usr/bin/env python3
"""Synthetic two-stream TP8 tile publication/reduction oracle."""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time

import torch
import torch.distributed as dist
import aiter as aiter_ops

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from sglang.kernels.ops.attention.dsv4.gfx90a_tile_epoch_pipeline_oracle import (
    _jit_module,
    ack,
    end,
    load_only,
    producer,
    reduce,
    wait_only,
)

M, H, WORLD = 32, 4096, 8


def expected(epoch: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(M, device=device, dtype=torch.int32)[:, None]
    col = torch.arange(H, device=device, dtype=torch.int32)[None, :]
    acc = None
    for rank in range(WORLD):
        code = (col * 17 + row * 29 + epoch * 13 + rank * 37) & 255
        value = ((code - 128).float() / 64.0 + rank * 0.25).bfloat16()
        acc = value.float() if acc is None else acc + value.float()
    return acc.bfloat16()


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("eager", "diagnose", "capture", "replay"), default="replay"
    )
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--timing-iters", type=int, default=200)
    parser.add_argument("--timing-reps", type=int, default=7)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--diagnose-stage", choices=("wait", "load", "both"), default="both"
    )
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != WORLD:
        raise RuntimeError(f"requires 8 ranks, got {world}")

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter custom all-reduce did not initialize")
    _jit_module()  # compile before registration/capture
    # Direct HIP allocation: unlike a caching-allocator tensor, its IPC handle
    # maps the same base address that register_buffer records with offset zero.
    workspace = aiter_ops.allocate_meta_buffer(2 * 1024 * 1024)
    data_bytes = M * H * 2
    signal_bytes = (H // 256) * WORLD * 4
    produced_offset = data_bytes
    consumed_offset = produced_offset + signal_bytes
    end_offset = consumed_offset + signal_bytes
    output_offset = (end_offset + WORLD * 4 + 255) // 256 * 256
    data = workspace[:data_bytes].view(torch.bfloat16).view(M, H)
    produced = workspace[produced_offset:consumed_offset].view(torch.int32).view(H // 256, WORLD)
    consumed = workspace[consumed_offset:end_offset].view(torch.int32).view(H // 256, WORLD)
    end_epoch = workspace[end_offset:end_offset + WORLD * 4].view(torch.int32)
    output = workspace[output_offset:output_offset + data_bytes].view(torch.bfloat16).view(M, H)
    data.zero_(); produced.zero_(); consumed.zero_(); end_epoch.zero_(); output.zero_()
    waited = torch.zeros((H // 256,), device="cuda", dtype=torch.int32)
    comm.register_buffer(workspace)
    dist.barrier()
    if rank == 0:
        print("stage=registered", flush=True)

    p_stream = torch.cuda.Stream()
    c_stream = torch.cuda.Stream()

    if args.mode in ("eager", "diagnose"):
        if rank == 0:
            print("stage=eager_producer_enqueue", flush=True)
        with torch.cuda.stream(p_stream):
            producer(comm._ptr, workspace, data, produced, consumed, rank)
        p_stream.synchronize()
        if rank == 0:
            print("stage=eager_producer_complete", flush=True)
        dist.barrier()
        local_produced = produced.cpu()
        produced_inboxes = [None] * world
        dist.all_gather_object(produced_inboxes, local_produced.tolist())
        if rank == 0:
            print(
                f"stage=eager_produced_inbox min={int(local_produced.min())} "
                f"max={int(local_produced.max())} all={produced_inboxes}",
                flush=True,
            )
        publication_ok = all(
            all(row[owner] >= 1 for row in inbox)
            for owner, inbox in enumerate(produced_inboxes)
        )
        if not publication_ok:
            comm.close()
            dist.barrier()
            dist.destroy_process_group()
            raise AssertionError("producer local publication slot contains stale zero")
        if args.mode == "diagnose":
            if rank == 0:
                print("stage=eager_wait_only_enqueue", flush=True)
            with torch.cuda.stream(c_stream):
                wait_only(comm._ptr, workspace, produced, consumed, waited, rank)
            c_stream.synchronize()
            if rank == 0:
                print(
                    f"stage=eager_wait_only_complete waited={waited.cpu().tolist()}",
                    flush=True,
                )
            if args.diagnose_stage == "wait":
                comm.close()
                dist.barrier()
                dist.destroy_process_group()
                return
            if rank == 0:
                print("stage=eager_load_only_enqueue", flush=True)
            with torch.cuda.stream(c_stream):
                load_only(comm._ptr, workspace, data, output, 1)
            c_stream.synchronize()
            ref = expected(1, output.device)
            local_exact = torch.equal(output, ref)
            states = [None] * world
            dist.all_gather_object(states, (rank, local_exact, digest(output)))
            if rank == 0:
                print({"stage": "eager_load_only_complete", "states": states}, flush=True)
            comm.close()
            dist.barrier()
            dist.destroy_process_group()
            if not local_exact:
                raise AssertionError(f"rank={rank} load-only mismatch")
            return
        if rank == 0:
            print("stage=eager_reduce_enqueue", flush=True)
        with torch.cuda.stream(c_stream):
            reduce(comm._ptr, workspace, data, produced, consumed, output, rank)
        c_stream.synchronize()
        if rank == 0:
            print("stage=eager_reduce_complete", flush=True)
        dist.barrier()
        local_consumed = consumed.cpu()
        if rank == 0:
            print(
                f"stage=eager_consumed_inbox min={int(local_consumed.min())} "
                f"max={int(local_consumed.max())}",
                flush=True,
            )
        if rank == 0:
            print("stage=eager_end_enqueue", flush=True)
        with torch.cuda.stream(c_stream):
            end(comm._ptr, workspace, end_epoch, rank)
        c_stream.synchronize()
        if rank == 0:
            print("stage=eager_end_complete", flush=True)
        ref = expected(1, output.device)
        torch.testing.assert_close(output, ref, rtol=0, atol=0)
        gathered_hashes = [None] * world
        dist.all_gather_object(gathered_hashes, digest(output))
        if rank == 0:
            print(
                {"eager_exact": True, "rank_hashes": gathered_hashes}, flush=True
            )
        comm.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    p_graph = torch.cuda.CUDAGraph()
    c_graph = torch.cuda.CUDAGraph()
    if rank == 0:
        print("stage=producer_capture_begin", flush=True)
    with torch.cuda.graph(p_graph, stream=p_stream):
        producer(comm._ptr, workspace, data, produced, consumed, rank)
    if rank == 0:
        print("stage=producer_capture_complete", flush=True)
    dist.barrier()
    if rank == 0:
        print("stage=consumer_capture_begin", flush=True)
    with torch.cuda.graph(c_graph, stream=c_stream):
        reduce(comm._ptr, workspace, data, produced, consumed, output, rank)
        end(comm._ptr, workspace, end_epoch, rank)
    torch.cuda.synchronize()
    if rank == 0:
        print("stage=consumer_capture_complete", flush=True)
    dist.barrier()
    captured_epoch = int(end_epoch[rank].item())
    if rank == 0:
        print(f"stage=post_capture epoch={captured_epoch}", flush=True)
    if args.mode == "capture":
        comm.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    hashes = []
    for replay_index in range(1, args.replays + 1):
        epoch = captured_epoch + replay_index
        if rank == 0 and replay_index <= 2:
            print(f"stage=replay_enqueue epoch={epoch}", flush=True)
        p_graph.replay()
        c_graph.replay()
        torch.cuda.synchronize()
        if rank == 0 and replay_index <= 2:
            print(f"stage=replay_complete epoch={epoch}", flush=True)
        ref = expected(epoch, output.device)
        local_exact = torch.equal(output, ref)
        all_exact = torch.tensor(int(local_exact), dtype=torch.int32)
        dist.all_reduce(all_exact, op=dist.ReduceOp.MIN)
        if not bool(all_exact.item()):
            errors = {
                candidate_epoch: float(
                    (output.float() - expected(candidate_epoch, output.device).float())
                    .abs()
                    .max()
                )
                for candidate_epoch in (max(1, epoch - 1), epoch, epoch + 1)
            }
            state = {
                "rank": rank,
                "epoch": epoch,
                "errors": errors,
                "produced_diag": produced[:, rank].cpu().tolist(),
                "consumed_diag": consumed[:, rank].cpu().tolist(),
                "end_diag": end_epoch.cpu().tolist(),
                "output_hash": digest(output),
                "local_exact": local_exact,
            }
            states = [None] * world
            dist.all_gather_object(states, state)
            if rank == 0:
                print({"replay_mismatch": states}, flush=True)
            comm.close()
            dist.barrier()
            dist.destroy_process_group()
            raise AssertionError(
                f"rank={rank} epoch={epoch} mismatch errors={errors}"
            )
        if epoch in (1, args.replays):
            hashes.append((epoch, digest(output)))

    if args.skip_timing:
        gathered_hashes = [None] * world
        dist.all_gather_object(gathered_hashes, hashes)
        if rank == 0:
            print(
                {"replays_exact": args.replays, "hashes": gathered_hashes},
                flush=True,
            )
        comm.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    # Fair serial baseline: the same producer and epoch protocol, followed by
    # the existing registered AIter AR.  A separate direct allocation keeps A
    # and B epoch state independent.
    baseline_workspace = aiter_ops.allocate_meta_buffer(2 * 1024 * 1024)
    baseline_data = baseline_workspace[:data_bytes].view(torch.bfloat16).view(M, H)
    baseline_produced = baseline_workspace[produced_offset:consumed_offset].view(torch.int32).view(H // 256, WORLD)
    baseline_consumed = baseline_workspace[consumed_offset:end_offset].view(torch.int32).view(H // 256, WORLD)
    baseline_end = baseline_workspace[end_offset:end_offset + WORLD * 4].view(torch.int32)
    baseline_output = torch.empty_like(output)
    baseline_waited = torch.zeros_like(waited)
    baseline_data.zero_(); baseline_produced.zero_(); baseline_consumed.zero_(); baseline_end.zero_()
    comm.register_buffer(baseline_workspace)
    dist.barrier()

    bp_stream, bc_stream = torch.cuda.Stream(), torch.cuda.Stream()
    bp_graph, bc_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(bp_graph, stream=bp_stream):
        producer(comm._ptr, baseline_workspace, baseline_data,
                 baseline_produced, baseline_consumed, rank)
    with torch.cuda.graph(bc_graph, stream=bc_stream):
        wait_only(comm._ptr, baseline_workspace, baseline_produced,
                  baseline_consumed, baseline_waited, rank)
        comm.all_reduce(baseline_data, out=baseline_output, registered=True)
        ack(baseline_consumed, rank)
        end(comm._ptr, baseline_workspace, baseline_end, rank)

    ar_only_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ar_only_graph):
        comm.all_reduce(baseline_data, out=baseline_output, registered=True)
    torch.cuda.synchronize()

    def replay_pair(pg: torch.cuda.CUDAGraph, cg: torch.cuda.CUDAGraph,
                    count: int) -> None:
        for _ in range(count):
            pg.replay()
            cg.replay()

    for _ in range(20):
        replay_pair(p_graph, c_graph, 1)
        replay_pair(bp_graph, bc_graph, 1)
    torch.cuda.synchronize()
    candidate_epoch = int(end_epoch[rank].item())
    baseline_epoch = int(baseline_end[rank].item())
    candidate_exact = torch.equal(output, expected(candidate_epoch, output.device))
    baseline_error = float((baseline_output.float() - expected(baseline_epoch, output.device).float()).abs().max())
    gates = [None] * world
    dist.all_gather_object(gates, (rank, candidate_exact, baseline_error,
                                   candidate_epoch, baseline_epoch))
    if rank == 0:
        print({"timing_correctness_gate": gates}, flush=True)
    if not candidate_exact or baseline_error > 0.125:
        raise AssertionError(
            f"timing gate failed rank={rank} candidate={candidate_exact} "
            f"baseline_max_abs={baseline_error}"
        )

    local_samples = []
    for _ in range(args.timing_reps):
        for label, pg, cg in (
            ("candidate", p_graph, c_graph),
            ("baseline", bp_graph, bc_graph),
            ("baseline", bp_graph, bc_graph),
            ("candidate", p_graph, c_graph),
        ):
            dist.barrier()
            start = time.perf_counter()
            replay_pair(pg, cg, args.timing_iters)
            torch.cuda.synchronize()
            elapsed_us = (time.perf_counter() - start) * 1e6 / args.timing_iters
            local_samples.append((label, elapsed_us))

    local_ar_only = []
    for _ in range(args.timing_reps):
        dist.barrier()
        start = time.perf_counter()
        for _ in range(args.timing_iters):
            ar_only_graph.replay()
        torch.cuda.synchronize()
        local_ar_only.append((time.perf_counter() - start) * 1e6 / args.timing_iters)

    gathered = [None] * world
    dist.all_gather_object(gathered, (local_samples, local_ar_only))
    if rank == 0:
        slowest = []
        for sample_index, (label, _) in enumerate(local_samples):
            slowest.append((label, max(item[0][sample_index][1] for item in gathered)))
        candidate_samples = [value for label, value in slowest if label == "candidate"]
        baseline_samples = [value for label, value in slowest if label == "baseline"]
        candidate_median = statistics.median(candidate_samples)
        baseline_median = statistics.median(baseline_samples)
        ar_only_samples = [max(item[1][i] for item in gathered)
                           for i in range(args.timing_reps)]
        print({
            "abba_rounds": args.timing_reps,
            "replays_per_segment": args.timing_iters,
            "candidate_slowest_rank_median_us": candidate_median,
            "baseline_slowest_rank_median_us": baseline_median,
            "candidate_speedup_pct": (baseline_median / candidate_median - 1.0) * 100.0,
            "candidate_samples_us": candidate_samples,
            "baseline_samples_us": baseline_samples,
            "ar_only_slowest_rank_median_us": statistics.median(ar_only_samples),
            "ar_only_samples_us": ar_only_samples,
        }, flush=True)
    comm.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
