#!/usr/bin/env python3
"""Validate two independent TP4 custom-AR instances in one HIP graph.

This is a capture-lifecycle harness, not a model benchmark.  The primary
communicator runs the production-style M128 all-reduce, while a secondary
communicator runs the exact progressive M128 oracle on separate metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import aiter as aiter_ops
import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

from sglang.kernels.ops.communication.gfx90a_tp4_m128_progressive_ar_oracle import (
    _jit_module,
    anchor_end,
    arm,
    begin_draft,
    progressive,
)


ROWS, HIDDEN, WORLD = 128, 4096, 4
BLOCKS = 12
WORKSPACE_U32 = BLOCKS + 2 * BLOCKS * WORLD + WORLD + 3
PAYLOAD_BYTES = ROWS * HIDDEN * torch.bfloat16.itemsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--payload-mode",
        choices=("graph", "registered", "uncached"),
        default="graph",
    )
    parser.add_argument(
        "--primary-frequency", choices=("once", "layer"), default="once"
    )
    parser.add_argument(
        "--progressive-mode", choices=("full", "split"), default="full"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def graph_only_progressive(
    comm: CustomAllreduce,
    inp: torch.Tensor,
    sync: torch.Tensor,
    out: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Mirror the required production warmup guard."""
    if comm._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
        return out.zero_()
    arm(sync)
    progressive(comm._ptr, inp, sync, out, rank)
    return out


def aggregate_int(value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != WORLD:
        raise RuntimeError(f"requires TP4, got {world}")
    arch = torch.cuda.get_device_properties(local_rank).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"requires gfx90a, got {arch}")

    primary = None
    secondary = None
    try:
        primary = CustomAllreduce(
            dist.group.WORLD, torch.device("cuda", local_rank), max_size=PAYLOAD_BYTES
        )
        secondary = CustomAllreduce(
            dist.group.WORLD, torch.device("cuda", local_rank), max_size=PAYLOAD_BYTES
        )
        if primary.disabled or secondary.disabled:
            raise RuntimeError("one or both CustomAllreduce instances are disabled")
        if primary._ptr == secondary._ptr:
            raise RuntimeError("secondary communicator unexpectedly aliases primary")

        # Payload deliberately comes from the caching allocator, as it does in
        # the full graph backend.  Only the polling workspace is uncached and
        # explicitly registered with the secondary communicator.
        inp = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device="cuda")
        primary_out = torch.empty_like(inp)
        shadow_out = torch.empty_like(inp)
        dedicated_storage = None
        if args.payload_mode == "graph":
            shadow_input = inp
        elif args.payload_mode == "registered":
            shadow_input = (
                secondary.buffer[:PAYLOAD_BYTES]
                .view(torch.bfloat16)
                .view(ROWS, HIDDEN)
            )
        else:
            dedicated_storage = aiter_ops.allocate_meta_buffer(PAYLOAD_BYTES)
            shadow_input = dedicated_storage.view(torch.bfloat16).view(ROWS, HIDDEN)
            secondary.register_buffer(dedicated_storage)
        sync = aiter_ops.allocate_meta_buffer(WORKSPACE_U32 * 4).view(torch.uint32)
        sync.zero_()
        secondary.register_buffer(sync)
        split_input = (
            torch.empty_like(inp) if args.payload_mode == "graph" else shadow_input
        )
        routed_zero = torch.zeros(
            (ROWS // 4, HIDDEN), dtype=inp.dtype, device=inp.device
        )
        side = torch.cuda.Stream()

        generator = torch.Generator(device="cuda")
        generator.manual_seed(91000 + rank)
        base = torch.randn(
            (ROWS, HIDDEN), generator=generator, device="cuda", dtype=torch.bfloat16
        ).mul_(0.0625)
        mutation = torch.linspace(
            -1.0, 1.0, inp.numel(), device="cuda", dtype=torch.float32
        ).view_as(inp).to(torch.bfloat16)
        inp.copy_(base)

        _jit_module()
        torch.cuda.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        # Reproduce full-backend ordering: both capture lifecycles own the
        # complete session, including eager warmups, and register only after
        # device stream capture has ended.
        with primary.capture(), secondary.capture():
            for _ in range(2):
                primary.custom_all_reduce(inp)
                graph_only_progressive(secondary, inp, sync, shadow_out, rank)
            torch.cuda.synchronize()
            dist.barrier()
            with torch.cuda.graph(graph):
                sync.zero_()
                if args.primary_frequency == "once":
                    primary.all_reduce(inp, out=primary_out, registered=True)
                for _ in range(args.layers):
                    if args.primary_frequency == "layer":
                        primary.all_reduce(inp, out=primary_out, registered=True)
                    if args.progressive_mode == "full":
                        if args.payload_mode == "registered":
                            shadow_input.copy_(inp)
                        graph_only_progressive(
                            secondary, shadow_input, sync, shadow_out, rank
                        )
                    else:
                        main = torch.cuda.current_stream()
                        fork = torch.cuda.Event()
                        fork.record(main)
                        side.wait_event(fork)
                        with torch.cuda.stream(side):
                            split_input.copy_(inp)
                            arm(sync)
                            begin_draft(
                                secondary._ptr,
                                split_input,
                                sync,
                                shadow_out,
                                rank,
                            )
                        main.wait_stream(side)
                        anchor_end(
                            secondary._ptr,
                            split_input,
                            routed_zero,
                            sync,
                            shadow_out,
                            rank,
                        )
        torch.cuda.synchronize()
        dist.barrier()

        mutation_failures = 0
        max_abs = 0.0
        for iteration in range(args.mutations):
            inp.copy_(base)
            inp.add_(
                mutation,
                alpha=((iteration * 1543 + 17) % 2047 - 1023) / 32768.0,
            )
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(primary_out, shadow_out):
                mutation_failures += 1
            max_abs = max(
                max_abs,
                float((primary_out.float() - shadow_out.float()).abs().max().item()),
            )

        inp.copy_(base)
        graph.replay()
        torch.cuda.synchronize()
        stable_reference = shadow_out.clone()
        replay_failures = 0
        for _ in range(args.replays):
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(primary_out, shadow_out) or not torch.equal(
                shadow_out, stable_reference
            ):
                replay_failures += 1

        mutation_failures_all = aggregate_int(mutation_failures)
        replay_failures_all = aggregate_int(replay_failures)
        max_abs_values = [None] * world
        dist.all_gather_object(max_abs_values, max_abs)
        result = {
            "world": world,
            "rows": ROWS,
            "hidden": HIDDEN,
            "payload_bytes": PAYLOAD_BYTES,
            "payload_mode": args.payload_mode,
            "layers_per_replay": args.layers,
            "primary_frequency": args.primary_frequency,
            "progressive_mode": args.progressive_mode,
            "primary_secondary_distinct": primary._ptr != secondary._ptr,
            "sync_uncached_registered": True,
            "mutations": args.mutations,
            "mutation_failures_all_ranks": mutation_failures_all,
            "replays": args.replays,
            "replay_failures_all_ranks": replay_failures_all,
            "max_abs_all_ranks": max(float(v) for v in max_abs_values),
            "passed": mutation_failures_all == 0 and replay_failures_all == 0,
        }
        if rank == 0:
            payload = json.dumps(result, indent=2, sort_keys=True)
            print(payload, flush=True)
            if args.output is not None:
                args.output.write_text(payload + "\n", encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError(f"secondary custom-AR graph validation failed: {result}")
    finally:
        if secondary is not None:
            secondary.close()
        if primary is not None:
            primary.close()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
