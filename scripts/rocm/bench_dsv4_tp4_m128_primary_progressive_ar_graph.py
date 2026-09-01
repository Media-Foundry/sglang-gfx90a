#!/usr/bin/env python3
"""Validate a progressive M128 op inside the primary AIter communicator.

The graph issues the production two-stage M128 all-reduce and then the
candidate begin/anchor-end sequence through the same CustomAllreduce object.
Inputs change between replays and are rank-distinct; constant-input smoke tests
are insufficient for peer-read synchronization validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


ROWS, HIDDEN, WORLD = 128, 4096, 4
PAYLOAD_BYTES = ROWS * HIDDEN * torch.bfloat16.itemsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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

    comm = None
    try:
        comm = CustomAllreduce(
            dist.group.WORLD,
            torch.device("cuda", local_rank),
            max_size=PAYLOAD_BYTES,
        )
        if comm.disabled:
            raise RuntimeError("primary CustomAllreduce is disabled")
        if not hasattr(comm, "dsv4_progressive_m128_begin"):
            raise RuntimeError("AIter primary progressive M128 API is unavailable")

        generator = torch.Generator(device="cuda")
        generator.manual_seed(92000 + rank)
        states = []
        for layer in range(args.layers):
            shared = torch.empty(
                (ROWS, HIDDEN), dtype=torch.bfloat16, device="cuda"
            )
            routed = torch.empty(
                (ROWS // 4, HIDDEN), dtype=torch.bfloat16, device="cuda"
            )
            states.append(
                {
                    "shared": shared,
                    "routed": routed,
                    "baseline_input": torch.empty_like(shared),
                    "candidate_input": torch.empty_like(shared),
                    "baseline_out": torch.empty_like(shared),
                    "shared_base": torch.randn(
                        shared.shape,
                        generator=generator,
                        device="cuda",
                        dtype=torch.bfloat16,
                    ).mul_(0.0625),
                    "routed_base": torch.randn(
                        routed.shape,
                        generator=generator,
                        device="cuda",
                        dtype=torch.bfloat16,
                    ).mul_(0.03125),
                    "layer_scale": (layer + 1) / max(args.layers, 1),
                }
            )
        shared_delta = torch.linspace(
            -1.0, 1.0, ROWS * HIDDEN, device="cuda", dtype=torch.float32
        ).view(ROWS, HIDDEN).to(torch.bfloat16)
        routed_delta = torch.linspace(
            1.0, -1.0, (ROWS // 4) * HIDDEN, device="cuda", dtype=torch.float32
        ).view(ROWS // 4, HIDDEN).to(torch.bfloat16)

        def prepare(iteration: int) -> None:
            alpha = ((iteration * 1543 + 17) % 2047 - 1023) / 32768.0
            for state in states:
                scaled = alpha * state["layer_scale"]
                state["shared"].copy_(state["shared_base"]).add_(
                    shared_delta, alpha=scaled
                )
                state["routed"].copy_(state["routed_base"]).add_(
                    routed_delta, alpha=-scaled
                )
                state["baseline_input"].copy_(state["shared"])
                state["baseline_input"][::4].add_(state["routed"])
                state["candidate_input"].copy_(state["shared"])

        prepare(0)
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        candidate_outputs = []
        draft_snapshots = []
        fork_events = [torch.cuda.Event() for _ in states]
        draft_done_events = [torch.cuda.Event() for _ in states]
        with comm.capture():
            for _ in range(2):
                state = states[0]
                comm.custom_all_reduce(state["baseline_input"])
                warm = comm.dsv4_progressive_m128_begin(state["candidate_input"])
                comm.dsv4_progressive_m128_anchor_end(
                    state["candidate_input"], state["routed"], warm
                )
            torch.cuda.synchronize()
            dist.barrier()
            with torch.cuda.graph(graph):
                main = torch.cuda.current_stream()
                for layer, state in enumerate(states):
                    fork_events[layer].record(main)
                    side.wait_event(fork_events[layer])
                    with torch.cuda.stream(side):
                        candidate_out = comm.dsv4_progressive_m128_begin(
                            state["candidate_input"]
                        )
                        draft_done_events[layer].record(side)
                        draft_snapshot = candidate_out.view(32, 4, HIDDEN)[
                            :, 1:
                        ].contiguous()
                    main.wait_event(draft_done_events[layer])
                    comm.dsv4_progressive_m128_anchor_end(
                        state["candidate_input"],
                        state["routed"],
                        candidate_out,
                    )
                    # A stock primary collective after every progressive epoch
                    # catches divergence between Signal block 0 and blocks 1-11.
                    comm.all_reduce(
                        state["baseline_input"],
                        out=state["baseline_out"],
                        registered=True,
                    )
                    candidate_outputs.append(candidate_out)
                    draft_snapshots.append(draft_snapshot)
                main.wait_stream(side)
        assert len(candidate_outputs) == args.layers
        torch.cuda.synchronize()
        dist.barrier()

        mutation_failures = 0
        draft_failures = 0
        max_abs = 0.0
        for iteration in range(args.mutations):
            prepare(iteration)
            graph.replay()
            torch.cuda.synchronize()
            for state, candidate_out, draft_snapshot in zip(
                states, candidate_outputs, draft_snapshots
            ):
                baseline_out = state["baseline_out"]
                if not torch.equal(baseline_out, candidate_out):
                    mutation_failures += 1
                expected_draft = baseline_out.view(32, 4, HIDDEN)[:, 1:]
                if not torch.equal(expected_draft, draft_snapshot.view_as(expected_draft)):
                    draft_failures += 1
                max_abs = max(
                    max_abs,
                    float(
                        (baseline_out.float() - candidate_out.float()).abs().max()
                    ),
                )

        prepare(0)
        graph.replay()
        torch.cuda.synchronize()
        stable_references = [out.clone() for out in candidate_outputs]
        replay_failures = 0
        for _ in range(args.replays):
            prepare(0)
            graph.replay()
            torch.cuda.synchronize()
            for state, candidate_out, stable_reference in zip(
                states, candidate_outputs, stable_references
            ):
                if not torch.equal(
                    state["baseline_out"], candidate_out
                ) or not torch.equal(candidate_out, stable_reference):
                    replay_failures += 1

        mutation_failures_all = aggregate_int(mutation_failures)
        draft_failures_all = aggregate_int(draft_failures)
        replay_failures_all = aggregate_int(replay_failures)
        max_abs_values = [None] * world
        dist.all_gather_object(max_abs_values, max_abs)
        result = {
            "world": world,
            "rows": ROWS,
            "hidden": HIDDEN,
            "layers_per_replay": args.layers,
            "primary_communicator_only": True,
            "mutations": args.mutations,
            "mutation_failures_all_ranks": mutation_failures_all,
            "draft_snapshot_failures_all_ranks": draft_failures_all,
            "replays": args.replays,
            "replay_failures_all_ranks": replay_failures_all,
            "max_abs_all_ranks": max(float(v) for v in max_abs_values),
            "passed": (
                mutation_failures_all == 0
                and draft_failures_all == 0
                and replay_failures_all == 0
            ),
        }
        if rank == 0:
            payload = json.dumps(result, indent=2, sort_keys=True)
            print(payload, flush=True)
            if args.output is not None:
                args.output.write_text(payload + "\n", encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError(f"primary progressive M128 validation failed: {result}")
    finally:
        if comm is not None:
            comm.close()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
