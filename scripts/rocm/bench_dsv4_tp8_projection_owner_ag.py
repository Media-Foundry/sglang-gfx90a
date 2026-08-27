#!/usr/bin/env python3
"""Bitwise TP8 projection-owner peer-read oracle.

Each of ranks 0..3 executes exactly one original production-shaped BF16 GEMM;
ranks 4..7 execute none.  Owners publish their original-shape output in a
registered slot; consumers GLC-load only useful segments and directly pack
[M,4160].  Epoch handshakes prevent stale reads and owner overwrite.
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import aiter as aiter_ops

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from sglang.kernels.ops.attention.dsv4.gfx90a_projection_owner_peer_oracle import (
    _jit_module as _jit_peer_module,
    end as peer_end,
    pack as peer_pack,
    publish as peer_publish,
)

from bench_dsv4_tp8_output_n_projection_ag import (
    HIDDEN,
    PROJECTIONS,
    capture,
    digest,
    load_dump,
    rank_max_samples,
    reference_projection,
)


PAD_N = 2048
DATA_BYTES = 32 * PAD_N * 2
PRODUCED_OFFSET = DATA_BYTES
SIGNAL_BYTES = 8 * 4
CONSUMED_OFFSET = PRODUCED_OFFSET + SIGNAL_BYTES
END_OFFSET = CONSUMED_OFFSET + SIGNAL_BYTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-dir", type=Path, default=Path("/tmp/dsv4-layer20-m32")
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--mutations", type=int, default=1000)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--require-gain-us", type=float, default=30.0)
    return parser.parse_args()


def owner_candidate(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    rank: int,
    comm: CustomAllreduce,
    local: torch.Tensor,
    gathered: torch.Tensor,
) -> torch.Tensor:
    if rank < len(PROJECTIONS):
        width = PROJECTIONS[rank][1]
        # This is intentionally the original full projection shape.  The copy
        # changes only the destination stride, not the GEMM reduction tree.
        local[:, :width].copy_(F.linear(x, weights[rank]))
    comm.all_gather_reg(local, out=gathered)
    ranked = gathered.view(8, x.shape[0], PAD_N)
    return torch.cat(
        [ranked[index, :, :width] for index, (_, width) in enumerate(PROJECTIONS)],
        dim=1,
    )


def owner_compute_only(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    rank: int,
    local: torch.Tensor,
) -> torch.Tensor:
    if rank < len(PROJECTIONS):
        width = PROJECTIONS[rank][1]
        local[:, :width].copy_(F.linear(x, weights[rank]))
    else:
        # Keep a non-empty graph on idle owner ranks; rank-max is determined by
        # ranks 0..3 and this one-element store is deliberately negligible.
        local[0, 0].zero_()
    return local


def gather_pack_only(
    local: torch.Tensor,
    gathered: torch.Tensor,
    comm: CustomAllreduce,
) -> torch.Tensor:
    comm.all_gather_reg(local, out=gathered)
    ranked = gathered.view(8, local.shape[0], PAD_N)
    return torch.cat(
        [ranked[index, :, :width] for index, (_, width) in enumerate(PROJECTIONS)],
        dim=1,
    )


def owner_peer_candidate(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    rank: int,
    comm: CustomAllreduce,
    workspace: torch.Tensor,
    data: torch.Tensor,
    produced: torch.Tensor,
    consumed: torch.Tensor,
    end_epoch: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if rank < len(PROJECTIONS):
        owner_output = F.linear(x, weights[rank])
        peer_publish(
            comm._ptr, workspace, owner_output, data, produced, rank
        )
    peer_pack(comm._ptr, workspace, consumed, output, rank)
    peer_end(comm._ptr, workspace, end_epoch, rank)
    return output


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"requires 8 ranks, got {world}")

    x_cpu, weights_cpu = load_dump(args)
    x = x_cpu.cuda()
    base_x = x.clone()
    mutation = torch.linspace(-1.0, 1.0, x.numel(), device="cuda").view_as(x)
    mutation = mutation.to(torch.bfloat16)
    weights = [weight.cuda() for weight in weights_cpu]

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter CustomAllreduce did not initialize")
    local = torch.empty((x.shape[0], PAD_N), dtype=torch.bfloat16, device="cuda")
    gathered = torch.empty(
        world * x.shape[0] * PAD_N, dtype=torch.bfloat16, device="cuda"
    )
    _jit_peer_module()
    peer_workspace = aiter_ops.allocate_meta_buffer(2 * 1024 * 1024)
    peer_data = peer_workspace[:DATA_BYTES].view(torch.bfloat16).view(x.shape[0], PAD_N)
    peer_produced = peer_workspace[PRODUCED_OFFSET:CONSUMED_OFFSET].view(torch.int32)
    peer_consumed = peer_workspace[CONSUMED_OFFSET:END_OFFSET].view(torch.int32)
    peer_end_epoch = peer_workspace[END_OFFSET:END_OFFSET + SIGNAL_BYTES].view(torch.int32)
    peer_output = torch.empty(
        (x.shape[0], sum(width for _, width in PROJECTIONS)),
        dtype=torch.bfloat16,
        device="cuda",
    )
    peer_data.zero_(); peer_produced.zero_(); peer_consumed.zero_(); peer_end_epoch.zero_(); peer_output.zero_()
    comm.register_buffer(peer_workspace)
    dist.barrier()
    input_bytes = local.numel() * local.element_size()
    if input_bytes % 16 or input_bytes > comm.max_size // (world * 2):
        raise RuntimeError(
            f"AG input violates pool gate: {input_bytes=} "
            f"limit={comm.max_size // (world * 2)}"
        )

    # Library initialization must happen outside capture.
    _ = reference_projection(x, weights)
    if rank < len(PROJECTIONS):
        width = PROJECTIONS[rank][1]
        local[:, :width].copy_(F.linear(x, weights[rank]))
    comm.all_gather_unreg(local, out=gathered)
    eager_peer = owner_peer_candidate(
        x, weights, rank, comm, peer_workspace, peer_data,
        peer_produced, peer_consumed, peer_end_epoch, peer_output
    )
    torch.cuda.synchronize()
    eager_reference = reference_projection(x, weights)
    if not torch.equal(eager_reference, eager_peer):
        raise AssertionError(
            f"eager peer-read mismatch rank={rank} "
            f"max_abs={(eager_reference.float() - eager_peer.float()).abs().max().item()}"
        )

    graph_a, output_a = capture(lambda: reference_projection(x, weights))
    graph_b, output_b = capture(
        lambda: owner_peer_candidate(
            x, weights, rank, comm, peer_workspace, peer_data,
            peer_produced, peer_consumed, peer_end_epoch, peer_output
        ),
        comm,
    )
    graph_owner, _ = capture(lambda: owner_compute_only(x, weights, rank, local))
    graph_gather, _ = capture(
        lambda: gather_pack_only(local, gathered, comm), comm
    )

    graph_a.replay()
    graph_b.replay()
    torch.cuda.synchronize()
    initial_exact = torch.equal(output_a, output_b)
    initial_max = (output_a.float() - output_b.float()).abs().max().item()

    mismatches = 0
    max_abs = 0.0
    first_mismatch = -1
    for iteration in range(args.mutations):
        state = (iteration * 1543 + 17) % 2047
        alpha = (state - 1023) / 32768.0
        x.copy_(base_x)
        x.add_(mutation, alpha=alpha)
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        if not torch.equal(output_a, output_b):
            if first_mismatch < 0:
                first_mismatch = iteration
            mismatches += 1
            max_abs = max(
                max_abs, (output_a.float() - output_b.float()).abs().max().item()
            )

    x.copy_(base_x)
    graph_a.replay()
    torch.cuda.synchronize()
    replay_mismatches = 0
    replay_first_mismatch = -1
    for iteration in range(args.replays):
        graph_b.replay()
        torch.cuda.synchronize()
        if not torch.equal(output_a, output_b):
            if replay_first_mismatch < 0:
                replay_first_mismatch = iteration
            replay_mismatches += 1

    report = {
        "rank": rank,
        "initial_exact": initial_exact,
        "initial_max_abs": initial_max,
        "mutations": args.mutations,
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "max_abs": max_abs,
        "replays": args.replays,
        "replay_mismatches": replay_mismatches,
        "replay_first_mismatch": replay_first_mismatch,
        "a_hash": digest(output_a),
        "b_hash": digest(output_b),
    }
    reports: list[dict | None] = [None] * world
    dist.all_gather_object(reports, report)
    if rank == 0:
        print(f"correctness={reports}", flush=True)

    x.copy_(base_x)
    torch.cuda.synchronize()
    schedule = (("A", graph_a), ("B", graph_b), ("B", graph_b), ("A", graph_a))
    timings: list[tuple[str, float]] = []
    for slot, (name, graph) in enumerate(schedule, 1):
        median, samples = rank_max_samples(
            graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        timings.append((name, median))
        if rank == 0:
            print(
                f"slot={slot} mode={name} median_us={median:.3f} "
                f"samples={[round(v, 3) for v in samples]}",
                flush=True,
            )
    a = statistics.mean(value for name, value in timings if name == "A")
    b = statistics.mean(value for name, value in timings if name == "B")
    if rank == 0:
        print(
            f"ABBA A_us={a:.3f} B_us={b:.3f} saved_us={a-b:.3f} "
            f"delta_pct={(b/a-1)*100:+.2f}",
            flush=True,
        )

    for label, graph in (("owner_only", graph_owner), ("gather_pack_only", graph_gather)):
        median, samples = rank_max_samples(
            graph,
            warmup=args.warmup,
            iters=args.iters,
            reps=args.reps,
            world=world,
        )
        if rank == 0:
            print(
                f"component={label} median_us={median:.3f} "
                f"samples={[round(v, 3) for v in samples]}",
                flush=True,
            )

    gain = a - b
    bad = any(
        item is None or item["mismatches"] or item["replay_mismatches"]
        for item in reports
    )
    comm.close()
    dist.barrier()
    dist.destroy_process_group()
    if bad:
        raise AssertionError("projection-owner oracle is not bitwise exact")
    if gain < args.require_gain_us:
        raise AssertionError(
            f"projection-owner peer-read saved {gain:.3f} us, below "
            f"required {args.require_gain_us:.3f} us"
        )


if __name__ == "__main__":
    main()
