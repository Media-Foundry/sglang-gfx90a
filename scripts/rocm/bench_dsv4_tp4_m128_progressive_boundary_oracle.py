#!/usr/bin/env python3
"""Four-rank DSpark semantic-boundary oracle with progressive M128 AR.

The baseline overlaps the production TP4-sharded shared expert with routed M32,
joins anchor rows, runs the production M128 all-reduce, then consumes draft M96.
The candidate starts an exact draft-row reduction as soon as shared M128 is
ready, overlaps the draft consumer with routed M32, and reduces anchor rows in
the same logical collective epoch after the routed join.
"""

from __future__ import annotations

import argparse
import os
import statistics

import aiter as aiter_ops
import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

import sglang.kernels.ops.layernorm.mhc as mhc_module
from scripts.rocm.bench_dsv4_tp4_m32_attn_moe_overlap_oracle import metadata
from sglang.kernels.ops.attention.dsv4.gfx90a_unified_sparse_decode import (
    run as run_ck_sparse,
    workspace_size_bytes,
)
from sglang.kernels.ops.communication.gfx90a_tp4_m128_progressive_ar_oracle import (
    _jit_module as progressive_module,
    anchor_end,
    begin_draft,
    wait_draft,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


WORLD, M, DRAFT_M, H, I, TOPK, E = 4, 32, 96, 4096, 512, 6, 256
A, R, W, LDS = 4, 2, 8, 2
BLOCKS = 12
WORKSPACE_U32 = BLOCKS + 2 * BLOCKS * WORLD + WORLD + 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument(
        "--recorder",
        default="/tmp/expert_distribution_recorder_1788187926.1154153_0.pt",
    )
    parser.add_argument("--record", type=int, default=40)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--entry-mhc-only",
        action="store_true",
        help="consume draft-ready M96 only with the next-layer entry MHC",
    )
    return parser.parse_args()


def capture(comm, fn):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture():
        # The entry-MHC oracle allocates graph-owned outputs on a joined side
        # stream. HIP rejects the allocator call in global capture mode even
        # though the allocation is graph-private and prewarmed.
        with torch.cuda.graph(graph, capture_error_mode="relaxed"):
            outputs = fn()
    dist.barrier()
    return graph, outputs


def rankmax_once(graph, iters, world):
    dist.barrier()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    local = start.elapsed_time(end) * 1000.0 / iters
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    return max(float(v) for v in gathered)


def main():
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

    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter CustomAllreduce did not initialize")

    input_storage_a = aiter_ops.allocate_meta_buffer(128 * H * 2)
    input_storage_b = aiter_ops.allocate_meta_buffer(128 * H * 2)
    ar_input_a = input_storage_a.view(torch.bfloat16).view(128, H)
    ar_input_b = input_storage_b.view(torch.bfloat16).view(128, H)
    sync = aiter_ops.allocate_meta_buffer(WORKSPACE_U32 * 4)
    sync.zero_()
    comm.register_buffer(input_storage_a)
    comm.register_buffer(input_storage_b)
    comm.register_buffer(sync)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260901 + rank * 1009)
    shared_x = torch.randn(
        (128, H), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    def shared_buffers():
        gate_out = torch.empty((128, I), dtype=torch.bfloat16, device="cuda")
        return (
            gate_out,
            torch.empty_like(gate_out),
            torch.empty_like(gate_out),
            torch.empty_like(gate_out),
        )

    shared_bufs_a = shared_buffers()
    shared_bufs_b = shared_buffers()
    sg = torch.randn((I, H), generator=generator, device="cuda", dtype=torch.bfloat16)
    su = torch.randn((I, H), generator=generator, device="cuda", dtype=torch.bfloat16)
    sd = torch.randn((H, I), generator=generator, device="cuda", dtype=torch.bfloat16)
    sg.mul_(0.015625); su.mul_(0.015625); sd.mul_(0.015625)

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    record = payload["records"][args.record]["topk_ids_of_layer"]
    topk = record[args.layer, 0::4].contiguous().to(device="cuda", dtype=torch.int32)
    if topk.shape != (M, TOPK):
        raise RuntimeError(f"expected M32 anchor route, got {topk.shape}")
    ids, experts = metadata(topk)
    valid = torch.tensor([ids.numel(), M], dtype=torch.int32, device="cuda")
    routed_x = torch.randn((M, H), generator=generator, device="cuda", dtype=torch.bfloat16)
    routed_xq, routed_xs = per_token_group_quant_int8(routed_x, 32)
    topk_weights = torch.rand((M, TOPK), generator=generator, device="cuda")
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="cuda")
    s2 = torch.full((E, H, I // 32), 127, dtype=torch.uint8, device="cuda")
    def routed_buffers():
        return (
            torch.empty((M, TOPK, I), dtype=torch.bfloat16, device="cuda"),
            torch.empty((M, TOPK, H), dtype=torch.float32, device="cuda"),
            torch.empty((M, H), dtype=torch.bfloat16, device="cuda"),
        )

    routed_bufs_a = routed_buffers()
    routed_bufs_b = routed_buffers()
    gate = _jit_gate_up_grouped(E, M, TOPK, I, H, A, R, W, 2080, LDS)
    down = _jit_down_grouped(E, M, TOPK, H, I, A, R, W, 832, LDS)

    projection_ns = (1536, 2048, 512, 64)
    projections = [
        torch.randn((n, H), generator=generator, device="cuda", dtype=torch.bfloat16)
        .mul_(0.015625)
        for n in projection_ns
    ]
    projection_a = [
        torch.empty((DRAFT_M, n), dtype=torch.bfloat16, device="cuda")
        for n in projection_ns
    ]
    projection_b = [torch.empty_like(t) for t in projection_a]
    draft_a = torch.empty((M, 3, H), dtype=torch.bfloat16, device="cuda")
    draft_b = torch.empty_like(draft_a)
    q = torch.randn((DRAFT_M, 16, 512), generator=generator, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(
        (DRAFT_M * args.context, 512),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = torch.arange(DRAFT_M * args.context, dtype=torch.int32, device="cuda")
    indptr = torch.arange(
        0, (DRAFT_M + 1) * args.context, args.context,
        dtype=torch.int32, device="cuda",
    )
    sink = torch.randn((16,), generator=generator, device="cuda")
    sparse_a = torch.empty_like(q)
    sparse_b = torch.empty_like(q)
    workspace_a = torch.empty(
        workspace_size_bytes(tokens=DRAFT_M), dtype=torch.uint8, device="cuda"
    )
    workspace_b = torch.empty_like(workspace_a)
    out_a = torch.empty_like(ar_input_a)
    out_b = torch.empty_like(ar_input_b)

    if args.entry_mhc_only:
        # The local oracle does not use SGLang's symmetric output allocator.
        mhc_module.get_tp_group = lambda: None
        residual = torch.randn(
            (DRAFT_M, 4, H),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        post = torch.sigmoid(
            torch.randn((DRAFT_M, 4), generator=generator, device="cuda")
        )
        comb = torch.softmax(
            torch.randn((DRAFT_M, 4, 4), generator=generator, device="cuda"),
            dim=1,
        )
        mhc_weight = (
            torch.randn((24, 4 * H), generator=generator, device="cuda")
            * 0.0078125
        )
        mhc_scale = torch.ones((3,), dtype=torch.float32, device="cuda")
        mhc_base = torch.zeros((24,), dtype=torch.float32, device="cuda")
        norm_weight = torch.ones((H,), dtype=torch.bfloat16, device="cuda")

    def shared(ar_input, bufs):
        shared_gate, shared_up, shared_mid, shared_clamped = bufs
        torch.mm(shared_x, sg.t(), out=shared_gate)
        torch.mm(shared_x, su.t(), out=shared_up)
        torch.sigmoid(shared_gate, out=shared_mid)
        torch.clamp(shared_up, -10.0, 10.0, out=shared_clamped)
        shared_mid.mul_(shared_gate).mul_(shared_clamped)
        torch.mm(shared_mid, sd.t(), out=ar_input)

    def routed(bufs):
        inter, partial, routed_out = bufs
        gate.run(routed_xq, routed_xs, w13, s13, ids, experts, valid, inter, 10.0)
        iq, isc = per_token_group_quant_int8(inter, 32)
        down.run_partial(iq, isc, w2, s2, ids, experts, valid, topk_weights, partial)
        down.reduce(partial, routed_out)

    def consume(draft, projection_out, sparse_out, workspace):
        if args.entry_mhc_only:
            return mhc_fused_post_pre(
                draft.view(DRAFT_M, H),
                residual,
                post,
                comb,
                mhc_weight,
                mhc_scale,
                mhc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                20,
                norm_weight=norm_weight,
                norm_eps=1e-6,
            )
        for weight, output in zip(projections, projection_out):
            torch.mm(draft.view(DRAFT_M, H), weight.t(), out=output)
        run_ck_sparse(
            q, kv, indices, indptr, sink, sparse_out, workspace,
            1.0 / (512**0.5),
        )
        return (*projection_out, sparse_out)

    anchor_view_a = ar_input_a.view(M, 4, H)[:, 0]
    anchor_view_b = ar_input_b.view(M, 4, H)[:, 0]
    main = torch.cuda.current_stream()
    shared_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()

    a_fork, a_shared_done = torch.cuda.Event(), torch.cuda.Event()
    b_fork = torch.cuda.Event()
    b_shared_done = torch.cuda.Event()
    b_consumer_done = torch.cuda.Event()

    def baseline():
        a_fork.record(main)
        shared_stream.wait_event(a_fork)
        with torch.cuda.stream(shared_stream):
            shared(ar_input_a, shared_bufs_a); a_shared_done.record(shared_stream)
        routed(routed_bufs_a)
        main.wait_event(a_shared_done)
        anchor_view_a.add_(routed_bufs_a[2])
        comm.all_reduce(ar_input_a, out=out_a, registered=True)
        draft_a.copy_(out_a.view(M, 4, H)[:, 1:])
        consumer_a = consume(draft_a, projection_a, sparse_a, workspace_a)
        return out_a, draft_a, consumer_a

    def candidate():
        b_fork.record(main)
        shared_stream.wait_event(b_fork)
        comm_stream.wait_event(b_fork)
        consumer_stream.wait_event(b_fork)
        with torch.cuda.stream(shared_stream):
            shared(ar_input_b, shared_bufs_b); b_shared_done.record(shared_stream)
        with torch.cuda.stream(consumer_stream):
            wait_draft(sync)
            draft_b.copy_(out_b.view(M, 4, H)[:, 1:])
            consumer_b = consume(draft_b, projection_b, sparse_b, workspace_b)
            b_consumer_done.record(consumer_stream)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(b_shared_done)
            begin_draft(comm._ptr, ar_input_b, sync, out_b, rank)
        routed(routed_bufs_b)
        main.wait_event(b_shared_done)
        main.wait_stream(comm_stream)
        anchor_end(comm._ptr, ar_input_b, routed_bufs_b[2], sync, out_b, rank)
        main.wait_event(b_consumer_done)
        return out_b, draft_b, consumer_b

    progressive_module()
    # hipBLASLt lazily materializes per-shape/per-stream plans and workspaces.
    # Force every GEMM spelling on every stream before graph capture.
    shared(ar_input_a, shared_bufs_a); routed(routed_bufs_a)
    shared(ar_input_b, shared_bufs_b); routed(routed_bufs_b)
    draft_a.zero_(); consume(draft_a, projection_a, sparse_a, workspace_a)
    comm.all_reduce(ar_input_a, out=out_a, registered=True)
    with torch.cuda.stream(shared_stream):
        shared(ar_input_a, shared_bufs_a)
        shared(ar_input_b, shared_bufs_b)
    with torch.cuda.stream(consumer_stream):
        draft_b.zero_(); consume(draft_b, projection_b, sparse_b, workspace_b)
    torch.cuda.synchronize(); dist.barrier()
    graph_a, outputs_a = capture(comm, baseline)
    graph_b, outputs_b = capture(comm, candidate)
    sync.zero_(); torch.cuda.synchronize(); dist.barrier()
    graph_a.replay(); graph_b.replay(); torch.cuda.synchronize()
    def compare_outputs():
        return [
            torch.equal(outputs_a[0], outputs_b[0]),
            torch.equal(outputs_a[1], outputs_b[1]),
            *[torch.equal(a, b) for a, b in zip(outputs_a[2], outputs_b[2])],
        ]

    exact = compare_outputs()
    exact_all = [None] * world
    dist.all_gather_object(exact_all, exact)
    if not all(all(v) for v in exact_all):
        local_debug = {
            "pre_ar_max": float(
                (ar_input_a.float() - ar_input_b.float()).abs().max().item()
            ),
            "output_max": float(
                (outputs_a[0].float() - outputs_b[0].float()).abs().max().item()
            ),
            "candidate_anchor_max": float(
                (
                    outputs_b[0].view(M, 4, H)[:, 0].float()
                    - outputs_a[0].view(M, 4, H)[:, 0].float()
                ).abs().max().item()
            ),
        }
        debug_all = [None] * world
        dist.all_gather_object(debug_all, local_debug)
        raise RuntimeError(f"boundary mismatch {exact_all} debug={debug_all}")

    mutation_delta = torch.sin(
        torch.arange(shared_x.numel(), dtype=torch.float32, device="cuda")
    ).view_as(shared_x).to(torch.bfloat16)
    mutation_failures = 0
    for mutation in range(args.mutations):
        alpha = ((mutation * 1543 + 17) % 2047 - 1023) / 32768.0
        shared_x.add_(mutation_delta, alpha=alpha)
        graph_a.replay(); graph_b.replay(); torch.cuda.synchronize()
        mutation_exact = compare_outputs()
        if mutation == 0 and not all(mutation_exact):
            mutation_debug = [
                float((a.float() - b.float()).abs().max())
                for a, b in zip(
                    (outputs_a[0], outputs_a[1], *outputs_a[2]),
                    (outputs_b[0], outputs_b[1], *outputs_b[2]),
                )
            ]
            print(
                f"rank={rank} first_mutation_exact={mutation_exact} "
                f"max_abs={mutation_debug}",
                flush=True,
            )
        mutation_failures += int(not all(mutation_exact))
    mutation_failures_all = [None] * world
    dist.all_gather_object(mutation_failures_all, mutation_failures)
    if any(mutation_failures_all):
        raise RuntimeError(f"mutation mismatch {mutation_failures_all}")

    graph_b.replay(); torch.cuda.synchronize()
    stable_b = tuple(tensor.clone() for tensor in outputs_b[2])
    for _ in range(args.graph_replays):
        graph_b.replay()
    torch.cuda.synchronize()
    replay_stable = all(
        torch.equal(expected, actual)
        for expected, actual in zip(stable_b, outputs_b[2])
    )
    replay_stable_all = [None] * world
    dist.all_gather_object(replay_stable_all, replay_stable)
    if not all(replay_stable_all):
        raise RuntimeError(f"graph replay instability {replay_stable_all}")

    for _ in range(args.warmup):
        graph_a.replay(); graph_b.replay()
    torch.cuda.synchronize()
    a1, b1, b2, a2 = [], [], [], []
    for _ in range(args.rounds):
        a1.append(rankmax_once(graph_a, args.iters, world))
        b1.append(rankmax_once(graph_b, args.iters, world))
        b2.append(rankmax_once(graph_b, args.iters, world))
        a2.append(rankmax_once(graph_a, args.iters, world))
    if rank == 0:
        av, bv = a1 + a2, b1 + b2
        am, bm = statistics.median(av), statistics.median(bv)
        print(f"route_blocks={experts.numel()} exact={exact_all}", flush=True)
        print(f"entry_mhc_only={args.entry_mhc_only}", flush=True)
        print(
            f"mutations={args.mutations} failures={mutation_failures_all} "
            f"graph_replays={args.graph_replays} stable={replay_stable_all}",
            flush=True,
        )
        print(f"A1_rankmax_us={[round(v,3) for v in a1]}", flush=True)
        print(f"B1_rankmax_us={[round(v,3) for v in b1]}", flush=True)
        print(f"B2_rankmax_us={[round(v,3) for v in b2]}", flush=True)
        print(f"A2_rankmax_us={[round(v,3) for v in a2]}", flush=True)
        print(
            f"baseline_us={am:.3f} progressive_us={bm:.3f} "
            f"saving_us={am-bm:.3f} gain_pct={(am-bm)/am*100:.3f} "
            f"gate_100us={'pass' if am-bm >= 100 else 'fail'}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
