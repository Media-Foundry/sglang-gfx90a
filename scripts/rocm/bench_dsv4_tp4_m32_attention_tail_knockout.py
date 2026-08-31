#!/usr/bin/env python3
"""TP4 attention-tail graph knockout oracle for M32 or M64.

A: inverse RoPE + wo_a + wo_b + registered all-reduce
B: exact pre-generated wo_a output alias + wo_b + all-reduce
C: exact pre-generated wo_b partial alias + all-reduce
D: exact reduced-output alias, empty graph (graph launch floor)

The knockout aliases are prepared outside capture/replay.  No candidate graph
contains a copy used to simulate removed work.
"""

from __future__ import annotations

import argparse
import os
import statistics

import aiter as aiter_ops
import torch
import torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


M, G, D, R, H, WORLD = 32, 2, 4096, 1024, 4096, 4
ROPE = 64


def capture(comm, fn):
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with comm.capture():
        with torch.cuda.graph(graph):
            outputs = fn()
    dist.barrier()
    return graph, outputs


def rankmax(graph, iters, world):
    dist.barrier()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        graph.replay()
    b.record(); b.synchronize()
    local = a.elapsed_time(b) * 1000.0 / iters
    values = [None] * world
    dist.all_gather_object(values, local)
    return max(float(x) for x in values)


def trim(xs):
    return statistics.mean(sorted(xs)[1:-1])


def main():
    global M

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", default="/tmp/dsv4_ffn_dump.f3ZQ89")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--mutations", type=int, default=100)
    p.add_argument("--replays", type=int, default=1000)
    p.add_argument(
        "--tokens",
        type=int,
        choices=(32, 64, 128),
        default=32,
        help="decode graph token tier to measure",
    )
    args = p.parse_args()
    M = args.tokens

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != WORLD:
        raise RuntimeError(f"requires TP4, got {world}")
    comm = CustomAllreduce(dist.group.WORLD, torch.device("cuda", local_rank))
    if comm.disabled:
        raise RuntimeError("AIter CustomAllreduce unavailable")

    # Build a TP4-shaped real activation from adjacent TP8 dump ranks.
    p0 = os.path.join(args.dump, f"layer_20_rank_{2*rank}_attn_inverse_rope.pt")
    p1 = os.path.join(args.dump, f"layer_20_rank_{2*rank+1}_attn_inverse_rope.pt")
    if os.path.exists(p0) and os.path.exists(p1):
        x0 = torch.load(p0, map_location="cpu", weights_only=False)
        x1 = torch.load(p1, map_location="cpu", weights_only=False)
        host = torch.cat((x0, x1), dim=1).reshape(-1, G, D)
        if host.shape[0] < M:
            repeats = (M + host.shape[0] - 1) // host.shape[0]
            host = host.repeat(repeats, 1, 1)
        host = host[:M].contiguous()
        source = f"real_tp8_pair_tiled_to_m{M}"
    else:
        gen = torch.Generator().manual_seed(20260830 + rank)
        host = torch.randn((M, G, D), generator=gen, dtype=torch.bfloat16)
        source = "deterministic_fallback"
    x = host.cuda().contiguous()
    base = x.clone()
    mutation = torch.linspace(-1, 1, x.numel(), device="cuda").view_as(x).to(torch.bfloat16)

    gen = torch.Generator(device="cuda").manual_seed(31000 + rank)
    wa = torch.randn((G, R, D), generator=gen, device="cuda", dtype=torch.bfloat16).mul_(1/64)
    wb = torch.randn((H, G*R), generator=gen, device="cuda", dtype=torch.bfloat16).mul_(1/64)
    theta = torch.linspace(0.001, 0.25, ROPE // 2, device="cuda", dtype=torch.float32)
    cos, sin = theta.cos().to(torch.bfloat16), theta.sin().neg().to(torch.bfloat16)

    nbytes = M * H * torch.bfloat16.itemsize
    storages = [aiter_ops.allocate_meta_buffer(nbytes) for _ in range(6)]
    partial_a, partial_b, alias_partial, reduced_a, reduced_b, reduced_c = [
        s.view(torch.bfloat16).view(M, H) for s in storages
    ]
    for s in storages:
        comm.register_buffer(s)

    alias_mid = torch.empty((M, G, R), device="cuda", dtype=torch.bfloat16)
    alias_inv = torch.empty((M, G, D), device="cuda", dtype=torch.bfloat16)
    alias_reduced = torch.empty((M, H), device="cuda", dtype=torch.bfloat16)
    whole_wob = torch.empty((M, H), device="cuda", dtype=torch.bfloat16)
    chunked_wob = torch.empty((M, H), device="cuda", dtype=torch.bfloat16)

    def inverse(inp):
        nope = inp[..., :-ROPE]
        tail = inp[..., -ROPE:].reshape(M, G, ROPE // 2, 2)
        even, odd = tail[..., 0], tail[..., 1]
        rot = torch.stack((even*cos - odd*sin, odd*cos + even*sin), dim=-1)
        return torch.cat((nope, rot.flatten(-2)), dim=-1)

    def prepare_aliases():
        inv = inverse(x)
        # Alias preparation is deliberately outside every captured graph.
        alias_inv.copy_(inv)
        alias_mid.copy_(torch.einsum("mgd,grd->mgr", inv, wa))
        torch.mm(alias_mid.flatten(1), wb.t(), out=alias_partial)
        comm.all_reduce(alias_partial, out=alias_reduced, registered=True)

    def case_a():
        inv = inverse(x)
        mid = torch.einsum("mgd,grd->mgr", inv, wa)
        torch.mm(mid.flatten(1), wb.t(), out=partial_a)
        comm.all_reduce(partial_a, out=reduced_a, registered=True)
        return mid, partial_a, reduced_a

    def case_b():
        torch.mm(alias_mid.flatten(1), wb.t(), out=partial_b)
        comm.all_reduce(partial_b, out=reduced_b, registered=True)
        return partial_b, reduced_b

    def case_c():
        comm.all_reduce(alias_partial, out=reduced_c, registered=True)
        return reduced_c

    def case_d():
        return alias_reduced

    def case_e():
        torch.mm(alias_mid.flatten(1), wb.t(), out=whole_wob)
        return whole_wob

    def case_f():
        # A two-stage wo_b -> all-reduce pipeline first needs row-chunked wo_b
        # to remain cheap.  Measure its real producer cost before implementing
        # any new collective protocol.
        half = M // 2
        flat = alias_mid.flatten(1)
        torch.mm(flat[:half], wb.t(), out=chunked_wob[:half])
        torch.mm(flat[half:], wb.t(), out=chunked_wob[half:])
        return chunked_wob

    def case_g():
        return torch.einsum("mgd,grd->mgr", alias_inv, wa)

    def case_h():
        return torch.bmm(
            alias_inv.transpose(0, 1), wa.transpose(1, 2)
        ).transpose(0, 1)

    prepare_aliases(); torch.cuda.synchronize()
    ga, oa = capture(comm, case_a)
    gb, ob = capture(comm, case_b)
    gc, oc = capture(comm, case_c)
    gd, od = capture(comm, case_d)
    ge, oe = capture(comm, case_e)
    gf, of = capture(comm, case_f)
    gg, og = capture(comm, case_g)
    gh, oh = capture(comm, case_h)

    def check(label):
        checks = {
            "mid": torch.equal(oa[0], alias_mid),
            "partial_ab": torch.equal(oa[1], ob[0]),
            "partial_ac": torch.equal(oa[1], alias_partial),
            "reduced_ab": torch.equal(oa[2], ob[1]),
            "reduced_ac": torch.equal(oa[2], oc),
            "reduced_ad": torch.equal(oa[2], od),
            "whole_wob": torch.equal(oe, alias_partial),
            "einsum_woa": torch.equal(og, alias_mid),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{label} mismatch {checks}")

    for i in range(args.mutations):
        x.copy_(base).add_(mutation, alpha=((i*1543+17)%2047-1023)/32768.0)
        prepare_aliases()
        ga.replay(); gb.replay(); gc.replay(); gd.replay(); ge.replay(); gf.replay()
        gg.replay(); gh.replay()
        torch.cuda.synchronize()
        check(f"mutation={i}")
    if rank == 0:
        print(
            f"CORRECTNESS tokens={M} source={source} "
            f"mutations={args.mutations} all_exact=True",
            flush=True,
        )

    prepare_aliases()
    ga.replay(); gb.replay(); gc.replay(); gd.replay(); ge.replay(); gf.replay()
    gg.replay(); gh.replay()
    torch.cuda.synchronize(); check("fixed")
    chunk_exact = torch.equal(of, alias_partial)
    chunk_max_abs = float((of.float() - alias_partial.float()).abs().max().item())
    bmm_exact = torch.equal(oh, alias_mid)
    bmm_max_abs = float((oh.float() - alias_mid.float()).abs().max().item())
    for i in range(args.replays):
        ga.replay(); gb.replay(); gc.replay(); gd.replay(); ge.replay(); gf.replay()
        gg.replay(); gh.replay()
        if (i + 1) % 100 == 0:
            torch.cuda.synchronize(); check(f"replay={i+1}")
    if rank == 0:
        print(f"CORRECTNESS graph_replays={args.replays} all_exact=True graph_copy=False", flush=True)

    for _ in range(args.warmup):
        ga.replay(); gb.replay(); gc.replay(); gd.replay()
    torch.cuda.synchronize()
    vals = {k: [] for k in "ABCDEFGH"}
    graphs = {
        "A": ga, "B": gb, "C": gc, "D": gd,
        "E": ge, "F": gf, "G": gg, "H": gh,
    }
    for _ in range(args.rounds):
        for name in (
            "A", "B", "C", "D", "E", "F", "G", "H",
            "H", "G", "F", "E", "D", "C", "B", "A",
        ):
            vals[name].append(rankmax(graphs[name], args.iters, world))
    if rank == 0:
        t = {k: trim(v) for k, v in vals.items()}
        for k in "ABCDEFGH":
            print(f"RESULT profile={k} samples_us={','.join(f'{x:.3f}' for x in vals[k])} trimmed_rankmax_us={t[k]:.3f}")
        print(
            f"KNOCKOUT inverse_plus_woa_gross_us={t['A']-t['B']:.3f} "
            f"wob_gross_us={t['B']-t['C']:.3f} ar_gross_us={t['C']-t['D']:.3f} "
            f"launch_floor_us={t['D']:.3f} total_tail_us={t['A']-t['D']:.3f}",
            flush=True,
        )
        print(
            f"ROW_CHUNK whole_wob_us={t['E']:.3f} "
            f"two_half_wob_us={t['F']:.3f} penalty_us={t['F']-t['E']:.3f} "
            f"exact={chunk_exact} max_abs={chunk_max_abs:.8f}",
            flush=True,
        )
        print(
            f"WOA_LAYOUT einsum_us={t['G']:.3f} bmm_us={t['H']:.3f} "
            f"delta_us={t['H']-t['G']:.3f} exact={bmm_exact} "
            f"max_abs={bmm_max_abs:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
