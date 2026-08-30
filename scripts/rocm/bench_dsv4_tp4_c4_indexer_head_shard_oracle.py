#!/usr/bin/env python3
"""TP4 C4-indexer head-shard oracle (synthetic production shapes, no service)."""

from __future__ import annotations

import argparse
import os
import statistics

os.environ.setdefault("SGLANG_OPT_USE_TRITON_INDEXER_FULL", "1")

import torch
import torch.distributed as dist
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fnuz


B, H, D, K, PAGE, TOPK = 32, 64, 128, 1024, 64, 512


@triton.jit
def _local_logits_kernel(
    q_u8, kv_u8, weights, seq_lens, page_table, out,
    max_pages: tl.constexpr, max_len: tl.constexpr, heads: tl.constexpr,
    qs0: tl.constexpr, qsh: tl.constexpr, qsd: tl.constexpr,
    kvs: tl.constexpr, ws0: tl.constexpr, wsh: tl.constexpr,
    pts0: tl.constexpr, ptsp: tl.constexpr, os0: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    bid, block = tl.program_id(0), tl.program_id(1)
    s = block * BLOCK_S + tl.arange(0, BLOCK_S)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, 128)
    seq = tl.load(seq_lens + bid)
    po = s // 64
    pi = tl.load(page_table + bid * pts0 + po * ptsp, mask=po < max_pages, other=0)
    valid = (s < seq) & (po < max_pages)
    base = pi * kvs
    ko = base[:, None] + (s % 64)[:, None] * 128 + d[None, :]
    kb = tl.load(kv_u8 + ko, mask=valid[:, None], other=0)
    kval = kb.to(tl.float8e4b8, bitcast=True).to(tl.float16)
    qb = tl.load(q_u8 + bid * qs0 + h[:, None] * qsh + d[None, :] * qsd,
                 mask=(h < heads)[:, None], other=0)
    qval = qb.to(tl.float8e4b8, bitcast=True).to(tl.float16)
    scores = tl.dot(kval, tl.trans(qval)).to(tl.float32)
    w = tl.load(weights + bid * ws0 + h * wsh, mask=h < heads, other=0.0)
    reduced = tl.sum(tl.maximum(scores, 0.0) * w[None, :], axis=1)
    sb = base + 8192 + (s % 64) * 4
    b0 = tl.load(kv_u8 + sb + 0, mask=valid, other=0).to(tl.uint32)
    b1 = tl.load(kv_u8 + sb + 1, mask=valid, other=0).to(tl.uint32)
    b2 = tl.load(kv_u8 + sb + 2, mask=valid, other=0).to(tl.uint32)
    b3 = tl.load(kv_u8 + sb + 3, mask=valid, other=0).to(tl.uint32)
    scale = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.float32, bitcast=True)
    tl.store(out + bid * os0 + s, tl.where(valid, reduced * scale, 0.0),
             mask=s < max_len)


def index_logits(q, kv, weights, seq_lens, page_table, length):
    heads = q.shape[2]
    q_u8 = q.view(torch.uint8)
    kv_u8 = kv.view(-1, PAGE * (D + 4)).view(torch.uint8)
    out = torch.empty((B, length), device=q.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(heads)
    _local_logits_kernel[(B, triton.cdiv(length, 16))](
        q_u8, kv_u8, weights, seq_lens, page_table, out,
        page_table.shape[1], length, heads,
        q_u8.stride(0), q_u8.stride(2), q_u8.stride(3), kv_u8.stride(0),
        weights.stride(0), weights.stride(1), page_table.stride(0),
        page_table.stride(1), out.stride(0), BLOCK_S=16, BLOCK_H=block_h,
        num_warps=4,
    )
    return out


def make_kv(length: int, device: torch.device):
    pages_per_req = (length + PAGE - 1) // PAGE
    total_pages = B * pages_per_req
    values = torch.randn(total_pages, PAGE, D, device=device, dtype=torch.bfloat16)
    values = values.to(FP8_DTYPE)
    scales = torch.rand(total_pages, PAGE, device=device, dtype=torch.float32) + 0.5
    packed = torch.empty(total_pages, PAGE * (D + 4), device=device, dtype=torch.uint8)
    packed[:, : PAGE * D].copy_(values.view(torch.uint8).reshape(total_pages, -1))
    packed[:, PAGE * D :].copy_(scales.view(torch.uint8).reshape(total_pages, -1))
    kv = packed.view(FP8_DTYPE).view(total_pages, PAGE, 1, D + 4)
    table = torch.arange(total_pages, device=device, dtype=torch.int32).reshape(
        B, pages_per_req
    )
    lens = torch.full((B,), length, device=device, dtype=torch.int32)
    return kv, table, lens


def rank_max_time_us(fn, warmup: int, inner: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    end.synchronize()
    value = torch.tensor(
        [start.elapsed_time(end) * 1000.0 / inner], device="cuda"
    )
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def set_overlap(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float, int]:
    rows = []
    for i in range(a.shape[0]):
        rows.append(len(set(a[i].cpu().tolist()) & set(b[i].cpu().tolist())))
    return sum(x == TOPK for x in rows), statistics.mean(rows), min(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="513,640")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--inner", type=int, default=20)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"oracle requires TP4, got world={world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(20260830)

    q_lora = torch.randn(B, K, device=device, dtype=torch.bfloat16)
    # Replicated synthetic wq_b, matching the production [8192,1024] shape.
    wq_b = torch.randn(H * D, K, device=device, dtype=torch.bfloat16) * 0.02
    head_weight = torch.rand(B, H, device=device, dtype=torch.float32) + 0.25
    h0, h1 = rank * (H // world), (rank + 1) * (H // world)
    w_local = wq_b[h0 * D : h1 * D].contiguous()
    weight_local = head_weight[:, h0:h1].contiguous()
    q_full_bf16 = torch.empty(B, H * D, device=device, dtype=torch.bfloat16)
    q_local_bf16 = torch.empty(B, (H // world) * D, device=device, dtype=torch.bfloat16)
    q_full = torch.empty(B, 1, H, D, device=device, dtype=FP8_DTYPE)
    q_local = torch.empty(B, 1, H // world, D, device=device, dtype=FP8_DTYPE)

    def projection_a():
        torch.mm(q_lora, wq_b.t(), out=q_full_bf16)
        q_full.copy_(q_full_bf16.view(B, 1, H, D))

    def projection_b():
        torch.mm(q_lora, w_local.t(), out=q_local_bf16)
        q_local.copy_(q_local_bf16.view(B, 1, H // world, D))

    projection_a()
    projection_b()

    for length in [int(v) for v in args.lengths.split(",")]:
        kv, page_table, seq_lens = make_kv(length, device)
        def logits_a():
            return index_logits(q_full, kv, head_weight, seq_lens, page_table, length)

        def logits_b_local():
            return index_logits(q_local, kv, weight_local, seq_lens, page_table, length)

        score_a = logits_a()
        score_b = logits_b_local()
        dist.all_reduce(score_b, op=dist.ReduceOp.SUM)
        raw_a = torch.topk(score_a, TOPK, dim=1, sorted=False).indices
        raw_b = torch.topk(score_b, TOPK, dim=1, sorted=False).indices
        torch.cuda.synchronize()
        diff = (score_a - score_b).abs()
        rel_l2 = float(torch.linalg.vector_norm(score_a - score_b) /
                       torch.linalg.vector_norm(score_a).clamp_min(1e-12))
        exact_rows, mean_overlap, min_overlap = set_overlap(raw_a, raw_b)
        if rank == 0:
            print(
                f"CORRECTNESS L={length} score_max_abs={float(diff.max()):.8g} "
                f"score_mean_abs={float(diff.mean()):.8g} rel_l2={rel_l2:.8g} "
                f"topk_exact_rows={exact_rows}/{B} topk_mean_overlap={mean_overlap:.3f} "
                f"topk_min_overlap={min_overlap}", flush=True,
            )

        collective_buf = torch.zeros_like(score_b)

        def collective_only():
            dist.all_reduce(collective_buf, op=dist.ReduceOp.SUM)

        def topk_a():
            return torch.topk(score_a, TOPK, dim=1, sorted=False).indices

        def topk_b():
            return torch.topk(score_b, TOPK, dim=1, sorted=False).indices

        def full_a():
            projection_a()
            s = logits_a()
            return torch.topk(s, TOPK, dim=1, sorted=False).indices

        def full_b():
            projection_b()
            s = logits_b_local()
            dist.all_reduce(s, op=dist.ReduceOp.SUM)
            return torch.topk(s, TOPK, dim=1, sorted=False).indices

        fns = {
            "proj_full64": projection_a,
            "proj_local16": projection_b,
            "logits_full64": logits_a,
            "logits_local16": logits_b_local,
            "score_ar": collective_only,
            "topk_full": topk_a,
            "topk_shard": topk_b,
            "A_full": full_a,
            "B_shard_ar": full_b,
        }
        samples = {name: [] for name in fns}
        order = list(fns)
        for _ in range(args.rounds):
            for name in order + list(reversed(order)):
                samples[name].append(
                    rank_max_time_us(fns[name], args.warmup, args.inner)
                )
        if rank == 0:
            for name, values in samples.items():
                trimmed = sorted(values)[1:-1]
                print(
                    f"RESULT L={length} profile={name} samples_us="
                    + ",".join(f"{v:.3f}" for v in values)
                    + f" median_us={statistics.median(values):.3f}"
                    + f" trimmed_mean_us={statistics.mean(trimmed):.3f}",
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
