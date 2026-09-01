#!/usr/bin/env python3
"""Optimistic single-GCD TP4/M32 attention--MoE overlap oracle.

This is deliberately disconnected from production.  It replays the current
production kernels and shapes over real layer-20 M32 activations/weights, but
excludes TP collectives and dynamic paged-KV metadata.  Consequently it is an
optimistic continuation gate for a two-microbatch decode wavefront: failure to
overlap here closes the more expensive four-rank implementation.
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)
from sglang.kernels.ops.attention.dsv4.gfx90a_unified_sparse_decode import (
    run as run_ck_sparse,
    workspace_size_bytes as ck_workspace_size_bytes,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8
from sglang.srt.runtime_context import get_parallel


E, M, TOPK, H, I, N = 256, 32, 6, 4096, 512, 4096
A, R, W, LDS = 4, 2, 8, 2


def require_physical_gpu(expected: int) -> None:
    visible = os.getenv("HIP_VISIBLE_DEVICES") or os.getenv("ROCR_VISIBLE_DEVICES")
    if visible != str(expected):
        raise RuntimeError(f"set HIP_VISIBLE_DEVICES={expected} (one physical GPU only)")
    if not torch.version.hip:
        raise RuntimeError("ROCm PyTorch required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"expected gfx90a, got {arch}")


def dequant_block_fp8(handle, weight: str, scale: str, rows: slice, cols: slice):
    block = 128
    w = handle.get_slice(weight)[rows, cols]
    r0, r1 = rows.start or 0, rows.stop
    c0, c1 = cols.start or 0, cols.stop
    s = handle.get_tensor(scale)[r0 // block : r1 // block, c0 // block : c1 // block].float()
    return (w.float() * s.repeat_interleave(block, 0).repeat_interleave(block, 1)).bfloat16().contiguous()


def reconstruct_topk(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.int64).cpu()
    rows: list[list[int]] = [[] for _ in range(M)]
    for expert in torch.argsort(counts, descending=True).tolist():
        for _ in range(int(counts[expert])):
            choices = [t for t in range(M) if len(rows[t]) < TOPK and expert not in rows[t]]
            token = min(choices, key=lambda t: (len(rows[t]), t))
            rows[token].append(expert)
    result = torch.tensor(rows, dtype=torch.int32)
    if result.shape != (M, TOPK):
        raise RuntimeError(f"invalid reconstructed route {result.shape}")
    return result


def metadata(topk: torch.Tensor):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(topk.cpu().tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    ids, experts = [], []
    sentinel = (TOPK << 24) | M
    for expert, bucket in enumerate(buckets):
        for off in range(0, len(bucket), A):
            chunk = bucket[off : off + A]
            ids.extend(chunk + [sentinel] * (A - len(chunk)))
            experts.append(expert)
    dev = topk.device
    return (
        torch.tensor(ids, dtype=torch.int32, device=dev),
        torch.tensor(experts, dtype=torch.int32, device=dev),
    )


def timed(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record(); end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def trim(values: list[float]) -> float:
    ordered = sorted(values)
    return statistics.mean(ordered[1:-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, default=Path("/tmp/dsv4_ffn_dump.f3ZQ89"))
    p.add_argument("--recorder", type=Path, default=Path("/tmp/expert_distribution_recorder_1787803355.1855972.pt"))
    p.add_argument("--model-shard", type=Path, default=Path("/home/pc/models/modelscope/model-00022-of-00048.safetensors"))
    p.add_argument("--pass-index", type=int, default=37)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--recorder-ranks", type=int, default=8)
    p.add_argument("--context", type=int, default=256)
    p.add_argument(
        "--attention-tokens", type=int, choices=(32, 96, 128), default=32
    )
    p.add_argument("--moe-routed-only", action="store_true")
    p.add_argument(
        "--shared-tokens", type=int, choices=(32, 96, 128), default=32
    )
    p.add_argument(
        "--shared-with-attention",
        action="store_true",
        help="put the draft-lane shared expert before attention on the same stream",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="use fixed random activations/projection weights/routes when old dumps are unavailable",
    )
    p.add_argument("--physical-gpu", type=int, default=0)
    p.add_argument(
        "--ck-m32",
        action="store_true",
        help="oracle-only: run the CK sparse kernel for the M32 attention arm",
    )
    p.add_argument(
        "--anchor-attention-before-moe",
        action="store_true",
        help="prepend a CK M32 attention chain to the routed-M32 lane",
    )
    p.add_argument(
        "--production-baseline",
        action="store_true",
        help="also time full attention followed by routed/shared internal overlap",
    )
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    args = p.parse_args()
    if args.moe_routed_only and args.shared_with_attention:
        raise RuntimeError("shared-with-attention requires the shared expert")
    require_physical_gpu(args.physical_gpu)
    # The paged-decode selector reads live topology even in a standalone
    # oracle.  Use its documented test-only override instead of initializing a
    # fake distributed group.
    topology = get_parallel().override(attn_tp_size=4, attn_tp_rank=0)
    topology.__enter__()
    dev = torch.device("cuda")

    def dump(rank: int, name: str):
        return torch.load(args.dump / f"layer_{args.layer}_rank_{rank}_{name}.pt", map_location="cpu", weights_only=True)

    torch.manual_seed(20260901)
    x = (
        torch.randn((M, H), dtype=torch.bfloat16, device=dev)
        if args.synthetic
        else dump(0, "attn_norm").to(dev)
    )
    if args.attention_tokens % M != 0:
        raise RuntimeError("attention tokens must be a multiple of the M32 dump")
    attn_m = args.attention_tokens
    attn_x = x.repeat(attn_m // M, 1).contiguous()
    projection_specs = (
        ("projection_wqkv_a", 1536),
        ("projection_core_compressor", 2048),
        ("projection_index_compressor", 512),
        ("projection_index_weights", 64),
    )
    projections = [
        (
            torch.randn((n, H), dtype=torch.bfloat16, device=dev)
            if args.synthetic
            else dump(0, name).to(dev)
        )
        for name, n in projection_specs
    ]
    projection_out = [torch.empty((attn_m, w.shape[0]), dtype=torch.bfloat16, device=dev) for w in projections]
    anchor_projection_out = [
        torch.empty((M, w.shape[0]), dtype=torch.bfloat16, device=dev)
        for w in projections
    ]
    q = (
        torch.randn((M, 16, 512), dtype=torch.bfloat16, device=dev)
        if args.synthetic
        else torch.cat((dump(0, "q"), dump(1, "q")), dim=1).to(dev)
    )
    q = q.repeat(attn_m // M, 1, 1).contiguous()
    inv = (
        torch.randn((M, 2, H), dtype=torch.bfloat16, device=dev)
        if args.synthetic
        else torch.cat(
            (dump(0, "attn_inverse_rope"), dump(1, "attn_inverse_rope")),
            dim=1,
        ).to(dev)
    )
    inv = inv.repeat(attn_m // M, 1, 1).contiguous()

    prefix = f"layers.{args.layer}"
    with safe_open(args.model_shard, framework="pt", device="cpu") as f:
        wo_a = dequant_block_fp8(f, f"{prefix}.attn.wo_a.weight", f"{prefix}.attn.wo_a.scale", slice(0, 2048), slice(0, H))
        wo_b = dequant_block_fp8(f, f"{prefix}.attn.wo_b.weight", f"{prefix}.attn.wo_b.scale", slice(0, H), slice(0, 2048))
        sg = dequant_block_fp8(f, f"{prefix}.ffn.shared_experts.w1.weight", f"{prefix}.ffn.shared_experts.w1.scale", slice(0, 512), slice(0, H))
        su = dequant_block_fp8(f, f"{prefix}.ffn.shared_experts.w3.weight", f"{prefix}.ffn.shared_experts.w3.scale", slice(0, 512), slice(0, H))
        sd = dequant_block_fp8(f, f"{prefix}.ffn.shared_experts.w2.weight", f"{prefix}.ffn.shared_experts.w2.scale", slice(0, H), slice(0, 512))
    wo_a, wo_b, sg, su, sd = (t.to(dev) for t in (wo_a, wo_b, sg, su, sd))
    mid_attn = torch.empty((attn_m, 2048), dtype=torch.bfloat16, device=dev)
    out_attn = torch.empty((attn_m, H), dtype=torch.bfloat16, device=dev)
    sparse_out = torch.empty_like(q)
    anchor_sparse_out = torch.empty_like(q[:M])
    ck_workspace = torch.empty(
        ck_workspace_size_bytes(tokens=attn_m), dtype=torch.uint8, device=dev
    )
    anchor_ck_workspace = torch.empty(
        ck_workspace_size_bytes(tokens=M), dtype=torch.uint8, device=dev
    )
    kv = torch.randn((attn_m * args.context, 512), dtype=torch.bfloat16, device=dev)
    indices = torch.arange(attn_m * args.context, dtype=torch.int32, device=dev)
    indptr = torch.arange(0, (attn_m + 1) * args.context, args.context, dtype=torch.int32, device=dev)
    sink = torch.randn((16,), dtype=torch.float32, device=dev)
    anchor_mid_attn = torch.empty((M, 2048), dtype=torch.bfloat16, device=dev)
    anchor_out_attn = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    if args.synthetic:
        topk = torch.topk(
            torch.rand((M, E), device=dev), TOPK, dim=1, sorted=False
        ).indices.to(torch.int32)
    else:
        payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
        raw = payload["logical_count"][args.pass_index, args.layer]
        if torch.any(raw.remainder(args.recorder_ranks) != 0):
            raise RuntimeError("recorder counts are not replicated across requested ranks")
        counts = raw // args.recorder_ranks
        if int(counts.sum()) != M * TOPK:
            raise RuntimeError(f"expected {M * TOPK} M32 assignments, got {int(counts.sum())}")
        topk = reconstruct_topk(counts).to(dev)
    ids, experts = metadata(topk)
    valid = torch.tensor([ids.numel(), M], dtype=torch.int32, device=dev)
    torch.manual_seed(20260830)
    moe_x = (
        torch.randn((M, H), dtype=torch.bfloat16, device=dev)
        if args.synthetic
        else dump(0, "ffn_mhc_residual")[:, 0].contiguous().to(dev)
    )
    xq, xs = per_token_group_quant_int8(moe_x, 32)
    tw = torch.rand((M, TOPK), dtype=torch.float32, device=dev)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=dev)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=dev)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=dev)
    inter = torch.empty((M, TOPK, I), dtype=torch.bfloat16, device=dev)
    partial = torch.empty((M, TOPK, N), dtype=torch.float32, device=dev)
    routed_out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    shared_x = moe_x.repeat(args.shared_tokens // M, 1).contiguous()
    shared_gate = torch.empty(
        (args.shared_tokens, 512), dtype=torch.bfloat16, device=dev
    )
    shared_up = torch.empty_like(shared_gate)
    shared_mid = torch.empty_like(shared_gate)
    shared_out = torch.empty(
        (args.shared_tokens, H), dtype=torch.bfloat16, device=dev
    )
    gate = _jit_gate_up_grouped(E, M, TOPK, I, H, A, R, W, 2080, LDS)
    down = _jit_down_grouped(E, M, TOPK, N, I, A, R, W, 832, LDS)

    def shared():
        torch.mm(shared_x, sg.t(), out=shared_gate)
        torch.mm(shared_x, su.t(), out=shared_up)
        torch.sigmoid(shared_gate, out=shared_mid)
        shared_mid.mul_(shared_gate).mul_(shared_up.clamp(-10.0, 10.0))
        torch.mm(shared_mid, sd.t(), out=shared_out)

    def attention():
        if args.shared_with_attention:
            shared()
        for w, out in zip(projections, projection_out):
            torch.mm(attn_x, w.t(), out=out)
        if attn_m in (96, 128) or args.ck_m32:
            run_ck_sparse(
                q,
                kv,
                indices,
                indptr,
                sink,
                sparse_out,
                ck_workspace,
                1.0 / (512**0.5),
            )
        else:
            _sparse_attn_v4_paged_decode_triton(q, kv, indices, indptr, sink, 1.0 / (512 ** 0.5), block_h=16, kv_splits=4, block_k=16)
        # M32 intentionally falls through the wave64 GEMV selector (M<=8)
        # to this production einsum/GEMM spelling.
        grouped = torch.einsum(
            "tgd,grd->tgr", inv.reshape(attn_m, 2, 4096), wo_a.reshape(2, 1024, 4096)
        )
        mid_attn.copy_(grouped.flatten(1))
        torch.mm(mid_attn, wo_b.t(), out=out_attn)

    def anchor_attention():
        for w, out in zip(projections, anchor_projection_out):
            torch.mm(attn_x[:M], w.t(), out=out)
        run_ck_sparse(
            q[:M],
            kv[: M * args.context],
            indices[: M * args.context],
            indptr[: M + 1],
            sink,
            anchor_sparse_out,
            anchor_ck_workspace,
            1.0 / (512**0.5),
        )
        grouped = torch.einsum(
            "tgd,grd->tgr",
            inv[:M].reshape(M, 2, 4096),
            wo_a.reshape(2, 1024, 4096),
        )
        anchor_mid_attn.copy_(grouped.flatten(1))
        torch.mm(anchor_mid_attn, wo_b.t(), out=anchor_out_attn)

    def routed():
        gate.run(xq, xs, w13, s13, ids, experts, valid, inter, 10.0)
        iq, isc = per_token_group_quant_int8(inter, 32)
        down.run_partial(iq, isc, w2, s2, ids, experts, valid, tw, partial)
        down.reduce(partial, routed_out)

    def moe():
        if args.anchor_attention_before_moe:
            anchor_attention()
        routed()
        if not args.moe_routed_only and not args.shared_with_attention:
            shared()

    main_stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    done = torch.cuda.Event()

    def serial():
        attention(); moe()

    def overlap():
        side.wait_stream(main_stream)
        with torch.cuda.stream(side):
            moe(); done.record(side)
        attention()
        main_stream.wait_event(done)

    def production_baseline():
        attention()
        side.wait_stream(main_stream)
        with torch.cuda.stream(side):
            shared(); done.record(side)
        routed()
        main_stream.wait_event(done)

    serial(); torch.cuda.synchronize()
    refs = [out_attn.clone(), routed_out.clone()]
    if args.anchor_attention_before_moe:
        refs.append(anchor_out_attn.clone())
    if not args.moe_routed_only:
        refs.append(shared_out.clone())
    overlap(); torch.cuda.synchronize()
    outputs = [out_attn, routed_out]
    if args.anchor_attention_before_moe:
        outputs.append(anchor_out_attn)
    if not args.moe_routed_only:
        outputs.append(shared_out)
    exact = [torch.equal(a, b) for a, b in zip(refs, outputs)]
    if not all(exact):
        raise RuntimeError(f"overlap output race/mismatch exact={exact}")

    attn_samples, moe_samples, serial_samples, overlap_samples = [], [], [], []
    for _ in range(args.rounds):
        attn_samples.append(timed(attention, args.warmup, args.iterations))
        moe_samples.append(timed(moe, args.warmup, args.iterations))
        serial_samples.append(timed(serial, args.warmup, args.iterations))
        overlap_samples.append(timed(overlap, args.warmup, args.iterations))
        overlap_samples.append(timed(overlap, args.warmup, args.iterations))
        serial_samples.append(timed(serial, args.warmup, args.iterations))
    a, m, s, o = map(trim, (attn_samples, moe_samples, serial_samples, overlap_samples))
    production_samples = []
    if args.production_baseline:
        production_baseline(); torch.cuda.synchronize()
        production_samples = [
            timed(production_baseline, args.warmup, args.iterations)
            for _ in range(args.rounds)
        ]
    print(
        f"route_blocks={experts.numel()} context={args.context} "
        f"attention_tokens={attn_m} shared_tokens={args.shared_tokens} "
        f"routed_only={args.moe_routed_only} exact={exact}"
    )
    print(f"attention_us={a:.3f} moe_us={m:.3f} serial_us={s:.3f} overlap_us={o:.3f}")
    print(f"saved_pct={(s-o)/s*100:.3f} ideal_us={max(a,m):.3f} overlap_efficiency={(s-o)/(s-max(a,m))*100:.3f}")
    if production_samples:
        print(
            f"production_baseline_us={trim(production_samples):.3f} "
            f"samples={[round(v,3) for v in production_samples]}"
        )
    print(f"continue={o <= 0.8*s}")


if __name__ == "__main__":
    main()
